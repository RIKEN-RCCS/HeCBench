#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>

// ---- Transpose helper (host) -----------------------------------------------
static double *transpose(const double *idata, const int width, const int height) {
  double *odata = (double *)malloc(sizeof(double) * width * height);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      odata[y + height * x] = idata[x + width * y];
  return odata;
}

// ---- Kernel 1: compute_probs ----------------------------------------------
// One thread per i (row); non-unit stride access
static void compute_probs(Kokkos::View<double *> d_alphas,
                          Kokkos::View<double *> d_rands,
                          Kokkos::View<double *> d_probs,
                          int n, int K, int M, int repeat) {
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "compute_probs",
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const int i) {
          double w[21];
          double M_d = (double)M;
          for (int k = 0; k < K; k++) d_probs(i * K + k) = 0.0;

          for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++)
              w[k] = d_alphas(i * K + k) + d_rands(m * K + k);

            int    maxind = K - 1;
            double maxval = w[K - 1];
            for (int k = 0; k < K - 1; k++)
              if (w[k] > maxval) { maxind = k; maxval = w[k]; }
            d_probs(i * K + maxind) += 1.0;
          }
          for (int k = 0; k < K; k++) d_probs(i * K + k) /= M_d;
        });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
}

// ---- Kernel 2: compute_probs_unitStrides ----------------------------------
// Transposed storage for unit-stride inner loop access
static void compute_probs_unitStrides(Kokkos::View<double *> d_alphas,
                                      Kokkos::View<double *> d_rands,
                                      Kokkos::View<double *> d_probs,
                                      int n, int K, int M, int repeat) {
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "compute_probs_unitStrides",
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const int i) {
          double w[21];
          double M_d = (double)M;
          for (int k = 0; k < K; k++) d_probs(k * n + i) = 0.0;

          for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++)
              w[k] = d_alphas(k * n + i) + d_rands(k * M + m);

            int    maxind = K - 1;
            double maxval = w[K - 1];
            for (int k = 0; k < K - 1; k++)
              if (w[k] > maxval) { maxind = k; maxval = w[k]; }
            d_probs(maxind * n + i) += 1.0;
          }
          for (int k = 0; k < K; k++) d_probs(k * n + i) /= M_d;
        });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
}

// ---- Kernel 3: compute_probs_unitStrides_sharedMem ------------------------
// Team-based with scratch memory; threads = 96, shared = 2*K*threads doubles
static void compute_probs_unitStrides_sharedMem(
    Kokkos::View<double *> d_alphas, Kokkos::View<double *> d_rands,
    Kokkos::View<double *> d_probs,
    int n, int K, int M, int threads_per_block, int num_blocks, int repeat) {

  // Scratch per team: 2 * K * threads_per_block doubles
  const int scratch_bytes = 2 * K * threads_per_block * (int)sizeof(double);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<double *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(num_blocks, threads_per_block, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "compute_probs_sharedMem", policy,
        KOKKOS_LAMBDA(const member_type &team) {
          int tidx = team.team_rank();
          int i    = team.league_rank() * threads_per_block + tidx;

          ScratchView sdata(team.team_scratch(0),
                            2 * K * threads_per_block);
          double *probs_shared = sdata.data();
          double *w            = probs_shared + K * threads_per_block;

          if (i < n) {
            double M_d = (double)M;

            for (int k = 0; k < K; k++)
              probs_shared[k * threads_per_block + tidx] = 0.0;

            for (int m = 0; m < M; m++) {
              for (int k = 0; k < K; k++)
                w[k * threads_per_block + tidx] =
                    d_alphas(k * n + i) + d_rands(k * M + m);

              int    maxind = K - 1;
              double maxval = w[(K - 1) * threads_per_block + tidx];
              for (int k = 0; k < K - 1; k++)
                if (w[k * threads_per_block + tidx] > maxval) {
                  maxind = k;
                  maxval = w[k * threads_per_block + tidx];
                }
              probs_shared[maxind * threads_per_block + tidx] += 1.0;
            }

            for (int k = 0; k < K; k++) {
              probs_shared[k * threads_per_block + tidx] /= M_d;
              d_probs(k * n + i) = probs_shared[k * threads_per_block + tidx];
            }
          }
        });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
}

// ---- Main -----------------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <path to filename> <repeat>\n", argv[0]);
    return 1;
  }
  char      *filename = argv[1];
  const int  repeat   = atoi(argv[2]);

  const int n = 26280, K = 21, M = 10000;
  const int alphas_size      = n * K;
  const int rands_size       = M * K;
  const int alphas_size_byte = alphas_size * sizeof(double);
  const int rands_size_byte  = rands_size  * sizeof(double);

  FILE *fp = fopen(filename, "r");
  if (!fp) { printf("Error: failed to open file %s\n", filename); return 1; }

  double *alphas = (double *)malloc(alphas_size_byte);
  double *rands  = (double *)malloc(rands_size_byte);
  double *probs  = (double *)malloc(alphas_size_byte);

  for (int i = 0; i < alphas_size; i++) fscanf(fp, "%lf", &alphas[i]);
  fclose(fp);

  std::mt19937 gen(19937);
  std::normal_distribution<double> norm_dist(0.0, 1.0);
  for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double *> d_alphas("d_alphas", alphas_size);
    Kokkos::View<double *> d_rands("d_rands",   rands_size);
    Kokkos::View<double *> d_probs("d_probs",   alphas_size);

    // Upload initial alphas and rands (row-major layout)
    {
      auto h_a = Kokkos::create_mirror_view(d_alphas);
      auto h_r = Kokkos::create_mirror_view(d_rands);
      memcpy(h_a.data(), alphas, alphas_size_byte);
      memcpy(h_r.data(), rands,  rands_size_byte);
      Kokkos::deep_copy(d_alphas, h_a);
      Kokkos::deep_copy(d_rands,  h_r);
    }

    // ---- Kernel 1 ---------------------------------------------------------
    int threads_per_block = 192;
    int num_blocks        = (int)ceil(1.0 * n / threads_per_block);

    Kokkos::deep_copy(d_probs, 0.0);
    compute_probs(d_alphas, d_rands, d_probs, n, K, M, repeat);

    {
      auto h_p = Kokkos::create_mirror_view(d_probs);
      Kokkos::deep_copy(h_p, d_probs);
      double s = 0.0;
      for (int i = 0; i < alphas_size; i++) s += h_p(i);
      printf("compute_probs: checksum = %lf\n", s);
    }

    // ---- Kernel 2: transpose storage --------------------------------------
    double *t_rands  = transpose(rands,  K, M);
    double *t_alphas = transpose(alphas, K, n);
    memcpy(rands,  t_rands,  rands_size_byte);
    memcpy(alphas, t_alphas, alphas_size_byte);
    free(t_rands);
    free(t_alphas);

    {
      auto h_a = Kokkos::create_mirror_view(d_alphas);
      auto h_r = Kokkos::create_mirror_view(d_rands);
      memcpy(h_a.data(), alphas, alphas_size_byte);
      memcpy(h_r.data(), rands,  rands_size_byte);
      Kokkos::deep_copy(d_alphas, h_a);
      Kokkos::deep_copy(d_rands,  h_r);
    }

    Kokkos::deep_copy(d_probs, 0.0);
    compute_probs_unitStrides(d_alphas, d_rands, d_probs, n, K, M, repeat);

    {
      auto h_p = Kokkos::create_mirror_view(d_probs);
      Kokkos::deep_copy(h_p, d_probs);
      double s = 0.0;
      for (int i = 0; i < alphas_size; i++) s += h_p(i);
      printf("compute_probs_unitStrides: checksum = %lf\n", s);
    }

    // ---- Kernel 3: shared memory version ----------------------------------
    threads_per_block = 96;
    num_blocks        = (int)ceil(1.0 * n / threads_per_block);

    Kokkos::deep_copy(d_probs, 0.0);
    compute_probs_unitStrides_sharedMem(d_alphas, d_rands, d_probs,
                                        n, K, M,
                                        threads_per_block, num_blocks,
                                        repeat);

    {
      auto h_p = Kokkos::create_mirror_view(d_probs);
      Kokkos::deep_copy(h_p, d_probs);
      double s = 0.0;
      for (int i = 0; i < alphas_size; i++) s += h_p(i);
      printf("compute_probs_unitStrides_sharedMem: checksum = %lf\n", s);
    }
  }
  Kokkos::finalize();

  free(alphas);
  free(rands);
  free(probs);
  return 0;
}
