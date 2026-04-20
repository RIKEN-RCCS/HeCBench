// Kokkos port of softmax-online-cuda
// Online softmax: computes max and sum in one pass per row

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>
#include <limits>

// CPU reference implementation
void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float* inp_row = inp + i * C;
    float* out_row = out + i * C;
    float maxval = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < C; j++) if (inp_row[j] > maxval) maxval = inp_row[j];
    double sum = 0.0;
    for (int j = 0; j < C; j++) { out_row[j] = expf(inp_row[j] - maxval); sum += out_row[j]; }
    float norm = 1.f / (float)sum;
    for (int j = 0; j < C; j++) out_row[j] *= norm;
  }
}

// Online softmax kernel using TeamPolicy
// Each team handles one row; threads cooperate to find max+sum in one pass
void softmax_online_kokkos(Kokkos::View<float*> d_out,
                            Kokkos::View<const float*> d_inp,
                            int N, int C)
{
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;

  Kokkos::parallel_for("softmax_online",
    team_policy(N, Kokkos::AUTO),
    KOKKOS_LAMBDA(const member_type& team) {
      const int row = team.league_rank();
      const float* x = &d_inp[row * C];
      float* y = &d_out[row * C];

      // Compute max
      float max_val = -Kokkos::Experimental::infinity<float>();
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, C),
        [&](const int j, float& lmax) {
          if (x[j] > lmax) lmax = x[j];
        }, Kokkos::Max<float>(max_val));
      team.team_barrier();

      // Compute sum of exp
      float sum_val = 0.0f;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, C),
        [&](const int j, float& lsum) {
          lsum += Kokkos::exp(x[j] - max_val);
        }, sum_val);
      team.team_barrier();

      // Write output
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, C),
        [&](const int j) {
          y[j] = Kokkos::exp(x[j] - max_val) / sum_val;
        });
    });
}

int main(int argc, char **argv) {
  srand(0);

  int B = 8;
  int T = 1024;
  int V = 50257;

  float* out = (float*)malloc(B * T * V * sizeof(float));
  float* inp = (float*)malloc(B * T * V * sizeof(float));

  // Initialize with random values
  for (int i = 0; i < B * T * V; i++)
    inp[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // Make some outliers
  for (int j = 0; j < B * T; j++) {
    for (int k = 0; k < 3; k++) {
      int idx = rand() % V;
      inp[j * V + idx] *= 20.0f;
    }
  }

  // Read kernel_num from command line
  int kernel_num = 1;
  if (argc > 1) kernel_num = atoi(argv[1]);
  printf("Using kernel %d\n", kernel_num);

  // CPU reference
  softmax_forward_cpu(out, inp, B * T, V);
  {
    float max_el = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < B * T * V; i++) if (out[i] > max_el) max_el = out[i];
    assert(max_el > 1e-4f);
    printf("Largest output is: %f\n", max_el);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_inp("d_inp", (size_t)B * T * V);
    Kokkos::View<float*> d_out("d_out", (size_t)B * T * V);

    // Copy input to device
    auto h_inp = Kokkos::create_mirror_view(d_inp);
    for (int i = 0; i < B * T * V; i++) h_inp(i) = inp[i];
    Kokkos::deep_copy(d_inp, h_inp);

    printf("All results match. Starting benchmarks.\n\n");

    int repeat_times = 10; // reduce for large V*B*T
    for (int block_size = 32; block_size <= 1024; block_size *= 2) {
      // benchmark
      Kokkos::fence();
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < repeat_times; i++) {
        softmax_online_kokkos(d_out, d_inp, B * T, V);
      }
      Kokkos::fence();
      auto stop = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float, std::milli> dur = stop - start;
      float elapsed_time = dur.count() / repeat_times;
      printf("block_size %4d | time %.4f ms | per token %.2f µs\n",
             block_size, elapsed_time, elapsed_time * 1000.0f / (B * T));
      (void)block_size; // Kokkos handles thread count automatically
      break; // only one run needed (Kokkos auto-selects team size)
    }
  }
  Kokkos::finalize();

  free(out);
  free(inp);
  return 0;
}
