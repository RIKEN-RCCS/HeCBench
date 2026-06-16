// RSC (RANSAC clustering) – Kokkos port
// Original OpenMP target offload code ported to Kokkos.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <unistd.h>

#include "support/common.h"
#include "support/setup.h"
#include "support/verify.h"

// ─── Params ───────────────────────────────────────────────────────────────
struct Params {
  int         device;
  int         n_gpu_threads;
  int         n_gpu_blocks;
  int         n_threads;
  int         n_warmup;
  int         n_reps;
  const char *file_name;
  int         max_iter;
  int         error_threshold;
  float       convergence_threshold;

  Params(int argc, char **argv) {
    device                = 0;
    n_gpu_threads         = 256;
    n_gpu_blocks          = 64;
    n_threads             = 1;
    n_warmup              = 5;
    n_reps                = 1000;
    file_name             = "input/vectors.csv";
    max_iter              = 2000;
    error_threshold       = 3;
    convergence_threshold = 0.75f;
    int opt;
    while ((opt = getopt(argc, argv, "hi:g:t:w:r:f:m:e:c:")) >= 0) {
      switch (opt) {
      case 'h': usage(); exit(0);
      case 'i': n_gpu_threads         = atoi(optarg); break;
      case 'g': n_gpu_blocks          = atoi(optarg); break;
      case 't': n_threads             = atoi(optarg); break;
      case 'w': n_warmup              = atoi(optarg); break;
      case 'r': n_reps                = atoi(optarg); break;
      case 'f': file_name             = optarg;        break;
      case 'm': max_iter              = atoi(optarg); break;
      case 'e': error_threshold       = atoi(optarg); break;
      case 'c': convergence_threshold = atof(optarg); break;
      default: fprintf(stderr,"Unrecognized option!\n"); usage(); exit(0);
      }
    }
    assert(n_gpu_threads > 0);
    assert(n_gpu_blocks  > 0);
    assert(n_threads     > 0);
  }
  void usage() {
    fprintf(stderr,
            "\nUsage:  ./rsct [options]\n"
            "\n  -h        help\n"
            "  -i <I>    # device threads per block (default=256)\n"
            "  -g <G>    # device blocks (default=64)\n"
            "  -t <T>    # host threads (default=1)\n"
            "  -w <W>    # warmup iterations (default=5)\n"
            "  -r <R>    # timed repetitions (default=1000)\n"
            "  -f <F>    input file (default=input/vectors.csv)\n"
            "  -m <M>    max iterations (default=2000)\n"
            "  -e <E>    error threshold (default=3)\n"
            "  -c <C>    convergence threshold (default=0.75)\n");
  }
};

// ─── run_cpu_threads ──────────────────────────────────────────────────────
void run_cpu_threads(float *model_param_local, flowvector *flowvectors,
                     int flowvector_count, int *random_numbers,
                     int max_iter, int error_threshold,
                     float convergence_threshold, int *g_out_id, int num_threads)
{
  std::vector<std::thread> threads;
  for (int k = 0; k < num_threads; k++) {
    threads.emplace_back([=]() {
      for (int loop_count = k; loop_count < max_iter; loop_count += num_threads) {
        float *mp = &model_param_local[4 * loop_count];
        flowvector fv[2];
        fv[0] = flowvectors[random_numbers[loop_count * 2 + 0]];
        fv[1] = flowvectors[random_numbers[loop_count * 2 + 1]];
        int vx1 = fv[0].vx - fv[0].x, vy1 = fv[0].vy - fv[0].y;
        int vx2 = fv[1].vx - fv[1].x, vy2 = fv[1].vy - fv[1].y;
        // gen_model_param is defined inline in support/verify.h
        int ret = gen_model_param(fv[0].x, fv[0].y, vx1, vy1,
                                   fv[1].x, fv[1].y, vx2, vy2, mp);
        if (ret == 0) mp[0] = -2011.0f;
      }
    });
  }
  for (auto &t : threads) t.join();
}

// ─── call_RANSAC_kernel_block (Kokkos) ───────────────────────────────────
void call_RANSAC_kernel_block(
    int blocks, int threads,
    const Kokkos::View<float *>      &d_model_params,
    const Kokkos::View<flowvector *> &d_flowvectors,
    int flowvector_count, int max_iter,
    int error_threshold, float convergence_threshold,
    Kokkos::View<int *>  &d_g_out_id,
    Kokkos::View<int *>  &d_model_candidate,
    Kokkos::View<int *>  &d_outliers_candidate)
{
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;
  using ScratchInt  = Kokkos::View<int *,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const int scratch_bytes = ScratchInt::shmem_size(1);

  Kokkos::parallel_for(
    team_policy(blocks, threads).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const member_type &team) {
      ScratchInt outlier_sh(team.team_scratch(0), 1);

      const int tx         = team.team_rank();
      const int bx         = team.league_rank();
      const int num_blocks = team.league_size();
      const int block_dim  = team.team_size();

      float vx_error, vy_error;
      int   outlier_local_count;

      for (int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {
        const float *model_param = d_model_params.data() + 4 * loop_count;

        if (tx == 0) outlier_sh(0) = 0;
        team.team_barrier();

        if (model_param[0] == -2011.0f) continue;

        outlier_local_count = 0;
        for (int i = tx; i < flowvector_count; i += block_dim) {
          flowvector fv = d_flowvectors(i);
          vx_error = fv.x + (int)((fv.x - model_param[0]) * model_param[2])
                           - (int)((fv.y - model_param[1]) * model_param[3]) - fv.vx;
          vy_error = fv.y + (int)((fv.y - model_param[1]) * model_param[2])
                           + (int)((fv.x - model_param[0]) * model_param[3]) - fv.vy;
          if (fabsf(vx_error) >= error_threshold || fabsf(vy_error) >= error_threshold)
            outlier_local_count++;
        }

        Kokkos::atomic_add(&outlier_sh(0), outlier_local_count);
        team.team_barrier();

        if (tx == 0) {
          if (outlier_sh(0) < (int)((float)flowvector_count * convergence_threshold)) {
            int index = Kokkos::atomic_fetch_add(&d_g_out_id(0), 1);
            d_model_candidate(index)    = loop_count;
            d_outliers_candidate(index) = outlier_sh(0);
          }
        }
      }
    });
}

// ─── Input file reading ───────────────────────────────────────────────────
static int read_input_size(const Params &p) {
  FILE *f = fopen(p.file_name, "r");
  if (!f) { puts("Error opening file!"); exit(-1); }
  int n; fscanf(f, "%d", &n); fclose(f); return n;
}

static void read_input(flowvector *v, int *r, const Params &p) {
  FILE *f = fopen(p.file_name, "r");
  if (!f) { puts("Error opening file!"); exit(-1); }
  int n; fscanf(f, "%d", &n);
  int ic = 0;
  while (fscanf(f, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
    ic++;
    if (ic > n) { puts("Error: inconsistent file!"); exit(-1); }
  }
  if (ic < n) { puts("Error: inconsistent file!"); exit(-1); }
  fclose(f);
  srand(123);
  for (int i = 0; i < 2 * p.max_iter; i++) r[i] = ((int)rand()) % n;
}

// ─── main ─────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
  const Params p(argc, argv);
  assert(p.n_gpu_threads <= 256 &&
         "Thread block size exceeds maximum allowed (256)");

  int         n_flow_vectors = read_input_size(p);
  int         candidates     = 0;
  int         best_model     = -1;
  int         best_outliers  = n_flow_vectors;

  flowvector *h_flowvectors       = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
  int        *h_random_numbers    = (int *)malloc(2 * p.max_iter * sizeof(int));
  int        *h_model_candidate   = (int *)malloc(p.max_iter * sizeof(int));
  int        *h_outliers_candidate= (int *)malloc(p.max_iter * sizeof(int));
  float      *h_model_param_local = (float *)malloc(4 * p.max_iter * sizeof(float));
  int        *h_g_out_id          = (int *)malloc(sizeof(int));
  ALLOC_ERR(h_flowvectors, h_random_numbers, h_model_candidate,
            h_outliers_candidate, h_model_param_local, h_g_out_id);

  read_input(h_flowvectors, h_random_numbers, p);

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<flowvector *> d_flowvectors      ("flowvectors",       n_flow_vectors);
    Kokkos::View<float *>      d_model_params     ("model_params",      4 * p.max_iter);
    Kokkos::View<int *>        d_g_out_id         ("g_out_id",          1);
    Kokkos::View<int *>        d_model_candidate  ("model_candidate",   p.max_iter);
    Kokkos::View<int *>        d_outliers_candidate("outliers_cand",    p.max_iter);

    // Host mirrors
    auto h_d_flowvectors = Kokkos::create_mirror_view(d_flowvectors);
    auto h_d_model_params= Kokkos::create_mirror_view(d_model_params);
    auto h_d_g_out_id    = Kokkos::create_mirror_view(d_g_out_id);
    auto h_d_model_cand  = Kokkos::create_mirror_view(d_model_candidate);
    auto h_d_out_cand    = Kokkos::create_mirror_view(d_outliers_candidate);

    // Upload flow vectors (constant across iterations)
    for (int i = 0; i < n_flow_vectors; i++) h_d_flowvectors(i) = h_flowvectors[i];
    Kokkos::deep_copy(d_flowvectors, h_d_flowvectors);

    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
      // Reset host arrays
      memset(h_model_candidate,    0, p.max_iter * sizeof(int));
      memset(h_outliers_candidate, 0, p.max_iter * sizeof(int));
      memset(h_model_param_local,  0, 4 * p.max_iter * sizeof(float));
      h_g_out_id[0] = 0;

      // CPU computes model parameters
      std::thread t(run_cpu_threads, h_model_param_local, h_flowvectors,
                    n_flow_vectors, h_random_numbers, p.max_iter,
                    p.error_threshold, p.convergence_threshold,
                    h_g_out_id, p.n_threads);
      t.join();

      // Upload model params and reset device arrays
      for (int i = 0; i < 4 * p.max_iter; i++) h_d_model_params(i) = h_model_param_local[i];
      Kokkos::deep_copy(d_model_params, h_d_model_params);
      h_d_g_out_id(0) = 0;
      Kokkos::deep_copy(d_g_out_id, h_d_g_out_id);

      // GPU RANSAC kernel
      call_RANSAC_kernel_block(p.n_gpu_blocks, p.n_gpu_threads,
                               d_model_params, d_flowvectors,
                               n_flow_vectors, p.max_iter,
                               p.error_threshold, p.convergence_threshold,
                               d_g_out_id, d_model_candidate, d_outliers_candidate);
      Kokkos::fence();

      // Copy back candidates count
      Kokkos::deep_copy(h_d_g_out_id, d_g_out_id);
      candidates = h_d_g_out_id(0);

      // Copy back candidate arrays
      Kokkos::deep_copy(h_d_model_cand, d_model_candidate);
      Kokkos::deep_copy(h_d_out_cand,   d_outliers_candidate);
      for (int i = 0; i < candidates; i++) {
        h_model_candidate[i]    = h_d_model_cand(i);
        h_outliers_candidate[i] = h_d_out_cand(i);
      }

      // Post-processing
      for (int i = 0; i < candidates; i++) {
        if (h_outliers_candidate[i] < best_outliers) {
          best_outliers = h_outliers_candidate[i];
          best_model    = h_model_candidate[i];
        }
      }
    }

    auto end  = std::chrono::steady_clock::now();
    double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total task execution time for %d iterations: %f (ms)\n",
           p.n_reps + p.n_warmup, ns * 1e-6f);
    printf("Best model (test) %d\n", best_model);

    verify(h_flowvectors, n_flow_vectors, h_random_numbers, p.max_iter,
           p.error_threshold, p.convergence_threshold, candidates, best_outliers);
  }
  Kokkos::finalize();

  free(h_model_candidate);
  free(h_outliers_candidate);
  free(h_model_param_local);
  free(h_g_out_id);
  free(h_flowvectors);
  free(h_random_numbers);
  return 0;
}
