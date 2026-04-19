#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ─── Constants ───────────────────────────────────────────────────────────────
static constexpr int nsources = 1024;
static constexpr int SX = 16;   // targets per team
static constexpr int SY = 4;    // sources per team step (tile height)

// ─── CPU reference ───────────────────────────────────────────────────────────
static void matern_kernel_reference(int ns, int nt, float l,
                                    const float* sources,
                                    const float* targets,
                                    const float* weights,
                                    float* result) {
  for (int t = 0; t < nt; t++) {
    float sum = 0.f;
    for (int s = 0; s < ns; s++) {
      float sq = 0.f;
      for (int i = 0; i < 3; i++) {
        float d = sources[s * 3 + i] - targets[t * 3 + i];
        sq += d * d;
      }
      float diff = sqrtf(sq);
      sum += (1.f + sqrtf(5.f) * diff / l + 5.f * sq / (3.f * l * l)) *
             expf(-sqrtf(5.f) * diff / l) * weights[s];
    }
    result[t] = sum;
  }
}

// ─── Simple device kernel (one thread per target) ────────────────────────────
void matern_kernel(int ntargets, float l,
                   Kokkos::View<const float*> d_sources,
                   Kokkos::View<const float*> d_targets,
                   Kokkos::View<const float*> d_weights,
                   Kokkos::View<float*>       d_result) {
  Kokkos::parallel_for("matern_kernel", ntargets, KOKKOS_LAMBDA(int t) {
    float sum = 0.f;
    for (int s = 0; s < nsources; s++) {
      float sq = 0.f;
      for (int i = 0; i < 3; i++) {
        float d = d_sources(s * 3 + i) - d_targets(t * 3 + i);
        sq += d * d;
      }
      float diff = sqrtf(sq);
      sum += (1.f + sqrtf(5.f) * diff / l + 5.f * sq / (3.f * l * l)) *
             expf(-sqrtf(5.f) * diff / l) * d_weights(s);
    }
    d_result(t) = sum;
  });
}

// ─── Tiled device kernel (TeamPolicy + TeamThreadRange) ──────────────────────
// One team per target; all team threads cooperatively reduce over sources.
// On GPU this exploits warp/wavefront-level parallelism; on CPU it works
// with a single-thread team via the reduction fallback.
void matern_kernel2(int ntargets, float l,
                    Kokkos::View<const float*> d_sources,
                    Kokkos::View<const float*> d_targets,
                    Kokkos::View<const float*> d_weights,
                    Kokkos::View<float*>       d_result) {
  using policy_t = Kokkos::TeamPolicy<>;

  Kokkos::parallel_for(
    "matern_kernel2",
    policy_t(ntargets, Kokkos::AUTO),
    KOKKOS_LAMBDA(const policy_t::member_type& team) {
      const int t = team.league_rank();

      float sum = 0.f;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nsources),
        [=](int s, float& lsum) {
          float sq = 0.f;
          for (int i = 0; i < 3; i++) {
            float d = d_targets(t * 3 + i) - d_sources(s * 3 + i);
            sq += d * d;
          }
          float diff = sqrtf(sq);
          lsum += (1.f + sqrtf(5.f) * diff / l + 5.f * sq / (3.f * l * l)) *
                  expf(-sqrtf(5.f) * diff / l) * d_weights(s);
        },
        sum);

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        d_result(t) = sum;
      });
    });
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <npoints> <repeat>\n", argv[0]);
    return 1;
  }
  const int npoints = atoi(argv[1]);
  const int repeat  = atoi(argv[2]);

  const int ntargets = npoints * npoints * npoints;

  float* sources   = (float*) malloc(nsources * 3 * sizeof(float));
  float* targets   = (float*) malloc(ntargets * 3 * sizeof(float));
  float* weights   = (float*) malloc(nsources     * sizeof(float));
  float* result    = (float*) malloc(ntargets     * sizeof(float));
  float* result_ref = (float*) malloc(ntargets    * sizeof(float));

  srand(123);
  for (int i = 0; i < nsources * 3; i++) sources[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < nsources;     i++) weights[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < ntargets * 3; i++) targets[i] = rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_sources("d_sources", nsources * 3);
    Kokkos::View<float*> d_targets("d_targets", ntargets * 3);
    Kokkos::View<float*> d_weights("d_weights", nsources);
    const int ntargets_alloc = (ntargets > ntargets_small) ? ntargets : ntargets_small;
    Kokkos::View<float*> d_result ("d_result",  ntargets_alloc);

    {
      auto h = Kokkos::create_mirror_view(d_sources);
      memcpy(h.data(), sources, nsources * 3 * sizeof(float));
      Kokkos::deep_copy(d_sources, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_targets);
      memcpy(h.data(), targets, ntargets * 3 * sizeof(float));
      Kokkos::deep_copy(d_targets, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_weights);
      memcpy(h.data(), weights, nsources * sizeof(float));
      Kokkos::deep_copy(d_weights, h);
    }

    auto d_s_c = Kokkos::View<const float*>(d_sources);
    auto d_t_c = Kokkos::View<const float*>(d_targets);
    auto d_w_c = Kokkos::View<const float*>(d_weights);

    // ── Verification with small problem ─────────────────────────────────────
    const int ntargets_small = 16 * 16 * 16;
    printf("------------------------------------------------------------\n");
    printf("Verifying the kernel results with the problem size (16 cube)\n");
    printf("------------------------------------------------------------\n");

    float l = 0.1f;
    while (l <= 1e5f) {
      matern_kernel_reference(nsources, ntargets_small, l, sources, targets,
                              weights, result_ref);
      matern_kernel2(ntargets_small, l, d_s_c, d_t_c, d_w_c, d_result);
      Kokkos::fence();

      auto h_res = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(h_res, d_result);

      bool ok = true;
      for (int i = 0; i < ntargets_small; i++) {
        if (fabsf(h_res(i) - result_ref[i]) > 1e-3f) {
          printf("@%d actual=%f expected=%f\n", i, h_res(i), result_ref[i]);
          ok = false;
          break;
        }
      }
      printf("Length scale = %.1e check = %s\n", l, ok ? "PASS" : "FAIL");
      l *= 10.f;
    }

    // ── Timing with full problem ─────────────────────────────────────────────
    printf("--------------------------------------------------------------------\n");
    printf("Timing the kernel execution with the problem size (%d cube)\n", npoints);
    printf("--------------------------------------------------------------------\n");

    l = 0.1f;
    while (l <= 1e5f) {
      printf("Warmup..\n");
      for (int i = 0; i < repeat; i++)
        matern_kernel2(ntargets, l, d_s_c, d_t_c, d_w_c, d_result);
      Kokkos::fence();

      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        matern_kernel2(ntargets, l, d_s_c, d_t_c, d_w_c, d_result);
      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();

      printf("Length scale = %.1e ", l);
      printf("Average kernel execution time: %f (us)\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
             * 1e-3f / repeat);
      l *= 10.f;
    }
  }
  Kokkos::finalize();

  free(sources);
  free(weights);
  free(targets);
  free(result);
  free(result_ref);
  return 0;
}
