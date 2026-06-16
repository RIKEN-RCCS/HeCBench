// ************************************************
// original authors: Lee Howes and David B. Thomas
// Kokkos port
// ************************************************

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// ---- Constants ------------------------------------------------------------
#define TAUSWORTHE_NUM_THREADS 256
#define TAUSWORTHE_NUM_BLOCKS  4096
#define TAUSWORTHE_TOTAL_NUM_THREADS  (TAUSWORTHE_NUM_THREADS * TAUSWORTHE_NUM_BLOCKS)
#define TAUSWORTHE_NUM_SEEDS_PER_GENERATOR 4
#define TAUSWORTHE_NUM_SEEDS  (TAUSWORTHE_TOTAL_NUM_THREADS * TAUSWORTHE_NUM_SEEDS_PER_GENERATOR)

// LOOKBACK_MAX_T must fit in shared memory per team
// = (TAUSWORTHE_NUM_BLOCKS * TAUSWORTHE_NUM_THREADS - team0_offset) / threads_per_team
// Original: (4096-256)/256 = 15
#define LOOKBACK_MAX_T   ((4096 - 256) / TAUSWORTHE_NUM_THREADS)
#define LOOKBACK_NUM_PARAMETER_VALUES  TAUSWORTHE_TOTAL_NUM_THREADS
#define LOOKBACK_TAUSWORTHE_NUM_BLOCKS  TAUSWORTHE_NUM_BLOCKS
#define LOOKBACK_TAUSWORTHE_NUM_THREADS TAUSWORTHE_NUM_THREADS
#define LOOKBACK_PATHS_PER_SIM 512

#define PI 3.14159265f

// ---- PRNG helpers (device) -------------------------------------------------
KOKKOS_INLINE_FUNCTION
unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M) {
  unsigned b = (((z << S1) ^ z) >> S2);
  return z = (((z & M) << S3) ^ b);
}

KOKKOS_INLINE_FUNCTION
unsigned LCGStep(unsigned &z) {
  return z = (1664525 * z + 1013904223);
}

KOKKOS_INLINE_FUNCTION
float getRandomValueTauswortheUniform(unsigned &z1, unsigned &z2,
                                      unsigned &z3, unsigned &z4) {
  unsigned taus =
      TausStep(z1, 13, 19, 12, 4294967294U) ^
      TausStep(z2, 2,  25,  4, 4294967288U) ^
      TausStep(z3, 3,  11, 17, 4294967280U);
  unsigned lcg = LCGStep(z4);
  return 2.3283064365387e-10f * (taus ^ lcg);
}

KOKKOS_INLINE_FUNCTION
void boxMuller(float u1, float u2, float &uo1, float &uo2) {
  float z1 = sqrtf(-2.0f * logf(u1));
  uo1 = z1 * sinf(2.0f * PI * u2);
  uo2 = z1 * cosf(2.0f * PI * u2);
}

KOKKOS_INLINE_FUNCTION
float getRandomValueTausworthe(unsigned &z1, unsigned &z2, unsigned &z3,
                               unsigned &z4, float &temporary, unsigned phase) {
  if (phase & 1) return temporary;
  float t1, t2, t3;
  t1 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
  t2 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
  boxMuller(t1, t2, t3, temporary);
  return t3;
}

// ---- Simulation path function (device) ------------------------------------
// path layout: path[thread_id + t * LOOKBACK_TAUSWORTHE_NUM_THREADS]
KOKKOS_INLINE_FUNCTION
float tausworthe_lookback_sim(unsigned T, float VOL_0, float EPS_0,
                              float A_0, float A_1, float A_2, float S_0,
                              float MU, unsigned &z1, unsigned &z2,
                              unsigned &z3, unsigned &z4,
                              float *path, int thread_id) {
  float temp_random_value = 0.0f;
  float vol = VOL_0, eps = EPS_0, s = S_0;
  int   base = thread_id;

  for (unsigned t = 0; t < T; t++) {
    path[base] = s;
    base += LOOKBACK_TAUSWORTHE_NUM_THREADS;

    vol = sqrtf(A_0 + A_1 * vol * vol + A_2 * eps * eps);
    eps = getRandomValueTausworthe(z1, z2, z3, z4, temp_random_value, t) * vol;
    eps = fmaxf(fminf(eps, 1.f), -1.f);
    s  *= expf(MU + eps);
  }

  float sum = 0;
  for (unsigned t = 0; t < T; t++) {
    base -= LOOKBACK_TAUSWORTHE_NUM_THREADS;
    sum  += fmaxf(path[base] - s, 0.f);
  }
  return sum;
}

// ---- Main kernel ----------------------------------------------------------
void tausworthe_lookback(
    unsigned num_cycles,
    Kokkos::View<unsigned *> d_seeds,
    Kokkos::View<float *>    d_mean,
    Kokkos::View<float *>    d_variance,
    Kokkos::View<float *>    d_VOL_0,
    Kokkos::View<float *>    d_EPS_0,
    Kokkos::View<float *>    d_A_0,
    Kokkos::View<float *>    d_A_1,
    Kokkos::View<float *>    d_A_2,
    Kokkos::View<float *>    d_S_0,
    Kokkos::View<float *>    d_MU) {

  // Scratch memory per team: LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_MAX_T floats
  const int path_scratch = LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_MAX_T * (int)sizeof(float);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<float *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(LOOKBACK_TAUSWORTHE_NUM_BLOCKS,
                    LOOKBACK_TAUSWORTHE_NUM_THREADS, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(path_scratch));

  Kokkos::parallel_for(
      "tausworthe_lookback", policy,
      KOKKOS_LAMBDA(const member_type &team) {
        int team_id   = team.league_rank();
        int thread_id = team.team_rank();
        unsigned address = team_id * LOOKBACK_TAUSWORTHE_NUM_THREADS + thread_id;

        // Load seeds
        unsigned z1 = d_seeds[address];
        unsigned z2 = d_seeds[address +     TAUSWORTHE_TOTAL_NUM_THREADS];
        unsigned z3 = d_seeds[address + 2 * TAUSWORTHE_TOTAL_NUM_THREADS];
        unsigned z4 = d_seeds[address + 3 * TAUSWORTHE_TOTAL_NUM_THREADS];

        float VOL_0 = d_VOL_0[address];
        float EPS_0 = d_EPS_0[address];
        float A_0   = d_A_0[address];
        float A_1   = d_A_1[address];
        float A_2   = d_A_2[address];
        float S_0   = d_S_0[address];
        float MU    = d_MU[address];

        // Path scratch per team
        ScratchView path(team.team_scratch(0),
                         LOOKBACK_TAUSWORTHE_NUM_THREADS * LOOKBACK_MAX_T);

        float mean = 0.f, variance = 0.f;
        for (unsigned i = 1; i <= LOOKBACK_PATHS_PER_SIM; i++) {
          float res = tausworthe_lookback_sim(
              num_cycles, VOL_0, EPS_0, A_0, A_1, A_2, S_0, MU,
              z1, z2, z3, z4, path.data(), thread_id);
          float delta = res - mean;
          mean     += delta / (float)i;
          variance += delta * (res - mean);
        }

        d_mean[address]     = mean;
        d_variance[address] = variance / (LOOKBACK_PATHS_PER_SIM - 1);
      });
  Kokkos::fence();
}

// ---- Simple LCG for parameter initialisation ------------------------------
static float Rand() {
  static unsigned s = 12345;
  s = 1664525 * s + 1013904223;
  return (float)s / (float)0xffffffffU;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dump> <repeat>\n", argv[0]);
    return 1;
  }
  const int dump   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    const size_t N    = LOOKBACK_NUM_PARAMETER_VALUES;
    const size_t Nseed = TAUSWORTHE_NUM_SEEDS;

    // Host arrays
    std::vector<float>    h_VOL_0(N), h_A_0(N), h_A_1(N), h_A_2(N);
    std::vector<float>    h_S_0(N), h_EPS_0(N), h_MU(N);
    std::vector<float>    h_mean(N), h_variance(N);
    std::vector<unsigned> h_seeds(Nseed);

    for (size_t i = 0; i < N; i++) {
      h_VOL_0[i] = Rand(); h_A_0[i]   = Rand(); h_A_1[i] = Rand();
      h_A_2[i]   = Rand(); h_S_0[i]   = Rand();
      h_EPS_0[i] = Rand(); h_MU[i]    = Rand();
    }
    srand(42);
    for (size_t i = 0; i < Nseed; i++)
      h_seeds[i] = (unsigned)rand() + 16;

    // Device views
    Kokkos::View<unsigned *> d_seeds("d_seeds",   Nseed);
    Kokkos::View<float *>    d_VOL_0("d_VOL_0",   N);
    Kokkos::View<float *>    d_A_0("d_A_0",       N);
    Kokkos::View<float *>    d_A_1("d_A_1",       N);
    Kokkos::View<float *>    d_A_2("d_A_2",       N);
    Kokkos::View<float *>    d_S_0("d_S_0",       N);
    Kokkos::View<float *>    d_EPS_0("d_EPS_0",   N);
    Kokkos::View<float *>    d_MU("d_MU",         N);
    Kokkos::View<float *>    d_mean("d_mean",     N);
    Kokkos::View<float *>    d_variance("d_var",  N);

    // Copy to device
    auto copy_to = [&](Kokkos::View<float *> dv, std::vector<float> &hv) {
      auto mv = Kokkos::create_mirror_view(dv);
      memcpy(mv.data(), hv.data(), hv.size() * sizeof(float));
      Kokkos::deep_copy(dv, mv);
    };
    {
      auto mv = Kokkos::create_mirror_view(d_seeds);
      memcpy(mv.data(), h_seeds.data(), Nseed * sizeof(unsigned));
      Kokkos::deep_copy(d_seeds, mv);
    }
    copy_to(d_VOL_0, h_VOL_0); copy_to(d_A_0, h_A_0);
    copy_to(d_A_1,   h_A_1);   copy_to(d_A_2, h_A_2);
    copy_to(d_S_0,   h_S_0);   copy_to(d_EPS_0, h_EPS_0);
    copy_to(d_MU,    h_MU);

    const unsigned num_cycles = LOOKBACK_MAX_T;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      tausworthe_lookback(num_cycles, d_seeds, d_mean, d_variance,
                          d_VOL_0, d_EPS_0, d_A_0, d_A_1, d_A_2,
                          d_S_0, d_MU);
    }
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

    if (dump) {
      auto h_m = Kokkos::create_mirror_view(d_mean);
      auto h_v = Kokkos::create_mirror_view(d_variance);
      Kokkos::deep_copy(h_m, d_mean);
      Kokkos::deep_copy(h_v, d_variance);
      for (size_t i = 0; i < N; i++)
        printf("%zu %.3f %.3f\n", i, h_m(i), h_v(i));
    }
  }
  Kokkos::finalize();
  return 0;
}
