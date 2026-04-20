#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// Option types
typedef float real;

typedef struct {
  real S, X, T, R, V;
} TOptionData;

#define NUM_STEPS    2048
#define MAX_OPTIONS  1024
#define NUM_ITERATIONS 1000
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS / THREADBLOCK_SIZE)

static_assert(NUM_STEPS % THREADBLOCK_SIZE == 0, "NUM_STEPS must be divisible by THREADBLOCK_SIZE");

typedef struct {
  real S, X, vDt, puByDf, pdByDf;
} __TOptionData;

KOKKOS_INLINE_FUNCTION
real expiryCallValue(real S, real X, real vDt, int i) {
  real d = S * Kokkos::exp(vDt * (real)(2 * i - NUM_STEPS)) - X;
  return (d > 0.0f) ? d : 0.0f;
}

static real CND(real d) {
  const real A1 = 0.31938153f, A2 = -0.356563782f, A3 = 1.781477937f;
  const real A4 = -1.821255978f, A5 = 1.330274429f;
  const real RSQRT2PI = 0.39894228040143267793994605993438f;
  real K = 1.0f / (1.0f + 0.2316419f * fabsf(d));
  real cnd = RSQRT2PI * expf(-0.5f * d * d) *
      (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
  if (d > 0) cnd = 1.0f - cnd;
  return cnd;
}

void BlackScholesCall(real &callResult, TOptionData o) {
  real sqrtT = sqrtf(o.T);
  real d1 = (logf(o.S / o.X) + (o.R + 0.5f*o.V*o.V)*o.T) / (o.V*sqrtT);
  real d2 = d1 - o.V*sqrtT;
  real expRT = expf(-o.R * o.T);
  callResult = o.S * CND(d1) - o.X * expRT * CND(d2);
}

static real expiryCallValueCPU(real S, real X, real vDt, int i) {
  real d = S * expf(vDt * (real)(2*i - NUM_STEPS)) - X;
  return (d > 0) ? d : 0;
}

void binomialOptionsCPU(real &callResult, TOptionData o) {
  static real Call[NUM_STEPS + 1];
  const real dt = o.T / (real)NUM_STEPS;
  const real vDt = o.V * sqrtf(dt);
  const real rDt = o.R * dt;
  const real If = expf(rDt), Df = expf(-rDt);
  const real u = expf(vDt), d = expf(-vDt);
  const real pu = (If - d) / (u - d);
  const real pd = 1.0f - pu;
  const real puByDf = pu * Df, pdByDf = pd * Df;
  for (int i = 0; i <= NUM_STEPS; i++)
    Call[i] = expiryCallValueCPU(o.S, o.X, vDt, i);
  for (int i = NUM_STEPS; i > 0; i--)
    for (int j = 0; j <= i - 1; j++)
      Call[j] = puByDf * Call[j+1] + pdByDf * Call[j];
  callResult = Call[0];
}

int main(int argc, char **argv) {
  printf("[%s] - Starting...\n", argv[0]);
  const int OPT_N = MAX_OPTIONS;

  TOptionData optionData[MAX_OPTIONS];
  real callValueBS[MAX_OPTIONS], callValueGPU[MAX_OPTIONS], callValueCPU[MAX_OPTIONS];

  srand(123);
  for (int i = 0; i < OPT_N; i++) {
    optionData[i].S = 5.0f + (30.0f - 5.0f) * rand() / RAND_MAX;
    optionData[i].X = 1.0f + (100.0f - 1.0f) * rand() / RAND_MAX;
    optionData[i].T = 0.25f + (10.0f - 0.25f) * rand() / RAND_MAX;
    optionData[i].R = 0.06f;
    optionData[i].V = 0.10f;
    BlackScholesCall(callValueBS[i], optionData[i]);
  }

  printf("Running GPU binomial tree...\n");

  // Prepare processed option data
  __TOptionData d_OptionData[MAX_OPTIONS];
  for (int i = 0; i < OPT_N; i++) {
    const real T = optionData[i].T, R = optionData[i].R, V = optionData[i].V;
    const real dt = T / (real)NUM_STEPS;
    const real vDt = V * sqrtf(dt), rDt = R * dt;
    const real If = expf(rDt), Df = expf(-rDt);
    const real u = expf(vDt), d = expf(-vDt);
    const real pu = (If - d) / (u - d);
    const real pd = 1.0f - pu;
    d_OptionData[i].S      = optionData[i].S;
    d_OptionData[i].X      = optionData[i].X;
    d_OptionData[i].vDt    = vDt;
    d_OptionData[i].puByDf = pu * Df;
    d_OptionData[i].pdByDf = pd * Df;
  }

  auto start = std::chrono::high_resolution_clock::now();

  Kokkos::initialize(argc, argv);
  {
    using ViewOpt  = Kokkos::View<__TOptionData*>;
    using ViewReal = Kokkos::View<real*>;

    ViewOpt  d_opts("opts",  MAX_OPTIONS);
    ViewReal d_call("call",  MAX_OPTIONS);

    auto h_opts = Kokkos::create_mirror_view(d_opts);
    for (int i = 0; i < OPT_N; i++) h_opts(i) = d_OptionData[i];
    Kokkos::deep_copy(d_opts, h_opts);

    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchView = Kokkos::View<real*, scratch_space, Kokkos::MemoryUnmanaged>;

    size_t scratch_bytes = ScratchView::shmem_size(THREADBLOCK_SIZE + 1);
    using team_policy = Kokkos::TeamPolicy<>;

    auto policy = team_policy(OPT_N, THREADBLOCK_SIZE)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    auto kstart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
      Kokkos::parallel_for("binomial", policy,
        KOKKOS_LAMBDA(const team_policy::member_type &team) {
          const int bid = team.league_rank();
          const int tid = team.team_rank();

          ScratchView call_exchange(team.team_scratch(0), THREADBLOCK_SIZE + 1);

          const real S      = d_opts(bid).S;
          const real X      = d_opts(bid).X;
          const real vDt    = d_opts(bid).vDt;
          const real puByDf = d_opts(bid).puByDf;
          const real pdByDf = d_opts(bid).pdByDf;

          real call[ELEMS_PER_THREAD + 1];
          for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

          if (tid == 0)
            call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

          int final_it = (tid * ELEMS_PER_THREAD - 1 > 0) ? tid * ELEMS_PER_THREAD - 1 : 0;

          for (int i = NUM_STEPS; i > 0; --i) {
            call_exchange[tid] = call[0];
            team.team_barrier();
            call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
            team.team_barrier();

            if (i > final_it) {
              for (int j = 0; j < ELEMS_PER_THREAD; ++j)
                call[j] = puByDf * call[j + 1] + pdByDf * call[j];
            }
          }

          if (tid == 0)
            d_call(bid) = call[0];
        });
    }
    Kokkos::fence();

    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::microseconds>(kend - kstart).count();
    printf("Average kernel execution time : %f (us)\n", (float)ktime / NUM_ITERATIONS);

    auto h_call = Kokkos::create_mirror_view(d_call);
    Kokkos::deep_copy(h_call, d_call);
    for (int i = 0; i < OPT_N; i++) callValueGPU[i] = h_call(i);
  }
  Kokkos::finalize();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<real> elapsed = end - start;
  printf("Options count            : %i\n", OPT_N);
  printf("Time steps               : %i\n", NUM_STEPS);
  printf("Total time: %f msec\n", elapsed.count() * 1000);

  printf("Running CPU binomial tree...\n");
  for (int i = 0; i < OPT_N; i++)
    binomialOptionsCPU(callValueCPU[i], optionData[i]);

  real sumDelta = 0, sumRef = 0;
  printf("GPU binomial vs. Black-Scholes\n");
  for (int i = 0; i < OPT_N; i++) {
    sumDelta += fabsf(callValueBS[i] - callValueGPU[i]);
    sumRef   += fabsf(callValueBS[i]);
  }
  if (sumRef > 1E-5f) printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
  else printf("Avg. diff: %E\n", (double)(sumDelta / OPT_N));

  sumDelta = sumRef = 0;
  printf("CPU binomial vs. GPU binomial\n");
  for (int i = 0; i < OPT_N; i++) {
    sumDelta += fabsf(callValueGPU[i] - callValueCPU[i]);
    sumRef   += callValueCPU[i];
  }
  real errorVal = sumDelta / sumRef;
  printf("L1 norm: %E\n", (double)errorVal);

  if (errorVal > 5e-4f) { printf("Test failed!\n"); exit(EXIT_FAILURE); }
  printf("Test passed\n");
  return 0;
}
