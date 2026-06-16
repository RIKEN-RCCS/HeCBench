#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

static float* make_random_float(size_t n) {
  float* arr = new float[n];
  for (size_t i = 0; i < n; i++)
    arr[i] = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
  return arr;
}

// CPU reference: out(BT, OC) = inp(BT, C) * weight^T(OC, C) + bias(OC)
static void matmul_cpu(float* out,
                       const float* inp, const float* weight, const float* bias,
                       int BT, int C, int OC) {
  for (int bt = 0; bt < BT; bt++) {
    for (int oc = 0; oc < OC; oc++) {
      float val = bias ? bias[oc] : 0.f;
      for (int c = 0; c < C; c++)
        val += inp[bt * C + c] * weight[oc * C + c];
      out[bt * OC + oc] = val;
    }
  }
}

static void validate(const Kokkos::View<float*>& d_out,
                     const float* h_ref, const char* name,
                     size_t n, float tol) {
  Kokkos::View<float*, Kokkos::HostSpace> h_out("h_out_val", n);
  Kokkos::deep_copy(h_out, d_out);
  int errs = 0;
  for (size_t i = 0; i < n; i++) {
    if (std::fabs(h_out[i] - h_ref[i]) > tol) {
      if (++errs <= 5)
        printf("  Mismatch %s[%zu]: got %.6f ref %.6f\n",
               name, i, h_out[i], h_ref[i]);
    }
  }
  if (errs == 0)
    printf("  All results match for %s.\n", name);
  else
    printf("  FAIL: %d mismatches in %s.\n", errs, name);
}

// --------------------------------------------------------------------------
// Kernel 1 – naive: MDRangePolicy, each thread owns one (bt, oc) output
// --------------------------------------------------------------------------
static void matmul_kernel1(const Kokkos::View<float*>& d_out,
                           const Kokkos::View<float*>& d_inp,
                           const Kokkos::View<float*>& d_weight,
                           const Kokkos::View<float*>& d_bias,
                           int BT, int C, int OC) {
  Kokkos::parallel_for(
      "matmul_k1",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {BT, OC}),
      KOKKOS_LAMBDA(int bt, int oc) {
        float val = d_bias[oc];
        for (int c = 0; c < C; c++)
          val += d_inp[bt * C + c] * d_weight[oc * C + c];
        d_out[bt * OC + oc] = val;
      });
  Kokkos::fence();
}

// --------------------------------------------------------------------------
// Kernel 4 – tiled: TeamPolicy, each team owns one BT row.
//   The inp row is cached in scratch memory; threads share OC work.
// --------------------------------------------------------------------------
static void matmul_kernel4(const Kokkos::View<float*>& d_out,
                           const Kokkos::View<float*>& d_inp,
                           const Kokkos::View<float*>& d_weight,
                           const Kokkos::View<float*>& d_bias,
                           int BT, int C, int OC) {
  using ExecSpace  = Kokkos::DefaultExecutionSpace;
  using ScrSpace   = ExecSpace::scratch_memory_space;
  using ScrView    = Kokkos::View<float*, ScrSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // Level-0 scratch holds one inp row (C floats)
  const int scratch_bytes = sizeof(float) * C;

  auto policy = Kokkos::TeamPolicy<>(BT, Kokkos::AUTO)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  Kokkos::parallel_for(
      "matmul_k4", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int bt = team.league_rank();

        // Load inp[bt, :] into scratch cooperatively
        ScrView inp_s(team.team_scratch(0), C);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, C),
                             [&](int c) { inp_s[c] = d_inp[bt * C + c]; });
        team.team_barrier();

        // Each thread computes one or more OC outputs using cached inp row
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, OC), [&](int oc) {
          float val = d_bias[oc];
          for (int c = 0; c < C; c++)
            val += inp_s[c] * d_weight[oc * C + c];
          d_out[bt * OC + oc] = val;
        });
      });
  Kokkos::fence();
}

// --------------------------------------------------------------------------
// add_bias – add bias vector to every row of out (BT x OC)
// --------------------------------------------------------------------------
static void add_bias_kernel(const Kokkos::View<float*>& d_out,
                            const Kokkos::View<float*>& d_bias,
                            int BT, int OC) {
  Kokkos::parallel_for(
      "add_bias", BT * OC,
      KOKKOS_LAMBDA(int i) { d_out[i] += d_bias[i % OC]; });
  Kokkos::fence();
}

// --------------------------------------------------------------------------
// Timing helper
// --------------------------------------------------------------------------
static double bench_ms(const std::function<void()>& fn, int reps) {
  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; i++) fn();
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
         * 1e-6 / reps;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc != 5) {
    printf("Usage: %s <B> <T> <C> <repeat>\n", argv[0]);
    return 1;
  }
  const int B      = atoi(argv[1]);
  const int T      = atoi(argv[2]);
  const int C      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int OC     = C * 4; // typical GPT-2 MLP expansion
  const int BT     = B * T;

  printf("B=%d T=%d C=%d OC=%d repeat=%d\n", B, T, C, OC, repeat);

  srand(0);

  float* h_inp    = make_random_float((size_t)BT * C);
  float* h_weight = make_random_float((size_t)OC * C);
  float* h_bias   = make_random_float(OC);
  float* h_ref    = new float[(size_t)BT * OC]();

  // CPU reference
  matmul_cpu(h_ref, h_inp, h_weight, h_bias, BT, C, OC);

  Kokkos::initialize(argc, argv);
  {
    // --- Allocate device views ---
    Kokkos::View<float*> d_inp   ("d_inp",    (size_t)BT * C);
    Kokkos::View<float*> d_weight("d_weight", (size_t)OC * C);
    Kokkos::View<float*> d_bias  ("d_bias",   OC);
    Kokkos::View<float*> d_out   ("d_out",    (size_t)BT * OC);

    // --- Copy H→D ---
    {
      auto hv = Kokkos::View<float*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_inp, (size_t)BT * C);
      Kokkos::deep_copy(d_inp, hv);
    }
    {
      auto hv = Kokkos::View<float*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_weight, (size_t)OC * C);
      Kokkos::deep_copy(d_weight, hv);
    }
    {
      auto hv = Kokkos::View<float*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_bias, OC);
      Kokkos::deep_copy(d_bias, hv);
    }

    // ----- Kernel 1 -----
    printf("\n--- Kernel 1 (MDRangePolicy naive) ---\n");
    matmul_kernel1(d_out, d_inp, d_weight, d_bias, BT, C, OC);
    validate(d_out, h_ref, "kernel1", (size_t)BT * OC, 1e-1f);

    double ms1 = bench_ms(
        [&]() { matmul_kernel1(d_out, d_inp, d_weight, d_bias, BT, C, OC); },
        repeat);
    double tflops1 = (double)BT * OC * C * 2 / ms1 * 1e-9;
    printf("  time %.4f ms | tflops %.2f\n", ms1, tflops1);

    // ----- Kernel 4 -----
    printf("\n--- Kernel 4 (TeamPolicy tiled, inp row in scratch) ---\n");
    matmul_kernel4(d_out, d_inp, d_weight, d_bias, BT, C, OC);
    validate(d_out, h_ref, "kernel4", (size_t)BT * OC, 1e-1f);

    double ms4 = bench_ms(
        [&]() { matmul_kernel4(d_out, d_inp, d_weight, d_bias, BT, C, OC); },
        repeat);
    double tflops4 = (double)BT * OC * C * 2 / ms4 * 1e-9;
    printf("  time %.4f ms | tflops %.2f\n", ms4, tflops4);

    // ----- add_bias (standalone) -----
    printf("\n--- add_bias (standalone) ---\n");
    Kokkos::deep_copy(d_out, 0.f); // zero out
    add_bias_kernel(d_out, d_bias, BT, OC);
    double ms_ab = bench_ms(
        [&]() { add_bias_kernel(d_out, d_bias, BT, OC); }, repeat);
    printf("  time %.4f ms\n", ms_ab);
  }
  Kokkos::finalize();

  delete[] h_inp;
  delete[] h_weight;
  delete[] h_bias;
  delete[] h_ref;
  return 0;
}
