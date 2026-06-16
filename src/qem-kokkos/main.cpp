/*
 * Kokkos port of qem-cuda benchmark.
 *
 * Solves f(x) = A*x^4 + B*x^3 + C*x^2 + D*x + E for its minimum using a
 * two-pass quartic solver:
 *   Pass 1 (QRdel):       compute depressed-cubic coefficients b, c, d and
 *                         discriminant quantities Q, R, Qint, Rint, del.
 *   Pass 2 (QuarticSolver): find roots of the depressed cubic, then select
 *                            the x-value that minimises f.
 *
 * A CPU reference is computed inline and used for verification.
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

// ---------------------------------------------------------------------------
// CPU reference  (logic from reference.h / quarticSolver_cpu)
// ---------------------------------------------------------------------------
static void QuarticMinimumCPU(int N,
                               const float* A, const float* B,
                               const float* C, const float* D,
                               float* min_cpu)
{
  const float TWO_PI  = 2.f * 3.1415927f;
  const float FOUR_PI = 4.f * 3.1415927f;

  auto b_v    = std::make_unique<float[]>(N);
  auto c_v    = std::make_unique<float[]>(N);
  auto d_v    = std::make_unique<float[]>(N);
  auto Q_v    = std::make_unique<float[]>(N);
  auto R_v    = std::make_unique<float[]>(N);
  auto del_v  = std::make_unique<float[]>(N);
  auto x1_v   = std::make_unique<float[]>(N);
  auto x2_v   = std::make_unique<float[]>(N);
  auto x3_v   = std::make_unique<float[]>(N);
  auto tmp_v  = std::make_unique<float[]>(N);

  for (int i = 0; i < N; ++i) {
    b_v[i] = 0.75f * (B[i] / A[i]);
    c_v[i] = 0.50f * (C[i] / A[i]);
    d_v[i] = 0.25f * (D[i] / A[i]);

    Q_v[i] = (c_v[i] / 3.f) - ((b_v[i] * b_v[i]) / 9.f);
    R_v[i] = (b_v[i] * c_v[i]) / 6.f
           - (b_v[i] * b_v[i] * b_v[i]) / 27.f
           - 0.5f * d_v[i];

    Q_v[i]   = roundf(Q_v[i] * 1e5f) / 1e5f;
    R_v[i]   = roundf(R_v[i] * 1e5f) / 1e5f;
    del_v[i] = R_v[i] * R_v[i] + Q_v[i] * Q_v[i] * Q_v[i];
  }

  for (int i = 0; i < N; ++i) {
    if (del_v[i] <= 1e-5f) {
      float theta = acosf(R_v[i] / sqrtf(-(Q_v[i] * Q_v[i] * Q_v[i])));
      float sq    = 2.f * sqrtf(-Q_v[i]);

      x1_v[i] = sq * cosf( theta                / 3.f) - b_v[i] / 3.f;
      x2_v[i] = sq * cosf((theta + TWO_PI ) / 3.f) - b_v[i] / 3.f;
      x3_v[i] = sq * cosf((theta + FOUR_PI) / 3.f) - b_v[i] / 3.f;

      // bubble sort descending so x1 >= x2 >= x3
      if (x1_v[i] < x2_v[i]) { tmp_v[i] = x1_v[i]; x1_v[i] = x2_v[i]; x2_v[i] = tmp_v[i]; }
      if (x2_v[i] < x3_v[i]) { tmp_v[i] = x2_v[i]; x2_v[i] = x3_v[i]; x3_v[i] = tmp_v[i]; }
      if (x1_v[i] < x2_v[i]) { tmp_v[i] = x1_v[i]; x1_v[i] = x2_v[i]; x2_v[i] = tmp_v[i]; }

      // compare f(x1) - f(x3): if <= 0, x1 is the minimum; otherwise x3
      float diff = A[i] * (x1_v[i]*x1_v[i]*x1_v[i]*x1_v[i]
                         - x3_v[i]*x3_v[i]*x3_v[i]*x3_v[i]) / 4.f
                 + B[i] * (x1_v[i]*x1_v[i]*x1_v[i] - x3_v[i]*x3_v[i]*x3_v[i]) / 3.f
                 + C[i] * (x1_v[i]*x1_v[i] - x3_v[i]*x3_v[i]) / 2.f
                 + D[i] * (x1_v[i] - x3_v[i]);
      min_cpu[i] = (diff <= 0.f) ? x1_v[i] : x3_v[i];
    } else {
      x1_v[i]  = cbrtf(R_v[i] + sqrtf(del_v[i]))
               + cbrtf(R_v[i] - sqrtf(del_v[i]))
               - b_v[i] / 3.f;
      min_cpu[i] = x1_v[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Data generator
// ---------------------------------------------------------------------------
static void generate_data(int size, int lo, int hi, float* data)
{
  std::mt19937_64 rng{1993764};
  std::uniform_int_distribution<int> dist{lo, hi};
  for (int i = 0; i < size; ++i)
    data[i] = (float)dist(rng);
}

// ---------------------------------------------------------------------------
// Device quartic solver (Kokkos version)
// ---------------------------------------------------------------------------
static void QuarticMinimumKokkos(int N,
                                  const float* h_A, const float* h_B,
                                  const float* h_C, const float* h_D,
                                  float* h_min)
{
  using FView = Kokkos::View<float*>;
  using HView = Kokkos::View<float*, Kokkos::HostSpace>;

  // Device views for input
  FView d_A("A", N), d_B("B", N), d_C("C", N), d_D("D", N);

  // Intermediate views (results of QRdel)
  FView d_b("b", N), d_Q("Q", N), d_R("R", N), d_del("del", N);

  // Output view
  FView d_min("min", N);

  // Upload inputs
  {
    HView ha(const_cast<float*>(h_A), N);
    HView hb(const_cast<float*>(h_B), N);
    HView hc(const_cast<float*>(h_C), N);
    HView hd(const_cast<float*>(h_D), N);
    Kokkos::deep_copy(d_A, ha);
    Kokkos::deep_copy(d_B, hb);
    Kokkos::deep_copy(d_C, hc);
    Kokkos::deep_copy(d_D, hd);
  }

  // ---- QRdel kernel ----
  Kokkos::parallel_for(
      "QRdel",
      Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(int i) {
        float bi = 0.75f * (d_B(i) / d_A(i));
        float ci = 0.50f * (d_C(i) / d_A(i));
        float di = 0.25f * (d_D(i) / d_A(i));

        float Qi = (ci / 3.f) - (bi * bi / 9.f);
        float Ri = (bi * ci) / 6.f - (bi * bi * bi) / 27.f - 0.5f * di;

        Qi = roundf(Qi * 1e5f) / 1e5f;
        Ri = roundf(Ri * 1e5f) / 1e5f;

        d_b(i)   = bi;
        d_Q(i)   = Qi;
        d_R(i)   = Ri;
        d_del(i) = Ri * Ri + Qi * Qi * Qi;
      });
  Kokkos::fence();

  // ---- QuarticSolver kernel ----
  Kokkos::parallel_for(
      "QuarticSolver",
      Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(int i) {
        const float Ai   = d_A(i);
        const float Bi   = d_B(i);
        const float Ci   = d_C(i);
        const float Di   = d_D(i);
        const float bi   = d_b(i);
        const float Qi   = d_Q(i);
        const float Ri   = d_R(i);
        const float deli = d_del(i);

        float x1, x3;

        if (deli <= 1e-5f) {
          // Three real roots
          float theta = acosf(Ri / sqrtf(-(Qi * Qi * Qi)));
          float sq    = 2.f * sqrtf(-Qi);

          x1 = sq * cosf( theta                  / 3.f) - bi / 3.f;
          float x2 = sq * cosf((theta + 2.f * 3.1415927f) / 3.f) - bi / 3.f;
          x3 = sq * cosf((theta + 4.f * 3.1415927f) / 3.f) - bi / 3.f;

          // Bubble sort descending
          float tmp;
          if (x1 < x2) { tmp = x1; x1 = x2; x2 = tmp; }
          if (x2 < x3) { tmp = x2; x2 = x3; x3 = tmp; }
          if (x1 < x2) { tmp = x1; x1 = x2; x2 = tmp; }

          // Compare f(x1) vs f(x3) to find the minimum
          float diff = Ai * (x1*x1*x1*x1 - x3*x3*x3*x3) / 4.f
                     + Bi * (x1*x1*x1    - x3*x3*x3   ) / 3.f
                     + Ci * (x1*x1       - x3*x3       ) / 2.f
                     + Di * (x1          - x3          );
          d_min(i) = (diff <= 0.f) ? x1 : x3;
        } else {
          // One real root
          x1 = cbrtf(Ri + sqrtf(deli)) + cbrtf(Ri - sqrtf(deli)) - bi / 3.f;
          d_min(i) = x1;
        }
      });
  Kokkos::fence();

  // Download result
  {
    HView h_out(h_min, N);
    Kokkos::deep_copy(h_out, d_min);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int N      = 1999999;
  printf("N = %d\n", N);

  auto A           = std::make_unique<float[]>(N);
  auto B           = std::make_unique<float[]>(N);
  auto C           = std::make_unique<float[]>(N);
  auto D           = std::make_unique<float[]>(N);
  auto E           = std::make_unique<float[]>(N);
  auto minimum_ref = std::make_unique<float[]>(N);
  auto minimum     = std::make_unique<float[]>(N);

  printf("Generating data...\n");
  generate_data(N, -100, 100, A.get());
  generate_data(N, -100, 100, B.get());
  generate_data(N, -100, 100, C.get());
  generate_data(N, -100, 100, D.get());
  generate_data(N, -100, 100, E.get());

  for (int i = 0; i < N; ++i)
    if (A[i] == 0.f) A[i] = 1.f; // avoid A=0 undefined behaviour

  Kokkos::initialize(argc, argv);
  {
    // ---- CPU reference ----
    printf("####################### Reference #############\n");
    double avg_cpu = 0.0;
    for (int k = 0; k < repeat; ++k) {
      auto t0 = std::chrono::high_resolution_clock::now();
      QuarticMinimumCPU(N, A.get(), B.get(), C.get(), D.get(), minimum_ref.get());
      auto t1 = std::chrono::high_resolution_clock::now();
      avg_cpu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    printf("Execution time (ms): %f\n", avg_cpu / repeat);

    // ---- Kokkos device solver ----
    printf("####################### Kokkos #############\n");
    double avg_gpu = 0.0;
    for (int k = 0; k < repeat; ++k) {
      auto t0 = std::chrono::high_resolution_clock::now();
      QuarticMinimumKokkos(N, A.get(), B.get(), C.get(), D.get(), minimum.get());
      auto t1 = std::chrono::high_resolution_clock::now();
      avg_gpu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    printf("Execution time (ms): %f\n", avg_gpu / repeat);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
      if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
