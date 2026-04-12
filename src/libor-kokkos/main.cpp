//////////////////////////////////////////////////////////////////
//                                                              //
// Kokkos port of libor-sycl benchmark.                         //
// Original copyright University of Oxford (BSD3).              //
//                                                              //
//////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define BLOCK_SIZE 64
#define GRID_SIZE  1500

#define NN     80
#define NMAT   40
#define L2_SIZE 3280   // NN*(NMAT+1)
#define NOPT   15
#define NPATH  96000

KOKKOS_INLINE_FUNCTION
void path_calc(float* L, const float* z, const float* lambda,
               float delta, int Nmat, int N)
{
  for (int n = 0; n < Nmat; n++) {
    float sqez = Kokkos::sqrt(delta) * z[n];
    float v = 0.f;
    for (int i = n + 1; i < N; i++) {
      float lam  = lambda[i - n - 1];
      float con1 = delta * lam;
      v   += con1 * L[i] / (1.f + delta * L[i]);
      float vrat = Kokkos::exp(con1 * v + lam * (sqez - 0.5f * con1));
      L[i] = L[i] * vrat;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void path_calc_b1(float* L, const float* z, float* L2,
                  const float* lambda, float delta, int Nmat, int N)
{
  for (int i = 0; i < N; i++) L2[i] = L[i];

  for (int n = 0; n < Nmat; n++) {
    float sqez = Kokkos::sqrt(delta) * z[n];
    float v = 0.f;
    for (int i = n + 1; i < N; i++) {
      float lam  = lambda[i - n - 1];
      float con1 = delta * lam;
      v   += con1 * L[i] / (1.f + delta * L[i]);
      float vrat = Kokkos::exp(con1 * v + lam * (sqez - 0.5f * con1));
      L[i] = L[i] * vrat;
      L2[i + (n + 1) * N] = L[i];
    }
  }
}

KOKKOS_INLINE_FUNCTION
void path_calc_b2(float* L_b, const float* z, const float* L2,
                  const float* lambda, float delta, int Nmat, int N)
{
  for (int n = Nmat - 1; n >= 0; n--) {
    float v1 = 0.f;
    for (int i = N - 1; i > n; i--) {
      v1    += lambda[i - n - 1] * L2[i + (n + 1) * N] * L_b[i];
      float faci   = delta / (1.f + delta * L2[i + n * N]);
      L_b[i] = L_b[i] * (L2[i + (n + 1) * N] / L2[i + n * N])
               + v1 * lambda[i - n - 1] * faci * faci;
    }
  }
}

KOKKOS_INLINE_FUNCTION
float portfolio_b(float* L, float* L_b,
                  const float* lambda,
                  const int* maturities,
                  const float* swaprates,
                  float delta, int Nmat, int N, int Nopt)
{
  float B[NMAT], S[NMAT], B_b[NMAT], S_b[NMAT];

  float b = 1.f, s = 0.f;
  for (int m = 0; m < N - Nmat; m++) {
    int n = m + Nmat;
    b    = b / (1.f + delta * L[n]);
    s    = s + delta * b;
    B[m] = b;
    S[m] = s;
  }

  float v = 0.f;
  for (int m = 0; m < NMAT; m++) { B_b[m] = 0.f; S_b[m] = 0.f; }

  for (int n = 0; n < Nopt; n++) {
    int m = maturities[n] - 1;
    float swapval = B[m] + swaprates[n] * S[m] - 1.f;
    if (swapval < 0) {
      v      += -100.f * swapval;
      S_b[m] += -100.f * swaprates[n];
      B_b[m] += -100.f;
    }
  }

  for (int m = N - Nmat - 1; m >= 0; m--) {
    int n = m + Nmat;
    B_b[m]  += delta * S_b[m];
    L_b[n]   = -B_b[m] * B[m] * delta / (1.f + delta * L[n]);
    if (m > 0) {
      S_b[m - 1] += S_b[m];
      B_b[m - 1] += B_b[m] / (1.f + delta * L[n]);
    }
  }

  b = 1.f;
  for (int n = 0; n < Nmat; n++) b = b / (1.f + delta * L[n]);
  v = b * v;

  for (int n = 0; n < Nmat; n++)
    L_b[n] = -v * delta / (1.f + delta * L[n]);
  for (int n = Nmat; n < N; n++)
    L_b[n] = b * L_b[n];

  return v;
}

KOKKOS_INLINE_FUNCTION
float portfolio(float* L,
                const float* lambda,
                const int* maturities,
                const float* swaprates,
                float delta, int Nmat, int N, int Nopt)
{
  float B[40], S[40];
  float b = 1.f, s = 0.f;

  for (int n = Nmat; n < N; n++) {
    b = b / (1.f + delta * L[n]);
    s = s + delta * b;
    B[n - Nmat] = b;
    S[n - Nmat] = s;
  }

  float v = 0.f;
  for (int i = 0; i < Nopt; i++) {
    int m = maturities[i] - 1;
    float swapval = B[m] + swaprates[i] * S[m] - 1.f;
    if (swapval < 0) v += -100.f * swapval;
  }

  b = 1.f;
  for (int n = 0; n < Nmat; n++) b = b / (1.f + delta * L[n]);
  v = b * v;
  return v;
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 0;
    }
    const int repeat = atoi(argv[1]);

    float  h_lambda[NN];
    float  h_delta  = 0.25f;
    int    h_N      = NN, h_Nmat = NMAT, h_Nopt = NOPT;
    int    h_maturities[] = {4,4,4,8,8,8,20,20,20,28,28,28,40,40,40};
    float  h_swaprates[]  = {.045f,.05f,.055f,.045f,.05f,.055f,.045f,.05f,
                             .055f,.045f,.05f,.055f,.045f,.05f,.055f};

    for (int i = 0; i < NN; i++) h_lambda[i] = 0.2f;

    // Device views
    Kokkos::View<int*>   d_maturities("d_maturities", NOPT);
    Kokkos::View<float*> d_swaprates("d_swaprates",   NOPT);
    Kokkos::View<float*> d_lambda("d_lambda",         NN);
    Kokkos::View<float*> d_v("d_v",                   NPATH);
    Kokkos::View<float*> d_Lb("d_Lb",                 NPATH);

    auto h_mat  = Kokkos::create_mirror_view(d_maturities);
    auto h_swap = Kokkos::create_mirror_view(d_swaprates);
    auto h_lam  = Kokkos::create_mirror_view(d_lambda);
    auto h_v    = Kokkos::create_mirror_view(d_v);
    auto h_Lb   = Kokkos::create_mirror_view(d_Lb);

    for (int i = 0; i < NOPT; i++) { h_mat(i)  = h_maturities[i]; h_swap(i) = h_swaprates[i]; }
    for (int i = 0; i < NN;   i++) h_lam(i)  = h_lambda[i];

    Kokkos::deep_copy(d_maturities, h_mat);
    Kokkos::deep_copy(d_swaprates,  h_swap);
    Kokkos::deep_copy(d_lambda,     h_lam);

    const int threadN = GRID_SIZE * BLOCK_SIZE;
    bool ok = true;

    // Kernel 1: no greeks
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("nogreek", threadN, KOKKOS_LAMBDA(int tid) {
        float L[NN], z[NN];
        for (int path = tid; path < NPATH; path += threadN) {
          for (int i = 0; i < h_N; i++) { z[i] = 0.3f; L[i] = 0.05f; }
          path_calc(L, z, d_lambda.data(), h_delta, h_Nmat, h_N);
          d_v(path) = portfolio(L, d_lambda.data(), d_maturities.data(),
                                d_swaprates.data(), h_delta, h_Nmat, h_N, h_Nopt);
        }
      });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

    Kokkos::deep_copy(h_v, d_v);
    double v = 0.0;
    for (int i = 0; i < NPATH; i++) v += h_v(i);
    v /= NPATH;
    if (fabs(v - 224.323) > 1e-3) {
      ok = false;
      printf("Expected: 224.323 Actual %15.3f\n", v);
    }

    // Kernel 2: greeks
    Kokkos::fence();
    start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("greek", threadN, KOKKOS_LAMBDA(int tid) {
        float L[NN], L2[L2_SIZE], z[NN];
        float* L_b = L;

        for (int path = tid; path < NPATH; path += threadN) {
          for (int i = 0; i < h_N; i++) { z[i] = 0.3f; L[i] = 0.05f; }
          path_calc_b1(L, z, L2, d_lambda.data(), h_delta, h_Nmat, h_N);
          d_v(path)  = portfolio_b(L, L_b, d_lambda.data(), d_maturities.data(),
                                   d_swaprates.data(), h_delta, h_Nmat, h_N, h_Nopt);
          path_calc_b2(L_b, z, L2, d_lambda.data(), h_delta, h_Nmat, h_N);
          d_Lb(path) = L_b[NN - 1];
        }
      });
    }

    Kokkos::fence();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

    Kokkos::deep_copy(h_v,  d_v);
    Kokkos::deep_copy(h_Lb, d_Lb);

    v = 0.0;
    for (int i = 0; i < NPATH; i++) v += h_v(i);
    v /= NPATH;

    double Lb = 0.0;
    for (int i = 0; i < NPATH; i++) Lb += h_Lb(i);
    Lb /= NPATH;

    if (fabs(v  - 224.323) > 1e-3) { ok = false; printf("Expected: 224.323 Actual %15.3f\n", v); }
    if (fabs(Lb -  21.348) > 1e-3) { ok = false; printf("Expected:  21.348 Actual %15.3f\n", Lb); }

    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
