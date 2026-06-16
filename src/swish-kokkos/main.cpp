#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

// Inline reference
template <typename T>
void reference(const int N, T *X, T *Y, T *dX, T *dY) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] / (T(1) + exp(-X[i]));
    dX[i] = dY[i] * (Y[i] + (T(1) - Y[i]) / (T(1) + exp(-X[i])));
  }
}

template <typename T>
void eval_swish(const int N, const int repeat)
{
  T *h_X  = (T*) malloc(N * sizeof(T));
  T *h_Y  = (T*) malloc(N * sizeof(T));
  T *h_dY = (T*) malloc(N * sizeof(T));
  T *h_dX = (T*) malloc(N * sizeof(T));
  T *r_Y  = (T*) malloc(N * sizeof(T));
  T *r_dX = (T*) malloc(N * sizeof(T));

  std::default_random_engine gen(123);
  std::uniform_real_distribution<float> distr(-2.f, 2.f);
  for (int i = 0; i < N; i++) {
    h_X[i]  = (T)distr(gen);
    h_dY[i] = (T)distr(gen);
  }

  Kokkos::View<T*> d_X("X", N);
  Kokkos::View<T*> d_Y("Y", N);
  Kokkos::View<T*> d_dY("dY", N);
  Kokkos::View<T*> d_dX("dX", N);

  auto hv_X  = Kokkos::create_mirror_view(d_X);
  auto hv_dY = Kokkos::create_mirror_view(d_dY);
  for (int i = 0; i < N; i++) { hv_X(i) = h_X[i]; hv_dY(i) = h_dY[i]; }
  Kokkos::deep_copy(d_X,  hv_X);
  Kokkos::deep_copy(d_dY, hv_dY);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("swish", N, KOKKOS_LAMBDA(int idx) {
      d_Y(idx) = d_X(idx) / (T(1) + Kokkos::exp(-d_X(idx)));
    });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Swish kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("swish_grad", N, KOKKOS_LAMBDA(int idx) {
      d_dX(idx) = d_dY(idx) * (d_Y(idx) + (T(1) - d_Y(idx)) / (T(1) + Kokkos::exp(-d_X(idx))));
    });
    Kokkos::fence();
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SwishGradient kernel: %f (us)\n", (time * 1e-3f) / repeat);

  auto hv_Y  = Kokkos::create_mirror_view(d_Y);
  auto hv_dX = Kokkos::create_mirror_view(d_dX);
  Kokkos::deep_copy(hv_Y,  d_Y);
  Kokkos::deep_copy(hv_dX, d_dX);
  for (int i = 0; i < N; i++) { h_Y[i] = hv_Y(i); h_dX[i] = hv_dX(i); }

  // Verify
  reference(N, h_X, r_Y, r_dX, h_dY);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabs((double)(h_dX[i] - r_dX[i])) > 1e-3 ||
        fabs((double)(h_Y[i]  - r_Y[i]))  > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_X); free(h_Y); free(h_dX); free(h_dY);
  free(r_dX); free(r_Y);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    eval_swish<float>(N, repeat);
  }
  Kokkos::finalize();
  return 0;
}
