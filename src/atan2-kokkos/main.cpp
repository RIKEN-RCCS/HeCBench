// Kokkos port of the approximate atan2 benchmark.
// Three output types: float (f32), int (i32), short (i16).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "atan2_funcs.h"

// ─── Reference (host) ────────────────────────────────────────────────────────

void reference_f(const int n, const float *x, const float *y, float *r) {
  for (int i = 0; i < n; i++) {
    const float vy = y[i], vx = x[i];
    r[i] = safe_atan2f< 3>(vy, vx) + safe_atan2f< 5>(vy, vx) +
           safe_atan2f< 7>(vy, vx) + safe_atan2f< 9>(vy, vx) +
           safe_atan2f<11>(vy, vx) + safe_atan2f<13>(vy, vx) +
           safe_atan2f<15>(vy, vx);
  }
}

void reference_i(const int n, const float *x, const float *y, int *r) {
  for (int i = 0; i < n; i++) {
    const float vy = y[i], vx = x[i];
    r[i] = unsafe_atan2i< 3>(vy, vx) + unsafe_atan2i< 5>(vy, vx) +
           unsafe_atan2i< 7>(vy, vx) + unsafe_atan2i< 9>(vy, vx) +
           unsafe_atan2i<11>(vy, vx) + unsafe_atan2i<13>(vy, vx) +
           unsafe_atan2i<15>(vy, vx);
  }
}

void reference_s(const int n, const float *x, const float *y, short *r) {
  for (int i = 0; i < n; i++) {
    const float vy = y[i], vx = x[i];
    r[i] = (short)(unsafe_atan2s<3>(vy, vx) + unsafe_atan2s<5>(vy, vx) +
                   unsafe_atan2s<7>(vy, vx) + unsafe_atan2s<9>(vy, vx));
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of coordinates> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  float *x  = (float*) malloc(n * sizeof(float));
  float *y  = (float*) malloc(n * sizeof(float));
  float *hf = (float*) malloc(n * sizeof(float));
  int   *hi = (int*)   malloc(n * sizeof(int));
  short *hs = (short*) malloc(n * sizeof(short));
  float *rf = (float*) malloc(n * sizeof(float));
  int   *ri = (int*)   malloc(n * sizeof(int));
  short *rs = (short*) malloc(n * sizeof(short));

  srand(123);
  for (int i = 0; i < n; i++) {
    x[i] = rand() / (float)RAND_MAX + 1.57f;
    y[i] = rand() / (float)RAND_MAX + 1.57f;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> dx("x",  n);
    Kokkos::View<float*> dy("y",  n);
    Kokkos::View<float*> df("rf", n);
    Kokkos::View<int*>   di("ri", n);
    Kokkos::View<short*> ds("rs", n);

    {
      auto hx = Kokkos::View<float*, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(x, n);
      auto hy = Kokkos::View<float*, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(y, n);
      Kokkos::deep_copy(dx, hx);
      Kokkos::deep_copy(dy, hy);
    }

    // ---- f32 kernel ----
    printf("\n======== output type is f32 ========\n");
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < repeat; k++) {
      Kokkos::parallel_for("atan2_f", n, KOKKOS_LAMBDA(const int i) {
        const float vy = dy(i), vx = dx(i);
        df(i) = safe_atan2f< 3>(vy, vx) + safe_atan2f< 5>(vy, vx) +
                safe_atan2f< 7>(vy, vx) + safe_atan2f< 9>(vy, vx) +
                safe_atan2f<11>(vy, vx) + safe_atan2f<13>(vy, vx) +
                safe_atan2f<15>(vy, vx);
      });
      Kokkos::fence();
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat);

    {
      auto h = Kokkos::View<float*, Kokkos::HostSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hf, n);
      Kokkos::deep_copy(h, df);
    }
    reference_f(n, x, y, rf);
    float err = 0.f;
    for (int i = 0; i < n; i++) {
      float d = rf[i] - hf[i];
      if (fabsf(d) > 1e-3f) err += d * d;
    }
    printf("RMSE: %f\n", sqrtf(err / n));

    // ---- i32 kernel ----
    printf("\n======== output type is i32 ========\n");
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < repeat; k++) {
      Kokkos::parallel_for("atan2_i", n, KOKKOS_LAMBDA(const int i) {
        const float vy = dy(i), vx = dx(i);
        di(i) = unsafe_atan2i< 3>(vy, vx) + unsafe_atan2i< 5>(vy, vx) +
                unsafe_atan2i< 7>(vy, vx) + unsafe_atan2i< 9>(vy, vx) +
                unsafe_atan2i<11>(vy, vx) + unsafe_atan2i<13>(vy, vx) +
                unsafe_atan2i<15>(vy, vx);
      });
      Kokkos::fence();
    }
    t1 = std::chrono::steady_clock::now();
    printf("Average execution time: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat);

    {
      auto h = Kokkos::View<int*, Kokkos::HostSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hi, n);
      Kokkos::deep_copy(h, di);
    }
    reference_i(n, x, y, ri);
    err = 0.f;
    for (int i = 0; i < n; i++) {
      float d = (float)(ri[i] - hi[i]);
      if (abs(ri[i] - hi[i]) > 0) err += d * d;
    }
    printf("RMSE: %f\n", sqrtf(err / n));

    // ---- i16 kernel ----
    printf("\n======== output type is i16 ========\n");
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < repeat; k++) {
      Kokkos::parallel_for("atan2_s", n, KOKKOS_LAMBDA(const int i) {
        const float vy = dy(i), vx = dx(i);
        ds(i) = (short)(unsafe_atan2s<3>(vy, vx) + unsafe_atan2s<5>(vy, vx) +
                        unsafe_atan2s<7>(vy, vx) + unsafe_atan2s<9>(vy, vx));
      });
      Kokkos::fence();
    }
    t1 = std::chrono::steady_clock::now();
    printf("Average execution time: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat);

    {
      auto h = Kokkos::View<short*, Kokkos::HostSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>(hs, n);
      Kokkos::deep_copy(h, ds);
    }
    reference_s(n, x, y, rs);
    err = 0.f;
    for (int i = 0; i < n; i++) {
      float d = (float)(rs[i] - hs[i]);
      if (abs(rs[i] - hs[i]) > 0) err += d * d;
    }
    printf("RMSE: %f\n", sqrtf(err / n));
  }
  Kokkos::finalize();

  free(x); free(y);
  free(hf); free(hi); free(hs);
  free(rf); free(ri); free(rs);
  return 0;
}
