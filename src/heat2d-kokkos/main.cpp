#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <Lx> <Ly> <niter>\n", argv[0]);
    return 1;
  }

  const int Lx    = atoi(argv[1]);
  const int Ly    = atoi(argv[2]);
  const int niter = atoi(argv[3]);

  if (niter < 1) {
    printf("niter must be >= 1\n");
    return 1;
  }

  // Fixed diffusion coefficient
  const float sigma = 0.01f;
  const float delta = sigma / (1.0f + 4.0f * sigma);
  const float norm  = 1.0f  / (1.0f + 4.0f * sigma);

  printf(" Ly,Lx = %d,%d\n", Ly, Lx);
  printf(" niter = %d\n", niter);

  const int N = Lx * Ly;

  // Initialise with a sinusoidal field on the host
  float *h_arr = (float *) malloc(N * sizeof(float));
  for (int y = 0; y < Ly; y++)
    for (int x = 0; x < Lx; x++)
      h_arr[y * Lx + x] = sinf(2.0f * (float)M_PI * x / Lx)
                         * cosf(2.0f * (float)M_PI * y / Ly);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_in ("d_in",  N);
    Kokkos::View<float*> d_out("d_out", N);

    auto h_in = Kokkos::create_mirror_view(d_in);
    for (int i = 0; i < N; i++) h_in(i) = h_arr[i];
    Kokkos::deep_copy(d_in, h_in);
    Kokkos::fence();

    auto kstart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < niter; iter++) {
      Kokkos::parallel_for("lapl_iter", N,
        KOKKOS_LAMBDA(const int i) {
          const int x   = i % Lx;
          const int y   = i / Lx;
          const int v00 = y * Lx + x;
          const int v0p = y * Lx + (x + 1) % Lx;
          const int v0m = y * Lx + (Lx + x - 1) % Lx;
          const int vp0 = ((y + 1) % Ly) * Lx + x;
          const int vm0 = ((Ly + y - 1) % Ly) * Lx + x;
          d_out(v00) = norm * d_in(v00)
                     + delta * (d_in(v0p) + d_in(v0m) + d_in(vp0) + d_in(vm0));
        });
      Kokkos::fence();
      // Swap in/out for next iteration
      Kokkos::deep_copy(d_in, d_out);
    }

    auto kend  = std::chrono::steady_clock::now();
    double t   = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count()
                 * 1e-3 / niter;  // microseconds per iteration

    printf("Device: iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter,"
           " BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
           niter, Lx, Ly, t,
           N * sizeof(float) * 2.0 / (t * 1.0e3),
           N * 6.0 / (t * 1.0e3));

    auto h_out = Kokkos::create_mirror_view(d_in);
    Kokkos::deep_copy(h_out, d_in);
    for (int i = 0; i < N; i++) h_arr[i] = h_out(i);
  }
  Kokkos::finalize();

  // Basic sanity check: all values finite
  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (!isfinite(h_arr[i])) {
      printf("Non-finite value at index %d\n", i);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_arr);
  return 0;
}
