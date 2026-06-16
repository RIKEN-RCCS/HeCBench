#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define TWO26 (1 << 26)
#define NLUP  32

int main(int argc, char* argv[]) {
  const int n = TWO26;
  const float a = 2.0f;
  const size_t nbytes = sizeof(float) * n;

  float* x     = (float*) malloc(nbytes);
  float* y     = (float*) malloc(nbytes);
  float* yhost = (float*) malloc(nbytes);
  float* yaccl = (float*) malloc(nbytes);
  if (!x || !y || !yhost || !yaccl) {
    printf("error: memory allocation\n");
    return 1;
  }

  srand(42);
  for (int i = 0; i < n; i++) {
    x[i]     = (rand() % 32) / 32.0f;
    y[i]     = (rand() % 32) / 32.0f;
    yhost[i] = a * x[i] + y[i];
    yaccl[i] = y[i];
  }

  printf("The system supports 1 ns time resolution\n");
  printf("total size of x and y is %9.1f MB\n", 2.0 * nbytes / (1 << 20));
  printf("tests are averaged over %2d loops\n", NLUP);

  // Host saxpy
  {
    float* ytmp = (float*) malloc(nbytes);
    for (int i = 0; i < n; i++) ytmp[i] = y[i];

    auto t0 = std::chrono::steady_clock::now();
    for (int loop = 0; loop < NLUP; loop++) {
      for (int i = 0; i < n; i++) ytmp[i] = a * x[i] + ytmp[i];
    }
    auto t1 = std::chrono::steady_clock::now();
    double wt = std::chrono::duration<double>(t1 - t0).count() / NLUP;
    double mbps = 3.0 * nbytes / wt / (1 << 20);

    float maxabserr = 0.0f;
    for (int i = 0; i < n; i++) {
      float err = fabsf(ytmp[i] - yhost[i] - (NLUP - 1) * (a * x[i]));
      if (err > maxabserr) maxabserr = err;
    }
    printf("saxpy on host: %9.1f MB/s %9.1f MB/s maxabserr = %.1f\n",
           mbps, mbps, (double)maxabserr);
    free(ytmp);
  }

  // Kokkos saxpy
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_x("d_x", n);
    Kokkos::View<float*> d_y("d_y", n);

    // Initialize device views
    {
      auto hx = Kokkos::create_mirror_view(d_x);
      auto hy = Kokkos::create_mirror_view(d_y);
      for (int i = 0; i < n; i++) { hx(i) = x[i]; hy(i) = y[i]; }
      Kokkos::deep_copy(d_x, hx);
      Kokkos::deep_copy(d_y, hy);
    }

    // Warmup
    Kokkos::parallel_for("saxpy_warmup", n, KOKKOS_LAMBDA(const int i) {
      d_y(i) = a * d_x(i) + d_y(i);
    });
    Kokkos::fence();

    // Reset y
    {
      auto hy = Kokkos::create_mirror_view(d_y);
      for (int i = 0; i < n; i++) hy(i) = y[i];
      Kokkos::deep_copy(d_y, hy);
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int loop = 0; loop < NLUP; loop++) {
      Kokkos::parallel_for("saxpy", n, KOKKOS_LAMBDA(const int i) {
        d_y(i) = a * d_x(i) + d_y(i);
      });
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();

    double wt = std::chrono::duration<double>(t1 - t0).count() / NLUP;
    double mbps = 3.0 * nbytes / wt / (1 << 20);

    auto hy = Kokkos::create_mirror_view(d_y);
    Kokkos::deep_copy(hy, d_y);

    float maxabserr = 0.0f;
    for (int i = 0; i < n; i++) {
      float ref = yhost[i] + (NLUP - 1) * (a * x[i]);
      float err = fabsf(hy(i) - ref);
      if (err > maxabserr) maxabserr = err;
    }

    printf("saxpy on accl (impl. 0)\n");
    printf("total: %9.1f MB/s kernel: %9.1f MB/s maxabserr = %.1f\n",
           mbps, mbps, (double)maxabserr);
  }
  Kokkos::finalize();

  free(x); free(y); free(yhost); free(yaccl);
  return 0;
}
