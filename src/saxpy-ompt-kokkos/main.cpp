#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>

#include "saxpy_kernels.h"

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

  run_host_saxpy(x, y, yhost, n, a, NLUP, nbytes);

  Kokkos::initialize(argc, argv);
  {
    run_kokkos_saxpy(x, y, yhost, n, a, NLUP, nbytes);
  }
  Kokkos::finalize();

  free(x); free(y); free(yhost); free(yaccl);
  return 0;
}
