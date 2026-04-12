// Kokkos port of lebesgue-cuda
// Utility functions and main() from lebesgue-cuda (pure C++, no CUDA).
// Only lebesgue_function() is replaced with a Kokkos implementation.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <chrono>

#include <Kokkos_Core.hpp>

// ---------------------------------------------------------------------------
// lebesgue_function: Kokkos implementation replacing the CUDA kernel
// ---------------------------------------------------------------------------
double lebesgue_function(int n, double x[], int nfun, double xfun[]) {
  Kokkos::View<double*> d_x("d_x", n);
  Kokkos::View<double*> d_xfun("d_xfun", nfun);

  auto h_x    = Kokkos::create_mirror_view(d_x);
  auto h_xfun = Kokkos::create_mirror_view(d_xfun);

  for (int i = 0; i < n;    i++) h_x(i)    = x[i];
  for (int i = 0; i < nfun; i++) h_xfun(i) = xfun[i];

  Kokkos::deep_copy(d_x,    h_x);
  Kokkos::deep_copy(d_xfun, h_xfun);

  double lmax = 0.0;

  Kokkos::parallel_reduce("lebesgue", nfun,
    KOKKOS_LAMBDA(int j, double& local_max) {
      // n_max is 11 in tests; use a fixed-size local array
      double linterp[12];
      for (int i = 0; i < n; i++) linterp[i] = 1.0;

      for (int i1 = 0; i1 < n; i1++)
        for (int i2 = 0; i2 < n; i2++)
          if (i1 != i2)
            linterp[i1] *= (d_xfun(j) - d_x(i2)) / (d_x(i1) - d_x(i2));

      double t = 0.0;
      for (int i = 0; i < n; i++)
        t += fabs(linterp[i]);

      if (t > local_max) local_max = t;
    },
    Kokkos::Max<double>(lmax));

  Kokkos::fence();
  return lmax;
}

// ---------------------------------------------------------------------------
// Utility functions (unchanged from lebesgue-cuda/utils.cpp)
// ---------------------------------------------------------------------------

double *chebyshev1(int n) {
  const double r8_pi = 3.141592653589793;
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    double angle = r8_pi * (double)(2*i+1) / (double)(2*n);
    x[i] = cos(angle);
  }
  return x;
}

double *chebyshev2(int n) {
  const double r8_pi = 3.141592653589793;
  double *x = (double *) malloc(n * sizeof(double));
  if (n == 1) {
    x[0] = 0.0;
  } else {
    for (int i = 0; i < n; i++) {
      double angle = r8_pi * (double)(n-i-1) / (double)(n-1);
      x[i] = cos(angle);
    }
  }
  return x;
}

double *chebyshev3(int n) {
  const double r8_pi = 3.141592653589793;
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    double angle = r8_pi * (double)(2*n - 2*i - 1) / (double)(2*n + 1);
    x[i] = cos(angle);
  }
  return x;
}

double *chebyshev4(int n) {
  const double r8_pi = 3.141592653589793;
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    double angle = r8_pi * (double)(2*n - 2*i) / (double)(2*n + 1);
    x[i] = cos(angle);
  }
  return x;
}

double *equidistant1(int n) {
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++)
    x[i] = (double)(-n + 1 + 2*i) / (double)(n + 1);
  return x;
}

double *equidistant2(int n) {
  double *x = (double *) malloc(n * sizeof(double));
  if (n == 1) {
    x[0] = 0.0;
  } else {
    for (int i = 0; i < n; i++)
      x[i] = (double)(-n + 1 + 2*i) / (double)(n - 1);
  }
  return x;
}

double *equidistant3(int n) {
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++)
    x[i] = (double)(-n + 1 + 2*i) / (double)(n);
  return x;
}

double *fejer1(int n) {
  const double r8_pi = 3.141592653589793;
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    double theta = r8_pi * (double)(2*n - 1 - 2*i) / (double)(2*n);
    x[i] = cos(theta);
  }
  return x;
}

double *fejer2(int n) {
  const double r8_pi = 3.141592653589793;
  double *x = (double *) malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    double theta = r8_pi * (double)(n - i) / (double)(n + 1);
    x[i] = cos(theta);
  }
  return x;
}

double lebesgue_constant(int n, double x[], int nfun, double xfun[]) {
  if (n > 1)
    return lebesgue_function(n, x, nfun, xfun);
  else
    return 1.0;
}

double *r8vec_linspace_new(int n, double a, double b) {
  double *x = (double *) malloc(n * sizeof(double));
  if (n == 1) {
    x[0] = (a + b) / 2.0;
  } else {
    for (int i = 0; i < n; i++)
      x[i] = ((double)(n-1-i) * a + (double)(i) * b) / (double)(n-1);
  }
  return x;
}

void r8vec_print(int n, double a[], const char *title) {
  fprintf(stdout, "\n");
  fprintf(stdout, "%s\n", title);
  fprintf(stdout, "\n");
  for (int i = 0; i < n; i++)
    fprintf(stdout, "  %8d: %14g\n", i, a[i]);
}

void timestamp() {
#define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;
  now = time(NULL);
  tm = localtime(&now);
  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);
  fprintf(stdout, "%s\n", time_buffer);
#undef TIME_SIZE
}

// ---------------------------------------------------------------------------
// Test functions (unchanged from lebesgue-cuda/main.cpp)
// ---------------------------------------------------------------------------

#define RUN_TEST(name, point_fn, label_str, n_max_val) \
void name(int nfun) { \
  double *l; \
  int n; \
  const int n_max = n_max_val; \
  double *x; \
  double *xfun; \
  printf("\n"); \
  printf(#name ":\n"); \
  xfun = r8vec_linspace_new(nfun, -1.0, +1.0); \
  l = (double *) malloc(n_max * sizeof(double)); \
  float total_time = 0.f; \
  for (n = 1; n <= n_max; n++) { \
    x = point_fn(n); \
    auto start = std::chrono::steady_clock::now(); \
    l[n-1] = lebesgue_constant(n, x, nfun, xfun); \
    auto end = std::chrono::steady_clock::now(); \
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); \
    free(x); \
  } \
  printf("  Total kernel execution time %f (s)\n", total_time * 1e-9f); \
  r8vec_print(n_max, l, label_str " Lebesgue constants for N = 1 to 11:"); \
  n = 11; x = point_fn(n); \
  r8vec_print(n, x, label_str " points for N = 11"); \
  free(l); free(x); free(xfun); \
}

RUN_TEST(test01, chebyshev1,   "Chebyshev1",   11)
RUN_TEST(test02, chebyshev2,   "Chebyshev2",   11)
RUN_TEST(test03, chebyshev3,   "Chebyshev3",   11)
RUN_TEST(test04, chebyshev4,   "Chebyshev4",   11)
RUN_TEST(test05, equidistant1, "Equidistant1", 11)
RUN_TEST(test06, equidistant2, "Equidistant2", 11)
RUN_TEST(test07, equidistant3, "Equidistant3", 11)
RUN_TEST(test08, fejer1,       "Fejer1",       11)
RUN_TEST(test09, fejer2,       "Fejer2",       11)

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of points in an interval> <repeat>\n", argv[0]);
    return 1;
  }
  int nfun   = atoi(argv[1]);
  int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    printf("\nLEBESGUE_TEST\n");

    for (int i = 0; i < repeat; i++) {
      timestamp();
      test01(nfun); test02(nfun); test03(nfun);
      test04(nfun); test05(nfun); test06(nfun);
      test07(nfun); test08(nfun); test09(nfun);
      timestamp();
    }
  }
  Kokkos::finalize();
  return 0;
}
