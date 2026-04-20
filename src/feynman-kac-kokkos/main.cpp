/*
  Purpose:

    MAIN is the main program for FEYNMAN_KAC_2D (Kokkos port).

  Licensing:
    This code is distributed under the GNU LGPL license.

  Original C 2D version by John Burkardt.
  Kokkos port.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
int i4_ceiling(double x) {
  int value = (int)x;
  if (value < x) value = value + 1;
  return value;
}

KOKKOS_INLINE_FUNCTION
double potential(double a, double b, double x, double y) {
  return 2.0 * (pow(x / a / a, 2.0) + pow(y / b / b, 2.0))
             + 1.0 / a / a + 1.0 / b / b;
}

// LCG random number generator (thread-local, non-atomic)
KOKKOS_INLINE_FUNCTION
double r8_uniform_01(int *seed) {
  int k = *seed / 127773;
  *seed = 16807 * (*seed - k * 127773) - k * 2836;
  if (*seed < 0) *seed = *seed + 2147483647;
  return (double)(*seed) * 4.656612875E-10;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  double a = 2.0;
  double b = 1.0;
  int dim = 2;
  double err;
  double h = 0.001;
  int N = 1000;
  int n_inside;
  int ni;
  int nj;
  double rth;
  int seed = 123456789;

  printf("\n");
  printf("FEYNMAN_KAC_2D:\n");
  printf("\n");
  printf("  Program parameters:\n");
  printf("\n");
  printf("  The calculation takes place inside a 2D ellipse.\n");
  printf("  A rectangular grid of points will be defined.\n");
  printf("  The solution will be estimated for those grid points\n");
  printf("  that lie inside the ellipse.\n");
  printf("\n");
  printf("  Each solution will be estimated by computing %d trajectories\n", N);
  printf("  from the point to the boundary.\n");
  printf("\n");
  printf("    (X/A)^2 + (Y/B)^2 = 1\n");
  printf("\n");
  printf("  The ellipse parameters A, B are set to:\n");
  printf("\n");
  printf("    A = %f\n", a);
  printf("    B = %f\n", b);
  printf("  Stepsize H = %6.4f\n", h);

  rth = sqrt((double)dim * h);

  nj = 128;
  ni = 1 + i4_ceiling(a / b) * (nj - 1);

  printf("\n");
  printf("  X coordinate marked by %d points\n", ni);
  printf("  Y coordinate marked by %d points\n", nj);

  Kokkos::initialize(argc, argv);
  {
    long total_time = 0;

    for (int iter = 0; iter < repeat; iter++) {
      // Use Views with atomic add for reductions
      Kokkos::View<double> d_err("err");
      Kokkos::View<int> d_n_inside("n_inside");
      Kokkos::deep_copy(d_err, 0.0);
      Kokkos::deep_copy(d_n_inside, 0);

      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for(
          "feynman_kac",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nj, ni}),
          KOKKOS_LAMBDA(int j, int i) {
            double x = ((double)(nj - j) * (-a) + (double)(j - 1) * a) /
                       (double)(nj - 1);
            double y = ((double)(ni - i) * (-b) + (double)(i - 1) * b) /
                       (double)(ni - 1);

            double chk = pow(x / a, 2.0) + pow(y / b, 2.0);

            if (1.0 < chk) {
              // outside: do nothing (w_exact = wt = 1, contribution = 0)
            } else {
              Kokkos::atomic_add(&d_n_inside(), 1);

              double w_exact = exp(pow(x / a, 2.0) + pow(y / b, 2.0) - 1.0);
              double wt = 0.0;

              // Per-thread seed derived from position
              int local_seed = seed + j * ni + i;

              for (int k = 0; k < N; k++) {
                double x1 = x;
                double x2 = y;
                double w = 1.0;
                double chk2 = 0.0;
                while (chk2 < 1.0) {
                  double ut = r8_uniform_01(&local_seed);
                  double dx = 0.0;
                  if (ut < 0.5) {
                    double us = r8_uniform_01(&local_seed) - 0.5;
                    dx = (us < 0.0) ? -rth : rth;
                  }

                  ut = r8_uniform_01(&local_seed);
                  double dy = 0.0;
                  if (ut < 0.5) {
                    double us = r8_uniform_01(&local_seed) - 0.5;
                    dy = (us < 0.0) ? -rth : rth;
                  }

                  double vs = potential(a, b, x1, x2);
                  x1 = x1 + dx;
                  x2 = x2 + dy;
                  double vh = potential(a, b, x1, x2);

                  double we = (1.0 - h * vs) * w;
                  w = w - 0.5 * h * (vh * we + vs * w);

                  chk2 = pow(x1 / a, 2.0) + pow(x2 / b, 2.0);
                }
                wt += w;
              }
              wt /= (double)(N);
              double contrib = pow(w_exact - wt, 2.0);
              Kokkos::atomic_add(&d_err(), contrib);
            }
          });
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      // Copy results back (last iteration)
      if (iter == repeat - 1) {
        auto h_err = Kokkos::create_mirror_view(d_err);
        auto h_n_inside = Kokkos::create_mirror_view(d_n_inside);
        Kokkos::deep_copy(h_err, d_err);
        Kokkos::deep_copy(h_n_inside, d_n_inside);
        err = h_err();
        n_inside = h_n_inside();
      }
    }
    printf("Average kernel time: %lf (s)\n", total_time * 1e-9 / repeat);

    err = sqrt(err / (double)(n_inside));
    printf("\n");
    printf("  RMS absolute error in solution = %e\n", err);
    printf("\n");
    printf("FEYNMAN_KAC_2D:\n");
    printf("  Normal end of execution.\n");
    printf("\n");
  }
  Kokkos::finalize();

  return 0;
}
