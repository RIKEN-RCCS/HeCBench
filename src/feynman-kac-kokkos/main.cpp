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
  int n_inside = 0;
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

  const double inv_a2 = 1.0 / (a * a);
  const double inv_b2 = 1.0 / (b * b);
  const double potential_constant = inv_a2 + inv_b2;
  const double potential_x_factor = 2.0 * inv_a2 * inv_a2;
  const double potential_y_factor = 2.0 * inv_b2 * inv_b2;

  for (int j = 0; j < nj; j++) {
    const double x = ((double)(nj - j) * (-a) + (double)(j - 1) * a) /
                     (double)(nj - 1);
    for (int i = 0; i < ni; i++) {
      const double y = ((double)(ni - i) * (-b) + (double)(i - 1) * b) /
                       (double)(ni - 1);
      if (x * x * inv_a2 + y * y * inv_b2 <= 1.0) {
        n_inside++;
      }
    }
  }

  Kokkos::initialize(argc, argv);
  {
    long total_time = 0;

    for (int iter = 0; iter < repeat; iter++) {
      auto start = std::chrono::steady_clock::now();

      double iteration_err = 0.0;

      Kokkos::parallel_reduce(
          "feynman_kac",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nj, ni}),
          KOKKOS_LAMBDA(int j, int i, double &local_err) {
            double x = ((double)(nj - j) * (-a) + (double)(j - 1) * a) /
                       (double)(nj - 1);
            double y = ((double)(ni - i) * (-b) + (double)(i - 1) * b) /
                       (double)(ni - 1);

            double chk = x * x * inv_a2 + y * y * inv_b2;

            if (chk <= 1.0) {
              double w_exact = exp(chk - 1.0);
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

                  double vs = potential_x_factor * x1 * x1 +
                              potential_y_factor * x2 * x2 + potential_constant;
                  x1 = x1 + dx;
                  x2 = x2 + dy;
                  double vh = potential_x_factor * x1 * x1 +
                              potential_y_factor * x2 * x2 + potential_constant;

                  double we = (1.0 - h * vs) * w;
                  w = w - 0.5 * h * (vh * we + vs * w);

                  chk2 = x1 * x1 * inv_a2 + x2 * x2 * inv_b2;
                }
                wt += w;
              }
              wt /= (double)(N);
              double diff = w_exact - wt;
              double contrib = diff * diff;
              local_err += contrib;
            }
          }, iteration_err);
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      err = iteration_err;
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
