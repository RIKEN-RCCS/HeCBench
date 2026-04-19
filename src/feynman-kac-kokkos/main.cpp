/*
  Kokkos port of FEYNMAN_KAC_2D.

  Solves (1/2) Laplacian U - V(X,Y) * U = 0 inside an elliptic domain
  using the Feynman-Kac formula and Monte Carlo random walks.

  Original C version by Wesley Petersen / John Burkardt.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "util.h"

// Combined reduction value: accumulated squared error and count of inside points
struct ErrCount {
  double err;
  int count;

  KOKKOS_INLINE_FUNCTION ErrCount() : err(0.0), count(0) {}
  KOKKOS_INLINE_FUNCTION ErrCount(double e, int c) : err(e), count(c) {}

  KOKKOS_INLINE_FUNCTION ErrCount& operator+=(const ErrCount& rhs) {
    err   += rhs.err;
    count += rhs.count;
    return *this;
  }
};

namespace Kokkos {
  template<>
  struct reduction_identity<ErrCount> {
    KOKKOS_FORCEINLINE_FUNCTION static ErrCount sum() { return ErrCount(0.0, 0); }
  };
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const double a = 2.0;
  const double b = 1.0;
  const int    dim = 2;
  const double h = 0.001;
  const int    N = 1000;
  const int    seed = 123456789;

  printf("\n");
  printf("FEYNMAN_KAC_2D (Kokkos):\n");
  printf("\n");
  printf("  The calculation takes place inside a 2D ellipse.\n");
  printf("\n");
  printf("  Each solution will be estimated by computing %d trajectories\n", N);
  printf("\n");
  printf("    (X/A)^2 + (Y/B)^2 = 1\n");
  printf("\n");
  printf("  A = %f\n", a);
  printf("  B = %f\n", b);
  printf("  Stepsize H = %6.4f\n", h);

  const double rth = sqrt((double)dim * h);

  const int nj = 128;
  const int ni = 1 + i4_ceiling(a / b) * (nj - 1);

  printf("\n");
  printf("  X coordinate marked by %d points\n", ni);
  printf("  Y coordinate marked by %d points\n", nj);

  Kokkos::initialize(argc, argv);
  {
    long total_time = 0;
    double err = 0.0;
    int    n_inside = 0;

    for (int rep = 0; rep < repeat; rep++) {
      ErrCount result(0.0, 0);

      auto kstart = std::chrono::steady_clock::now();

      Kokkos::parallel_reduce(
        "feynman_kac",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nj, ni}),
        KOKKOS_LAMBDA(int j, int i, ErrCount& lval) {
          const double x = ((double)(nj - j    ) * (-a)
                          + (double)(     j - 1) *   a)
                          / (double)(nj      - 1);

          const double y = ((double)(ni - i    ) * (-b)
                          + (double)(     i - 1) *   b)
                          / (double)(ni      - 1);

          double chk = pow(x / a, 2.0) + pow(y / b, 2.0);

          if (1.0 < chk) {
            // outside ellipse — no contribution
            return;
          }

          lval.count++;

          const double w_exact = exp(pow(x / a, 2.0) + pow(y / b, 2.0) - 1.0);

          // per-thread seed derived from global seed and thread index
          const int tid = j * ni + i;
          int seed_i = seed + tid;

          double wt = 0.0;
          for (int k = 0; k < N; k++) {
            double x1 = x;
            double x2 = y;
            double w  = 1.0;
            double chk2 = 0.0;

            while (chk2 < 1.0) {
              double dx, dy;

              double ut = r8_uniform_01(&seed_i);
              if (ut < 1.0 / 2.0) {
                double us = r8_uniform_01(&seed_i) - 0.5;
                dx = (us < 0.0) ? -rth : rth;
              } else {
                dx = 0.0;
              }

              ut = r8_uniform_01(&seed_i);
              if (ut < 1.0 / 2.0) {
                double us = r8_uniform_01(&seed_i) - 0.5;
                dy = (us < 0.0) ? -rth : rth;
              } else {
                dy = 0.0;
              }

              double vs = potential(a, b, x1, x2);
              x1 += dx;
              x2 += dy;

              double vh = potential(a, b, x1, x2);
              double we = (1.0 - h * vs) * w;
              w = w - 0.5 * h * (vh * we + vs * w);

              chk2 = pow(x1 / a, 2.0) + pow(x2 / b, 2.0);
            }
            wt += w;
          }

          wt /= (double)N;
          lval.err += pow(w_exact - wt, 2.0);
        },
        result);

      Kokkos::fence();
      auto kend = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

      err      = result.err;
      n_inside = result.count;
    }

    printf("Average kernel time: %lf (s)\n", total_time * 1e-9 / repeat);

    err = sqrt(err / (double)n_inside);
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
