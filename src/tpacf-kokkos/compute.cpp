#include "tpacf.h"

void compute_histogram(
    const Kokkos::View<double*> &x1, const Kokkos::View<double*> &y1, const Kokkos::View<double*> &z1,
    const Kokkos::View<double*> &x2, const Kokkos::View<double*> &y2, const Kokkos::View<double*> &z2,
    int n1, int n2,
    Kokkos::View<long long*> &histo,
    const Kokkos::View<double*> &d_binb, int nbins)
{
  Kokkos::parallel_for("acf", n1, KOKKOS_LAMBDA(const int i) {
    double xi = x1(i), yi = y1(i), zi = z1(i);
    for (int j = 0; j < n2; j++) {
      double dot = xi * x2(j) + yi * y2(j) + zi * z2(j);
      if (dot >  1.0) dot =  1.0;
      if (dot < -1.0) dot = -1.0;

      int bin = nbins - 1;
      for (int b = 0; b < nbins - 1; b++) {
        if (dot > d_binb(b)) { bin = b; break; }
      }
      Kokkos::atomic_add(&histo(bin), 1LL);
    }
  });
  Kokkos::fence();
}
