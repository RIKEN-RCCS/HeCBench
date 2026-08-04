#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <vector>

#define NUMBINS 64
#define D2R (M_PI / 180.0)

struct CartesianData {
  std::vector<double> x, y, z;
  int n;
  CartesianData(int n_) : x(n_), y(n_), z(n_), n(n_) {}
};

double* init_bins(int bins_per_dec, float min_angle, float max_angle,
                  int angle_units, int *nbins);
CartesianData generate_data(int n, unsigned seed);

void compute_histogram(
    const Kokkos::View<double*> &x1, const Kokkos::View<double*> &y1, const Kokkos::View<double*> &z1,
    const Kokkos::View<double*> &x2, const Kokkos::View<double*> &y2, const Kokkos::View<double*> &z2,
    int n1, int n2,
    Kokkos::View<long long*> &histo,
    const Kokkos::View<double*> &d_binb, int nbins);
