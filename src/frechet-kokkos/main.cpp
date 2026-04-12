/*
 * Compute the discrete Frechet distance between two curves specified by
 * discrete ordered points in n-dimensional space.
 *
 * Based on `DiscreteFrechetDist` by Zachary Danziger,
 * http://www.mathworks.com/matlabcentral/fileexchange/ \
 * 31922-discrete-frechet-distance
 *
 * Ported to Kokkos from CUDA by HeCBench, 2024.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <Kokkos_Core.hpp>

#define n_d 10000  /* Number of dimensions */

// ─── norm functions ──────────────────────────────────────────────────────────

KOKKOS_INLINE_FUNCTION
double norm1(int i, int j, const double *c1, const double *c2) {
  double dist = 0.0;
  for (int k = 0; k < n_d; k++) {
    double diff = c1[(i - 1) * n_d + k] - c2[(j - 1) * n_d + k];
    dist += fabs(diff);
  }
  return dist;
}

KOKKOS_INLINE_FUNCTION
double norm2(int i, int j, const double *c1, const double *c2) {
  double dist = 0.0;
  for (int k = 0; k < n_d; k++) {
    double diff = c1[(i - 1) * n_d + k] - c2[(j - 1) * n_d + k];
    dist += diff * diff;
  }
  return sqrt(dist);
}

KOKKOS_INLINE_FUNCTION
double norm3(int i, int j, const double *c1, const double *c2) {
  double dist = 0.0;
  for (int k = 0; k < n_d; k++) {
    double diff = c1[(i - 1) * n_d + k] - c2[(j - 1) * n_d + k];
    dist = fmax(dist, fabs(diff));
  }
  return dist;
}

// ─── recursive DFD functions (memoised into ca) ──────────────────────────────

KOKKOS_INLINE_FUNCTION
double recursive_norm1(int i, int j, int n_2, double *ca,
                       const double *c1, const double *c2) {
  double *ca_ij = ca + (i - 1) * n_2 + (j - 1);
  if (*ca_ij > -1.0) return *ca_ij;
  if (i == 1 && j == 1)
    *ca_ij = norm1(1, 1, c1, c2);
  else if (i > 1 && j == 1)
    *ca_ij = fmax(recursive_norm1(i-1, 1, n_2, ca, c1, c2), norm1(i, 1, c1, c2));
  else if (i == 1 && j > 1)
    *ca_ij = fmax(recursive_norm1(1, j-1, n_2, ca, c1, c2), norm1(1, j, c1, c2));
  else if (i > 1 && j > 1)
    *ca_ij = fmax(
        fmin(fmin(recursive_norm1(i-1, j,   n_2, ca, c1, c2),
                  recursive_norm1(i-1, j-1, n_2, ca, c1, c2)),
                  recursive_norm1(i,   j-1, n_2, ca, c1, c2)),
        norm1(i, j, c1, c2));
  else
    *ca_ij = INFINITY;
  return *ca_ij;
}

KOKKOS_INLINE_FUNCTION
double recursive_norm2(int i, int j, int n_2, double *ca,
                       const double *c1, const double *c2) {
  double *ca_ij = ca + (i - 1) * n_2 + (j - 1);
  if (*ca_ij > -1.0) return *ca_ij;
  if (i == 1 && j == 1)
    *ca_ij = norm2(1, 1, c1, c2);
  else if (i > 1 && j == 1)
    *ca_ij = fmax(recursive_norm2(i-1, 1, n_2, ca, c1, c2), norm2(i, 1, c1, c2));
  else if (i == 1 && j > 1)
    *ca_ij = fmax(recursive_norm2(1, j-1, n_2, ca, c1, c2), norm2(1, j, c1, c2));
  else if (i > 1 && j > 1)
    *ca_ij = fmax(
        fmin(fmin(recursive_norm2(i-1, j,   n_2, ca, c1, c2),
                  recursive_norm2(i-1, j-1, n_2, ca, c1, c2)),
                  recursive_norm2(i,   j-1, n_2, ca, c1, c2)),
        norm2(i, j, c1, c2));
  else
    *ca_ij = INFINITY;
  return *ca_ij;
}

KOKKOS_INLINE_FUNCTION
double recursive_norm3(int i, int j, int n_2, double *ca,
                       const double *c1, const double *c2) {
  double *ca_ij = ca + (i - 1) * n_2 + (j - 1);
  if (*ca_ij > -1.0) return *ca_ij;
  if (i == 1 && j == 1)
    *ca_ij = norm3(1, 1, c1, c2);
  else if (i > 1 && j == 1)
    *ca_ij = fmax(recursive_norm3(i-1, 1, n_2, ca, c1, c2), norm3(i, 1, c1, c2));
  else if (i == 1 && j > 1)
    *ca_ij = fmax(recursive_norm3(1, j-1, n_2, ca, c1, c2), norm3(1, j, c1, c2));
  else if (i > 1 && j > 1)
    *ca_ij = fmax(
        fmin(fmin(recursive_norm3(i-1, j,   n_2, ca, c1, c2),
                  recursive_norm3(i-1, j-1, n_2, ca, c1, c2)),
                  recursive_norm3(i,   j-1, n_2, ca, c1, c2)),
        norm3(i, j, c1, c2));
  else
    *ca_ij = INFINITY;
  return *ca_ij;
}

// ─── discrete Frechet distance driver ────────────────────────────────────────

void discrete_frechet_distance(const int s, const int n_1, const int n_2,
                                const int repeat) {
  int ca_size = n_1 * n_2;
  int c1_size = n_1 * n_d;
  int c2_size = n_2 * n_d;

  std::vector<double> ca_host(ca_size, -1.0);
  std::vector<double> c1_host(c1_size);
  std::vector<double> c2_host(c2_size);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  for (auto &v : c1_host) v = dis(gen);
  for (auto &v : c2_host) v = dis(gen);

  Kokkos::View<double *> d_ca("ca", ca_size);
  Kokkos::View<double *> d_c1("c1", c1_size);
  Kokkos::View<double *> d_c2("c2", c2_size);

  {
    auto h_ca = Kokkos::create_mirror_view(d_ca);
    auto h_c1 = Kokkos::create_mirror_view(d_c1);
    auto h_c2 = Kokkos::create_mirror_view(d_c2);
    for (int k = 0; k < ca_size; k++) h_ca(k) = ca_host[k];
    for (int k = 0; k < c1_size; k++) h_c1(k) = c1_host[k];
    for (int k = 0; k < c2_size; k++) h_c2(k) = c2_host[k];
    Kokkos::deep_copy(d_ca, h_ca);
    Kokkos::deep_copy(d_c1, h_c1);
    Kokkos::deep_copy(d_c2, h_c2);
  }

  Kokkos::fence();
  auto t_start = std::chrono::steady_clock::now();

  using Policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

  for (int k = 0; k < repeat; k++) {
    double *raw_ca = d_ca.data();
    double *raw_c1 = d_c1.data();
    double *raw_c2 = d_c2.data();

    if (s == 0) {
      Kokkos::parallel_for(
          "frechet_norm1",
          Policy2D({1, 1}, {n_1 + 1, n_2 + 1}),
          KOKKOS_LAMBDA(int i, int j) {
            recursive_norm1(i, j, n_2, raw_ca, raw_c1, raw_c2);
          });
    } else if (s == 1) {
      Kokkos::parallel_for(
          "frechet_norm2",
          Policy2D({1, 1}, {n_1 + 1, n_2 + 1}),
          KOKKOS_LAMBDA(int i, int j) {
            recursive_norm2(i, j, n_2, raw_ca, raw_c1, raw_c2);
          });
    } else {
      Kokkos::parallel_for(
          "frechet_norm3",
          Policy2D({1, 1}, {n_1 + 1, n_2 + 1}),
          KOKKOS_LAMBDA(int i, int j) {
            recursive_norm3(i, j, n_2, raw_ca, raw_c1, raw_c2);
          });
    }
    Kokkos::fence();
  }

  auto t_end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     t_end - t_start).count();
  printf("Average kernel execution time %f (s)\n", (elapsed * 1e-9) / repeat);

  auto h_ca = Kokkos::create_mirror_view(d_ca);
  Kokkos::deep_copy(h_ca, d_ca);

  double checkSum = 0.0;
  for (int k = 0; k < ca_size; k++) checkSum += h_ca(k);
  printf("checkSum: %lf\n", checkSum);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <n_1> <n_2> <repeat>\n", argv[0]);
    printf("  n_1: number of points of the 1st curve\n");
    printf("  n_2: number of points of the 2nd curve\n");
    return 1;
  }

  const int n_1    = atoi(argv[1]);
  const int n_2    = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  Kokkos::initialize(argc, argv);
  {
    for (int i = 0; i < 3; i++)
      discrete_frechet_distance(i, n_1, n_2, repeat);
  }
  Kokkos::finalize();
  return 0;
}
