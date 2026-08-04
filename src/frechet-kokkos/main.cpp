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

  for (int k = 0; k < repeat; k++) {
    Kokkos::deep_copy(d_ca, -1.0);
    double *raw_ca = d_ca.data();
    double *raw_c1 = d_c1.data();
    double *raw_c2 = d_c2.data();

    for (int diag = 0; diag <= n_1 + n_2 - 2; ++diag) {
      const int i_begin = diag < n_2 ? 0 : diag - (n_2 - 1);
      const int i_end = diag < n_1 ? diag : n_1 - 1;
      const int count = i_end - i_begin + 1;

      Kokkos::parallel_for("frechet_diag", count, KOKKOS_LAMBDA(int t) {
        const int i0 = i_begin + t;
        const int j0 = diag - i0;
        const int i = i0 + 1;
        const int j = j0 + 1;

        double dist;
        if (s == 0) dist = norm1(i, j, raw_c1, raw_c2);
        else if (s == 1) dist = norm2(i, j, raw_c1, raw_c2);
        else dist = norm3(i, j, raw_c1, raw_c2);

        double value;
        if (i0 == 0 && j0 == 0) {
          value = dist;
        } else if (i0 > 0 && j0 == 0) {
          value = fmax(raw_ca[(i0 - 1) * n_2], dist);
        } else if (i0 == 0) {
          value = fmax(raw_ca[j0 - 1], dist);
        } else {
          const double prev_i = raw_ca[(i0 - 1) * n_2 + j0];
          const double prev_ij = raw_ca[(i0 - 1) * n_2 + (j0 - 1)];
          const double prev_j = raw_ca[i0 * n_2 + (j0 - 1)];
          value = fmax(fmin(fmin(prev_i, prev_ij), prev_j), dist);
        }
        raw_ca[i0 * n_2 + j0] = value;
      });
      Kokkos::fence();
    }
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
