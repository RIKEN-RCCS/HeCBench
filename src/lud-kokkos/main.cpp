// Kokkos port of LUD (LU Decomposition) benchmark
// Original OMP target source: src/lud-omp/lud.cpp
// Uses simplified non-tiled LUD with parallel inner loops (functionally equivalent)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define BLOCK_SIZE 16

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

// Generate a diagonally dominant matrix for testing
void create_matrix(float **mp, int size) {
  float *m = (float *)malloc(size * size * sizeof(float));
  srand(7);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      m[i * size + j] = ((float)rand() / RAND_MAX) + 0.5f;
    }
    m[i * size + i] += size; // Make diagonally dominant
  }
  *mp = m;
}

void matrix_duplicate(float *src, float **dst, int matrix_dim) {
  *dst = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));
  memcpy(*dst, src, matrix_dim * matrix_dim * sizeof(float));
}

void lud_verify(float *m, float *lu, int size) {
  float *tmp = (float *)malloc(size * size * sizeof(float));
  // Reconstruct L*U and compare with original
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      float sum = 0.0f;
      int min_ij = (i < j) ? i : j;
      for (int k = 0; k <= min_ij; k++) {
        float L_ik = (i == k) ? 1.0f : ((i > k) ? lu[i * size + k] : 0.0f);
        float U_kj = (k <= j) ? lu[k * size + j] : 0.0f;
        sum += L_ik * U_kj;
      }
      tmp[i * size + j] = sum;
    }
  }
  float max_err = 0.0f;
  for (int i = 0; i < size * size; i++) {
    float err = fabsf(tmp[i] - m[i]);
    if (err > max_err) max_err = err;
  }
  printf("LUD verify: max error = %e  %s\n", max_err, (max_err < 1e-2f) ? "PASS" : "FAIL");
  free(tmp);
}

void lud_kokkos(float *h_m, int matrix_dim) {
  int n = matrix_dim;
  Kokkos::View<float**> d_m("d_m", n, n);
  auto h_view = Kokkos::create_mirror_view(d_m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      h_view(i, j) = h_m[i * n + j];
  Kokkos::deep_copy(d_m, h_view);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < n; i++) {
    // Compute column multipliers: m[j][i] /= m[i][i], for j > i
    Kokkos::parallel_for("lud_col",
      Kokkos::RangePolicy<>(i + 1, n),
      KOKKOS_LAMBDA(const int j) {
        d_m(j, i) /= d_m(i, i);
      });
    Kokkos::fence();

    // Update trailing submatrix: m[j][k] -= m[j][i] * m[i][k], for j,k > i
    Kokkos::parallel_for("lud_update",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i + 1, i + 1}, {n, n}),
      KOKKOS_LAMBDA(const int j, const int k) {
        d_m(j, k) -= d_m(j, i) * d_m(i, k);
      });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time : %f (s)\n", time * 1e-9f);

  Kokkos::deep_copy(h_view, d_m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      h_m[i * n + j] = h_view(i, j);
}

static int do_verify = 0;

int main(int argc, char *argv[]) {
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  int matrix_dim = 512;
  int opt;
  extern char *optarg;

  while ((opt = getopt(argc, argv, "vs:i:")) != -1) {
    switch (opt) {
      case 'v': do_verify = 1; break;
      case 's':
        matrix_dim = atoi(optarg);
        printf("Generate input matrix internally, size=%d\n", matrix_dim);
        break;
      case 'i':
        printf("Note: file input not supported in this port; using -s instead\n");
        break;
    }
  }

  // Ensure matrix_dim is a multiple of BLOCK_SIZE
  if (matrix_dim % BLOCK_SIZE != 0) {
    matrix_dim = ((matrix_dim / BLOCK_SIZE) + 1) * BLOCK_SIZE;
    printf("Adjusted matrix_dim to %d (multiple of %d)\n", matrix_dim, BLOCK_SIZE);
  }

  float *m;
  create_matrix(&m, matrix_dim);

  float *mm = NULL;
  if (do_verify) matrix_duplicate(m, &mm, matrix_dim);

  double offload_start = gettime();
  Kokkos::initialize(argc, argv);
  {
    lud_kokkos(m, matrix_dim);
  }
  Kokkos::finalize();
  double offload_end = gettime();

  printf("Device offloading time (s): %lf\n", offload_end - offload_start);

  if (do_verify) {
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim);
    free(mm);
  }

  free(m);
  return 0;
}
