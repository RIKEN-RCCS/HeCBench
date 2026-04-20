// Kokkos port of sps2d-cuda
// Sparse CSR matrix → dense matrix conversion

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static void init_matrix(float *matrix, int64_t num_rows, int64_t num_cols, int64_t nnz) {
  int64_t n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (int64_t i = 0; i < n; i++) d[i] = (float)i;
  for (int64_t i = n; i > 0; i--) {
    int64_t a = i-1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (int64_t i = 0; i < num_rows; i++)
    for (int64_t j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= (float)nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

static void init_csr(int64_t *row_offsets, float *values, int64_t *col_indices,
                     float *matrix, int64_t num_rows, int64_t num_cols, int64_t nnz) {
  row_offsets[0] = 0;
  int64_t *cnts = (int64_t*)calloc(num_rows, sizeof(int64_t));
  int64_t tmp = 0;
  for (int64_t i = 0; i < num_rows; i++)
    for (int64_t j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (int64_t i = 1; i <= num_rows; i++) row_offsets[i] = row_offsets[i-1] + cnts[i-1];
  free(cnts);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Sparse CSR matrix to dense matrix conversion\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  int64_t m = atol(argv[1]);
  int64_t n = atol(argv[2]);
  int64_t h_nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  float *h_dense = (float*)malloc(m * n * sizeof(float));
  float *h_dense_result = (float*)malloc(m * n * sizeof(float));
  float *h_csr_values = (float*)malloc(h_nnz * sizeof(float));
  int64_t *h_csr_columns = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  int64_t *h_csr_offsets = (int64_t*)malloc((m+1) * sizeof(int64_t));

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, m, n, h_nnz);
  init_csr(h_csr_offsets, h_csr_values, h_csr_columns, h_dense, m, n, h_nnz);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*>   d_dense("d_dense", m * n);
    Kokkos::View<int64_t*> d_csr_offsets("d_csr_offsets", m + 1);
    Kokkos::View<int64_t*> d_csr_columns("d_csr_columns", h_nnz);
    Kokkos::View<float*>   d_csr_values("d_csr_values", h_nnz);

    Kokkos::deep_copy(d_dense, 0.0f);
    {
      auto h_off = Kokkos::create_mirror_view(d_csr_offsets);
      auto h_col = Kokkos::create_mirror_view(d_csr_columns);
      auto h_val = Kokkos::create_mirror_view(d_csr_values);
      for (int64_t i = 0; i <= m; i++) h_off(i) = h_csr_offsets[i];
      for (int64_t i = 0; i < h_nnz; i++) { h_col(i) = h_csr_columns[i]; h_val(i) = h_csr_values[i]; }
      Kokkos::deep_copy(d_csr_offsets, h_off);
      Kokkos::deep_copy(d_csr_columns, h_col);
      Kokkos::deep_copy(d_csr_values, h_val);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::deep_copy(d_dense, 0.0f);
      Kokkos::parallel_for("sparse2dense", m, KOKKOS_LAMBDA(const int64_t row) {
        for (int64_t idx = d_csr_offsets(row); idx < d_csr_offsets(row+1); idx++) {
          int64_t col = d_csr_columns(idx);
          d_dense(row * n + col) = d_csr_values(idx);
        }
      });
      Kokkos::fence();
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SparseToDense_convert : %f (us)\n", (time * 1e-3f) / repeat);

    // Copy result
    auto h_d = Kokkos::create_mirror_view(d_dense);
    Kokkos::deep_copy(h_d, d_dense);
    for (int64_t i = 0; i < m*n; i++) h_dense_result[i] = h_d(i);
  }
  Kokkos::finalize();

  // Verify
  int correct = 1;
  int64_t nnz = 0;
  for (int64_t i = 0; i < m*n; i++) {
    if (h_dense_result[i] != 0) nnz++;
    if (h_dense[i] != h_dense_result[i]) { correct = 0; break; }
  }
  if (correct && nnz != h_nnz) correct = 0;
  printf("%s\n", correct ? "sparse2dense_csr_example test PASSED"
                          : "sparse2dense_csr_example test FAILED: wrong result");

  free(h_dense); free(h_dense_result); free(h_csr_values); free(h_csr_columns); free(h_csr_offsets);
  return 0;
}
