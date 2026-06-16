// Kokkos port of spd2s-cuda
// Dense-to-sparse (CSR) conversion

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

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
  int64_t tmp = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    int64_t cnt = 0;
    for (int64_t j = 0; j < num_cols; j++) {
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnt++;
      }
    }
    row_offsets[i+1] = row_offsets[i] + cnt;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("The function converts a dense MxN matrix into a sparse CSR matrix\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  int64_t m = atol(argv[1]);
  int64_t n = atol(argv[2]);
  int64_t h_nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  float *h_dense = (float*)malloc(m * n * sizeof(float));
  float *h_csr_values = (float*)malloc(h_nnz * sizeof(float));
  int64_t *h_csr_columns = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  int64_t *h_csr_offsets = (int64_t*)malloc((m+1) * sizeof(int64_t));
  float *h_csr_values_ref = (float*)malloc(h_nnz * sizeof(float));
  int64_t *h_csr_columns_ref = (int64_t*)malloc(h_nnz * sizeof(int64_t));
  int64_t *h_csr_offsets_ref = (int64_t*)malloc((m+1) * sizeof(int64_t));

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, m, n, h_nnz);
  init_csr(h_csr_offsets_ref, h_csr_values_ref, h_csr_columns_ref,
           h_dense, m, n, h_nnz);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*>   d_dense("d_dense", m * n);
    Kokkos::View<int64_t*> d_csr_offsets("d_csr_offsets", m + 1);
    Kokkos::View<int64_t*> d_csr_columns("d_csr_columns", h_nnz);
    Kokkos::View<float*>   d_csr_values("d_csr_values", h_nnz);

    // Copy dense to device
    {
      auto h_d = Kokkos::create_mirror_view(d_dense);
      for (int64_t i = 0; i < m*n; i++) h_d(i) = h_dense[i];
      Kokkos::deep_copy(d_dense, h_d);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      // Compute row nnz counts
      Kokkos::parallel_for("count_nnz", m, KOKKOS_LAMBDA(const int64_t row) {
        int64_t cnt = 0;
        for (int64_t j = 0; j < n; j++)
          if (d_dense(row * n + j) != 0.0f) cnt++;
        d_csr_offsets(row + 1) = cnt;
      });
      Kokkos::fence();

      // Prefix sum (exclusive scan on host for simplicity)
      auto h_off = Kokkos::create_mirror_view(d_csr_offsets);
      Kokkos::deep_copy(h_off, d_csr_offsets);
      h_off(0) = 0;
      for (int64_t i = 0; i < m; i++) h_off(i+1) += h_off(i);
      Kokkos::deep_copy(d_csr_offsets, h_off);

      // Fill CSR columns and values
      Kokkos::parallel_for("fill_csr", m, KOKKOS_LAMBDA(const int64_t row) {
        int64_t pos = d_csr_offsets(row);
        for (int64_t j = 0; j < n; j++) {
          float v = d_dense(row * n + j);
          if (v != 0.0f) {
            d_csr_columns(pos) = j;
            d_csr_values(pos) = v;
            pos++;
          }
        }
      });
      Kokkos::fence();
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of DenseToSparse_convert : %f (us)\n", (time * 1e-3f) / repeat);

    // Copy results back
    {
      auto h_off = Kokkos::create_mirror_view(d_csr_offsets);
      auto h_col = Kokkos::create_mirror_view(d_csr_columns);
      auto h_val = Kokkos::create_mirror_view(d_csr_values);
      Kokkos::deep_copy(h_off, d_csr_offsets);
      Kokkos::deep_copy(h_col, d_csr_columns);
      Kokkos::deep_copy(h_val, d_csr_values);
      for (int64_t i = 0; i <= m; i++) h_csr_offsets[i] = h_off(i);
      for (int64_t i = 0; i < h_nnz; i++) {
        h_csr_columns[i] = h_col(i);
        h_csr_values[i] = h_val(i);
      }
    }
  }
  Kokkos::finalize();

  // Verify
  int correct = 1;
  int64_t nnz = h_csr_offsets[m];
  if (nnz != h_nnz) { printf("nnz: %ld != %ld\n", nnz, h_nnz); correct = 0; }

  if (correct) {
    std::vector<int64_t> cols(h_csr_columns, h_csr_columns + h_nnz);
    std::vector<float>   vals(h_csr_values,  h_csr_values  + h_nnz);
    std::vector<int64_t> cols_ref(h_csr_columns_ref, h_csr_columns_ref + h_nnz);
    std::vector<float>   vals_ref(h_csr_values_ref,  h_csr_values_ref  + h_nnz);
    std::sort(cols.begin(), cols.end());
    std::sort(vals.begin(), vals.end());
    std::sort(cols_ref.begin(), cols_ref.end());
    std::sort(vals_ref.begin(), vals_ref.end());
    for (int64_t i = 0; i < h_nnz; i++) {
      if (cols[i] != cols_ref[i] || fabsf(vals[i] - vals_ref[i]) > 1e-5f) {
        correct = 0; break;
      }
    }
  }
  printf("%s\n", correct ? "dense2sparse_csr_example test PASSED"
                          : "dense2sparse_csr_example test FAILED: wrong result");

  free(h_dense); free(h_csr_values); free(h_csr_columns); free(h_csr_offsets);
  free(h_csr_values_ref); free(h_csr_columns_ref); free(h_csr_offsets_ref);
  return 0;
}
