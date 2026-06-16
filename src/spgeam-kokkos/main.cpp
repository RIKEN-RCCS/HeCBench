// Kokkos port of spgeam-cuda
// CSR to CSC conversion (equivalent to sparse matrix transpose)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

static void init_matrix(float *matrix, int num_rows, int num_cols, int nnz) {
  int n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  for (int i = n; i > 0; i--) {
    int a = i-1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

static void init_csr(int *row_offsets, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int nnz) {
  row_offsets[0] = 0;
  int *cnts = (int*)calloc(num_rows, sizeof(int));
  int tmp = 0;
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
    }
  }
  for (int i = 1; i <= num_rows; i++) row_offsets[i] = row_offsets[i-1] + cnts[i-1];
  free(cnts);
}

// Transpose dense matrix
static void transpose_dense(float *A, float *B, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      B[j*rows + i] = A[i*cols + j];
}

// CSR-to-CSC via Kokkos (compute CSC = transpose of CSR)
void csr2csc(Kokkos::View<int*> d_csr_offsets,
             Kokkos::View<int*> d_csr_columns,
             Kokkos::View<float*> d_csr_values,
             Kokkos::View<int*> d_csc_offsets,
             Kokkos::View<int*> d_csc_rows,
             Kokkos::View<float*> d_csc_values,
             int num_rows, int num_cols, int nnz)
{
  // Count nnz per column
  Kokkos::deep_copy(d_csc_offsets, 0);
  Kokkos::parallel_for("count_col", nnz, KOKKOS_LAMBDA(const int i) {
    Kokkos::atomic_increment(&d_csc_offsets(d_csr_columns(i) + 1));
  });
  Kokkos::fence();

  // Prefix scan on host
  auto h_csc_off = Kokkos::create_mirror_view(d_csc_offsets);
  Kokkos::deep_copy(h_csc_off, d_csc_offsets);
  for (int i = 1; i <= num_cols; i++) h_csc_off(i) += h_csc_off(i-1);
  Kokkos::deep_copy(d_csc_offsets, h_csc_off);

  // Fill CSC rows and values using atomic positions
  // We need a work array to track fill positions
  Kokkos::View<int*> d_work("work", num_cols + 1);
  Kokkos::deep_copy(d_work, d_csc_offsets);

  // Row-by-row fill (serial in terms of rows for correctness)
  auto h_csr_off = Kokkos::create_mirror_view(d_csr_offsets);
  auto h_csr_col = Kokkos::create_mirror_view(d_csr_columns);
  auto h_csr_val = Kokkos::create_mirror_view(d_csr_values);
  auto h_csc_row = Kokkos::create_mirror_view(d_csc_rows);
  auto h_csc_val = Kokkos::create_mirror_view(d_csc_values);
  auto h_work    = Kokkos::create_mirror_view(d_work);
  Kokkos::deep_copy(h_csr_off, d_csr_offsets);
  Kokkos::deep_copy(h_csr_col, d_csr_columns);
  Kokkos::deep_copy(h_csr_val, d_csr_values);
  Kokkos::deep_copy(h_work, d_work);

  for (int r = 0; r < num_rows; r++) {
    for (int idx = h_csr_off(r); idx < h_csr_off(r+1); idx++) {
      int c = h_csr_col(idx);
      int pos = h_work(c)++;
      h_csc_row(pos) = r;
      h_csc_val(pos) = h_csr_val(idx);
    }
  }
  Kokkos::deep_copy(d_csc_rows, h_csc_row);
  Kokkos::deep_copy(d_csc_values, h_csc_val);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("CSR to CSC conversion (sparse matrix transpose)\n");
    printf("Usage %s <M> <K> <nnz> <repeat> <verify>\n", argv[0]);
    return 1;
  }

  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int a_nnz = atoi(argv[3]);
  int repeat = atoi(argv[4]);
  int verify = atoi(argv[5]);

  float *hA = (float*)malloc(m * k * sizeof(float));
  float *hA_values = (float*)malloc(a_nnz * sizeof(float));
  int *hA_columns = (int*)malloc(a_nnz * sizeof(int));
  int *hA_offsets = (int*)malloc((m + 1) * sizeof(int));

  init_matrix(hA, m, k, a_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA, m, k, a_nnz);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*>   d_csr_offsets("d_csr_offsets", m + 1);
    Kokkos::View<int*>   d_csr_columns("d_csr_columns", a_nnz);
    Kokkos::View<float*> d_csr_values("d_csr_values", a_nnz);
    Kokkos::View<int*>   d_csc_offsets("d_csc_offsets", k + 1);
    Kokkos::View<int*>   d_csc_rows("d_csc_rows", a_nnz);
    Kokkos::View<float*> d_csc_values("d_csc_values", a_nnz);

    {
      auto h_off = Kokkos::create_mirror_view(d_csr_offsets);
      auto h_col = Kokkos::create_mirror_view(d_csr_columns);
      auto h_val = Kokkos::create_mirror_view(d_csr_values);
      for (int i = 0; i <= m; i++) h_off(i) = hA_offsets[i];
      for (int i = 0; i < a_nnz; i++) { h_col(i) = hA_columns[i]; h_val(i) = hA_values[i]; }
      Kokkos::deep_copy(d_csr_offsets, h_off);
      Kokkos::deep_copy(d_csr_columns, h_col);
      Kokkos::deep_copy(d_csr_values, h_val);
    }

    Kokkos::fence();
    auto start_t = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      csr2csc(d_csr_offsets, d_csr_columns, d_csr_values,
              d_csc_offsets, d_csc_rows, d_csc_values,
              m, k, a_nnz);
    }

    Kokkos::fence();
    auto end_t = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count();
    printf("Average execution time of cuSparse csr2cscEx2 : %f (us)\n", (time * 1e-3f) / repeat);

    if (verify) {
      printf("Computing the reference results..\n");
      float *hB = (float*)malloc(k * m * sizeof(float));
      transpose_dense(hA, hB, m, k);
      float *hB_values = (float*)malloc(a_nnz * sizeof(float));
      int *hB_rows = (int*)malloc(a_nnz * sizeof(int));
      int *hB_offsets = (int*)malloc((k+1) * sizeof(int));
      init_csr(hB_offsets, hB_values, hB_rows, hB, k, m, a_nnz);

      auto h_csc_off = Kokkos::create_mirror_view(d_csc_offsets);
      auto h_csc_row = Kokkos::create_mirror_view(d_csc_rows);
      auto h_csc_val = Kokkos::create_mirror_view(d_csc_values);
      Kokkos::deep_copy(h_csc_off, d_csc_offsets);
      Kokkos::deep_copy(h_csc_row, d_csc_rows);
      Kokkos::deep_copy(h_csc_val, d_csc_values);

      int correct = 1;
      for (int i = 0; i <= k; i++)
        if (h_csc_off(i) != hB_offsets[i]) { correct = 0; break; }
      if (correct) {
        for (int i = 0; i < a_nnz; i++)
          if (h_csc_row(i) != hB_rows[i] || fabsf(h_csc_val(i) - hB_values[i]) > 1e-2f)
            { correct = 0; break; }
      }
      printf("%s\n", correct ? "spgeam_example test PASSED" : "spgeam_example test FAILED: wrong result");
      free(hB); free(hB_values); free(hB_rows); free(hB_offsets);
    }
  }
  Kokkos::finalize();

  free(hA); free(hA_values); free(hA_columns); free(hA_offsets);
  return 0;
}
