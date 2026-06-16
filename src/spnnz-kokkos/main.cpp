// Kokkos port of spnnz-cuda
// Count non-zero elements per row and total in a dense matrix

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Counts non-zero elements per row and total in a dense matrix\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  int64_t m = atol(argv[1]);
  int64_t n = atol(argv[2]);
  int64_t h_nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  float *h_dense = (float*)malloc(m * n * sizeof(float));
  int *nnzPerRow = (int*)malloc(m * sizeof(int));

  printf("Initializing host matrices..\n");
  init_matrix(h_dense, m, n, h_nnz);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_dense("d_dense", m * n);
    Kokkos::View<int*>   d_nnzPerRow("d_nnzPerRow", m);

    {
      auto h = Kokkos::create_mirror_view(d_dense);
      for (int64_t i = 0; i < m*n; i++) h(i) = h_dense[i];
      Kokkos::deep_copy(d_dense, h);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("count_nnz_row", m, KOKKOS_LAMBDA(const int64_t row) {
        int cnt = 0;
        for (int64_t j = 0; j < n; j++)
          if (d_dense(row * n + j) != 0.0f) cnt++;
        d_nnzPerRow(row) = cnt;
      });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of cusparseSnnz : %f (us)\n", (time * 1e-3f) / repeat);

    // Count total nnz
    int nnzTotal = 0;
    Kokkos::parallel_reduce("total_nnz", m,
      KOKKOS_LAMBDA(const int64_t i, int& lsum) { lsum += d_nnzPerRow(i); },
      nnzTotal);

    auto h_nnzPerRow = Kokkos::create_mirror_view(d_nnzPerRow);
    Kokkos::deep_copy(h_nnzPerRow, d_nnzPerRow);
    for (int64_t i = 0; i < m; i++) nnzPerRow[i] = h_nnzPerRow(i);

    // Verify
    int correct = 1;
    if (h_nnz != (int64_t)nnzTotal) {
      printf("nnz: %d != %d\n", (int)h_nnz, nnzTotal);
      correct = 0;
    }
    if (correct) {
      for (int64_t i = 0; i < m; i++) {
        int ref_nnz = 0;
        for (int64_t j = 0; j < n; j++) if (h_dense[i*n+j] != 0) ref_nnz++;
        if (ref_nnz != nnzPerRow[i]) { correct = 0; break; }
      }
    }
    printf("%s\n", correct ? "sparse_nnz_example test PASSED" : "sparse_nnz_example test FAILED: wrong result");
  }
  Kokkos::finalize();

  free(h_dense); free(nnzPerRow);
  return 0;
}
