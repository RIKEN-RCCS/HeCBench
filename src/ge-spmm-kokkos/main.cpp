// Kokkos port of ge-spmm-cuda (General SpMM: Sparse×Dense matrix multiply)
// warp-shuffle/shared-memory tiled kernels → simple row-based parallel_for.
// The util/ headers (mmio.hpp, util.hpp) are reused via -I../ge-spmm-cuda.

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "./util/mmio.hpp"
#include "./util/util.hpp"

#define VALIDATE

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <matrix file> <tile row> <repeat>\n", argv[0]);
    return 1;
  }

  int A_nrows, A_ncols, nnz;
  const int max_ncols = 256;

  std::vector<int>   row_indices, col_indices;
  std::vector<float> values;

  int*   A_indptr  = nullptr;
  int*   A_indices = nullptr;
  float* A_data    = nullptr;
  float* B         = nullptr;
  float* C         = nullptr;
  float* golden    = nullptr;

  printf("reading data file ...\n");
  readMtx<float>(argv[1], row_indices, col_indices, values, A_nrows, A_ncols, nnz);

  const int tile_row = atoi(argv[2]);
  const int repeat   = atoi(argv[3]);

  A_data    = (float*)malloc(nnz             * sizeof(float));
  A_indptr  = (int*)  malloc((A_nrows + 1)   * sizeof(int));
  A_indices = (int*)  malloc(nnz             * sizeof(int));
  B         = (float*)malloc(max_ncols * A_ncols * sizeof(float));

  if (!A_data || !A_indices || !A_indptr || !B) {
    fprintf(stderr, "Host malloc failed\n"); return 1;
  }

#ifdef VALIDATE
  C      = (float*)malloc(A_nrows * max_ncols * sizeof(float));
  golden = (float*)malloc(A_nrows * max_ncols * sizeof(float));
  if (!C || !golden) { fprintf(stderr, "Host malloc failed\n"); return 1; }
#endif

  // COO → CSR
  for (int i = 0; i <= A_nrows; i++) A_indptr[i] = 0;
  for (int n = 0; n < nnz; n++) A_indptr[row_indices[n] + 1]++;
  for (int n = 1; n <= A_nrows; n++) A_indptr[n] += A_indptr[n-1];
  for (int n = 0; n < nnz; n++) {
    int ptr = A_indptr[row_indices[n]];
    A_indices[ptr] = col_indices[n];
    A_data[ptr]    = 1.f;   // binarise as in original
    A_indptr[row_indices[n]]++;
  }
  // restore row pointers
  for (int n = A_nrows - 1; n > 0; n--) A_indptr[n] = A_indptr[n-1];
  A_indptr[0] = 0;

  printf("read file ok. N=%d nnz=%d\n", A_nrows, nnz);

  srand(123);
  for (int i = 0; i < max_ncols * A_ncols; i++)
    B[i] = (float)(rand() % 100 - 50) / 100.f;

#ifdef VALIDATE
  for (int i = 0; i < A_nrows; i++)
    for (int k = 0; k < max_ncols; k++) {
      float acc = 0.f;
      for (int p = A_indptr[i]; p < A_indptr[i+1]; p++)
        acc += A_data[p] * B[max_ncols * A_indices[p] + k];
      golden[max_ncols * i + k] = acc;
    }
#endif

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*>   d_indptr  ("indptr",  A_nrows + 1);
    Kokkos::View<int*>   d_indices ("indices", nnz);
    Kokkos::View<float*> d_data    ("data",    nnz);
    Kokkos::View<float*> d_B       ("B",       max_ncols * A_ncols);
    Kokkos::View<float*> d_C       ("C",       A_nrows * max_ncols);

    {
      auto h = Kokkos::create_mirror_view(d_indptr);
      for (int i = 0; i <= A_nrows; i++) h(i) = A_indptr[i];
      Kokkos::deep_copy(d_indptr, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_indices);
      for (int i = 0; i < nnz; i++) h(i) = A_indices[i];
      Kokkos::deep_copy(d_indices, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_data);
      for (int i = 0; i < nnz; i++) h(i) = A_data[i];
      Kokkos::deep_copy(d_data, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_B);
      for (int i = 0; i < max_ncols * A_ncols; i++) h(i) = B[i];
      Kokkos::deep_copy(d_B, h);
    }

    bool ok = true;
    for (int B_ncols = 256; B_ncols <= max_ncols; B_ncols *= 2) {
      Kokkos::deep_copy(d_C, 0.f);

      Kokkos::fence();
      auto tstart = std::chrono::steady_clock::now();

      for (int r = 0; r < repeat; r++) {
        Kokkos::parallel_for(A_nrows, KOKKOS_LAMBDA(int row) {
          int lb = d_indptr(row), hb = d_indptr(row + 1);
          for (int col = 0; col < B_ncols; col++) {
            float acc = 0.f;
            for (int p = lb; p < hb; p++)
              acc += d_data(p) * d_B[max_ncols * d_indices(p) + col];
            d_C(row * B_ncols + col) = acc;
          }
        });
      }

      Kokkos::fence();
      auto tend = std::chrono::steady_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count() * 1e-3f;
      printf("Average kernel execution time %f (us), B_ncols=%d\n", us / repeat, B_ncols);

#ifdef VALIDATE
      {
        auto hC = Kokkos::create_mirror_view(d_C);
        Kokkos::deep_copy(hC, d_C);
        for (int i = 0; i < A_nrows && ok; i++)
          for (int j = 0; j < B_ncols && ok; j++)
            if (fabsf(hC(i * B_ncols + j) - golden[i * B_ncols + j]) > 1e-2f) {
              printf("b_ncols %d: mismatch at (%d,%d): %f vs %f\n",
                     B_ncols, i, j, hC(i*B_ncols+j), golden[i*B_ncols+j]);
              ok = false;
            }
      }
#endif
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(A_data); free(A_indptr); free(A_indices); free(B);
  if (C)      free(C);
  if (golden) free(golden);
  return 0;
}
