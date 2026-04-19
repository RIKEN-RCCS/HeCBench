#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <Kokkos_Core.hpp>

// Initialize a dense matrix with a shuffled pattern; entries >= nnz become 0,
// others become a random value in [1,2).
static void init_matrix(float *matrix, int num_rows, int num_cols, int nnz)
{
  int n = num_rows * num_cols;
  std::vector<float> d(n);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  srand(123);
  for (int i = n; i > 0; i--) {
    int a = i - 1;
    int b = rand() % i;
    if (a != b) std::swap(d[a], d[b]);
  }
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      matrix[i * num_cols + j] =
          (d[i * num_cols + j] >= nnz) ? 0.f : (float)(drand48() + 1.0);
}

// Build CSR structure from dense matrix.
static void init_csr(int *row_indices, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int nnz)
{
  row_indices[0] = 0;
  row_indices[num_rows] = nnz;
  int tmp = 0;
  for (int i = 0; i < num_rows; i++) {
    int nnz_per_row = 0;
    for (int j = 0; j < num_cols; j++) {
      if (matrix[i * num_cols + j] != 0.f) {
        values[tmp]      = matrix[i * num_cols + j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    row_indices[i + 1] = row_indices[i] + nnz_per_row;
  }
}

// CPU reference SDDMM for one batch.
static void sddmm_cpu(const float *A, const float *B, const float *C_dense,
                      float *C_values, const int *C_offsets, const int *C_columns,
                      int K, int M, int N)
{
  for (int i = 0; i < M; i++) {
    for (int j = C_offsets[i]; j < C_offsets[i + 1]; j++) {
      int col = C_columns[j];
      float val = 0.f;
      for (int l = 0; l < K; l++) val += A[i * K + l] * B[l * N + col];
      C_values[j] = val;
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc != 8) {
    printf("Usage: %s <num_batches> <M> <K> <N> <nnz> <repeat> <verify>\n",
           argv[0]);
    printf("SDDMM (A, B, C) where A: M x K, B: K x N, C: M x N (sparse CSR)\n");
    return 1;
  }

  const int b      = atoi(argv[1]);
  const int M      = atoi(argv[2]);
  const int K      = atoi(argv[3]);
  const int N      = atoi(argv[4]);
  const int nnz    = atoi(argv[5]);
  const int repeat = atoi(argv[6]);
  const int verify = atoi(argv[7]);

  // Allocate host arrays
  const int A_size = M * K;
  const int B_size = K * N;
  const int C_size = M * N;

  std::vector<float> hA(b * A_size);
  std::vector<float> hB(b * B_size);
  std::vector<float> hC_dense(C_size);       // dense template (same for all batches)
  std::vector<float> hC_values(b * nnz, 0.f);
  std::vector<int>   hC_offsets(b * (M + 1));
  std::vector<int>   hC_columns(b * nnz);

  // Reference output (CPU)
  std::vector<float> hC_result(b * nnz, 0.f);

  // Initialize A and B for each batch
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < A_size; j++) hA[i * A_size + j] = (float)(drand48() + 1.0);
    for (int j = 0; j < B_size; j++) hB[i * B_size + j] = (float)(drand48() + 1.0);
  }

  // Build CSR pattern once and replicate across batches
  init_matrix(hC_dense.data(), M, N, nnz);
  {
    std::vector<float> tmp_vals(nnz);
    std::vector<int>   tmp_cols(nnz);
    std::vector<int>   tmp_rows(M + 1);
    init_csr(tmp_rows.data(), tmp_vals.data(), tmp_cols.data(),
             hC_dense.data(), M, N, nnz);
    for (int i = 0; i < b; i++) {
      std::memcpy(hC_offsets.data() + i * (M + 1), tmp_rows.data(), (M + 1) * sizeof(int));
      std::memcpy(hC_columns.data() + i * nnz,     tmp_cols.data(), nnz * sizeof(int));
    }
  }

  // Compute CPU reference
  if (verify) {
    for (int i = 0; i < b; i++) {
      sddmm_cpu(hA.data() + i * A_size,
                hB.data() + i * B_size,
                hC_dense.data(),
                hC_result.data() + i * nnz,
                hC_offsets.data() + i * (M + 1),
                hC_columns.data() + i * nnz,
                K, M, N);
    }
  }

  Kokkos::initialize(argc, argv);
  {
    // Device views (1-D flat)
    Kokkos::View<float*>  dA("dA", b * A_size);
    Kokkos::View<float*>  dB("dB", b * B_size);
    Kokkos::View<float*>  dValues("dValues", b * nnz);
    Kokkos::View<int*>    dOffsets("dOffsets", b * (M + 1));
    Kokkos::View<int*>    dColumns("dColumns", b * nnz);

    // Host mirrors
    auto hA_m       = Kokkos::create_mirror_view(dA);
    auto hB_m       = Kokkos::create_mirror_view(dB);
    auto hOffsets_m = Kokkos::create_mirror_view(dOffsets);
    auto hColumns_m = Kokkos::create_mirror_view(dColumns);

    for (int i = 0; i < b * A_size; i++) hA_m(i)       = hA[i];
    for (int i = 0; i < b * B_size; i++) hB_m(i)       = hB[i];
    for (int i = 0; i < b * (M + 1); i++) hOffsets_m(i) = hC_offsets[i];
    for (int i = 0; i < b * nnz;    i++) hColumns_m(i) = hC_columns[i];

    Kokkos::deep_copy(dA,       hA_m);
    Kokkos::deep_copy(dB,       hB_m);
    Kokkos::deep_copy(dOffsets, hOffsets_m);
    Kokkos::deep_copy(dColumns, hColumns_m);

    // Capture sizes as scalars for lambda
    const int lM   = M;
    const int lK   = K;
    const int lN   = N;
    const int lNnz = nnz;
    const int lb   = b;

    // Warmup
    Kokkos::parallel_for("sddmm_warmup", lb * lM,
        KOKKOS_LAMBDA(const int bm) {
          int batch = bm / lM;
          int row   = bm % lM;
          int row_start = dOffsets[batch * (lM + 1) + row];
          int row_end   = dOffsets[batch * (lM + 1) + row + 1];
          for (int j = row_start; j < row_end; j++) {
            int col = dColumns[batch * lNnz + j];
            float val = 0.f;
            for (int l = 0; l < lK; l++)
              val += dA[batch * lM * lK + row * lK + l] *
                     dB[batch * lK * lN + l * lN + col];
            dValues[batch * lNnz + j] = val;
          }
        });
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("sddmm", lb * lM,
          KOKKOS_LAMBDA(const int bm) {
            int batch = bm / lM;
            int row   = bm % lM;
            int row_start = dOffsets[batch * (lM + 1) + row];
            int row_end   = dOffsets[batch * (lM + 1) + row + 1];
            for (int j = row_start; j < row_end; j++) {
              int col = dColumns[batch * lNnz + j];
              float val = 0.f;
              for (int l = 0; l < lK; l++)
                val += dA[batch * lM * lK + row * lK + l] *
                       dB[batch * lK * lN + l * lN + col];
              dValues[batch * lNnz + j] = val;
            }
          });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                * 1e-3 / repeat;
    printf("Average execution time of SDDMM: %f (us)\n", us);

    if (verify) {
      auto hValues_m = Kokkos::create_mirror_view(dValues);
      Kokkos::deep_copy(hValues_m, dValues);

      int correct = 1;
      for (int i = 0; i < b && correct; i++) {
        for (int j = 0; j < nnz; j++) {
          if (fabsf(hValues_m(i * nnz + j) - hC_result[i * nnz + j]) > 1e-2f) {
            printf("@batch%d index%d: %f != %f\n", i, j,
                   hValues_m(i * nnz + j), hC_result[i * nnz + j]);
            correct = 0;
            break;
          }
        }
      }
      if (correct)
        printf("sddmm_csr_batched_example test PASSED\n");
      else
        printf("sddmm_csr_batched_example test FAILED: wrong result\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
