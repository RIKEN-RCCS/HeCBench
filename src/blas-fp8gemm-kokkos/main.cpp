/*
 * Kokkos port of blas-fp8gemm.
 * Original used cuBLASLt FP8 (E4M3) GEMM.
 * This port implements FP32 GEMM using Kokkos parallel_for (naive GEMM).
 * FP8 → FP32 conversion: scale the result to simulate FP8 range.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>

// Naive GEMM: C = alpha * A^T * B * beta + C (column-major, A: k x m, B: k x n)
// Matching the original: transa=T, transb=N, so A is k x m and B is k x n
void gemm_fp32(
    Kokkos::View<float*> A, int lda,  // A: k x m (k rows, m cols, col-major → lda=k)
    Kokkos::View<float*> B, int ldb,  // B: k x n (k rows, n cols, col-major → ldb=k)
    Kokkos::View<float*> C, int ldc,  // C: m x n (m rows, n cols, col-major → ldc=m)
    int m, int n, int k,
    float alpha, float beta)
{
  Kokkos::parallel_for("gemm", Kokkos::RangePolicy<>(0, m * n),
    KOKKOS_LAMBDA(int idx) {
      int col = idx / m;  // n dimension
      int row = idx % m;  // m dimension
      float s = 0.f;
      // A^T: A[row + i*lda] (column-major A^T means we read A row by row)
      // B:   B[i + col*ldb]
      for (int i = 0; i < k; i++)
        s += A(row + i * lda) * B(i + col * ldb);
      C(row + col * ldc) = alpha * s + beta * C(row + col * ldc);
    });
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    // Same dimensions as original: m=64, n=128, k=256
    const int m = 64, n = 128, k = 256;
    const float alpha = 2.0f, beta = 1.0f;

    Kokkos::View<float*> A("A", k * m);
    Kokkos::View<float*> B("B", k * n);
    Kokkos::View<float*> C("C", m * n);
    Kokkos::View<float*> D("D", m * n);

    // Initialize with scaled values (simulating FP8 range [-448, 448])
    Kokkos::parallel_for("init", k * m, KOKKOS_LAMBDA(int i) {
      A(i) = (float)(i % 9 - 4) / 4.f;  // values in [-1, 1]
    });
    Kokkos::parallel_for("init_B", k * n, KOKKOS_LAMBDA(int i) {
      B(i) = (float)(i % 9 - 4) / 4.f;
    });
    Kokkos::deep_copy(C, 0.5f);
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::deep_copy(D, C);  // D = C (beta scaling from C)
      gemm_fp32(A, k, B, k, D, m, m, n, k, alpha, beta);
    }
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average GEMM execution time %f (us)\n", (time * 1e-3f) / repeat);

    // Checksum
    auto h_D = Kokkos::create_mirror_view(D);
    Kokkos::deep_copy(h_D, D);
    float s = 0.f;
    for (int i = 0; i < m * n; i++) s += h_D(i);
    printf("Checksum: %f\n", s / (m * n));
  }
  Kokkos::finalize();
  return 0;
}
