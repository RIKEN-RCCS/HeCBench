// Kokkos port of gels-cuda (batched least-squares via cublasXgelsBatched)
// cuBLAS gelsBatched replaced by a naive batched QR decomposition
// (Gram-Schmidt based) performed on the device via one work-item per batch.
// The test data is identical to the original; expected solution is all ones.

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <chrono>
#include <iostream>
#include <Kokkos_Core.hpp>

// ─── tiny serial QR least-squares  (Gram-Schmidt, real case) ─────────────────
// Solves min||Ax-b|| where A is m×n stored column-major with leading dim lda.
// b is in x on input; solution in x on output.
// Works only for well-conditioned square or overdetermined systems (m>=n).
static KOKKOS_INLINE_FUNCTION
void qr_solve(double* A, int m, int n, int lda, double* x)
{
  // In-place QR via modified Gram-Schmidt, storing Q in A and R separately.
  // For small n (<=5 here) this is fine.
  double R[5][5] = {};
  double Q[25];   // m×n flattened, col-major

  // Copy A into Q
  for (int c = 0; c < n; c++)
    for (int r = 0; r < m; r++)
      Q[r + c*m] = A[r + c*lda];

  // Modified Gram-Schmidt
  for (int j = 0; j < n; j++) {
    double norm = 0.0;
    for (int r = 0; r < m; r++) norm += Q[r + j*m] * Q[r + j*m];
    norm = Kokkos::sqrt(norm);
    R[j][j] = norm;
    if (norm < 1e-12) continue;
    double inv = 1.0 / norm;
    for (int r = 0; r < m; r++) Q[r + j*m] *= inv;
    for (int k = j+1; k < n; k++) {
      double dot = 0.0;
      for (int r = 0; r < m; r++) dot += Q[r + j*m] * Q[r + k*m];
      R[j][k] = dot;
      for (int r = 0; r < m; r++) Q[r + k*m] -= dot * Q[r + j*m];
    }
  }

  // Qt * b
  double Qtb[5] = {};
  for (int j = 0; j < n; j++)
    for (int r = 0; r < m; r++)
      Qtb[j] += Q[r + j*m] * x[r];

  // Back-substitution R x = Qtb
  for (int j = n-1; j >= 0; j--) {
    double s = Qtb[j];
    for (int k = j+1; k < n; k++) s -= R[j][k] * x[k];
    x[j] = (R[j][j] != 0.0) ? s / R[j][j] : 0.0;
  }
}

// ─── real-only runner ─────────────────────────────────────────────────────────
static int run_gels(int repeat)
{
  const int m = 5, n = 5, nrhs = 1, lda = m;
  const int stride_a = n * lda, stride_b = nrhs * m, batch_size = 2;

  // Matrix data (column-major, from the original CUDA benchmark)
  double A_host[] = {
     1.0,  1.0,  1.0,  1.0,  1.0,
     0.0,  0.2,  0.6,  1.0,  1.8,
     0.0, -0.4, -0.2, -1.0, -0.6,
     0.0, -0.4,  0.4,  0.6,  0.2,
     0.0, -0.8, -1.2, -0.8, -0.6,

     0.2,  0.4,  0.4,  0.8,  0.0,
    -0.4,  0.2, -0.8,  0.4,  0.0,
    -0.4,  0.8,  0.2, -0.4,  0.0,
    -0.8, -0.4,  0.4,  0.2,  0.0,
     0.0,  0.0,  0.0,  0.0,  1.0
  };

  double B_host[] = {
     5.0,  3.6, -2.2,  0.8, -3.4,
     1.8, -0.6,  0.2, -0.6,  1.0
  };

  const double X_expected[] = {1,1,1,1,1, 1,1,1,1,1};

  // Device views
  Kokkos::View<double*> d_A("A", stride_a * batch_size);
  Kokkos::View<double*> d_B("B", stride_b * batch_size);

  {
    auto hA = Kokkos::create_mirror_view(d_A);
    auto hB = Kokkos::create_mirror_view(d_B);
    for (int i = 0; i < stride_a * batch_size; i++) hA(i) = A_host[i];
    for (int i = 0; i < stride_b * batch_size; i++) hB(i) = B_host[i];
    Kokkos::deep_copy(d_A, hA);
    Kokkos::deep_copy(d_B, hB);
  }

  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // Restore inputs (QR destroys A in-place)
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
      for (int b = 0; b < batch_size; b++) {
        double* Ap = &d_A(b * stride_a);
        double* bp = &d_B(b * stride_b);
        // A is already column-major (lda=m)
        qr_solve(Ap, m, n, lda, bp);
      }
    });
    // Restore A and B for next iteration
    if (r < repeat - 1) {
      auto hA = Kokkos::create_mirror_view(d_A);
      auto hB = Kokkos::create_mirror_view(d_B);
      for (int i = 0; i < stride_a * batch_size; i++) hA(i) = A_host[i];
      for (int i = 0; i < stride_b * batch_size; i++) hB(i) = B_host[i];
      Kokkos::deep_copy(d_A, hA);
      Kokkos::deep_copy(d_B, hB);
    }
  }

  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  printf("Average kernel execution time : %f (us)\n", us / repeat);

  auto hB = Kokkos::create_mirror_view(d_B);
  Kokkos::deep_copy(hB, d_B);

  const double bound = 1e-8;
  bool passed = true;
  printf("Results:\n");
  for (int b = 0; b < batch_size; b++) {
    for (int j = 0; j < n; j++) {
      double v = hB(b * stride_b + j);
      printf("%6.2f ", v);
      if (std::fabs(v - X_expected[b * n + j]) > bound) passed = false;
    }
    printf("\n");
  }

  if (passed) printf("Calculations successfully finished\n");
  else        printf("ERROR: results mismatch!\n");

  return passed ? 0 : 1;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]); return 1;
  }
  int repeat = atoi(argv[1]);

  std::cout << "\n#############################################\n";
  std::cout << "# Batched GELS Kokkos port (real double)\n";
  std::cout << "#############################################\n\n";

  int ret = 0;
  Kokkos::initialize(argc, argv);
  {
    ret = run_gels(repeat);
  }
  Kokkos::finalize();
  return ret;
}
