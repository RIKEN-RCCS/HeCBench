// Kokkos port of determinant-cuda
// Computes the determinant of a symmetric positive-definite matrix
// via Cholesky factorization (L*L^T), then det = (product of diagonal)^2.
// Replaces cuSolver + Thrust with naive Kokkos parallel_for / parallel_reduce.

#include <stdio.h>
#include <chrono>
#include <Kokkos_Core.hpp>

static void choleskyDecomp(Kokkos::View<float**> A, int n)
{
  // Sequential Cholesky on device using a single-thread parallel_for
  // with range policy [0,1) to run on the device.
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
    for (int j = 0; j < n; j++) {
      float s = A(j, j);
      for (int k = 0; k < j; k++) s -= A(j, k) * A(j, k);
      A(j, j) = Kokkos::sqrt(s);
      for (int i = j + 1; i < n; i++) {
        float t = A(i, j);
        for (int k = 0; k < j; k++) t -= A(i, k) * A(j, k);
        A(i, j) = t / A(j, j);
      }
    }
  });
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int N = 11;

  // Row-major host data (lower-triangular Cholesky will use col-major view)
  float h_A[N * N] = {
     2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
    -2.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    -2.,  0.,  6.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    -2.,  0.,  2.,  8.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
    -2.,  0.,  2.,  4.,  10., 10., 10., 10., 10., 10., 10.,
    -2.,  0.,  2.,  4.,  10., 12., 12., 12., 12., 12., 12.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 14., 14., 14., 14.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 16., 16., 16.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 18., 18.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 20.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 22.
  };

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float**> d_A("A", N, N);
    auto h_A_view = Kokkos::create_mirror_view(d_A);

    float det = 0.f;

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      // Copy host matrix into 2-D view
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
          h_A_view(i, j) = h_A[i * N + j];
      Kokkos::deep_copy(d_A, h_A_view);

      choleskyDecomp(d_A, N);

      // Compute product of diagonal (Thrust::reduce equivalent)
      float prod = 1.f;
      Kokkos::parallel_reduce(N, KOKKOS_LAMBDA(int i, float& val) {
        val *= d_A(i, i);
      }, Kokkos::Prod<float>(prod));

      det = prod;
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
    printf("determinant = %f\n", det * det);
  }
  Kokkos::finalize();
  return 0;
}
