/*
 * Kokkos port of blas-dot.
 * Original used cuBLAS cublasDotEx. 
 * This port implements a parallel dot product using Kokkos::parallel_reduce.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

template <typename T>
void dot(const size_t iNumElements, const int iNumIterations) {
  std::vector<T> h_A(iNumElements), h_B(iNumElements);

  double sum = 0.0;
  double val = sqrt(65504.0 / iNumElements);
  for (size_t i = 0; i < iNumElements; i++) {
    h_A[i] = (T)val;
    h_B[i] = (T)val;
    sum += (double)h_A[i] * (double)h_B[i];
  }

  Kokkos::View<T*> d_A("A", iNumElements);
  Kokkos::View<T*> d_B("B", iNumElements);

  auto hA = Kokkos::create_mirror_view(d_A);
  auto hB = Kokkos::create_mirror_view(d_B);
  for (size_t i = 0; i < iNumElements; i++) { hA(i) = h_A[i]; hB(i) = h_B[i]; }
  Kokkos::deep_copy(d_A, hA);
  Kokkos::deep_copy(d_B, hB);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  double device_dot = 0.0;
  for (int i = 0; i < iNumIterations; i++) {
    double local_sum = 0.0;
    Kokkos::parallel_reduce("dot", iNumElements, KOKKOS_LAMBDA(size_t j, double& s) {
      s += (double)d_A(j) * (double)d_B(j);
    }, local_sum);
    device_dot = local_sum;
  }
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average dot execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  printf("Host: %lf  Device: %lf\n", sum, device_dot);
  printf("%s\n\n", (fabs(device_dot - sum) < 1e-1) ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 3) {
      printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const size_t iNumElements = atol(argv[1]);
    const int iNumIterations = atoi(argv[2]);

    printf("\nFP64 Dot\n");
    dot<double>(iNumElements, iNumIterations);
    printf("\nFP32 Dot\n");
    dot<float>(iNumElements, iNumIterations);
  }
  Kokkos::finalize();
  return EXIT_SUCCESS;
}
