#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

template <typename FP>
FP host_cost(const FP *A, const FP *B, const FP *scale_A, const FP *scale_B, int m, int n) {
  double sum = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      FP dist = 0;
      for (int d = 0; d < 2; d++) {
        FP diff = A[i + d*m] - B[j + d*n];
        dist += diff * diff;
      }
      sum += exp(-dist / (scale_A[i] + scale_B[j]));
    }
  }
  return (FP)sum;
}

template <typename FP>
void test(const int size, const int repeat) {
  int sz2 = size * 2;
  FP *A      = new FP[sz2];
  FP *B      = new FP[sz2];
  FP *scaleA = new FP[size];
  FP *scaleB = new FP[size];

  for (int i = 0; i < sz2; i++) { A[i] = 1; B[i] = 0; }
  for (int i = 0; i < size; i++) { scaleA[i] = 1; scaleB[i] = 1; }

  Kokkos::View<FP*> d_A("A", sz2), d_B("B", sz2);
  Kokkos::View<FP*> d_scaleA("scaleA", size), d_scaleB("scaleB", size);

  {
    auto h_A      = Kokkos::create_mirror_view(d_A);
    auto h_B      = Kokkos::create_mirror_view(d_B);
    auto h_scaleA = Kokkos::create_mirror_view(d_scaleA);
    auto h_scaleB = Kokkos::create_mirror_view(d_scaleB);
    for (int i = 0; i < sz2; i++) { h_A(i) = A[i]; h_B(i) = B[i]; }
    for (int i = 0; i < size; i++) { h_scaleA(i) = scaleA[i]; h_scaleB(i) = scaleB[i]; }
    Kokkos::deep_copy(d_A, h_A);
    Kokkos::deep_copy(d_B, h_B);
    Kokkos::deep_copy(d_scaleA, h_scaleA);
    Kokkos::deep_copy(d_scaleB, h_scaleB);
  }

  double output = 0;
  const int N = size;

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    double sum = 0;
    Kokkos::parallel_reduce("expdist",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{N,N}),
      KOKKOS_LAMBDA(int i, int j, double& lsum) {
        FP dist = 0;
        for (int d = 0; d < 2; d++) {
          FP diff = d_A(i + d*N) - d_B(j + d*N);
          dist += diff * diff;
        }
        lsum += Kokkos::exp(-dist / (d_scaleA(i) + d_scaleB(j)));
      }, sum);
    Kokkos::fence();
    output = sum;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
  printf("    device result: %lf\n", output);

  FP ref = host_cost<FP>(A, B, scaleA, scaleB, size, size);
  printf("      host result: %lf\n", (double)ref);
  printf("analytical result: %lf\n\n", (double)size * size * exp(-1.0));

  delete[] A; delete[] B; delete[] scaleA; delete[] scaleB;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage ./%s <size> <repeat>\n", argv[0]);
    return 1;
  }
  const int size   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    printf("Test single precision\n");
    test<float>(size, repeat);
    printf("Test double precision\n");
    test<double>(size, repeat);
  }
  Kokkos::finalize();
  return 0;
}
