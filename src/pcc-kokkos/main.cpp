#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

#define THREAD_BLOCK_SIZE 1024
#define FREE_MEMORY       (1024UL * 1024 * 1024 * 4)

// Normalize each row to zero mean, unit L2 norm (host side, matches original)
void preprocessing(float* data_t, int N, int L) {
  for (int i = 0; i < N; i++) {
    float* row = data_t + i * L;
    double sum1 = 0.0, sum2 = 0.0;
    for (int l = 0; l < L; l++) sum1 += row[l];
    sum1 /= L;
    for (int l = 0; l < L; l++) sum2 += (row[l] - sum1) * (row[l] - sum1);
    sum2 = std::sqrt(sum2);
    for (int l = 0; l < L; l++) {
      if (sum2 != 0.0)
        row[l] = (float)((row[l] - sum1) / sum2);
      else
        row[l] = 0.f;
    }
  }
}

// Compute full correlation matrix via manual GEMM, then extract upper triangle.
// cormat[i,j] = sum_k data[i,k] * data[j,k]  (inner product of row i and row j)
int CorMat_singlePass(float* upper_tri, float* data, int N, int L) {
  const size_t M1    = (size_t)(N - 1) * N / 2;
  const size_t total = (size_t)N * N;

  preprocessing(data, N, L);

  // Device views
  Kokkos::View<float**, Kokkos::LayoutRight> d_data("d_data", N, L);
  Kokkos::View<float**, Kokkos::LayoutRight> d_cormat("d_cormat", N, N);
  Kokkos::View<float*>                       d_upper("d_upper", M1);

  // Copy data H→D
  {
    auto hv = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data, N, L);
    Kokkos::deep_copy(d_data, hv);
  }

  // GEMM: cormat[i,j] = dot(row_i, row_j)
  auto t0 = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
      "gemm",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
      KOKKOS_LAMBDA(int i, int j) {
        float sum = 0.f;
        for (int k = 0; k < L; k++) sum += d_data(i, k) * d_data(j, k);
        d_cormat(i, j) = sum;
      });

  Kokkos::fence();
  auto t1       = std::chrono::steady_clock::now();
  auto gemm_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  // Extract upper triangle: for i < j, store cormat[i,j] into upper[t]
  // t = N*i - i*(i+1)/2 + j - i - 1
  t0 = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
      "extract_upper",
      (int)total,
      KOKKOS_LAMBDA(int idx) {
        int i = idx / N;
        int j = idx % N;
        if (i < j) {
          size_t t = (size_t)N * i - (size_t)i * (i + 1) / 2 + j - i - 1;
          d_upper[t] = d_cormat(i, j);
        }
      });

  Kokkos::fence();
  t1 = std::chrono::steady_clock::now();
  auto extract_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  // Copy upper triangle D→H
  {
    auto hv = Kokkos::View<float*, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(upper_tri, M1);
    Kokkos::deep_copy(hv, d_upper);
  }

  std::cout << "Kernel time (s)\n"
            << "GEMM: " << gemm_ns * 1e-9 << ", "
            << "Extract upper triangle: " << extract_ns * 1e-9 << "\n";

  return 1;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " <Number of voxels> <Length of time series>\n";
    return 1;
  }

  int N = atoi(argv[1]);
  int L = atoi(argv[2]);
  std::cout << "Number of voxels: " << N
            << "  Length of time series: " << L << "\n\n";

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-6.f, 6.f);

  float* data = new float[N * L];
  for (int k = 0; k < N; k++)
    for (int l = 0; l < L; l++)
      data[k * L + l] = distr(g);

  size_t M11 = (size_t)(N - 1) * N / 2;
  float* upper_tri = new float[M11]();

  std::cout << "\nComputing correlations ...\n";

  Kokkos::initialize(argc, argv);
  {
    auto t0 = std::chrono::steady_clock::now();
    CorMat_singlePass(upper_tri, data, N, L);
    auto t1   = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::cout << "\nRunning time for computing correlations:\n"
              << (time * 1e-9f) << " (s)\n";

    double checksum = 0.0;
    for (size_t i = 0; i < M11; i++) checksum += upper_tri[i];
    std::cout << "Checksum: " << checksum << "\n";

    if (N < 100 && L < 100) {
      std::cout << "\nWriting correlation values into corrs.txt ...\n";
      std::ofstream f("corrs.txt");
      for (size_t i = 0; i < M11; i++) f << upper_tri[i] << '\n';
      std::cout << "Correlations stored in corrs.txt\n";
    }
  }
  Kokkos::finalize();

  delete[] upper_tri;
  delete[] data;
  return 0;
}
