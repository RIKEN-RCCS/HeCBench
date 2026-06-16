#include <iostream>
#include <limits>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

using namespace std;

constexpr int m_size = 512 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int K = m_size / 2;

#include "verify.cpp"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[K] = new float[N][K];
  float(*c_host)[K] = new float[M][K];
  float(*c_back)[K] = new float[M][K];

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < K; j++)
      b_host[i][j] = rand() % 256;

  for (int j = 0; j < K; j++) {
    float sum = 0;
    for (int i = 0; i < N; i++) sum += b_host[i][j];
    for (int i = 0; i < N; i++) b_host[i][j] /= sum;
  }

  cout << "Problem size: c(" << M << "," << K << ") = a(" << M << "," << N
       << ") * b(" << N << "," << K << ")\n";

  Kokkos::initialize(argc, argv);
  {
    // Flatten 2D arrays into 1D Kokkos Views
    Kokkos::View<float *> d_a("a", M * N);
    Kokkos::View<float *> d_b("b", N * K);
    Kokkos::View<float *> d_c("c", M * K);

    {
      auto ha = Kokkos::create_mirror_view(d_a);
      auto hb = Kokkos::create_mirror_view(d_b);
      for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) ha(i * N + j) = a_host[i][j];
      for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++) hb(i * K + j) = b_host[i][j];
      Kokkos::deep_copy(d_a, ha);
      Kokkos::deep_copy(d_b, hb);
    }

    for (int m = 1; m <= 4; m++) {
      printf("Minkowski distance with p = %d\n", m);
      const float p = (float)m;
      const float one_over_p = 1.f / p;

      auto start = std::chrono::steady_clock::now();

      for (int r = 0; r < repeat; r++) {
        Kokkos::parallel_for(
            "minkowski",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, K}),
            KOKKOS_LAMBDA(const int i, const int j) {
              float sum = 0.f;
              for (int k = 0; k < N; k++) {
                sum += powf(fabsf(d_a(i * N + k) - d_b(k * K + j)), p);
              }
              d_c(i * K + j) = powf(sum, one_over_p);
            });
        Kokkos::fence();
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end - start).count();
      printf("Average kernel execution time: %f (s)\n",
             (time * 1e-9f) / repeat);

      // Copy result back
      auto hc = Kokkos::create_mirror_view(d_c);
      Kokkos::deep_copy(hc, d_c);
      for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++) c_back[i][j] = hc(i * K + j);

#ifdef VERIFY
      VerifyResult(a_host, b_host, c_host, c_back, p, one_over_p);
#endif
    }
  }
  Kokkos::finalize();

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return 0;
}
