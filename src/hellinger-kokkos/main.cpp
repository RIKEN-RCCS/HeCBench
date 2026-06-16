#include <iostream>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FABS fabs
  #define FP double
#else
  #define SQRT sqrtf
  #define FABS fabsf
  #define FP float
#endif

// Matrix size constants
constexpr int m_size = 768 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#ifdef VERIFY
#include <limits>

bool ValueSame(FP a, FP b) {
  return FABS(a - b) < std::numeric_limits<FP>::epsilon();
}

void VerifyResult(FP* a_host, FP* b_host, FP* c_host, FP* c_back) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) c_host[i * P + j] = (FP)0.0;

  for (int i = 0; i < M; i++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < P; j++)
        c_host[i * P + j] += SQRT(a_host[i * N + k] * b_host[k * P + j]);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      FP value = (FP)1.0 - c_host[i * P + j];
      FP gate = (value >= (FP)0.0) ? (FP)1.0 : (FP)0.0;
      c_host[i * P + j] = SQRT(gate * value);
    }
  }

  bool mismatch_found = false;
  int print_count = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      if (!ValueSame(c_back[i * P + j], c_host[i * P + j])) {
        std::cout << "Fail - The result is incorrect for element: [" << i << ", "
                  << j << "], expected: " << c_host[i * P + j]
                  << ", but found: " << c_back[i * P + j] << "\n";
        mismatch_found = true;
        if (++print_count == 5) break;
      }
    }
    if (print_count == 5) break;
  }
  std::cout << (mismatch_found ? "FAIL\n" : "PASS\n");
}
#endif

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
            << ") * b(" << N << "," << P << ")\n";

  // Host arrays (flat 1D)
  FP* a_host = new FP[M * N];
  FP* b_host = new FP[N * P];
  FP* c_back = new FP[M * P];
#ifdef VERIFY
  FP* c_host = new FP[M * P];
#endif

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      a_host[i * N + j] = (FP)1.0 / N;

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < P; j++)
      b_host[i * P + j] = (FP)(rand() % 256);

  for (int j = 0; j < P; j++) {
    FP sum = 0;
    for (int i = 0; i < N; i++) sum += b_host[i * P + j];
    for (int i = 0; i < N; i++) b_host[i * P + j] /= sum;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<FP**> d_a("a", M, N);
    Kokkos::View<FP**> d_b("b", N, P);
    Kokkos::View<FP**> d_c("c", M, P);

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_b = Kokkos::create_mirror_view(d_b);

    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        h_a(i, j) = a_host[i * N + j];

    for (int i = 0; i < N; i++)
      for (int j = 0; j < P; j++)
        h_b(i, j) = b_host[i * P + j];

    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_b, h_b);

    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < repeat; rep++) {
      Kokkos::parallel_for("hellinger",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, P}),
        KOKKOS_LAMBDA(int i, int j) {
          FP sum = (FP)0.0;
          for (int k = 0; k < N; k++)
            sum += SQRT(d_a(i, k) * d_b(k, j));
          const FP value = (FP)1.0 - sum;
          const FP gate = (value >= (FP)0.0) ? (FP)1.0 : (FP)0.0;
          d_c(i, j) = SQRT(gate * value);
        });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat << " (s)\n";

    auto h_c = Kokkos::create_mirror_view(d_c);
    Kokkos::deep_copy(h_c, d_c);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < P; j++)
        c_back[i * P + j] = h_c(i, j);
  }
  Kokkos::finalize();

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
  delete[] c_host;
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_back;
  return 0;
}
