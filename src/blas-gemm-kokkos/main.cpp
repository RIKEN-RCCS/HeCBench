// Kokkos port of blas-gemm from HeCBench
// Original CUDA version uses cuBLAS; this port uses KokkosBlas::gemm
// Half precision is omitted: KokkosBlas::gemm does not support half_t
// across all backends. Float and double are fully supported.

#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "utils.h"

// -----------------------------------------------------------------------
// Simple (reference) GEMM kernel: C = alpha * A * B + beta * C
// A is (M x K), B is (K x N), C is (M x N) — all row-major (LayoutRight)
// -----------------------------------------------------------------------
template <typename T>
struct MatMulFunctor {
  Kokkos::View<const T**, Kokkos::LayoutRight> A;
  Kokkos::View<const T**, Kokkos::LayoutRight> B;
  Kokkos::View<T**,       Kokkos::LayoutRight> C;
  int K;
  T alpha, beta;

  KOKKOS_INLINE_FUNCTION
  void operator()(int row, int col) const {
    T s = T(0);
    for (int k = 0; k < K; k++)
      s += A(row, k) * B(k, col);
    C(row, col) = alpha * s + beta * C(row, col);
  }
};

template <typename T>
void run_simple_gemm(
    Kokkos::View<const T**, Kokkos::LayoutRight> A,
    Kokkos::View<const T**, Kokkos::LayoutRight> B,
    Kokkos::View<T**,       Kokkos::LayoutRight> C,
    int M, int K, int N, T alpha, T beta)
{
  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "simple_gemm",
      Policy({0, 0}, {M, N}),
      MatMulFunctor<T>{A, B, C, K, alpha, beta});
}

// -----------------------------------------------------------------------
// Full benchmark: initialise, verify correctness, measure performance
// -----------------------------------------------------------------------
template <typename fp>
void run_gemm_example(int m, int k, int n, int repeat)
{
  const fp alpha = fp(2.0);
  const fp beta  = fp(0.5);

  // ---- Host data -------------------------------------------------------
  const size_t A_elems = static_cast<size_t>(m) * k;
  const size_t B_elems = static_cast<size_t>(k) * n;
  const size_t C_elems = static_cast<size_t>(m) * n;

  fp* h_a = static_cast<fp*>(aligned_alloc(64, A_elems * sizeof(fp)));
  fp* h_b = static_cast<fp*>(aligned_alloc(64, B_elems * sizeof(fp)));
  fp* h_c = static_cast<fp*>(aligned_alloc(64, C_elems * sizeof(fp)));

  srand(2);
  rand_matrix(h_a, m, k);
  rand_matrix(h_b, k, n);
  rand_matrix(h_c, m, n);

  // ---- Kokkos Views on device ------------------------------------------
  // A (m x k), B (k x n), C (m x n), R (m x n) — R is the reference result
  Kokkos::View<fp**, Kokkos::LayoutRight> dA("A", m, k);
  Kokkos::View<fp**, Kokkos::LayoutRight> dB("B", k, n);
  Kokkos::View<fp**, Kokkos::LayoutRight> dC("C", m, n);
  Kokkos::View<fp**, Kokkos::LayoutRight> dR("R", m, n); // reference

  // Host mirrors for upload/download
  auto hA = Kokkos::create_mirror_view(dA);
  auto hB = Kokkos::create_mirror_view(dB);
  auto hC = Kokkos::create_mirror_view(dC);

  // Fill host mirrors from raw arrays
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      hA(i, j) = h_a[i * k + j];

  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      hB(i, j) = h_b[i * n + j];

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      hC(i, j) = h_c[i * n + j];

  Kokkos::deep_copy(dA, hA);
  Kokkos::deep_copy(dB, hB);
  Kokkos::deep_copy(dC, hC);
  Kokkos::deep_copy(dR, hC); // R starts with the same initial C

  // ---- Reference simple GEMM (into dR) ---------------------------------
  std::cout << "Checking KokkosBlas GEMM.. ";
  {
    auto dA_const = Kokkos::View<const fp**, Kokkos::LayoutRight>(dA);
    auto dB_const = Kokkos::View<const fp**, Kokkos::LayoutRight>(dB);
    run_simple_gemm(dA_const, dB_const, dR, m, k, n, alpha, beta);
  }
  Kokkos::fence();

  // ---- KokkosBlas::gemm (into dC) --------------------------------------
  // Computes dC = alpha * dA * dB + beta * dC
  KokkosBlas::gemm("N", "N", alpha, dA, dB, beta, dC);
  Kokkos::fence();

  // ---- Correctness check -----------------------------------------------
  auto hR = Kokkos::create_mirror_view(dR);
  auto hC2 = Kokkos::create_mirror_view(dC);
  Kokkos::deep_copy(hR, dR);
  Kokkos::deep_copy(hC2, dC);

  int error = 0;
  for (int i = 0; i < m && !error; i++)
    for (int j = 0; j < n && !error; j++)
      if (hR(i, j) != hC2(i, j))
        error = 1;

  std::cout << (error ? "FAIL" : "PASS") << std::endl;

  // ---- Performance benchmark -------------------------------------------
  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    KokkosBlas::gemm("N", "N", alpha, dA, dB, beta, dC);

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  performance(m, n, k, false, static_cast<double>(time) / repeat);

  // ---- Debug output ----------------------------------------------------
#ifdef DEBUG
  std::cout << "\n\t\tOutputting 2x2 block of A, B, C matrices:" << std::endl;
  print_2x2_matrix_values(h_a, k, "A");
  print_2x2_matrix_values(h_b, n, "B");
  Kokkos::deep_copy(hC2, dC);
  fp* flat_c = new fp[C_elems];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      flat_c[i * n + j] = hC2(i, j);
  print_2x2_matrix_values(flat_c, n, "C");
  delete[] flat_c;
#endif

  free(h_a);
  free(h_b);
  free(h_c);
}

// -----------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc < 5) {
    printf("Usage: %s <m> <k> <n> <repeat> [--kokkos-...]\n", argv[0]);
    return 1;
  }

  const int m      = atoi(argv[1]);
  const int k      = atoi(argv[2]);
  const int n      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  Kokkos::initialize(argc, argv);
  {
    std::cout << "\tRunning with single precision data type:" << std::endl;
    run_gemm_example<float>(m, k, n, repeat);

    std::cout << "\tRunning with double precision data type:" << std::endl;
    run_gemm_example<double>(m, k, n, repeat);
  }
  Kokkos::finalize();

  return 0;
}
