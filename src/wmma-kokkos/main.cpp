// Port of wmma CUDA benchmark to Kokkos
// Original: GEMM using WMMA (tensor core) API
// Kokkos port: regular FP32 matrix multiply (D = alpha*(A*B) + beta*C)
// A is stored row-major [M x K], B col-major [K x N], C/D row-major [M x N]

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>

typedef float fp32;
// Use fp32 instead of fp16 — no tensor-core hardware in this environment
typedef float fp16;

template <typename DataT>
static void fill_mat(DataT *mat, uint32_t m, uint32_t n) {
  srand(m * n);
  for (uint32_t i = 0; i < m; i++)
    for (uint32_t j = 0; j < n; j++) {
      auto v = (i * n + j) % 13;
      mat[i * n + j] = (v % 3) ? -(DataT)v : (DataT)v;
    }
}

static void gemm_cpu(uint32_t m, uint32_t n, uint32_t k,
                     const fp16 *a, const fp16 *b, const fp32 *c, fp32 *d,
                     uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd,
                     fp32 alpha, fp32 beta) {
  for (uint32_t i = 0; i < m; i++)
    for (uint32_t j = 0; j < n; j++) {
      fp32 acc = 0.f;
      for (uint32_t h = 0; h < k; h++)
        acc += a[i * lda + h] * b[j * ldb + h]; // B is col-major
      d[i * ldd + j] = alpha * acc + beta * c[i * ldc + j];
    }
}

static void compareEqual(const fp32 *a, const fp32 *b, uint32_t size,
                         double tolerance = 10.0) {
  double max_rel = 0.0;
  for (uint32_t i = 0; i < size; i++) {
    double rel = fabs(a[i] - b[i]) / (fabs(a[i]) + fabs(b[i]) + 1.0);
    if (rel > max_rel || rel != rel) max_rel = rel;
  }
  auto eps = std::numeric_limits<fp32>::epsilon();
  if (max_rel != max_rel || max_rel > eps * tolerance)
    std::cout << "FAILED\n";
  else
    std::cout << "PASSED\n";
  std::cout << "Max relative error: " << max_rel << std::endl;
}

static void gemm_kokkos(uint32_t m, uint32_t n, uint32_t k, fp32 alpha, fp32 beta,
                        int32_t repeat, int32_t verify) {
  if (m == 0 || n == 0 || k == 0) { std::cout << "Unsupported size!\n"; return; }

  uint32_t lda = k, ldb = k, ldc = n, ldd = n;
  std::vector<fp16> A(m * k), B(k * n);
  std::vector<fp32> C(m * n), D(m * n, std::numeric_limits<fp32>::signaling_NaN());

  fill_mat(A.data(), m, k);
  fill_mat(B.data(), k, n);
  fill_mat(C.data(), m, n);

  std::cout << "Initializing host data..." << std::endl;
  std::cout << "Initializing device data..." << std::endl;

  Kokkos::View<fp16*> d_a("a", m * k);
  Kokkos::View<fp16*> d_b("b", k * n);
  Kokkos::View<fp32*> d_c("c", m * n);
  Kokkos::View<fp32*> d_d("d", m * n);

  {
    auto ha = Kokkos::create_mirror_view(d_a);
    auto hb = Kokkos::create_mirror_view(d_b);
    auto hc = Kokkos::create_mirror_view(d_c);
    auto hd = Kokkos::create_mirror_view(d_d);
    for (size_t i = 0; i < A.size(); i++) ha(i) = A[i];
    for (size_t i = 0; i < B.size(); i++) hb(i) = B[i];
    for (size_t i = 0; i < C.size(); i++) { hc(i) = C[i]; hd(i) = D[i]; }
    Kokkos::deep_copy(d_a, ha); Kokkos::deep_copy(d_b, hb);
    Kokkos::deep_copy(d_c, hc); Kokkos::deep_copy(d_d, hd);
  }

  std::cout << "Launching GEMM kernel..." << std::endl;

  // Warmup
  Kokkos::parallel_for("gemm_warmup", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{(int64_t)m,(int64_t)n}),
    KOKKOS_LAMBDA(const int64_t i, const int64_t j) {
      fp32 acc = 0.f;
      for (uint32_t h = 0; h < k; h++) acc += d_a(i * lda + h) * d_b(j * ldb + h);
      d_d(i * ldd + j) = alpha * acc + beta * d_c(i * ldc + j);
    });
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int32_t w = 0; w < repeat; w++) {
    Kokkos::parallel_for("gemm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{(int64_t)m,(int64_t)n}),
      KOKKOS_LAMBDA(const int64_t i, const int64_t j) {
        fp32 acc = 0.f;
        for (uint32_t h = 0; h < k; h++) acc += d_a(i * lda + h) * d_b(j * ldb + h);
        d_d(i * ldd + j) = alpha * acc + beta * d_c(i * ldc + j);
      });
  }
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6;

  auto gFlops      = 2.0 * m * n * k * 1e-9;
  auto tFlopsPerSec = gFlops * repeat / (elapsed_ms * 1e-3);

  std::cout << "BlkM, BlkN, BlkK, MatM, MatN, MatK, alpha, lda, ldb, beta, ldc, ldd, "
               "elapsedMs, Problem Size(GFlops), TFlops/s\n";
  std::cout << 16 << ", " << 16 << ", " << 16 << ", " << m << ", " << n << ", " << k
            << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", " << ldc
            << ", " << ldd << ", " << elapsed_ms << ", " << gFlops << ", " << tFlopsPerSec
            << "\n";

  if (verify) {
    std::cout << "Validating result with reference..." << std::endl;
    auto hd = Kokkos::create_mirror_view(d_d);
    Kokkos::deep_copy(hd, d_d);
    std::vector<fp32> D2(m * n, std::numeric_limits<fp32>::signaling_NaN());
    gemm_cpu(m, n, k, A.data(), B.data(), C.data(), D2.data(), lda, ldb, ldc, ldd, alpha, beta);
    for (size_t i = 0; i < D.size(); i++) D[i] = hd(i);
    compareEqual(D.data(), D2.data(), m * n);
  }
  std::cout << "Finished!" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0] << " <M> <N> <K> <repeat> <verify>\n";
    return -1;
  }
  const uint32_t m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
  const int32_t  repeat = atoi(argv[4]), verify = atoi(argv[5]);

  Kokkos::initialize(argc, argv);
  {
    gemm_kokkos(m, n, k, 0.5f, 2.0f, repeat, verify);
  }
  Kokkos::finalize();
  return 0;
}
