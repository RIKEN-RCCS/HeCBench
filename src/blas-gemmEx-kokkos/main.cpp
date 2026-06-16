/*
 * Kokkos port of blas-gemmEx.
 * Original tested cublasGemmEx with many data types (fp64, fp32, fp16, bf16, int8).
 * This port implements naive GEMM for fp64, fp32, and int32 using Kokkos.
 */

#include <sys/time.h>
#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <Kokkos_Core.hpp>

static int8_t float2int8(float f, float scale) {
  int8_t i = (int8_t)(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

static void performance(int m, int n, int k, bool is_integer, double avg_time_ms) {
  double total_ops = (double)m * n * k * 2;
  double perf = (total_ops / avg_time_ms) * 1e-9 * 1e3;
  const char* scale = "G";
  const char* unit = is_integer ? "OP/s" : "FLOP/s";
  if (perf >= 1000) { perf /= 1000; scale = "T"; }
  printf("%lf %s%s\n", perf, scale, unit);
}

// Generic GEMM: C = A*B (no transpose, row/col major assumed col-major like cuBLAS)
// A: m x k, B: k x n, C: m x n  (all col-major)
template<typename T, typename S>
void gemm_kokkos(Kokkos::View<T*> A, Kokkos::View<T*> B, Kokkos::View<S*> C,
                 int m, int n, int k) {
  Kokkos::parallel_for("gemm", m * n, KOKKOS_LAMBDA(int idx) {
    int col = idx / m;
    int row = idx % m;
    S s = S(0);
    for (int i = 0; i < k; i++)
      s += (S)A(row + i * m) * (S)B(i + col * k);
    C(row + col * m) = s;
  });
}

template<typename T, typename S>
void test_gemm(Kokkos::View<T*> A, Kokkos::View<T*> B, Kokkos::View<S*> C,
               int m, int n, int k, int iteration, bool is_int = false) {
  double total_time = 0;
  for (int i = 0; i < iteration; i++) {
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();
    gemm_kokkos(A, B, C, m, n, k);
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;
    if (i > 0) total_time += elapsed;
  }
  double avg_time = total_time / (iteration - 1);
  printf("algo 0: %.3f ms\n", avg_time);
  performance(m, n, k, is_int, avg_time);
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 5) {
      printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
      printf("C = A X B (A: M * K, B: K * N, C: M * N)\n");
      Kokkos::finalize();
      return 1;
    }
    const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    const int iteration = atoi(argv[4]);
    printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

    Kokkos::View<double*>  dA("dA", m*k), dB("dB", k*n), dC("dC", m*n);
    Kokkos::View<float*>   fA("fA", m*k), fB("fB", k*n), fC("fC", m*n);
    Kokkos::View<int32_t*> iC("iC", m*n);
    Kokkos::View<int8_t*>  iA("iA", m*k), iB("iB", k*n);

    // Initialize
    auto h_dA = Kokkos::create_mirror_view(dA); auto h_dB = Kokkos::create_mirror_view(dB);
    auto h_fA = Kokkos::create_mirror_view(fA); auto h_fB = Kokkos::create_mirror_view(fB);
    auto h_iA = Kokkos::create_mirror_view(iA); auto h_iB = Kokkos::create_mirror_view(iB);

    for (int i = 0; i < m*k; i++) {
      float v = (float)(i % 255 - 127) / 127.f;
      h_dA(i) = (double)v; h_fA(i) = v; h_iA(i) = float2int8(v, 127.f);
    }
    for (int i = 0; i < k*n; i++) {
      float v = (float)(i % 255 - 127) / 127.f;
      h_dB(i) = (double)v; h_fB(i) = v; h_iB(i) = float2int8(v, 127.f);
    }
    Kokkos::deep_copy(dA, h_dA); Kokkos::deep_copy(dB, h_dB);
    Kokkos::deep_copy(fA, h_fA); Kokkos::deep_copy(fB, h_fB);
    Kokkos::deep_copy(iA, h_iA); Kokkos::deep_copy(iB, h_iB);

    printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
    test_gemm(dA, dB, dC, m, n, k, iteration);

    printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
    test_gemm(fA, fB, fC, m, n, k, iteration);

    printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
    test_gemm(iA, iB, iC, m, n, k, iteration, true);

    printf(">>>>>>>>>>>>>>>>> compare first ten values >>>>>>>>>>>>>>>>>\n");
    auto h_dC = Kokkos::create_mirror_view(dC); Kokkos::deep_copy(h_dC, dC);
    auto h_fC = Kokkos::create_mirror_view(fC); Kokkos::deep_copy(h_fC, fC);
    auto h_iC = Kokkos::create_mirror_view(iC); Kokkos::deep_copy(h_iC, iC);
    printf("fp64: "); for (int i = 0; i < 10; i++) printf("%.5lf%c", h_dC(i), " \n"[i==9]);
    printf("fp32: "); for (int i = 0; i < 10; i++) printf("%.5f%c",  h_fC(i), " \n"[i==9]);
    printf("int8: "); for (int i = 0; i < 10; i++) printf("%.5f%c",  (float)h_iC(i)/127/127, " \n"[i==9]);
  }
  Kokkos::finalize();
  return 0;
}
