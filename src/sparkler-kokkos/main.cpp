// Kokkos port of sparkler-cuda
// Simplified: single-process GEMM benchmark (replaces MPI + cuBLAS)
// Computes C = A * B^T where A, B are sparse matrices stored in column-major

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// Distance between nonzero elements along a column
KOKKOS_INLINE_FUNCTION size_t nonzero_stride(const size_t i) {
  enum { MAX = 499 };
  return 1 + i % MAX;
}

// Set up sparse input matrix: column-major, element (r,c) = value if c % stride(r) == 0
void set_input_matrix(Kokkos::View<float*> mat, size_t nr, size_t nc,
                      size_t nru, size_t base_vector_num, float value) {
  Kokkos::parallel_for("set_input", nr * nc, KOKKOS_LAMBDA(const size_t index) {
    const size_t r = index % nr;
    const size_t c = index / nr;
    const size_t stride = nonzero_stride(r + base_vector_num);
    mat(r + nru * c) = (c % stride == 0) ? value : 0.0f;
  });
}

size_t gcd(size_t a, size_t b) { return a == 0 ? b : gcd(b % a, a); }
size_t lcm(size_t a, size_t b) { return (a * b) / gcd(a, b); }

// GEMM: C = A * B^T  (m x n = m x k * k x n^T)
// A: m x k (col-major, leading dim nru_a)
// B: n x k (col-major, leading dim nru_b)
// C: m x n (col-major, leading dim nru_c)
void perform_gemm(Kokkos::View<float*> A, Kokkos::View<float*> B,
                  Kokkos::View<float*> C,
                  size_t m, size_t n, size_t k,
                  size_t nru_a, size_t nru_b, size_t nru_c) {
  using policy2d = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("gemm", policy2d({0, 0}, {(int64_t)m, (int64_t)n}),
    KOKKOS_LAMBDA(const int64_t r, const int64_t c) {
      float sum = 0.0f;
      for (size_t kk = 0; kk < k; kk++)
        sum += A(r + nru_a * kk) * B(c + nru_b * kk);
      C(r + nru_c * c) = sum;
    });
}

int main(int argc, char** argv) {
  size_t num_vector = 0;
  size_t num_field = 0;
  int num_iterations = 1;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--num_vector") == 0 && i+1 < argc) num_vector = atol(argv[++i]);
    if (strcmp(argv[i], "--num_field") == 0 && i+1 < argc)  num_field  = atol(argv[++i]);
    if (strcmp(argv[i], "--num_iterations") == 0 && i+1 < argc) num_iterations = atoi(argv[++i]);
  }

  if (num_vector < 2 || num_field < 1) {
    printf("Usage: %s --num_vector <n> --num_field <n> [--num_iterations <n>]\n", argv[0]);
    printf("  num_vector >= 2, num_field >= 1\n");
    return 1;
  }

  printf("num_vector %zu num_field %zu num_iterations %d num_proc 1\n",
         num_vector, num_field, num_iterations);

  Kokkos::initialize(argc, argv);
  {
    enum { ROUNDUP = 8 };
    auto roundup = [](size_t x) { return ((x + ROUNDUP - 1) / ROUNDUP) * ROUNDUP; };

    const size_t m = 2 * roundup(num_vector); // padded
    const size_t k = num_field;
    const size_t n = m;
    const size_t nru = roundup(m);

    Kokkos::View<float*> A("A", roundup(m) * roundup(k));
    Kokkos::View<float*> B("B", roundup(n) * roundup(k));
    Kokkos::View<float*> C("C", roundup(m) * roundup(n));

    Kokkos::deep_copy(C, 0.0f);
    set_input_matrix(A, m, k, roundup(m), 0, 1.0f);
    set_input_matrix(B, n, k, roundup(n), 0, 1.0f);

    double timegemm = 0.0;
    double flops = 0.0;

    Kokkos::fence();
    auto tstart = std::chrono::steady_clock::now();

    for (int iteration = 1; iteration <= num_iterations; ++iteration) {
      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();

      perform_gemm(A, B, C, m, n, k, roundup(m), roundup(n), roundup(m));
      flops += 2.0 * m * n * k;

      Kokkos::fence();
      auto t2 = std::chrono::steady_clock::now();
      timegemm += std::chrono::duration<double>(t2 - t1).count();

      if (!(iteration & (iteration-1)) || iteration % 256 == 0 || iteration == num_iterations) {
        double elapsed = std::chrono::duration<double>(t2 - tstart).count();
        printf("Iteration %d of %d, elapsed sec %.3f\n", iteration, num_iterations, elapsed);
      }
    }

    Kokkos::fence();
    auto tend = std::chrono::steady_clock::now();
    double timetotal = std::chrono::duration<double>(tend - tstart).count();

    printf("TFLOPS %.3f\nGEMM time %.3f (s)\nGEMM TFLOPS/s %.3f\nTotal time %.3f (s)\n",
           flops/1e12, timegemm, flops*1e-12/timegemm, timetotal);
  }
  Kokkos::finalize();
  return 0;
}
