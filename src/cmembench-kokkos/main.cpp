// Kokkos port of cmembench CUDA benchmark.
// Benchmarks read-only memory bandwidth using scalar, 2-wide, and 4-wide access.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define VECTOR_SIZE 1024
#define BLOCK_SIZE  256

// Total number of ints processed per kernel launch
static constexpr long long TOTAL_INTS = 4096LL * VECTOR_SIZE;

// Run the benchmark for a given access stride and number of iterations.
// Each logical "thread" reads `stride` ints from data[], indexed with wrap-around.
// Total reads = TOTAL_INTS; since all elements are 1 the sum equals TOTAL_INTS.
// Returns measured bandwidth in GB/s.
double run_bench(const Kokkos::View<const int*>& data,
                 int stride, int repeat) {
  const long long n = TOTAL_INTS / stride;  // number of parallel work-items
  const int vs = VECTOR_SIZE;

  // Warm-up
  {
    long long sum = 0;
    Kokkos::parallel_reduce(
        "warmup", Kokkos::RangePolicy<>(0LL, n),
        KOKKOS_LAMBDA(const long long i, long long& acc) {
          long long s = 0;
          for (int k = 0; k < stride; ++k)
            s += data((i * stride + k) % vs);
          acc += s;
        },
        sum);
    Kokkos::fence();
  }

  long long verified_sum = -1;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; ++r) {
    long long sum = 0;
    Kokkos::parallel_reduce(
        "membench", Kokkos::RangePolicy<>(0LL, n),
        KOKKOS_LAMBDA(const long long i, long long& acc) {
          long long s = 0;
          for (int k = 0; k < stride; ++k)
            s += data((i * stride + k) % vs);
          acc += s;
        },
        sum);
    Kokkos::fence();
    if (r == 0) verified_sum = sum;
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  if (verified_sum == TOTAL_INTS)
    printf("  Verification PASS (sum=%lld)\n", verified_sum);
  else
    printf("  Verification FAIL: sum=%lld expected=%lld\n",
           verified_sum, TOTAL_INTS);

  double elapsed_s =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
      * 1e-9;
  double bytes = (double)TOTAL_INTS * sizeof(int) * repeat;
  return bytes / elapsed_s / 1e9;
}

int main(int argc, char* argv[]) {
  int repeat = (argc > 1) ? std::atoi(argv[1]) : 100;

  Kokkos::initialize(argc, argv);
  {
    // Build read-only data: all ones
    Kokkos::View<int*> d_mutable("constant_data", VECTOR_SIZE);
    auto h = Kokkos::create_mirror_view(d_mutable);
    for (int i = 0; i < VECTOR_SIZE; ++i) h(i) = 1;
    Kokkos::deep_copy(d_mutable, h);

    // Create a const view (read-only alias)
    Kokkos::View<const int*> d_data = d_mutable;

    printf("cmembench: VECTOR_SIZE=%d, TOTAL_INTS=%lld, repeat=%d\n",
           VECTOR_SIZE, TOTAL_INTS, repeat);
    printf("%-12s  %10s\n", "Access", "BW (GB/s)");

    double bw;
    bw = run_bench(d_data, 1, repeat);
    printf("%-12s  %10.3f\n", "int (x1)", bw);

    bw = run_bench(d_data, 2, repeat);
    printf("%-12s  %10.3f\n", "int2 (x2)", bw);

    bw = run_bench(d_data, 4, repeat);
    printf("%-12s  %10.3f\n", "int4 (x4)", bw);
  }
  Kokkos::finalize();
  return 0;
}
