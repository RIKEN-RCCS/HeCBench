// Kokkos port of concurrentKernels CUDA benchmark.
// Runs N "clock_block" kernels (each computing sum of j%3 over clock_count
// iterations) then sums the results and verifies the total.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>

static constexpr long long CLOCK_COUNT = 1000000LL;

int main(int argc, char* argv[]) {
  int nkernels = (argc > 1) ? std::atoi(argv[1]) : 4;
  if (nkernels < 1) { fprintf(stderr, "nkernels must be >= 1\n"); return 1; }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<long long*> d_results("results", nkernels);

    // Each "clock_block" kernel: one parallel_reduce computing
    // sum_{j=0}^{CLOCK_COUNT-1} (j % 3)
    for (int k = 0; k < nkernels; ++k) {
      long long kernel_sum = 0;
      Kokkos::parallel_reduce(
          "clock_block",
          Kokkos::RangePolicy<>(0LL, CLOCK_COUNT),
          KOKKOS_LAMBDA(const long long j, long long& acc) {
            acc += j % 3;
          },
          kernel_sum);
      auto sub = Kokkos::subview(d_results, k);
      Kokkos::deep_copy(sub, kernel_sum);
    }

    // Sum kernel: reduce the per-kernel results
    long long total = 0;
    Kokkos::parallel_reduce(
        "sum_kernel",
        Kokkos::RangePolicy<>(0, nkernels),
        KOKKOS_LAMBDA(const int i, long long& acc) {
          acc += d_results(i);
        },
        total);
    Kokkos::fence();

    // Expected: nkernels * sum_{j=0}^{CLOCK_COUNT-1} (j%3)
    // Pattern of j%3: 0,1,2,0,1,2,... sum per 3 = 3, so per CLOCK_COUNT:
    long long full_cycles = CLOCK_COUNT / 3;
    long long rem = CLOCK_COUNT % 3;
    long long single_sum = full_cycles * 3;
    for (long long r = 0; r < rem; ++r) single_sum += r;
    long long expected = (long long)nkernels * single_sum;

    printf("nkernels    : %d\n", nkernels);
    printf("clock_count : %lld\n", CLOCK_COUNT);
    printf("single_sum  : %lld\n", single_sum);
    printf("total       : %lld\n", total);
    printf("expected    : %lld\n", expected);
    printf("%s\n", (total == expected) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
