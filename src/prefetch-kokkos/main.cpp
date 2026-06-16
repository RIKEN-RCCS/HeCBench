// Port of prefetch-cuda to Kokkos.
//
// The CUDA version distinguishes "prefetch" (cudaMemPrefetchAsync) vs "naive"
// (no prefetch) with managed memory.  Kokkos handles memory placement
// transparently, so both variants use the same Kokkos::View pattern.
//
// Correctness: after `repeat` iterations of B[i] += A[i] starting from
//   A[i]=1, B[i]=2, we expect B[i] == repeat + 2.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static constexpr int NUM_ELEMENTS = 64 * 1024 * 1024;

void run_benchmark(const char *label, int numElements, int repeat)
{
  printf("%s\n", label);

  Kokkos::View<float *> A("A", numElements);
  Kokkos::View<float *> B("B", numElements);

  // Initialize: A[i]=1, B[i]=2
  Kokkos::parallel_for(
      "init", numElements, KOKKOS_LAMBDA(int i) {
        A(i) = 1.0f;
        B(i) = 2.0f;
      });
  Kokkos::fence();

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "add", numElements,
        KOKKOS_LAMBDA(int i) { B(i) += A(i); });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count();
  printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

  // Verify: B[i] should equal repeat + 2
  auto h_B =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);

  float maxError = 0.0f;
  for (int i = 0; i < numElements; i++)
    maxError = std::fmax(maxError, std::fabs(h_B(i) - (float)(repeat + 2)));

  printf("%s\n", (maxError == 0.0f) ? "PASS" : "FAIL");
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    for (int i = 0; i < 10; i++)
      run_benchmark("Concurrent managed access with prefetch",
                    NUM_ELEMENTS, repeat);

    for (int i = 0; i < 10; i++)
      run_benchmark("Concurrent managed access without prefetch",
                    NUM_ELEMENTS, repeat);
  }
  Kokkos::finalize();
  return 0;
}
