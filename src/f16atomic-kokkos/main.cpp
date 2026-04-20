// Kokkos port of f16atomic-cuda
// FP16/BF16 atomicAdd replaced with float32 atomic add.
// The benchmark measures atomic-add throughput on global memory.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define BLOCK_SIZE 256

// Kernel: each thread repeatedly performs a float atomic add
void atomicCost(int nelems, int repeat)
{
  const int atomics_count = 256;
  size_t result_size = sizeof(float) * nelems;
  Kokkos::View<float*> d_result("result", nelems);
  Kokkos::deep_copy(d_result, 0.f);

  const float val0 = 0.f;       // ZERO_FP16 → 0.0f
  const float val1 = 6.1035e-5f; // MIN_FP16  → smallest positive normal fp16

  // warm-up
  Kokkos::parallel_for(nelems / 2, KOKKOS_LAMBDA(int tid) {
    float v0 = val0, v1 = val1;
    for (int i = 0; i < atomics_count; i++) {
      Kokkos::atomic_add(&d_result(tid * 2),     v0);
      Kokkos::atomic_add(&d_result(tid * 2 + 1), v1);
    }
  });
  Kokkos::fence();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(nelems / 2, KOKKOS_LAMBDA(int tid) {
      float v0 = val0, v1 = val1;
      for (int i = 0; i < atomics_count; i++) {
        Kokkos::atomic_add(&d_result(tid * 2),     v0);
        Kokkos::atomic_add(&d_result(tid * 2 + 1), v1);
      }
    });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of float atomic add on global memory: %f (us)\n",
         time * 1e-3f / repeat);

  auto h_result = Kokkos::create_mirror_view(d_result);
  Kokkos::deep_copy(h_result, d_result);
  printf("First two elements: %f %f\n\n", h_result(0), h_result(1));
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  if (nelems <= 0 || nelems % 2 != 0) {
    printf("N must be positive and a multiple of 2\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    printf("\nFP32 atomic add (proxy for FP16/BF16)\n");
    atomicCost(nelems, repeat);
  }
  Kokkos::finalize();
  return 0;
}
