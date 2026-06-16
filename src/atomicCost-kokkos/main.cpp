#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define BLOCK_SIZE 256

template <typename T>
void woAtomicOnGlobalMem(Kokkos::View<T*> d_result, int size, int n)
{
  Kokkos::parallel_for("woAtomic", n, KOKKOS_LAMBDA(int tid) {
    for (unsigned int i = tid * size; i < (unsigned int)(tid + 1) * size; i++) {
      d_result(tid) += (T)(i % 2);
    }
  });
  Kokkos::fence();
}

template <typename T>
void wiAtomicOnGlobalMem(Kokkos::View<T*> d_result, int size, int n)
{
  Kokkos::parallel_for("wiAtomic", n, KOKKOS_LAMBDA(int tid) {
    for (unsigned int i = tid * size; i < (unsigned int)(tid + 1) * size; i++) {
      Kokkos::atomic_add(&d_result(tid), (T)(i % 2));
    }
  });
  Kokkos::fence();
}

template <typename T>
void atomicCost(int length, int size, int repeat)
{
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  Kokkos::View<T*> d_result_wi("result_wi", num_threads);
  Kokkos::View<T*> d_result_wo("result_wo", num_threads);
  Kokkos::deep_copy(d_result_wi, (T)0);
  Kokkos::deep_copy(d_result_wo, (T)0);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    wiAtomicOnGlobalMem<T>(d_result_wi, size, num_threads);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
         time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    woAtomicOnGlobalMem<T>(d_result_wo, size, num_threads);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
         time * 1e-3f / repeat);

  auto h_result_wi = Kokkos::create_mirror_view(d_result_wi);
  auto h_result_wo = Kokkos::create_mirror_view(d_result_wo);
  Kokkos::deep_copy(h_result_wi, d_result_wi);
  Kokkos::deep_copy(h_result_wo, d_result_wo);

  // Normalize by number of repeats before comparison
  bool diff = false;
  for (int i = 0; i < num_threads; i++) {
    if (h_result_wi(i) != h_result_wo(i)) {
      diff = true;
      break;
    }
  }
  printf("%s\n", diff ? "FAIL" : "PASS");
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  Kokkos::initialize(argc, argv);
  {
    printf("\nFP64 atomic add\n");
    atomicCost<double>(length, nelems, repeat);

    printf("\nINT32 atomic add\n");
    atomicCost<int>(length, nelems, repeat);

    printf("\nFP32 atomic add\n");
    atomicCost<float>(length, nelems, repeat);
  }
  Kokkos::finalize();
  return 0;
}
