/*
 * Atomic performance benchmark.
 * Tests various patterns of atomic operations on global memory.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define BLOCK_SIZE 256

template <typename T>
void BlockRangeAtomicOnGlobalMem(Kokkos::View<T*> data, int n) {
  Kokkos::parallel_for("BlockRange", n, KOKKOS_LAMBDA(int i) {
    Kokkos::atomic_increment(&data(i % BLOCK_SIZE));
  });
  Kokkos::fence();
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(Kokkos::View<T*> data, int n) {
  Kokkos::parallel_for("WarpRange", n, KOKKOS_LAMBDA(int i) {
    Kokkos::atomic_increment(&data(i & 0x1F));
  });
  Kokkos::fence();
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(Kokkos::View<T*> data, int n) {
  Kokkos::parallel_for("SingleRange", n, KOKKOS_LAMBDA(int i) {
    Kokkos::atomic_increment(&data(0));
  });
  Kokkos::fence();
}

template <typename T>
void atomicPerf(int n, int t, int repeat) {
  Kokkos::View<T*> d_data("data", t);
  auto h_data = Kokkos::create_mirror_view(d_data);

  auto reset = [&]() {
    for (int i = 0; i < t; i++) h_data(i) = (T)(i % 1024 + 1);
    Kokkos::deep_copy(d_data, h_data);
  };

  double time;

  reset();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) BlockRangeAtomicOnGlobalMem<T>(d_data, n);
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n", time * 1e-3f / repeat);

  reset();
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) WarpRangeAtomicOnGlobalMem<T>(d_data, n);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n", time * 1e-3f / repeat);

  reset();
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) SingleRangeAtomicOnGlobalMem<T>(d_data, n);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n", time * 1e-3f / repeat);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int n   = 3 * 4 * 7 * 8 * 9 * 256;
  const int len = 1024;

  Kokkos::initialize(argc, argv);
  {
    printf("\nFP64 atomic add\n");
    atomicPerf<double>(n, len, repeat);

    printf("\nINT32 atomic add\n");
    atomicPerf<int>(n, len, repeat);

    printf("\nFP32 atomic add\n");
    atomicPerf<float>(n, len, repeat);
  }
  Kokkos::finalize();
  return 0;
}
