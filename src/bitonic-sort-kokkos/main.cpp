// Kokkos port of the parallel bitonic-sort benchmark.
// Based on the Intel SYCL bitonic-sort sample (MIT licence).
// Algorithm overview:
//   Decompose a random sequence of size 2^n into bitonic sequences,
//   then merge them step by step into a single sorted sequence.
#include <math.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <Kokkos_Core.hpp>

// One parallel-for launch per (step, stage) pair.
// Matches the CUDA kernel exactly.
void bitonicSortStep(
    Kokkos::View<int*> a,
    const int seq_len,
    const int two_power,
    const int size)
{
  Kokkos::parallel_for(
    "bitonic_sort",
    size,
    KOKKOS_LAMBDA(const int i) {
      const int seq_num   = i / seq_len;
      const int h_len     = seq_len / 2;
      int swapped_ele = -1;
      if (i < (seq_len * seq_num) + h_len)
        swapped_ele = i + h_len;

      const int odd        = seq_num / two_power;
      const bool increasing = (odd % 2) == 0;

      if (swapped_ele != -1) {
        if (((a(i) > a(swapped_ele)) && increasing) ||
            ((a(i) < a(swapped_ele)) && !increasing)) {
          int tmp          = a(i);
          a(i)             = a(swapped_ele);
          a(swapped_ele)   = tmp;
        }
      }
    }
  );
  Kokkos::fence();
}

void parallelBitonicSort(int *input, const int n) {
  const int size  = 1 << n;   // 2^n
  const size_t nb = (size_t)size * sizeof(int);

  Kokkos::View<int*> d_a("a", size);
  {
    auto h_a = Kokkos::View<int*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(input, size);
    Kokkos::deep_copy(d_a, h_a);
  }

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int step = 0; step < n; step++) {
    for (int stage = step; stage >= 0; stage--) {
      const int seq_len   = 1 << (stage + 1);
      const int two_power = 1 << (step - stage);
      bitonicSortStep(d_a, seq_len, two_power, size);
    }
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (ms)\n", time * 1e-6f);

  auto h_a = Kokkos::View<int*, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(input, size);
  Kokkos::deep_copy(h_a, d_a);
}

// ─── Serial reference ────────────────────────────────────────────────────────

void swapElements(int step, int stage, int num_sequence, int seq_len, int *array) {
  for (int seq_num = 0; seq_num < num_sequence; seq_num++) {
    const int odd        = seq_num / (1 << (step - stage));
    const bool increasing = (odd % 2) == 0;
    const int h_len      = seq_len / 2;
    for (int i = seq_num * seq_len; i < seq_num * seq_len + h_len; i++) {
      int swapped_ele = i + h_len;
      if (((array[i] > array[swapped_ele]) && increasing) ||
          ((array[i] < array[swapped_ele]) && !increasing)) {
        int tmp = array[i]; array[i] = array[swapped_ele]; array[swapped_ele] = tmp;
      }
    }
  }
}

void bitonicSort(int *a, const int n) {
  for (int step = 0; step < n; step++)
    for (int stage = step; stage >= 0; stage--) {
      int num_seq  = 1 << (n - stage - 1);
      int seq_len  = 1 << (stage + 1);
      swapElements(step, stage, num_seq, seq_len, a);
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

void usage(const char *prog, int exp_max) {
  std::cout << "Usage: " << prog << " n seed\n"
            << "  n    : exponent for array size (array = 2^n, 0 <= n < " << exp_max << ")\n"
            << "  seed : RNG seed\n";
}

int main(int argc, char *argv[]) {
  const int exp_max = (int)log2((double)std::numeric_limits<int>::max());

  if (argc != 3) { usage(argv[0], exp_max); return -1; }

  int n, seed;
  try {
    n    = std::stoi(argv[1]);
    seed = std::stoi(argv[2]);
    if (n < 0 || n >= exp_max) { usage(argv[0], exp_max); return -1; }
  } catch (...) { usage(argv[0], exp_max); return -1; }

  const int size = 1 << n;
  std::cout << "\nArray size: " << size << ", seed: " << seed << "\n";

  int *data_cpu = (int*) malloc(size * sizeof(int));
  int *data_gpu = (int*) malloc(size * sizeof(int));

  srand(seed);
  for (int i = 0; i < size; i++)
    data_gpu[i] = data_cpu[i] = rand() % 1000;

  Kokkos::initialize(argc, argv);
  {
    std::cout << "Bitonic sort (parallel Kokkos)..\n";
    parallelBitonicSort(data_gpu, n);
  }
  Kokkos::finalize();

  std::cout << "Bitonic sort (serial)..\n";
  bitonicSort(data_cpu, n);

  int unequal = memcmp(data_gpu, data_cpu, (size_t)size * sizeof(int));
  std::cout << (unequal ? "FAIL" : "PASS") << std::endl;

  free(data_cpu);
  free(data_gpu);
  return 0;
}
