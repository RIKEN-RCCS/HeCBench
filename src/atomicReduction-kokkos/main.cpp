/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.

   Kokkos port of the atomicReduction benchmark.
*/

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

// Performs a sum reduction where each work-item processes VEC consecutive
// elements per stride, matching the vectorization variants in the CUDA version.
template<int VEC>
int reduction_variant(Kokkos::View<int*> d_in, int arrayLength, int block_size) {
  int n_blocks = std::min((arrayLength / VEC + block_size - 1) / block_size, 2048);
  int total_threads = n_blocks * block_size;
  int sum = 0;
  Kokkos::parallel_reduce("atomic_reduction",
    Kokkos::RangePolicy<>(0, total_threads),
    KOKKOS_LAMBDA(int idx, int& lsum) {
      int thread_sum = 0;
      for (int i = idx * VEC; i < arrayLength; i += total_threads * VEC) {
        for (int v = 0; v < VEC && i + v < arrayLength; v++) {
          thread_sum += d_in(i + v);
        }
      }
      lsum += thread_sum;
    }, sum);
  return sum;
}

template<int VEC>
void benchmark_variant(Kokkos::View<int*> d_in, int arrayLength, int block_size,
                       int N, int checksum, float GB) {
  int sum = 0;
  Kokkos::fence();
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    sum = reduction_variant<VEC>(d_in, arrayLength, block_size);
  }
  Kokkos::fence();
  auto t2 = std::chrono::high_resolution_clock::now();
  double times = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
  printf("Thread block size: %4d, vec_factor=%2d: perf = %.3f GBytes/sec\n",
         block_size, VEC, 1.0e-9 * GB / times);
  if (sum == checksum)
    printf("VERIFICATION: PASS\n\n");
  else
    printf("VERIFICATION: FAIL!! (got %d, expected %d)\n\n", sum, checksum);
}

int main(int argc, char** argv) {
  int arrayLength = 52428800;
  int N = 32;

  if (argc == 3) {
    arrayLength = atoi(argv[1]);
    N = atoi(argv[2]);
  }

  printf("Array size: %.2f MB\n", arrayLength * sizeof(int) / 1024.0 / 1024.0);
  printf("Repeat the kernel execution: %d times\n", N);

  int* array = (int*)malloc(arrayLength * sizeof(int));
  int checksum = 0;
  for (int i = 0; i < arrayLength; i++) {
    array[i] = rand() % 2;
    checksum += array[i];
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_in("d_in", arrayLength);
    {
      auto h_in = Kokkos::create_mirror_view(d_in);
      for (int i = 0; i < arrayLength; i++) h_in(i) = array[i];
      Kokkos::deep_copy(d_in, h_in);
      Kokkos::fence();
    }

    float GB = (float)arrayLength * sizeof(int) * N;

    // Warmup
    for (int i = 0; i < N; i++) {
      int sum = 0;
      Kokkos::parallel_reduce("warmup",
        Kokkos::RangePolicy<>(0, arrayLength),
        KOKKOS_LAMBDA(int idx, int& lsum) {
          lsum += d_in(idx);
        }, sum);
    }
    Kokkos::fence();

    int block_sizes[] = {128, 256, 512, 1024};
    for (int k = 0; k < 4; k++) {
      int block_size = block_sizes[k];
      benchmark_variant<1> (d_in, arrayLength, block_size, N, checksum, GB);
      benchmark_variant<2> (d_in, arrayLength, block_size, N, checksum, GB);
      benchmark_variant<4> (d_in, arrayLength, block_size, N, checksum, GB);
      benchmark_variant<8> (d_in, arrayLength, block_size, N, checksum, GB);
      benchmark_variant<16>(d_in, arrayLength, block_size, N, checksum, GB);
    }
  }
  Kokkos::finalize();
  free(array);
  return 0;
}
