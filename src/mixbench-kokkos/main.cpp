/**
 * This file is the modified read-only mixbench GPU micro-benchmark suite
 * ported to Kokkos.
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(Kokkos::View<float*> cd, int grid_dim, int block_dim, int compute_iterations) {
  const int total_threads = grid_dim * block_dim;
  const int big_stride = grid_dim * block_dim * granularity;

  Kokkos::parallel_for("mixbench", total_threads, KOKKOS_LAMBDA(int tid) {
    int idx = (tid / block_dim) * block_dim * granularity + (tid % block_dim);

    float tmps[granularity];
    for (int k = 0; k < fusion_degree; k++) {
      for (int j = 0; j < granularity; j++) {
        // Load elements (memory intensive part)
        tmps[j] = cd(idx + j * block_dim + k * big_stride);

        // Perform computations (compute intensive part)
        for (int i = 0; i < compute_iterations; i++)
          tmps[j] = tmps[j] * tmps[j] + (float)seed;
      }

      // Multiply add reduction
      float sum = 0.f;
      for (int j = 0; j < granularity; j += 2)
        sum += tmps[j] * tmps[j + 1];

      for (int j = 0; j < granularity; j++)
        cd(idx + k * big_stride) = sum;
    }
  });
  Kokkos::fence();
}

void mixbenchGPU(long size, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);

  Kokkos::View<float*> d_cd("d_cd", size);
  auto h_cd = Kokkos::create_mirror_view(d_cd);
  for (long i = 0; i < size; i++) h_cd(i) = 0;
  Kokkos::deep_copy(d_cd, h_cd);

  const long reduced_grid_size = size / granularity / 128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size / block_dim;

  // warmup
  for (int i = 0; i < repeat; i++) {
    benchmark_func(d_cd, grid_dim, block_dim, i);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    benchmark_func(d_cd, grid_dim, block_dim, i);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time: %f (s)\n", time * 1e-9f);

  Kokkos::deep_copy(h_cd, d_cd);

  // verification
  bool ok = true;
  for (long i = 0; i < size; i++) {
    if (h_cd(i) != 0) {
      if (fabsf(h_cd(i) - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %ld: %f\n", i, h_cd(i));
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE * sizeof(float);
  printf("Buffer size: %dMB\n", datasize / (1024 * 1024));

  Kokkos::initialize(argc, argv);
  {
    mixbenchGPU(VECTOR_SIZE, repeat);
  }
  Kokkos::finalize();

  return 0;
}
