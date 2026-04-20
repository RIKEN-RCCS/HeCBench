// Kokkos port of dispatch-cuda
// Measures kernel dispatch (enqueue) rate and execution latency
// for empty Kokkos parallel_for kernels.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

#define WARMUP_RUN_COUNT  100
#define TIMING_RUN_COUNT  1000
#define TOTAL_RUN_COUNT   (WARMUP_RUN_COUNT + TIMING_RUN_COUNT)
#define BATCH_SIZE        1000

void print_timing(const char* test,
                  const std::array<float, TOTAL_RUN_COUNT>& results,
                  int batch = 1)
{
  float total_us = 0.0f, mean_us = 0.0f, stddev_us = 0.0f;

  auto start_iter = results.begin() + WARMUP_RUN_COUNT;
  auto end_iter   = results.end();

  std::for_each(start_iter, end_iter, [&](const float& run_ms) {
    total_us += (run_ms * 1000) / batch;
  });
  mean_us = total_us / TIMING_RUN_COUNT;

  total_us = 0;
  std::for_each(start_iter, end_iter, [&](const float& run_ms) {
    float dev_us = ((run_ms * 1000) / batch) - mean_us;
    total_us += dev_us * dev_us;
  });
  stddev_us = std::sqrt(total_us / TIMING_RUN_COUNT);

  printf("\n %s: mean = %.1f us, stddev = %.1f us\n", test, mean_us, stddev_us);
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    std::array<float, TOTAL_RUN_COUNT> results;

    // Kernel launch enqueue rate
    for (int i = 0; i < TOTAL_RUN_COUNT; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {});
      auto t1 = std::chrono::high_resolution_clock::now();
      results[i] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    print_timing("Enqueue rate", results);

    // Single dispatch execution latency
    for (int i = 0; i < TOTAL_RUN_COUNT; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {});
      Kokkos::fence();
      auto t1 = std::chrono::high_resolution_clock::now();
      results[i] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    print_timing("Single dispatch latency", results);

    // Batch dispatch execution latency
    for (int i = 0; i < TOTAL_RUN_COUNT; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < BATCH_SIZE; j++)
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {});
      Kokkos::fence();
      auto t1 = std::chrono::high_resolution_clock::now();
      results[i] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
    print_timing("Batch dispatch latency", results, BATCH_SIZE);
  }
  Kokkos::finalize();
  return 0;
}
