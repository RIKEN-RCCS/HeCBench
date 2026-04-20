/*
 * Kokkos port of allreduce benchmark.
 * Original used MPI + custom NCCL-like ring-based all-reduce.
 * This port implements a single-device tree reduction using Kokkos,
 * simulating the bandwidth-bound nature of collective operations.
 */

#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

// Simple parallel sum reduction (simulates allreduce on a single device)
void allreduce_sum(Kokkos::View<float*> data, int n) {
  float sum = 0.f;
  Kokkos::parallel_reduce("allreduce", n,
    KOKKOS_LAMBDA(int i, float& lsum) { lsum += data(i); },
    sum);
  Kokkos::parallel_for("broadcast", n,
    KOKKOS_LAMBDA(int i) { data(i) = sum; });
}

void run_test(size_t n, int iterations) {
  Kokkos::View<float*> data("data", n);
  Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i) {
    data(i) = 1.0f;
  });
  Kokkos::fence();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    allreduce_sum(data, n);
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;

  auto h_data = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(h_data, data);
  // After reduce+broadcast every element should equal n (sum of n ones)
  bool ok = (fabsf(h_data(0) - (float)n) < 1.0f);
  printf("  size=%zu  avg=%.2f us  %s\n", n, time_us / iterations, ok ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    printf("Kokkos allreduce (single-device simulation)\n");
    std::vector<size_t> sizes = {1<<20, 1<<22, 1<<24, 1<<26};
    int iterations = 10;
    for (size_t s : sizes)
      run_test(s, iterations);
  }
  Kokkos::finalize();
  return 0;
}
