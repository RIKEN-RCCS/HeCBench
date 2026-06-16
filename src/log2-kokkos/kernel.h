#pragma once
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdint>
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
float binary_log(float input, int precision)
{
  union { float f; uint32_t i; } d1;
  d1.f = input;
  uint8_t exponent = (uint8_t)(((d1.i & 0x7F800000u) >> 23) - 127u);
  int m = 0;
  int sum_m = 0;
  float result = 0.f;
  int test = (1 << exponent);
  float y = input / (float)test;
  bool max_condition_met = false;
  uint64_t one = 1;
  uint64_t denom = 0;
  uint64_t prev_denom = 0;

  while ((sum_m < precision + 1 && y != 1.f) || max_condition_met) {
    m = 0;
    while ((y < 2.f) && (sum_m + m < precision + 1)) {
      y *= y;
      m++;
    }
    sum_m += m;
    prev_denom = denom;
    denom = one << sum_m;

    if (sum_m >= precision) break;
    if (prev_denom > denom) {
      max_condition_met = true;
      break;
    }
    result += 1.f / (float)denom;
    y /= 2.f;
  }
  return (float)exponent + result;
}

inline void log2_approx(
  std::vector<float> &inputs,
  std::vector<float> &outputs,
  std::vector<int>   &precision,
  const int           num_inputs,
  const int           precision_count,
  const int           repeat)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = Kokkos::DefaultExecutionSpace::memory_space;

  Kokkos::View<float*, MemSpace> d_inputs("d_inputs", num_inputs);
  Kokkos::View<float*, MemSpace> d_outputs("d_outputs", (size_t)num_inputs * precision_count);

  auto h_inputs = Kokkos::create_mirror_view(d_inputs);
  for (int i = 0; i < num_inputs; ++i) h_inputs(i) = inputs[i];
  Kokkos::deep_copy(d_inputs, h_inputs);

  for (int i = 0; i < precision_count; ++i) {
    int prec = precision[i];
    int offset = i * num_inputs;

    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < repeat; ++k) {
      Kokkos::parallel_for("log2_approx",
        Kokkos::RangePolicy<ExecSpace>(0, num_inputs),
        KOKKOS_LAMBDA(const int j) {
          d_outputs(offset + j) = binary_log(d_inputs(j), prec);
        });
      Kokkos::fence();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nIterative approximation with " << prec << " bits of precision\n";
    std::cout << "Average kernel execution time " << etime * 1e-3 / repeat << " (us)\n";
  }

  auto h_outputs = Kokkos::create_mirror_view(d_outputs);
  Kokkos::deep_copy(h_outputs, d_outputs);
  for (int i = 0; i < (int)outputs.size(); ++i)
    outputs[i] = h_outputs(i);
}
