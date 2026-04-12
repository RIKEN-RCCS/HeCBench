//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

// Stub out CUDA host/device annotations before including common.hpp
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "common.hpp"   // timing utilities from mandelbrot-cuda/
#include <Kokkos_Core.hpp>

constexpr int row_size = 1080;
constexpr int col_size = 1920;
constexpr int max_iterations = 100;

int repetitions;

typedef struct {
  float real;
  float imag;
} ComplexF;

struct MandelParameters {
  int row_count_;
  int col_count_;
  int max_iterations_;

  KOKKOS_INLINE_FUNCTION
  MandelParameters(int row_count, int col_count, int max_iters)
      : row_count_(row_count), col_count_(col_count), max_iterations_(max_iters) {}

  KOKKOS_INLINE_FUNCTION int row_count() const { return row_count_; }
  KOKKOS_INLINE_FUNCTION int col_count() const { return col_count_; }
  KOKKOS_INLINE_FUNCTION int max_iters()  const { return max_iterations_; }

  // Scale from [0, row_count) to [-1.5, 0.5)
  KOKKOS_INLINE_FUNCTION
  float ScaleRow(int i) const { return -1.5f + (i * (2.0f / row_count_)); }

  // Scale from [0, col_count) to [-1.0, 1.0)
  KOKKOS_INLINE_FUNCTION
  float ScaleCol(int j) const { return -1.0f + (j * (2.0f / col_count_)); }

  // Return iteration count before divergence (or max_iterations if bounded)
  KOKKOS_INLINE_FUNCTION
  int Point(const ComplexF& c) const {
    int count = 0;
    ComplexF z = {0.0f, 0.0f};
    for (int i = 0; i < max_iterations_; ++i) {
      float r  = z.real;
      float im = z.imag;
      if ((r * r + im * im) >= 4.0f) break;
      z.real = r * r - im * im + c.real;
      z.imag = 2.0f * r * im  + c.imag;
      ++count;
    }
    return count;
  }
};

void Execute() {
  MandelParameters p(row_size, col_size, max_iterations);
  const int total = row_size * col_size;

  Kokkos::View<int*> d_data("mandel_data", total);

  // Warm-up run (triggers any JIT / first-touch overhead)
  Kokkos::parallel_for("mandel_warmup",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {row_size, col_size}),
    KOKKOS_LAMBDA(int i, int j) {
      ComplexF c = {p.ScaleRow(i), p.ScaleCol(j)};
      d_data[i * col_size + j] = p.Point(c);
    });
  Kokkos::fence();

  double kernel_time = 0.0;
  common::MyTimer t_par;

  for (int rep = 0; rep < repetitions; ++rep) {
    common::MyTimer t_ker;
    Kokkos::parallel_for("mandel",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {row_size, col_size}),
      KOKKOS_LAMBDA(int i, int j) {
        ComplexF c = {p.ScaleRow(i), p.ScaleCol(j)};
        d_data[i * col_size + j] = p.Point(c);
      });
    Kokkos::fence();
    kernel_time += t_ker.elapsed().count();
  }

  common::Duration parallel_time = t_par.elapsed();

  // Copy result to host
  auto h_data = Kokkos::create_mirror_view(d_data);
  Kokkos::deep_copy(h_data, d_data);

  // Serial reference
  std::vector<int> h_serial(total);
  common::MyTimer t_ser;
  for (int i = 0; i < row_size; ++i) {
    for (int j = 0; j < col_size; ++j) {
      ComplexF c = {p.ScaleRow(i), p.ScaleCol(j)};
      h_serial[i * col_size + j] = p.Point(c);
    }
  }
  common::Duration serial_time = t_ser.elapsed();

  std::cout << std::setw(20) << "Serial time: "
            << serial_time.count() << " s\n";
  std::cout << std::setw(20) << "Average parallel time: "
            << (parallel_time / repetitions).count() * 1e3 << " ms\n";
  std::cout << std::setw(20) << "Average kernel execution time: "
            << kernel_time / repetitions * 1e3 << " ms\n";

  // Verify: allow up to 5% mismatch
  int diff = 0;
  for (int k = 0; k < total; ++k)
    if (h_data[k] != h_serial[k]) ++diff;

  double ratio = static_cast<double>(diff) / static_cast<double>(total);
  if (ratio > 0.05) {
    std::cout << "Fail verification - diff larger than tolerance" << std::endl;
    throw std::runtime_error("Verification failure");
  }
}

void Usage(const std::string& program_name) {
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << program_name << " <repeat>\n\n";
  exit(-1);
}

int main(int argc, char* argv[]) {
  if (argc != 2) Usage(argv[0]);

  Kokkos::initialize(argc, argv);
  {
    try {
      repetitions = atoi(argv[1]);
      Execute();
    } catch (...) {
      std::cout << "Failure\n";
      Kokkos::finalize();
      return 1;
    }
    std::cout << "Success\n";
  }
  Kokkos::finalize();
  return 0;
}
