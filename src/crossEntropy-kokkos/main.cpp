/*  Copyright (c) 2021-2022 Intel Corporation

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <Kokkos_Core.hpp>

// generate rand int64_t [a, b]
#define random_int(a, b) ( rand() % ((b) - (a)) + (a) )

// generate rand float [0, 1]
#define random_float() (rand() / static_cast<double>(RAND_MAX))

#define tolerance 4e-3f

constexpr int bs = 128;
constexpr int W  = 81;
constexpr int H  = 8732;
constexpr int PredictShape = bs * W * H;
constexpr int TargetShape  = bs * H;
constexpr int OutputShape  = bs * H;

// ---------------------------------------------------------------------------
// CPU reference (float/float only)
// ---------------------------------------------------------------------------
void loss_bwd_cpu(
    const float*   log_softmax,
    const int64_t* target,
    const float*   weight,
    const int64_t* mask,
    const float*   grad_output,
    const float*   grad_output_neg,
          float*   grad_predict)
{
  // Pass 1: set grad_predict for the "idx == k" slot, zero elsewhere
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset2d      = static_cast<int64_t>(i) * H + j;
        int64_t idx           = target[offset2d];
        int64_t predict_offset = static_cast<int64_t>(i) * W * H + static_cast<int64_t>(k) * H + j;

        if (idx == static_cast<int64_t>(k)) {
          grad_predict[predict_offset] =
            (-grad_output[offset2d] * weight[offset2d]) +
            (mask[offset2d] ? -grad_output_neg[offset2d] * weight[offset2d] : 0.0f);
        } else {
          grad_predict[predict_offset] = 0.0f;
        }
      }
    }
  }

  // Pass 2: compute per-(bs, H) sum_value = sum_k( grad_predict[b,k,h] * log_softmax[b,k,h] )
  std::vector<float> sum_value(static_cast<size_t>(bs) * H, 0.0f);
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = static_cast<int64_t>(i) * W * H + static_cast<int64_t>(k) * H + j;
        sum_value[static_cast<size_t>(i) * H + j] +=
          grad_predict[offset] * log_softmax[offset];
      }
    }
  }

  // Pass 3: grad_predict[b,k,h] -= exp(log_softmax[b,k,h]) * sum_value[b,h]
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = static_cast<int64_t>(i) * W * H + static_cast<int64_t>(k) * H + j;
        grad_predict[offset] -= std::exp(log_softmax[offset]) * sum_value[static_cast<size_t>(i) * H + j];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Kokkos kernel (float/float)
// ---------------------------------------------------------------------------
// The CUDA kernel fuses all three passes into one kernel per (bs, H) thread:
//   tmp_grad = -(grad_output + (mask ? grad_output_neg : 0)) * weight
//   sum_value = tmp_grad * log_softmax[b, idx, h]
//   for i in [0,W):
//     tmp_sfm = exp(log_softmax[b,i,h]) * sum_value
//     grad_predict[b,i,h] = (i==idx ? tmp_grad : 0) - tmp_sfm
//
// We replicate the same fused logic in Kokkos.
void loss_bwd_kokkos(
    Kokkos::View<const float*,   Kokkos::HostSpace> log_softmax_h,
    Kokkos::View<const float*,   Kokkos::HostSpace> grad_output_h,
    Kokkos::View<const float*,   Kokkos::HostSpace> grad_output_neg_h,
    Kokkos::View<const int64_t*, Kokkos::HostSpace> target_h,
    Kokkos::View<const float*,   Kokkos::HostSpace> weight_h,
    Kokkos::View<const int64_t*, Kokkos::HostSpace> mask_h,
    Kokkos::View<float*,         Kokkos::HostSpace> grad_predict_h,
    int iterations,
    double& kernel_time_ms)
{
  using DevSpace   = Kokkos::DefaultExecutionSpace;
  using DevView1f  = Kokkos::View<float*,   DevSpace>;
  using DevView1i  = Kokkos::View<int64_t*, DevSpace>;

  DevView1f d_log_softmax    ("log_softmax",    PredictShape);
  DevView1f d_grad_output    ("grad_output",    OutputShape);
  DevView1f d_grad_output_neg("grad_output_neg",OutputShape);
  DevView1i d_target         ("target",         TargetShape);
  DevView1f d_weight         ("weight",         OutputShape);
  DevView1i d_mask           ("mask",           TargetShape);
  DevView1f d_grad_predict   ("grad_predict",   PredictShape);

  Kokkos::deep_copy(d_log_softmax,     log_softmax_h);
  Kokkos::deep_copy(d_grad_output,     grad_output_h);
  Kokkos::deep_copy(d_grad_output_neg, grad_output_neg_h);
  Kokkos::deep_copy(d_target,          target_h);
  Kokkos::deep_copy(d_weight,          weight_h);
  Kokkos::deep_copy(d_mask,            mask_h);

  // warm-up
  for (int k = 0; k < 10; ++k) {
    Kokkos::parallel_for(
      "loss_bwd_warmup",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {bs, H}),
      KOKKOS_LAMBDA(int b, int h) {
        int64_t offset2d = static_cast<int64_t>(b) * H + h;
        int64_t idx      = d_target(offset2d);

        float tmp_grad = -(d_grad_output(offset2d) +
                          (d_mask(offset2d) ? d_grad_output_neg(offset2d) : 0.0f)) *
                         d_weight(offset2d);

        float sum_value = tmp_grad *
          d_log_softmax(static_cast<int64_t>(b) * W * H + idx * H + h);

        for (int i = 0; i < W; ++i) {
          int64_t in_offset = static_cast<int64_t>(b) * W * H + static_cast<int64_t>(i) * H + h;
          float tmp_sfm = Kokkos::exp(d_log_softmax(in_offset)) * sum_value;
          d_grad_predict(in_offset) = (i == idx ? tmp_grad : 0.0f) - tmp_sfm;
        }
      });
    Kokkos::fence();
  }

  // timed iterations
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < iterations; ++k) {
    Kokkos::parallel_for(
      "loss_bwd",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {bs, H}),
      KOKKOS_LAMBDA(int b, int h) {
        int64_t offset2d = static_cast<int64_t>(b) * H + h;
        int64_t idx      = d_target(offset2d);

        float tmp_grad = -(d_grad_output(offset2d) +
                          (d_mask(offset2d) ? d_grad_output_neg(offset2d) : 0.0f)) *
                         d_weight(offset2d);

        float sum_value = tmp_grad *
          d_log_softmax(static_cast<int64_t>(b) * W * H + idx * H + h);

        for (int i = 0; i < W; ++i) {
          int64_t in_offset = static_cast<int64_t>(b) * W * H + static_cast<int64_t>(i) * H + h;
          float tmp_sfm = Kokkos::exp(d_log_softmax(in_offset)) * sum_value;
          d_grad_predict(in_offset) = (i == idx ? tmp_grad : 0.0f) - tmp_sfm;
        }
      });
    Kokkos::fence();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  kernel_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  Kokkos::deep_copy(grad_predict_h, d_grad_predict);
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------
static int64_t errors = 0;

void verify(const float* cpu, const float* dev, size_t sz) {
  int count = 0;
  for (size_t i = 0; i < sz; ++i) {
    if (std::abs(cpu[i] - dev[i]) > tolerance) {
      ++count;
      if (count < 10)
        std::cout << "Error at i=" << i
                  << " cpu=" << cpu[i]
                  << " dev=" << dev[i]
                  << " gap=" << (cpu[i] - dev[i]) << "\n";
    }
  }
  errors += count;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    printf("Tensor size (BatchSize * Width * Height) = %d * %d * %d\n", bs, W, H);
    printf("=========== Data type is FP32 ===========\n");

    srand(42);

    // Host arrays (plain vectors)
    std::vector<float>   log_softmax(PredictShape);
    std::vector<float>   grad_output(OutputShape);
    std::vector<float>   grad_output_neg(OutputShape);
    std::vector<float>   weight(OutputShape);
    std::vector<int64_t> target(TargetShape);
    std::vector<int64_t> mask(TargetShape);
    std::vector<float>   grad_predict_cpu(PredictShape);
    std::vector<float>   grad_predict_dev(PredictShape, 0.0f);

    for (int i = 0; i < PredictShape; ++i)
      log_softmax[i] = static_cast<float>(random_float());
    for (int i = 0; i < OutputShape; ++i)
      grad_output[i] = static_cast<float>(random_float());
    for (int i = 0; i < OutputShape; ++i)
      grad_output_neg[i] = static_cast<float>(random_float());
    for (int i = 0; i < OutputShape; ++i)
      weight[i] = static_cast<float>(random_float());
    for (int i = 0; i < TargetShape; ++i)
      target[i] = static_cast<int64_t>(random_int(0, W - 1));
    for (int i = 0; i < TargetShape; ++i)
      mask[i] = static_cast<int64_t>(random_int(0, 1));

    // Wrap in HostSpace Views (unmanaged)
    using HView1f = Kokkos::View<const float*,   Kokkos::HostSpace>;
    using HView1i = Kokkos::View<const int64_t*, Kokkos::HostSpace>;
    using HView1fw= Kokkos::View<float*,         Kokkos::HostSpace>;

    HView1f  h_log_softmax    (log_softmax.data(),     PredictShape);
    HView1f  h_grad_output    (grad_output.data(),     OutputShape);
    HView1f  h_grad_output_neg(grad_output_neg.data(), OutputShape);
    HView1i  h_target         (target.data(),          TargetShape);
    HView1f  h_weight         (weight.data(),          OutputShape);
    HView1i  h_mask           (mask.data(),            TargetShape);
    HView1fw h_grad_predict   (grad_predict_dev.data(),PredictShape);

    double kernel_time_ms = 0.0;
    loss_bwd_kokkos(h_log_softmax, h_grad_output, h_grad_output_neg,
                    h_target, h_weight, h_mask, h_grad_predict,
                    repeat, kernel_time_ms);

    // CPU reference
    auto t_cpu0 = std::chrono::high_resolution_clock::now();
    loss_bwd_cpu(log_softmax.data(), target.data(), weight.data(),
                 mask.data(), grad_output.data(), grad_output_neg.data(),
                 grad_predict_cpu.data());
    auto t_cpu1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();

    verify(grad_predict_cpu.data(), grad_predict_dev.data(),
           static_cast<size_t>(PredictShape));

    double avg_kernel_ms = kernel_time_ms / repeat;
    double allBytes =
      static_cast<double>(sizeof(float))   * static_cast<double>(PredictShape * 2.0 + OutputShape * 3.0)
    + static_cast<double>(sizeof(int64_t)) * static_cast<double>(TargetShape  * 2.0);

    printf("Average kernel time (ms)   : %f\n", avg_kernel_ms);
    printf("CPU serial time (ms)       : %f\n", cpu_time);
    printf("BandWidth = %lf (GB/s)\n",
           allBytes / (avg_kernel_ms / 1000.0) / 1e9);
    printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
