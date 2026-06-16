/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Kokkos port: Replace FP8/bfloat16 ops with float equivalent.
 * Each element: out = (in * input_scale + bias) * output_scale
 */

#include <chrono>
#include <cstdio>
#include <Kokkos_Core.hpp>

// Kokkos port: replaces FP8TrtAddQKVBiasKernel
// Layout: [valid_word_num, 3, head_num, size_per_head]
// Transpose output to: [valid_word_num, head_num, 3, size_per_head]
void addBiasQKV(int batch_size, int seq_len, int hidden_units, int head_num, int repeat) {
  int m = batch_size * seq_len;          // valid_word_num
  int size_per_head = hidden_units / head_num;
  int total_size = m * 3 * hidden_units;

  Kokkos::View<float*> qkv_src("qkv_src", total_size);
  Kokkos::View<float*> qkv_tgt("qkv_tgt", total_size);
  Kokkos::View<float*> bias("bias", 3 * hidden_units);
  Kokkos::View<float[1]> input_scale("input_scale");
  Kokkos::View<float[1]> output_scale("output_scale");

  // Initialize
  Kokkos::parallel_for("init", total_size, KOKKOS_LAMBDA(int i) {
    qkv_src(i) = 0.5f;
  });
  Kokkos::parallel_for("init_bias", 3*hidden_units, KOKKOS_LAMBDA(int i) {
    bias(i) = 0.1f;
  });
  Kokkos::parallel_for("init_scale", 1, KOKKOS_LAMBDA(int) {
    input_scale(0) = 1.0f;
    output_scale(0) = 1.0f;
  });

  // Kernel: for each (word, qkv_idx, head, feature)
  // src index:  word * 3*hidden + qkv_idx * hidden + head * size_per_head + feat
  // tgt index:  word * 3*hidden + head * 3*size_per_head + qkv_idx * size_per_head + feat

  // Warmup
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("addBiasQKV",
      Kokkos::RangePolicy<>(0, m * 3 * head_num * size_per_head),
      KOKKOS_LAMBDA(int idx) {
        int feat     = idx % size_per_head;
        int tmp      = idx / size_per_head;
        int head_id  = tmp % head_num;
        int tmp2     = tmp / head_num;
        int qkv_idx  = tmp2 % 3;
        int word_id  = tmp2 / 3;

        int src_id = word_id * 3 * hidden_units + qkv_idx * hidden_units
                   + head_id * size_per_head + feat;
        int bias_id = qkv_idx * hidden_units + head_id * size_per_head + feat;
        int tgt_id  = word_id * 3 * hidden_units + head_id * 3 * size_per_head
                   + qkv_idx * size_per_head + feat;

        float val = (qkv_src(src_id) * input_scale(0) + bias(bias_id)) * output_scale(0);
        qkv_tgt(tgt_id) = val;
      });
  }
  Kokkos::fence();

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("addBiasQKV",
      Kokkos::RangePolicy<>(0, m * 3 * head_num * size_per_head),
      KOKKOS_LAMBDA(int idx) {
        int feat     = idx % size_per_head;
        int tmp      = idx / size_per_head;
        int head_id  = tmp % head_num;
        int tmp2     = tmp / head_num;
        int qkv_idx  = tmp2 % 3;
        int word_id  = tmp2 / 3;

        int src_id = word_id * 3 * hidden_units + qkv_idx * hidden_units
                   + head_id * size_per_head + feat;
        int bias_id = qkv_idx * hidden_units + head_id * size_per_head + feat;
        int tgt_id  = word_id * 3 * hidden_units + head_id * 3 * size_per_head
                   + qkv_idx * size_per_head + feat;

        float val = (qkv_src(src_id) * input_scale(0) + bias(bias_id)) * output_scale(0);
        qkv_tgt(tgt_id) = val;
      });
  }
  Kokkos::fence();
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    int batch_size = 8;
    int seq_len = 1024;
    int hidden_units = 768;
    int head_num = 12;
    int repeat = 1000;
    addBiasQKV(batch_size, seq_len, hidden_units, head_num, repeat);
  }
  Kokkos::finalize();
  return 0;
}
