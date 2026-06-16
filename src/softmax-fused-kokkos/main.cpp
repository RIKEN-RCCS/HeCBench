// Kokkos port of softmax-fused-cuda
// Scaled masked softmax kernel using float (replaces half/bfloat16)

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <limits>

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value)
    ++log2_value;
  return log2_value;
}

// Scaled masked softmax: for each batch row, apply mask and compute softmax
// We implement this with a Kokkos TeamPolicy kernel
void scaled_masked_softmax_forward(
    Kokkos::View<float*> dst,
    Kokkos::View<const float*> src,
    Kokkos::View<const uint8_t*> mask,
    float scale,
    int batch_count,   // total rows = batches*attn_heads*query_seq_len
    int element_count, // key_seq_len
    int pad_batches)
{
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;

  Kokkos::parallel_for("scaled_masked_softmax",
    team_policy(batch_count, Kokkos::AUTO),
    KOKKOS_LAMBDA(const member_type& team) {
      const int row = team.league_rank();
      const float neg_inf = -std::numeric_limits<float>::infinity();

      // Compute max
      float max_val = neg_inf;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, element_count),
        [&](const int j, float& lmax) {
          float v;
          uint8_t m = mask[row * element_count + j];
          if (m != 1)
            v = src[row * element_count + j] * scale;
          else
            v = -10000.0f;
          if (v > lmax) lmax = v;
        }, Kokkos::Max<float>(max_val));
      team.team_barrier();

      // Mask scale: if all masked (-10000), output 0
      float mask_scale = (max_val == -10000.0f) ? 0.0f : 1.0f;

      // Compute sum of exp
      float sum_val = 0.0f;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, element_count),
        [&](const int j, float& lsum) {
          float v;
          uint8_t m = mask[row * element_count + j];
          if (m != 1)
            v = src[row * element_count + j] * scale;
          else
            v = -10000.0f;
          lsum += Kokkos::exp(v - max_val);
        }, sum_val);
      team.team_barrier();

      // Write output
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, element_count),
        [&](const int j) {
          float v;
          uint8_t m = mask[row * element_count + j];
          if (m != 1)
            v = src[row * element_count + j] * scale;
          else
            v = -10000.0f;
          if (mask_scale != 0.0f)
            dst[row * element_count + j] = Kokkos::exp(v - max_val) / sum_val;
          else
            dst[row * element_count + j] = 0.0f;
        });
    });
}

void fused_softmax(int batches, int attn_heads, int query_seq_len,
                   int key_seq_len, int repeat) {
  uint64_t num_data_elems =
      (uint64_t)batches * attn_heads * query_seq_len * key_seq_len;
  float scale_factor = 1.0f / std::sqrt((float)key_seq_len);
  int pad_batches = 0;
  int batch_count = batches * attn_heads * query_seq_len;

  Kokkos::View<float*> d_input("d_input", num_data_elems);
  Kokkos::View<float*> d_output("d_output", num_data_elems);
  Kokkos::View<uint8_t*> d_mask("d_mask", num_data_elems);

  // Initialize: fill input with a constant, mask with 0
  Kokkos::deep_copy(d_mask, (uint8_t)0);
  Kokkos::deep_copy(d_input, 0.5f);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    scaled_masked_softmax_forward(d_output, d_input, d_mask,
                                  scale_factor, batch_count, key_seq_len, pad_batches);
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Usage: %s <batch> <head> <query length> <key length> <repeat>\n", argv[0]);
    return 1;
  }

  int batches = atoi(argv[1]);
  int attn_heads = atoi(argv[2]);
  int query_seq_len = atoi(argv[3]);
  int key_seq_len = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  Kokkos::initialize(argc, argv);
  {
    fused_softmax(batches, attn_heads, query_seq_len, key_seq_len, repeat);
  }
  Kokkos::finalize();
  return 0;
}
