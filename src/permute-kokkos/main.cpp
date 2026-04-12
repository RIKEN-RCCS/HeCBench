/*
 * Permute QKV tensors for attention mechanism.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

// CPU reference: permute QKV from (B, T, 3C) to 3x(B, NH, T, HS)
void permute_cpu(const float *inp, float *q, float *k, float *v,
                 int B, int T, int C, int NH)
{
  int i = 0;
  int d = C / NH;
  for (int b = 0; b < B; b++) {
    for (int n = 0; n < NH; n++) {
      for (int t = 0; t < T; t++) {
        for (int c = n * d; c < (n + 1) * d; c++) {
          q[i] = inp[b * T * 3 * C + t * 3 * C + c];
          k[i] = inp[b * T * 3 * C + t * 3 * C + C + c];
          v[i] = inp[b * T * 3 * C + t * 3 * C + 2 * C + c];
          i++;
        }
      }
    }
  }
}

// Kokkos permute kernel: inp (B, T, 3C) -> q,k,v (B, NH, T, HS)
void permute(Kokkos::View<float*> out, Kokkos::View<const float*> inp,
             int B, int T, int C, int NH)
{
  int d = C / NH;  // head size

  float *q = out.data() + 0 * B * T * C;
  float *k = out.data() + 1 * B * T * C;
  float *v = out.data() + 2 * B * T * C;

  Kokkos::parallel_for("permute", B * T * C, KOKKOS_LAMBDA(int idx) {
    int b    = idx / (C * T);
    int rest = idx % (C * T);
    int nh_  = rest / (T * d);
    rest     = rest % (T * d);
    int n    = rest / d;
    int d_   = rest % d;

    int inp_idx = b * T * 3 * C
                + n * 3 * C
                + nh_ * d
                + d_;

    q[idx] = inp(inp_idx);
    k[idx] = inp(inp_idx + C);
    v[idx] = inp(inp_idx + 2 * C);
  });
  Kokkos::fence();
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int B            = atoi(argv[1]);
  const int repeat_times = atoi(argv[2]);

  const int T  = 1024;
  const int C  = 768;
  const int NH = 12;

  size_t S = (size_t)B * T * C;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  float *inp = (float*) malloc(S * 3 * sizeof(float));
  float *q   = (float*) malloc(S * sizeof(float));
  float *k   = (float*) malloc(S * sizeof(float));
  float *v   = (float*) malloc(S * sizeof(float));

  for (size_t i = 0; i < S * 3; i++) inp[i] = dist(rng);

  permute_cpu(inp, q, k, v, B, T, C, NH);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_inp("d_inp", S * 3);
    Kokkos::View<float*> d_out("d_out", S * 3);

    auto h_inp = Kokkos::create_mirror_view(d_inp);
    for (size_t i = 0; i < S * 3; i++) h_inp(i) = inp[i];
    Kokkos::deep_copy(d_inp, h_inp);

    printf("Checking block size 256.\n");
    permute(d_out, d_inp, B, T, C, NH);

    auto h_out = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out, d_out);

    // Validate q, k, v
    bool ok = true;
    for (size_t i = 0; i < S && ok; i++) {
      if (fabsf(h_out(i) - q[i]) > 1e-5f) {
        printf("q mismatch at %zu: %f vs %f\n", i, (float)h_out(i), q[i]);
        ok = false;
      }
    }
    for (size_t i = 0; i < S && ok; i++) {
      if (fabsf(h_out(S + i) - k[i]) > 1e-5f) {
        printf("k mismatch at %zu: %f vs %f\n", i, (float)h_out(S + i), k[i]);
        ok = false;
      }
    }
    for (size_t i = 0; i < S && ok; i++) {
      if (fabsf(h_out(2 * S + i) - v[i]) > 1e-5f) {
        printf("v mismatch at %zu: %f vs %f\n", i, (float)h_out(2 * S + i), v[i]);
        ok = false;
      }
    }
    if (ok) printf("All results match. Starting benchmarks.\n\n");

    auto start = std::chrono::steady_clock::now();
    for (int j = 0; j < repeat_times; j++) {
      permute(d_out, d_inp, B, T, C, NH);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("block_size %4d | time %f ms\n", 256,
           (time * 1e-6f) / repeat_times);
  }
  Kokkos::finalize();

  free(inp);
  free(q);
  free(k);
  free(v);
  return 0;
}
