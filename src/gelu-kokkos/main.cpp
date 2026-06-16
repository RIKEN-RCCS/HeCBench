// Kokkos port of gelu-cuda (GELU + bias addition)
// fp16 (__half2) replaced with float throughout.
// Original kernel processes two elements per thread via half2;
// here each work-item processes one element.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
KOKKOS_INLINE_FUNCTION float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// CPU reference
static void gelu_bias_cpu(float* src, const float* bias,
                           int batch_size, int width, int height)
{
  for (int b = 0; b < batch_size; b++)
    for (int x = 0; x < height; x++)
      for (int y = 0; y < width; y++) {
        long long idx = (long long)b * width * height + (long long)x * width + y;
        float v = src[idx] + bias[y];
        src[idx] = gelu(v);
      }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch> <sequence length> <hidden dimension> <repeat>\n", argv[0]);
    printf("The hidden dimension must be even\n");
    return 1;
  }

  const int batch_size  = atoi(argv[1]);
  const int seq_len     = atoi(argv[2]);
  const int hidden_dim  = atoi(argv[3]);
  const int repeat      = atoi(argv[4]);

  if (hidden_dim % 2 != 0) {
    printf("Error: hidden dimension must be even\n"); return 1;
  }

  const long long src_size = (long long)batch_size * seq_len * hidden_dim;

  float* output     = (float*)malloc(src_size * sizeof(float));
  float* output_ref = (float*)malloc(src_size * sizeof(float));
  float* bias       = (float*)malloc(hidden_dim * sizeof(float));

  srand(42);
  for (long long i = 0; i < src_size; i++)
    output[i] = output_ref[i] = (float)(rand() % 24 - 12);
  for (int i = 0; i < hidden_dim; i++)
    bias[i] = (float)(rand() % 12 - 6);

  // CPU reference
  gelu_bias_cpu(output_ref, bias, batch_size, hidden_dim, seq_len);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_src  ("src",  src_size);
    Kokkos::View<float*> d_bias ("bias", hidden_dim);

    {
      auto hb = Kokkos::create_mirror_view(d_bias);
      for (int i = 0; i < hidden_dim; i++) hb(i) = bias[i];
      Kokkos::deep_copy(d_bias, hb);
    }

    // Warmup + verify
    {
      auto hs = Kokkos::create_mirror_view(d_src);
      for (long long i = 0; i < src_size; i++) hs(i) = output[i];
      Kokkos::deep_copy(d_src, hs);

      Kokkos::parallel_for(src_size, KOKKOS_LAMBDA(long long idx) {
        int y = (int)(idx % hidden_dim);
        float v = d_src(idx) + d_bias(y);
        d_src(idx) = gelu(v);
      });
      Kokkos::fence();

      Kokkos::deep_copy(hs, d_src);
      bool ok = true;
      for (long long i = 0; i < src_size && ok; i++)
        if (fabsf(hs(i) - output_ref[i]) > 1e-3f) { ok = false; }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    // Timed runs
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(src_size, KOKKOS_LAMBDA(long long idx) {
        int y = (int)(idx % hidden_dim);
        float v = d_src(idx) + d_bias(y);
        d_src(idx) = gelu(v);
      });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", ns * 1e-6 / repeat);
  }
  Kokkos::finalize();

  free(output);
  free(output_ref);
  free(bias);
  return 0;
}
