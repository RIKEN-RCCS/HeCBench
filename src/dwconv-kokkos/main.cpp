// Kokkos port of dwconv-cuda (2-D depthwise convolution forward)
// tensorAccessor.h removed; flat 1-D Views used with manual index arithmetic.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

template <typename scalar_t>
void dwconv2d_forward(
    int m, int n, int input_channels,
    int H, int W, int kH, int kW, int repeat,
    int padH = 1, int padW = 1,
    int strideH = 1, int strideW = 1,
    int dilateH = 1, int dilateW = 1)
{
  const int output_channels = input_channels * m;

  // Sizes
  int weight_size  = output_channels * kH * kW;
  int input_size   = n * input_channels * H * W;
  int bias_size    = output_channels;

  // Output dimensions
  int outputH = (H + 2 * padH - dilateH * (kH - 1) - 1) / strideH + 1;
  int outputW = (W + 2 * padW - dilateW * (kW - 1) - 1) / strideW + 1;
  int output_size = n * output_channels * outputH * outputW;

  // Host arrays
  std::vector<scalar_t> h_input(input_size), h_weight(weight_size),
                        h_bias(bias_size), h_output(output_size);

  srand(123);
  for (auto& v : h_input)  v = (scalar_t)(rand() / (float)RAND_MAX);
  for (auto& v : h_weight) v = (scalar_t)(rand() / (float)RAND_MAX);
  for (auto& v : h_bias)   v = (scalar_t)(rand() / (float)RAND_MAX);

  Kokkos::View<scalar_t*> d_input  ("input",  input_size);
  Kokkos::View<scalar_t*> d_weight ("weight", weight_size);
  Kokkos::View<scalar_t*> d_bias   ("bias",   bias_size);
  Kokkos::View<scalar_t*> d_output ("output", output_size);

  {
    auto hv = Kokkos::create_mirror_view(d_input);
    for (int i = 0; i < input_size; i++) hv(i) = h_input[i];
    Kokkos::deep_copy(d_input, hv);
  }
  {
    auto hv = Kokkos::create_mirror_view(d_weight);
    for (int i = 0; i < weight_size; i++) hv(i) = h_weight[i];
    Kokkos::deep_copy(d_weight, hv);
  }
  {
    auto hv = Kokkos::create_mirror_view(d_bias);
    for (int i = 0; i < bias_size; i++) hv(i) = h_bias[i];
    Kokkos::deep_copy(d_bias, hv);
  }

  int depthwiseMul = output_channels / input_channels;

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int rep = 0; rep < repeat; rep++) {
    Kokkos::parallel_for(output_size, KOKKOS_LAMBDA(int linearIndex) {
      int tmp1 = linearIndex / outputW;
      const int w_ = linearIndex - tmp1 * outputW;
      int tmp2 = tmp1 / outputH;
      const int h_ = tmp1 - tmp2 * outputH;
      tmp1 = tmp2;
      tmp2 = tmp1 / output_channels;
      const int c = tmp1 - tmp2 * output_channels;
      const int bn = tmp2;

      int inputChannel = c / depthwiseMul;

      scalar_t value = d_bias(c);
      const int offset0 = (bn * input_channels + inputChannel) * H * W;
      int weightOffset = c * kH * kW;

      for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
          int h_in = -padH + h_ * strideH + kh * dilateH;
          int w_in = -padW + w_ * strideW + kw * dilateW;
          if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            value += d_weight(weightOffset) * d_input(offset0 + h_in * W + w_in);
          }
          ++weightOffset;
        }
      }
      d_output(linearIndex) = value;
    });
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of dwconv2d_forward kernel: %f (ms)\n",
         time * 1e-6f / repeat);

  auto hv = Kokkos::create_mirror_view(d_output);
  Kokkos::deep_copy(hv, d_output);

  scalar_t sum = 0;
  for (int i = 0; i < output_size; i++) sum += hv(i);
  printf("Checksum = %f\n", (float)(sum / output_size));
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <batch size> <num input channels> "
           "<input height> <input width> <repeat>\n", argv[0]);
    return 1;
  }

  const int n      = atoi(argv[1]);
  const int c      = atoi(argv[2]);
  const int h      = atoi(argv[3]);
  const int w      = atoi(argv[4]);
  const int repeat = atoi(argv[5]);

  Kokkos::initialize(argc, argv);
  {
    for (int m = 1; m <= 4; m++) {
      for (int k = 1; k <= 5; k += 2) {
        printf("batch=%d, input channel=%d, height=%d, width=%d, "
               "kernel size=%d, output channel=%d\n", n, c, h, w, k, m * c);
        dwconv2d_forward<float>(m, n, c, h, w, k, k, repeat);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
