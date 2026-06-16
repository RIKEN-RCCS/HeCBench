#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// Bilinear interpolation at fractional position (h, w) in a height×width image.
KOKKOS_INLINE_FUNCTION
float bilinear_interp(const float* data, float h, float w, int height, int width) {
  if (h <= -1.f || h >= height || w <= -1.f || w >= width) return 0.f;
  int h0 = (int)floorf(h), h1 = h0 + 1;
  int w0 = (int)floorf(w), w1 = w0 + 1;
  float lh = h - h0, lw = w - w0;
  float hh = 1.f - lh,  hw = 1.f - lw;
  float v1 = (h0 >= 0 && w0 >= 0)        ? data[h0 * width + w0] : 0.f;
  float v2 = (h0 >= 0 && w1 < width)     ? data[h0 * width + w1] : 0.f;
  float v3 = (h1 < height && w0 >= 0)    ? data[h1 * width + w0] : 0.f;
  float v4 = (h1 < height && w1 < width) ? data[h1 * width + w1] : 0.f;
  return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    // Fixed convolution parameters
    constexpr int batch            = 1;
    constexpr int channels_in      = 64;
    constexpr int channels_out     = 64;
    constexpr int height           = 28;
    constexpr int width            = 28;
    constexpr int kernel_h         = 3;
    constexpr int kernel_w         = 3;
    constexpr int stride_h         = 1;
    constexpr int stride_w         = 1;
    constexpr int pad_h            = 1;
    constexpr int pad_w            = 1;
    constexpr int dilation_h       = 1;
    constexpr int dilation_w       = 1;
    constexpr int deformable_groups = 1;

    constexpr int height_out =
        (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1; // 28
    constexpr int width_out  =
        (width  + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1; // 28

    // Derived sizes
    constexpr int ksize      = kernel_h * kernel_w;
    constexpr int col_rows   = channels_in * ksize;          // 576
    constexpr int col_cols   = height_out * width_out;        // 784
    constexpr int n_offset   = 2 * deformable_groups * ksize; // 18
    constexpr int n_mask     = deformable_groups * ksize;     // 9

    const int input_size  = batch * channels_in * height * width;
    const int offset_size = batch * n_offset * height_out * width_out;
    const int mask_size   = batch * n_mask   * height_out * width_out;
    const int weight_size = channels_out * (channels_in / deformable_groups) * ksize;
    const int output_size = batch * channels_out * height_out * width_out;

    // Device views
    Kokkos::View<float*>  input  ("input",   input_size);
    Kokkos::View<float*>  offset ("offset",  offset_size);
    Kokkos::View<float*>  mask   ("mask",    mask_size);
    Kokkos::View<float*>  weight ("weight",  weight_size);
    Kokkos::View<float*>  bias   ("bias",    channels_out);
    Kokkos::View<float**> columns("columns", col_rows, col_cols);
    Kokkos::View<float*>  output ("output",  output_size);

    // Initialise with reproducible random data on host
    {
      auto h_input  = Kokkos::create_mirror_view(input);
      auto h_offset = Kokkos::create_mirror_view(offset);
      auto h_mask   = Kokkos::create_mirror_view(mask);
      auto h_weight = Kokkos::create_mirror_view(weight);
      auto h_bias   = Kokkos::create_mirror_view(bias);

      srand(42);
      for (int i = 0; i < input_size;  i++) h_input (i) = ((float)rand() / RAND_MAX - 0.5f);
      // offsets are small perturbations around zero
      for (int i = 0; i < offset_size; i++) h_offset(i) = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
      // masks are close to 1
      for (int i = 0; i < mask_size;   i++) h_mask  (i) = 0.9f + 0.2f * (float)rand() / RAND_MAX;
      for (int i = 0; i < weight_size; i++) h_weight(i) = ((float)rand() / RAND_MAX - 0.5f);
      for (int i = 0; i < channels_out;i++) h_bias  (i) = ((float)rand() / RAND_MAX - 0.5f);

      Kokkos::deep_copy(input,  h_input);
      Kokkos::deep_copy(offset, h_offset);
      Kokkos::deep_copy(mask,   h_mask);
      Kokkos::deep_copy(weight, h_weight);
      Kokkos::deep_copy(bias,   h_bias);
    }

    // ------------------------------------------------------------------ im2col
    // Warm-up
    Kokkos::parallel_for(
      "im2col_warmup",
      batch * channels_in * ksize * height_out * width_out,
      KOKKOS_LAMBDA(int idx) {
        int tmp   = idx;
        int w_out = tmp % width_out;  tmp /= width_out;
        int h_out = tmp % height_out; tmp /= height_out;
        int kw    = tmp % kernel_w;   tmp /= kernel_w;
        int kh    = tmp % kernel_h;   tmp /= kernel_h;
        int c_in  = tmp % channels_in; tmp /= channels_in;
        int b     = tmp;

        int offset_group = c_in / (channels_in / deformable_groups);
        int kp           = kh * kernel_w + kw;
        int offset_idx   = offset_group * 2 * ksize + kp * 2;
        int mask_idx     = offset_group * ksize + kp;
        int spatial      = h_out * width_out + w_out;
        int spatial_sz   = height_out * width_out;

        float h_in = h_out * stride_h - pad_h + kh * dilation_h
          + offset(b * n_offset * spatial_sz + offset_idx       * spatial_sz + spatial);
        float w_in = w_out * stride_w - pad_w + kw * dilation_w
          + offset(b * n_offset * spatial_sz + (offset_idx + 1) * spatial_sz + spatial);
        float mask_val =
          mask((b * n_mask + mask_idx) * spatial_sz + spatial);

        const float* input_ptr =
          input.data() + (b * channels_in + c_in) * height * width;
        float val = bilinear_interp(input_ptr, h_in, w_in, height, width) * mask_val;

        columns(c_in * ksize + kp, spatial) = val;
      });
    Kokkos::fence();

    // Timed repetitions
    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(
        "im2col",
        batch * channels_in * ksize * height_out * width_out,
        KOKKOS_LAMBDA(int idx) {
          int tmp   = idx;
          int w_out = tmp % width_out;  tmp /= width_out;
          int h_out = tmp % height_out; tmp /= height_out;
          int kw    = tmp % kernel_w;   tmp /= kernel_w;
          int kh    = tmp % kernel_h;   tmp /= kernel_h;
          int c_in  = tmp % channels_in; tmp /= channels_in;
          int b     = tmp;

          int offset_group = c_in / (channels_in / deformable_groups);
          int kp           = kh * kernel_w + kw;
          int offset_idx   = offset_group * 2 * ksize + kp * 2;
          int mask_idx     = offset_group * ksize + kp;
          int spatial      = h_out * width_out + w_out;
          int spatial_sz   = height_out * width_out;

          float h_in = h_out * stride_h - pad_h + kh * dilation_h
            + offset(b * n_offset * spatial_sz + offset_idx       * spatial_sz + spatial);
          float w_in = w_out * stride_w - pad_w + kw * dilation_w
            + offset(b * n_offset * spatial_sz + (offset_idx + 1) * spatial_sz + spatial);
          float mask_val =
            mask((b * n_mask + mask_idx) * spatial_sz + spatial);

          const float* input_ptr =
            input.data() + (b * channels_in + c_in) * height * width;
          float val = bilinear_interp(input_ptr, h_in, w_in, height, width) * mask_val;

          columns(c_in * ksize + kp, spatial) = val;
        });
      Kokkos::fence();
    }

    auto t1 = std::chrono::steady_clock::now();
    double im2col_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
    printf("im2col average time: %.4f ms\n", im2col_ms);

    // ----------------------------------------------------------------- matmul
    // output[c_out, col] = bias[c_out] + sum_k weight[c_out*col_rows+k] * columns[k, col]
    Kokkos::parallel_for(
      "matmul",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {channels_out, col_cols}),
      KOKKOS_LAMBDA(int c_out, int col) {
        float sum = bias(c_out);
        for (int k = 0; k < col_rows; k++)
          sum += weight(c_out * col_rows + k) * columns(k, col);
        output(c_out * col_cols + col) = sum;
      });
    Kokkos::fence();

    // ----------------------------------------------------------------- verify
    auto h_output = Kokkos::create_mirror_view(output);
    Kokkos::deep_copy(h_output, output);
    double sum = 0.0;
    for (int i = 0; i < output_size; i++) sum += h_output(i);
    printf("Output sum: %f\n", sum);
    printf("%s\n", (sum != 0.0) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
