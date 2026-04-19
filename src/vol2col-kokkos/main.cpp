#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

// Kernel for fast unfold+copy on volumes
template <typename T>
void vol2col_kernel(
    Kokkos::View<const T*> data_vol,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    Kokkos::View<T*> data_col)
{
  const int64_t total = (int64_t)channels * depth_col * height_col * width_col;
  Kokkos::parallel_for("vol2col", Kokkos::RangePolicy<>(0, total),
    KOKKOS_LAMBDA(int64_t idx) {
      const int w_out      = (int)(idx % width_col);
      const int h_out      = (int)(idx / width_col % height_col);
      const int t_out      = (int)(idx / ((int64_t)width_col * height_col) % depth_col);
      const int channel_in = (int)(idx / ((int64_t)width_col * height_col * depth_col));

      const int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
      const int t_in = t_out * stride_t - pad_t;
      const int h_in = h_out * stride_h - pad_h;
      const int w_in = w_out * stride_w - pad_w;

      // Base index into data_col for this output location
      const int64_t col_base =
          ((int64_t)(channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;

      for (int i = 0; i < ksize_t; ++i) {
        for (int j = 0; j < ksize_h; ++j) {
          for (int k = 0; k < ksize_w; ++k) {
            const int t = t_in + i * dilation_t;
            const int h = h_in + j * dilation_h;
            const int w = w_in + k * dilation_w;
            const int64_t col_idx =
                col_base +
                (int64_t)(i * ksize_h * ksize_w + j * ksize_w + k) *
                    depth_col * height_col * width_col;
            data_col(col_idx) =
                (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width)
                    ? data_vol(((int64_t)(channel_in * depth + t) * height + h) * width + w)
                    : static_cast<T>(0);
          }
        }
      }
    });
  Kokkos::fence();
}

template <typename T, typename accT>
void col2vol_kernel(
    Kokkos::View<const T*> data_col,
    const uint64_t n,
    const unsigned depth,
    const unsigned height,
    const unsigned width,
    const unsigned kernel_t,
    const unsigned kernel_h,
    const unsigned kernel_w,
    const unsigned pad_t,
    const unsigned pad_h,
    const unsigned pad_w,
    const unsigned stride_t,
    const unsigned stride_h,
    const unsigned stride_w,
    const unsigned dilation_t,
    const unsigned dilation_h,
    const unsigned dilation_w,
    const unsigned depth_col,
    const unsigned height_col,
    const unsigned width_col,
    Kokkos::View<T*> data_vol)
{
  Kokkos::parallel_for("col2vol", Kokkos::RangePolicy<>(0, (int64_t)n),
    KOKKOS_LAMBDA(int64_t index) {
      accT val = static_cast<accT>(0);
      const unsigned w_im = (unsigned)(index % width) + pad_w;
      const unsigned h_im = (unsigned)(index / width % height) + pad_h;
      const unsigned t_im = (unsigned)(index / width / height % depth) + pad_t;
      const unsigned c_im = (unsigned)(index / ((uint64_t)width * height * depth));

      const unsigned kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
      const unsigned kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
      const unsigned kernel_extent_t = (kernel_t - 1) * dilation_t + 1;

      const unsigned w_col_start =
          (w_im < kernel_extent_w) ? 0u : (w_im - kernel_extent_w) / stride_w + 1;
      const unsigned w_col_end = Kokkos::min(w_im / stride_w + 1, width_col);
      const unsigned h_col_start =
          (h_im < kernel_extent_h) ? 0u : (h_im - kernel_extent_h) / stride_h + 1;
      const unsigned h_col_end = Kokkos::min(h_im / stride_h + 1, height_col);
      const unsigned t_col_start =
          (t_im < kernel_extent_t) ? 0u : (t_im - kernel_extent_t) / stride_t + 1;
      const unsigned t_col_end = Kokkos::min(t_im / stride_t + 1, depth_col);

      for (unsigned t_col = t_col_start; t_col < t_col_end; t_col++) {
        for (unsigned h_col = h_col_start; h_col < h_col_end; h_col++) {
          for (unsigned w_col = w_col_start; w_col < w_col_end; w_col++) {
            uint64_t t_k = t_im - t_col * stride_t;
            uint64_t h_k = h_im - h_col * stride_h;
            uint64_t w_k = w_im - w_col * stride_w;
            if (t_k % dilation_t == 0 && h_k % dilation_h == 0 &&
                w_k % dilation_w == 0) {
              t_k /= dilation_t;
              h_k /= dilation_h;
              w_k /= dilation_w;
              const uint64_t idx_k =
                  ((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k;
              const uint64_t data_col_index =
                  ((idx_k * depth_col + t_col) * height_col + h_col) * width_col + w_col;
              val += data_col(data_col_index);
            }
          }
        }
      }
      data_vol(index) = static_cast<T>(val);
    });
  Kokkos::fence();
}

template <typename T>
void eval(
    const int repeat,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int depth_col,
    const int height_col,
    const int width_col,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w)
{
  const uint64_t vol_size =
      (uint64_t)channels * (2*pad_t+depth) * (2*pad_h+height) * (2*pad_w+width);
  const uint64_t col_size =
      ((uint64_t)channels * ksize_t * ksize_h * ksize_w + 1) *
      (depth_col+pad_t) * (height_col+pad_h) * (width_col+pad_w);

  const uint64_t n = (uint64_t)channels * depth_col * height_col * width_col;

  // Allocate device Views
  Kokkos::View<T*> d_data_vol("data_vol", vol_size);
  Kokkos::View<T*> d_data_col("data_col", col_size);

  // Initialize host data and copy to device
  {
    auto h_vol = Kokkos::create_mirror_view(d_data_vol);
    auto h_col = Kokkos::create_mirror_view(d_data_col);
    for (uint64_t i = 0; i < vol_size; i++) h_vol(i) = (T)1;
    for (uint64_t i = 0; i < col_size; i++) h_col(i) = (T)0;
    Kokkos::deep_copy(d_data_vol, h_vol);
    Kokkos::deep_copy(d_data_col, h_col);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    vol2col_kernel<T>(
        d_data_vol, channels, depth, height, width,
        ksize_t, ksize_h, ksize_w,
        pad_t, pad_h, pad_w,
        stride_t, stride_h, stride_w,
        dilation_t, dilation_h, dilation_w,
        depth_col, height_col, width_col,
        d_data_col);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of vol2col kernel: %f (us)\n", (time * 1e-3f) / repeat);

  {
    auto h_col = Kokkos::create_mirror_view(d_data_col);
    Kokkos::deep_copy(h_col, d_data_col);
    float checksum = 0;
    for (uint64_t i = 0; i < col_size; i++) checksum += h_col(i);
    printf("Checksum = %f\n", checksum / col_size);
  }

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    col2vol_kernel<T, T>(
        Kokkos::View<const T*>(d_data_col),
        n, depth, height, width,
        ksize_t, ksize_h, ksize_w,
        pad_t, pad_h, pad_w,
        stride_t, stride_h, stride_w,
        dilation_t, dilation_h, dilation_w,
        depth_col, height_col, width_col,
        d_data_vol);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of col2vol kernel: %f (us)\n", (time * 1e-3f) / repeat);

  {
    auto h_vol = Kokkos::create_mirror_view(d_data_vol);
    Kokkos::deep_copy(h_vol, d_data_vol);
    float checksum = 0;
    for (uint64_t i = 0; i < vol_size; i++) checksum += h_vol(i);
    printf("Checksum = %f\n", checksum / vol_size);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const int channels   = 4;
    const int depth      = 3;
    const int height     = 255;
    const int width      = 255;
    const int pad_t      = 1, pad_h = 1, pad_w = 1;
    const int stride_t   = 2, stride_h = 2, stride_w = 2;
    const int dilation_t = 2, dilation_h = 2, dilation_w = 2;
    const int depth_col  = 3, height_col = 255, width_col = 255;

    for (int k = 1; k <= 9; k += 2) {
      printf("\nkernel size: %d\n", k);
      eval<float>(repeat,
                  channels, depth, height, width,
                  depth_col, height_col, width_col,
                  k, k, k,
                  pad_t, pad_h, pad_w,
                  stride_t, stride_h, stride_w,
                  dilation_t, dilation_h, dilation_w);
    }
  }
  Kokkos::finalize();
  return 0;
}
