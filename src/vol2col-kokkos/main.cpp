#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <chrono>
#include <Kokkos_Core.hpp>

// Kernel for fast unfold+copy on volumes
template <typename T>
void vol2col(
    Kokkos::View<const T*, Kokkos::DefaultExecutionSpace::memory_space> data_vol,
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
    Kokkos::View<T*, Kokkos::DefaultExecutionSpace::memory_space> data_col)
{
  using exec_space = Kokkos::DefaultExecutionSpace;
  const int64_t total = (int64_t)channels * depth_col * height_col * width_col;

  Kokkos::parallel_for(
    "vol2col",
    Kokkos::RangePolicy<exec_space>(0, total),
    KOKKOS_LAMBDA(const int64_t idx) {
      int w_out = idx % width_col;
      int h_out = (idx / width_col) % height_col;
      int t_out = (idx / width_col / height_col) % depth_col;
      int channel_in  = idx / width_col / height_col / depth_col;
      int channel_out = channel_in * ksize_t * ksize_h * ksize_w;

      int t_in = t_out * stride_t - pad_t;
      int h_in = h_out * stride_h - pad_h;
      int w_in = w_out * stride_w - pad_w;

      int base_vol = ((channel_in * depth + t_in) * height + h_in) * width + w_in;
      int base_col = ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;

      for (int i = 0; i < ksize_t; ++i) {
        for (int j = 0; j < ksize_h; ++j) {
          for (int k = 0; k < ksize_w; ++k) {
            int t = t_in + i * dilation_t;
            int h = h_in + j * dilation_h;
            int w = w_in + k * dilation_w;
            data_col(base_col) =
              (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width)
              ? data_vol(base_vol + i * dilation_t * height * width +
                         j * dilation_h * width + k * dilation_w)
              : static_cast<T>(0);
            base_col += depth_col * height_col * width_col;
          }
        }
      }
    });
}

template <typename T>
void col2vol(
    Kokkos::View<const T*, Kokkos::DefaultExecutionSpace::memory_space> data_col,
    const int64_t n,
    const int depth,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
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
    Kokkos::View<T*, Kokkos::DefaultExecutionSpace::memory_space> data_vol)
{
  using exec_space = Kokkos::DefaultExecutionSpace;

  Kokkos::parallel_for(
    "col2vol",
    Kokkos::RangePolicy<exec_space, Kokkos::IndexType<int64_t>>(0, n),
    KOKKOS_LAMBDA(const int64_t index) {
      T val = static_cast<T>(0);
      int w_im = (int)(index % width) + pad_w;
      int h_im = (int)(index / width % height) + pad_h;
      int t_im = (int)(index / width / height % depth) + pad_t;
      int c_im = (int)(index / (width * height * depth));

      int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
      int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
      int kernel_extent_t = (kernel_t - 1) * dilation_t + 1;

      int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
      int w_col_end   = Kokkos::min(w_im / stride_w + 1, width_col);
      int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
      int h_col_end   = Kokkos::min(h_im / stride_h + 1, height_col);
      int t_col_start = (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
      int t_col_end   = Kokkos::min(t_im / stride_t + 1, depth_col);

      for (int t_col = t_col_start; t_col < t_col_end; ++t_col) {
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
          for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
            int64_t t_k = t_im - t_col * stride_t;
            int64_t h_k = h_im - h_col * stride_h;
            int64_t w_k = w_im - w_col * stride_w;
            if (t_k % dilation_t == 0 && h_k % dilation_h == 0 && w_k % dilation_w == 0) {
              t_k /= dilation_t; h_k /= dilation_h; w_k /= dilation_w;
              int64_t idx_k = ((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k;
              int64_t data_col_idx =
                ((idx_k * depth_col + t_col) * height_col + h_col) * width_col + w_col;
              val += data_col(data_col_idx);
            }
          }
        }
      }
      data_vol(index) = val;
    });
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
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space  = typename exec_space::memory_space;

  int64_t vol_size = (int64_t)channels * (2*pad_t+depth) * (2*pad_h+height) * (2*pad_w+width);
  int64_t col_size = ((int64_t)channels * ksize_t * ksize_h * ksize_w + 1) *
                     (depth_col+pad_t) * (height_col+pad_h) * (width_col+pad_w);

  Kokkos::View<T*, mem_space> d_vol("vol", vol_size);
  Kokkos::View<T*, mem_space> d_col("col", col_size);

  // Initialize vol on host then copy
  {
    auto h_vol = Kokkos::create_mirror_view(d_vol);
    for (int64_t i = 0; i < vol_size; i++) h_vol(i) = (T)1;
    Kokkos::deep_copy(d_vol, h_vol);
    Kokkos::deep_copy(d_col, (T)0);
  }

  int64_t n = (int64_t)channels * depth_col * height_col * width_col;

  auto cv = Kokkos::create_mirror_view(
    Kokkos::View<const T*, mem_space>(d_vol));

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    vol2col<T>(d_vol, channels, depth, height, width,
               ksize_t, ksize_h, ksize_w,
               pad_t, pad_h, pad_w,
               stride_t, stride_h, stride_w,
               dilation_t, dilation_h, dilation_w,
               depth_col, height_col, width_col, d_col);
  }
  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of vol2col kernel: %f (us)\n", (time * 1e-3f) / repeat);

  {
    auto h_col = Kokkos::create_mirror_view(d_col);
    Kokkos::deep_copy(h_col, d_col);
    float checksum = 0;
    for (int64_t i = 0; i < col_size; i++) checksum += h_col(i);
    printf("Checksum = %f\n", checksum / col_size);
  }

  Kokkos::fence();
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    auto d_col_const = Kokkos::View<const T*, mem_space>(d_col);
    col2vol<T>(d_col_const, n, depth, height, width,
               ksize_t, ksize_h, ksize_w,
               pad_t, pad_h, pad_w,
               stride_t, stride_h, stride_w,
               dilation_t, dilation_h, dilation_w,
               depth_col, height_col, width_col, d_vol);
  }
  Kokkos::fence();
  end  = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of col2vol kernel: %f (us)\n", (time * 1e-3f) / repeat);

  {
    auto h_vol = Kokkos::create_mirror_view(d_vol);
    Kokkos::deep_copy(h_vol, d_vol);
    float checksum = 0;
    for (int64_t i = 0; i < vol_size; i++) checksum += h_vol(i);
    printf("Checksum = %f\n", checksum / vol_size);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    int channels  = 4;
    int depth     = 3;
    int height    = 255;
    int width     = 255;
    int pad_t     = 1, pad_h = 1, pad_w = 1;
    int stride_t  = 2, stride_h = 2, stride_w = 2;
    int dilation_t = 2, dilation_h = 2, dilation_w = 2;
    int depth_col  = 3, height_col = 255, width_col = 255;

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
