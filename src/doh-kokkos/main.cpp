#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>

typedef float IMAGE_T;
typedef int   INT_T;

KOKKOS_INLINE_FUNCTION
int _clip(const int x, const int low, const int high) {
  if (x > high) return high;
  if (x < low)  return low;
  return x;
}

KOKKOS_INLINE_FUNCTION
IMAGE_T _integ(const IMAGE_T *img,
               const INT_T img_rows,
               const INT_T img_cols,
               int r, int c,
               const int rl, const int cl)
{
  r = _clip(r, 0, img_rows - 1);
  c = _clip(c, 0, img_cols - 1);
  const int r2 = _clip(r + rl, 0, img_rows - 1);
  const int c2 = _clip(c + cl, 0, img_cols - 1);
  IMAGE_T ans = img[r  * img_cols + c ] + img[r2 * img_cols + c2]
              - img[r  * img_cols + c2] - img[r2 * img_cols + c ];
  return fmaxf((IMAGE_T)0, ans);
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <height> <width> <repeat>\n", argv[0]);
    return 1;
  }
  const int h      = atoi(argv[1]);
  const int w      = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int img_size = h * w;

  float *input_img   = (float*)malloc(img_size * sizeof(float));
  float *integral_img = (float*)malloc(img_size * sizeof(float));

  std::default_random_engine rng(123);
  std::normal_distribution<float> norm_dist(0.f, 1.f);
  for (int i = 0; i < img_size; i++)
    input_img[i] = norm_dist(rng);

  printf("Integrating the input image...\n");
  // O(n^2) prefix sum: integral[i][j] = input[i][j]
  //   + integral[i-1][j] + integral[i][j-1] - integral[i-1][j-1]
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      IMAGE_T s = input_img[i * w + j];
      if (i > 0) s += integral_img[(i-1) * w + j];
      if (j > 0) s += integral_img[i * w + (j-1)];
      if (i > 0 && j > 0) s -= integral_img[(i-1) * w + (j-1)];
      integral_img[i * w + j] = s;
    }
  }

  free(input_img);

  Kokkos::initialize(argc, argv);
  {
    // Upload integral image to device
    Kokkos::View<IMAGE_T*> d_img("integral_img", img_size);
    Kokkos::View<IMAGE_T*> d_out("output_img",   img_size);
    {
      auto h_img = Kokkos::create_mirror_view(d_img);
      for (int i = 0; i < img_size; i++) h_img(i) = integral_img[i];
      Kokkos::deep_copy(d_img, h_img);
    }

    const IMAGE_T sigma = 4.0f;
    const INT_T   rows  = h;
    const INT_T   cols  = w;

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("hessian_matrix_det", img_size,
        KOKKOS_LAMBDA(int tid) {
          const int r = tid / cols;
          const int c = tid % cols;

          int size = (int)((IMAGE_T)3.0f * sigma);
          const int b   = (size - 1) / 2 + 1;
          const int l   = size / 3;
          const int w_k = size;

          const IMAGE_T w_i = (IMAGE_T)1.0f / (size * size);

          const IMAGE_T* img = d_img.data();

          const IMAGE_T tl = _integ(img, rows, cols, r - l,     c - l,     l,         l);
          const IMAGE_T br = _integ(img, rows, cols, r + 1,     c + 1,     l,         l);
          const IMAGE_T bl = _integ(img, rows, cols, r - l,     c + 1,     l,         l);
          const IMAGE_T tr = _integ(img, rows, cols, r + 1,     c - l,     l,         l);

          IMAGE_T dxy = bl + tr - tl - br;
          dxy = -dxy * w_i;

          IMAGE_T mid  = _integ(img, rows, cols, r - l + 1, c - l,     2 * l - 1, w_k);
          IMAGE_T side = _integ(img, rows, cols, r - l + 1, c - l / 2, 2 * l - 1, l);
          IMAGE_T dxx  = mid - (IMAGE_T)3 * side;
          dxx = -dxx * w_i;

          mid  = _integ(img, rows, cols, r - l,     c - b + 1, w_k,       2 * b - 1);
          side = _integ(img, rows, cols, r - b / 2, c - b + 1, b,         2 * b - 1);
          IMAGE_T dyy = mid - (IMAGE_T)3 * side;
          dyy = -dyy * w_i;

          d_out(tid) = dxx * dyy - (IMAGE_T)0.81f * (dxy * dxy);
        });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    long time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Copy back and compute checksum
    auto h_out = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out, d_out);

    double checksum = 0.0;
    for (int i = 0; i < img_size; i++)
      checksum += h_out(i);

    printf("Average kernel execution time : %f (us)\n", time_ns * 1e-3 / repeat);
    printf("Kernel checksum: %lf\n", checksum);
  }
  Kokkos::finalize();

  free(integral_img);
  return 0;
}
