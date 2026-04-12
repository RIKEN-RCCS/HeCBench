#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "utils.h"
#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    params p = {128, 3, 480, 640, 9, 1024, 1024};
    const int dim_b = p.output_dim_b;
    const int dim_c = p.output_dim_c;
    const int dim_h = p.output_dim_h;
    const int dim_w = p.output_dim_w;
    const int kernels_size = p.kernel_size;
    const int img_w = p.image_w;
    const int img_h = p.image_h;

    const int padding = 1;
    const int offset_unit = 1;

    const int k_size = (int)sqrtf((float)kernels_size);
    const int w = img_w - 2 * padding;
    const int h = img_h - 2 * padding;

    const size_t image_size  = (size_t)dim_b * dim_c * img_w * img_h;
    const size_t offset_size = (size_t)dim_b * kernels_size * dim_w * dim_h;
    const size_t kernel_size = (size_t)dim_b * kernels_size * dim_w * dim_h;
    const size_t output_size = (size_t)dim_b * dim_c * dim_w * dim_h;

    const size_t image_size_byte  = sizeof(float) * image_size;
    const size_t offset_size_byte = sizeof(float) * offset_size;
    const size_t kernel_size_byte = sizeof(float) * kernel_size;
    const size_t output_size_byte = sizeof(float) * output_size;

    float *img       = (float*) malloc(image_size_byte);
    float *offsets_h = (float*) malloc(offset_size_byte);
    float *offsets_v = (float*) malloc(offset_size_byte);
    float *kernel    = (float*) malloc(kernel_size_byte);
    float *output    = (float*) malloc(output_size_byte);
    float *output_ref= (float*) malloc(output_size_byte);

    unsigned long long seed = 123;
    for (size_t i = 0; i < image_size;  i++) img[i]      = (unsigned char)(256*LCG_random_double(&seed));
    for (size_t i = 0; i < kernel_size; i++) kernel[i]   = (unsigned char)(256*LCG_random_double(&seed));
    for (size_t i = 0; i < offset_size; i++) {
      offsets_h[i] = LCG_random_double(&seed);
      offsets_v[i] = LCG_random_double(&seed);
    }

    // Create device Views
    Kokkos::View<float*> d_img      ("d_img",       image_size);
    Kokkos::View<float*> d_offsets_h("d_offsets_h", offset_size);
    Kokkos::View<float*> d_offsets_v("d_offsets_v", offset_size);
    Kokkos::View<float*> d_kernel   ("d_kernel",    kernel_size);
    Kokkos::View<float*> d_output   ("d_output",    output_size);

    // Copy host data to device
    {
      auto hm = Kokkos::create_mirror_view(d_img);
      for (size_t i = 0; i < image_size; i++) hm(i) = img[i];
      Kokkos::deep_copy(d_img, hm);
    }
    {
      auto hm = Kokkos::create_mirror_view(d_offsets_h);
      for (size_t i = 0; i < offset_size; i++) hm(i) = offsets_h[i];
      Kokkos::deep_copy(d_offsets_h, hm);
    }
    {
      auto hm = Kokkos::create_mirror_view(d_offsets_v);
      for (size_t i = 0; i < offset_size; i++) hm(i) = offsets_v[i];
      Kokkos::deep_copy(d_offsets_v, hm);
    }
    {
      auto hm = Kokkos::create_mirror_view(d_kernel);
      for (size_t i = 0; i < kernel_size; i++) hm(i) = kernel[i];
      Kokkos::deep_copy(d_kernel, hm);
    }

    // Precompute strides for use inside the kernel
    const int vol_size = dim_c * dim_h * dim_w;
    const int img_size_2d = dim_h * dim_w;

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("car", (int)output_size, KOKKOS_LAMBDA(int global_idx) {
        const int idb = (global_idx / vol_size) % dim_b;
        const int idc = (global_idx / img_size_2d) % dim_c;
        const int idy = (global_idx / dim_w) % dim_h;
        const int idx = global_idx % dim_w;

        float result = 0.f;
        for (int k_y = 0; k_y < k_size; ++k_y) {
          for (int k_x = 0; k_x < k_size; ++k_x) {
            const int kk = k_size * k_y + k_x;
            const int off_base = idb*(int64_t)k_size*k_size*dim_w*dim_h
                                 + kk*(int64_t)dim_w*dim_h
                                 + idy*dim_w + idx;

            const float off_h = d_offsets_h(off_base) * offset_unit;
            const float off_v = d_offsets_v(off_base) * offset_unit;

            const float p_x = static_cast<float>(idx + 0.5f) / dim_w * w + k_x + off_h - 0.5f;
            const float p_y = static_cast<float>(idy + 0.5f) / dim_h * h + k_y + off_v - 0.5f;
            const float alpha = p_x - floorf(p_x);
            const float beta  = p_y - floorf(p_y);

            const int lim_x = w + 2 * padding - 1;
            const int lim_y = h + 2 * padding - 1;

            int tmp = (int)floorf(p_x);
            tmp = tmp < 0 ? 0 : (tmp > lim_x ? lim_x : tmp);
            const int xL = tmp;
            tmp = xL + 1;
            const int xR = tmp > lim_x ? lim_x : tmp;

            tmp = (int)floorf(p_y);
            tmp = tmp < 0 ? 0 : (tmp > lim_y ? lim_y : tmp);
            const int yT = tmp;
            tmp = yT + 1;
            const int yB = tmp > lim_y ? lim_y : tmp;

            const int64_t img_stride_b  = (int64_t)dim_c * img_w * img_h;
            const int64_t img_stride_c  = (int64_t)img_w * img_h;

            float val = (1.f - alpha) * (1.f - beta) * d_img(idb*img_stride_b + idc*img_stride_c + yT*img_w + xL);
            val += alpha * (1.f - beta) * d_img(idb*img_stride_b + idc*img_stride_c + yT*img_w + xR);
            val += (1.f - alpha) * beta * d_img(idb*img_stride_b + idc*img_stride_c + yB*img_w + xL);
            val += alpha * beta           * d_img(idb*img_stride_b + idc*img_stride_c + yB*img_w + xR);

            result += val * d_kernel(off_base);
          }
        }

        const int64_t out_idx = (int64_t)idb*dim_c*dim_w*dim_h
                                + (int64_t)idc*dim_w*dim_h
                                + idy*dim_w + idx;
        d_output(out_idx) = result;
      });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

    // Copy output back
    {
      auto hm = Kokkos::create_mirror_view(d_output);
      Kokkos::deep_copy(hm, d_output);
      for (size_t i = 0; i < output_size; i++) output[i] = hm(i);
    }

    // CPU reference
    reference(img, kernel, offsets_h, offsets_v, output_ref, p, offset_unit, padding);

    float rmse = 0.f;
    for (size_t i = 0; i < output_size; i++)
      rmse += (output_ref[i] - output[i]) * (output_ref[i] - output[i]);
    printf("RMSE: %f\n", sqrtf(rmse / output_size));

    free(img);
    free(offsets_h);
    free(offsets_v);
    free(kernel);
    free(output);
    free(output_ref);
  }
  Kokkos::finalize();
  return 0;
}
