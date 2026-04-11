// Kokkos port of the bilateral image-filter benchmark.
// Reference: https://en.wikipedia.org/wiki/Bilateral_filter
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include "reference.h"

// Bilateral filter kernel with radius R (runtime parameter)
void bilateralFilter(
    const Kokkos::View<const float*> in,
    Kokkos::View<float*> out,
    int w, int h,
    float a_square,
    float variance_I,
    float variance_spatial,
    int R)
{
  Kokkos::parallel_for(
    "bilateral",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, w}),
    KOKKOS_LAMBDA(const int idy, const int idx) {
      int id = idy * w + idx;
      float I = in(id);
      float res = 0.f;
      float normalization = 0.f;

      for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
          int idk = idx + i;
          int idl = idy + j;

          // mirror edges
          if (idk < 0)     idk = -idk;
          if (idl < 0)     idl = -idl;
          if (idk > w - 1) idk = w - 1 - i;
          if (idl > h - 1) idl = h - 1 - j;

          int id_w = idl * w + idk;
          float I_w = in(id_w);

          float range   = -(I - I_w) * (I - I_w) / (2.f * variance_I);
          float spatial = -((idk - idx) * (idk - idx) + (idl - idy) * (idl - idy))
                          / (2.f * variance_spatial);
          float weight  = a_square * Kokkos::exp(spatial + range);
          normalization += weight;
          res           += I_w * weight;
        }
      }
      out(id) = res / normalization;
    }
  );
  Kokkos::fence();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 6) {
      printf("Usage: %s <image width> <image height> <intensity variance> "
             "<spatial variance> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }

    int   w             = atoi(argv[1]);
    int   h             = atoi(argv[2]);
    float variance_I    = atof(argv[3]);
    float variance_spatial = atof(argv[4]);
    int   repeat        = atoi(argv[5]);

    float a_square = 0.5f / (variance_I * (float)M_PI);
    int   img_size = w * h;

    // Host data
    float *h_src = (float*) malloc(img_size * sizeof(float));
    float *h_dst = (float*) malloc(img_size * sizeof(float));
    float *r_dst = (float*) malloc(img_size * sizeof(float));

    srand(123);
    for (int i = 0; i < img_size; i++)
      h_src[i] = (float)(rand() % 256);

    // Device views
    Kokkos::View<float*> d_src("src", img_size);
    Kokkos::View<float*> d_dst("dst", img_size);

    auto h_src_view = Kokkos::View<const float*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_src, img_size);
    auto h_dst_view = Kokkos::View<float*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_dst, img_size);
    Kokkos::deep_copy(d_src, h_src_view);

    bool ok = true;

    // Radius 3
    {
      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        bilateralFilter(d_src, d_dst, w, h, a_square, variance_I, variance_spatial, 3);
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time (3x3): %f (ms)\n", time * 1e-6 / repeat);

      Kokkos::deep_copy(h_dst_view, d_dst);
      for (int i = 0; i < img_size; i++) h_dst[i] = h_dst_view(i);
      reference<3>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
      for (int i = 0; i < img_size; i++) {
        if (fabsf(r_dst[i] - h_dst[i]) > 1e-3f) { ok = false; break; }
      }
    }

    // Radius 6
    {
      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        bilateralFilter(d_src, d_dst, w, h, a_square, variance_I, variance_spatial, 6);
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time (6x6): %f (ms)\n", time * 1e-6 / repeat);

      Kokkos::deep_copy(h_dst_view, d_dst);
      for (int i = 0; i < img_size; i++) h_dst[i] = h_dst_view(i);
      reference<6>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
      for (int i = 0; i < img_size; i++) {
        if (fabsf(r_dst[i] - h_dst[i]) > 1e-3f) { ok = false; break; }
      }
    }

    // Radius 9
    {
      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++)
        bilateralFilter(d_src, d_dst, w, h, a_square, variance_I, variance_spatial, 9);
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time (9x9): %f (ms)\n", time * 1e-6 / repeat);

      Kokkos::deep_copy(h_dst_view, d_dst);
      for (int i = 0; i < img_size; i++) h_dst[i] = h_dst_view(i);
      reference<9>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
      for (int i = 0; i < img_size; i++) {
        if (fabsf(r_dst[i] - h_dst[i]) > 1e-3f) { ok = false; break; }
      }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");

    free(h_src);
    free(h_dst);
    free(r_dst);
  }
  Kokkos::finalize();
  return 0;
}
