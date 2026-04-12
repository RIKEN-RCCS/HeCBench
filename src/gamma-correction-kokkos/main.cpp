//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// Kokkos port of gamma-correction-cuda

#include <iomanip>
#include <iostream>
#include <chrono>

// Stub out CUDA annotations so utils/ImgPixel.hpp compiles without CUDA
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include <Kokkos_Core.hpp>
#include "utils.hpp"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int width     = atoi(argv[1]);
  const int height    = atoi(argv[2]);
  // block_size is accepted for interface compatibility but unused in Kokkos
  const int repeat    = atoi(argv[4]);

  Kokkos::initialize(argc, argv);
  {
    Img<ImgFormat::BMP> image{width, height};
    ImgFractal fractal{width, height};

    auto gamma_f = [](ImgPixel& pixel) {
      float v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.f;
      std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.f * v * v);
      if (gamma_pixel > 255) gamma_pixel = 255;
      pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
    };

    // Fill image with fractal
    int index = 0;
    image.fill([&index, width, &fractal](ImgPixel& pixel) {
      int x = index % width;
      int y = index / width;
      auto fractal_pixel = fractal(x, y);
      if (fractal_pixel < 0)   fractal_pixel = 0;
      if (fractal_pixel > 255) fractal_pixel = 255;
      pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);
      ++index;
    });

    Img<ImgFormat::BMP> image2 = image;
#ifdef DEBUG
    image.write("fractal_original.bmp");
#endif

    // Serial reference for correctness check
    image.fill(gamma_f);
#ifdef DEBUG
    image.write("fractal_gamma_serial.bmp");
#endif

    const int total_pixels = width * height;

    // Device view for pixel data
    Kokkos::View<ImgPixel*> d_pixel("d_pixel", total_pixels);
    auto h_pixel = Kokkos::create_mirror_view(d_pixel);

    float total_time = 0.f;

    for (int iter = 0; iter < repeat; iter++) {
      // Copy input to device
      for (int i = 0; i < total_pixels; i++) h_pixel(i) = image2.data()[i];
      Kokkos::deep_copy(d_pixel, h_pixel);

      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for("gamma_correction", total_pixels,
        KOKKOS_LAMBDA(int i) {
          const float v = (0.3f * d_pixel(i).r +
                           0.59f * d_pixel(i).g +
                           0.11f * d_pixel(i).b) / 255.f;
          std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.f * v * v);
          if (gamma_pixel > 255) gamma_pixel = 255;
          d_pixel(i).set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
        });

      Kokkos::fence();
      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);

    // Copy result back
    Kokkos::deep_copy(h_pixel, d_pixel);
    for (int i = 0; i < total_pixels; i++) image2.data()[i] = h_pixel(i);

    // Correctness check
    if (check(image.begin(), image.end(), image2.begin())) {
      std::cout << "PASS\n";
    } else {
      std::cout << "FAIL\n";
    }

#ifdef DEBUG
    image2.write("fractal_gamma_parallel.bmp");
#endif
  }
  Kokkos::finalize();
  return 0;
}
