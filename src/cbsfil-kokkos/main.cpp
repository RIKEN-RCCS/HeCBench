#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

#define POLE (sqrtf(3.0f) - 2.0f)

typedef unsigned int uint;
typedef unsigned char uchar;

KOKKOS_INLINE_FUNCTION
float InitialCausalCoefficient(float* c, uint DataLength, int step)
{
  const uint Horizon = 12 < DataLength ? 12 : DataLength;
  float zn = POLE;
  float Sum = *c;
  for (uint n = 0; n < Horizon; n++) {
    Sum += zn * *c;
    zn *= POLE;
    c = (float*)((uchar*)c + step);
  }
  return Sum;
}

KOKKOS_INLINE_FUNCTION
float InitialAntiCausalCoefficient(float* c, uint DataLength, int step)
{
  return (POLE / (POLE - 1.0f)) * *c;
}

KOKKOS_INLINE_FUNCTION
void ConvertToInterpolationCoefficients(float* coeffs, uint DataLength, int step)
{
  const float Lambda = (1.0f - POLE) * (1.0f - 1.0f / POLE);
  float* c = coeffs;
  float previous_c;
  *c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
  for (uint n = 1; n < DataLength; n++) {
    c = (float*)((uchar*)c + step);
    *c = previous_c = Lambda * *c + POLE * previous_c;
  }
  *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
  for (int n = (int)DataLength - 2; n >= 0; n--) {
    c = (float*)((uchar*)c - step);
    *c = previous_c = POLE * (previous_c - *c);
  }
}

int PowTwoDivider(int n)
{
  if (n == 0) return 0;
  int divider = 1;
  while ((n & divider) == 0) divider <<= 1;
  return divider;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width  = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int numPix = width * height;
  const int pitch  = width * (int)sizeof(float);

  float* image = (float*)malloc(numPix * sizeof(float));
  srand(123);
  for (int i = 0; i < numPix; i++) {
    uint x = rand() % 256, y = rand() % 256, z = rand() % 256, w = rand() % 256;
    *(uint*)(&image[i]) = (w << 24) | (z << 16) | (y << 8) | x;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_image("image", numPix);
    auto h_image = Kokkos::create_mirror_view(d_image);

    long total_time = 0;
    for (int iter = 0; iter < repeat; iter++) {
      for (int i = 0; i < numPix; i++) h_image(i) = image[i];
      Kokkos::deep_copy(d_image, h_image);

      auto start = std::chrono::steady_clock::now();

      // toCoef2DX: process rows horizontally
      Kokkos::parallel_for("toCoef2DX", height, KOKKOS_LAMBDA(int y) {
        float* line = d_image.data() + y * width;
        ConvertToInterpolationCoefficients(line, (uint)width, (int)sizeof(float));
      });
      Kokkos::fence();

      // toCoef2DY: process columns vertically
      Kokkos::parallel_for("toCoef2DY", width, KOKKOS_LAMBDA(int x) {
        float* line = d_image.data() + x;
        ConvertToInterpolationCoefficients(line, (uint)height, pitch);
      });
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    printf("Average kernel execution time %f (s)\n", total_time * 1e-9f / repeat);

    Kokkos::deep_copy(h_image, d_image);
    for (int i = 0; i < numPix; i++) image[i] = h_image(i);
  }
  Kokkos::finalize();

  float sum = 0.f;
  for (int i = 0; i < numPix; i++) {
    const uchar* t = (const uchar*)(&image[i]);
    sum += (t[0] + t[1] + t[2] + t[3]) / 4.f;
  }
  printf("Checksum: %f\n", sum / numPix);

  free(image);
  return 0;
}
