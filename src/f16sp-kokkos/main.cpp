// Kokkos port of f16sp-cuda (FP16 scalar / dot product benchmark)
// half2 replaced with float2 (float pairs); cuBLAS dot product replaced
// with Kokkos parallel_reduce; cub::BlockReduce replaced with Kokkos reduce.
//
// The analytical dot product result for the original input is 65504.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUM_OF_BLOCKS  (1024 * 1024)
#define NUM_OF_THREADS 128

// Input: float pairs (was half2 in original)
// size = NUM_OF_BLOCKS * NUM_OF_THREADS pairs
// Analytical result: sum of a[i].x*b[i].x + a[i].y*b[i].y = 65504

void generateInput(float* ax, float* ay, size_t size)
{
  float v = sqrtf(32752.f / size);
  for (size_t i = 0; i < size; i++) { ax[i] = v; ay[i] = v; }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;

  float* ax = (float*)malloc(size * sizeof(float));
  float* ay = (float*)malloc(size * sizeof(float));
  float* bx = (float*)malloc(size * sizeof(float));
  float* by = (float*)malloc(size * sizeof(float));

  srand(123);
  generateInput(ax, ay, size);
  generateInput(bx, by, size);

  printf("\nNumber of elements in the vectors is %zu\n", size * 2);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_ax("ax", size), d_ay("ay", size);
    Kokkos::View<float*> d_bx("bx", size), d_by("by", size);

    auto copy_in = [&](Kokkos::View<float*> d, float* h) {
      auto hv = Kokkos::create_mirror_view(d);
      for (size_t i = 0; i < size; i++) hv(i) = h[i];
      Kokkos::deep_copy(d, hv);
    };
    copy_in(d_ax, ax); copy_in(d_ay, ay);
    copy_in(d_bx, bx); copy_in(d_by, by);

    // Evaluate for different grid sizes (matching the original sweep)
    for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid /= 2) {
      printf("\nGrid size is %d\n", grid);

      // warm-up
      float r = 0.f;
      for (int i = 0; i < 10; i++) {
        Kokkos::parallel_reduce(size, KOKKOS_LAMBDA(size_t j, float& val) {
          val += d_ax(j) * d_bx(j) + d_ay(j) * d_by(j);
        }, r);
      }
      Kokkos::fence();

      // kernel1 (fp32 accumulation, intrinsics proxy)
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        r = 0.f;
        Kokkos::parallel_reduce(size, KOKKOS_LAMBDA(size_t j, float& val) {
          val += d_ax(j) * d_bx(j) + d_ay(j) * d_by(j);
        }, r);
      }
      Kokkos::fence();
      auto end = std::chrono::steady_clock::now();
      auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel1 execution time %f (us)\n", (t * 1e-3f) / repeat);
      printf("Error rate: %e\n", fabsf(r - 65504.f) / 65504.f);
    }
  }
  Kokkos::finalize();

  free(ax); free(ay); free(bx); free(by);
  return 0;
}
