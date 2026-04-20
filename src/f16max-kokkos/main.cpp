// Kokkos port of f16max-cuda
// FP16 half2 max replaced with float fmaxf.
// Element-wise max of two float arrays, validated against host reference.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUM_OF_BLOCKS  1048576
#define NUM_OF_THREADS 256

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // size = NUM_OF_BLOCKS * NUM_OF_THREADS floats (was half2 pairs in original)
  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;

  float* a = (float*)malloc(size * sizeof(float));
  float* b = (float*)malloc(size * sizeof(float));
  float* r = (float*)malloc(size * sizeof(float));

  srand(123);
  for (size_t i = 0; i < size; i++) {
    a[i] = (float)(rand() % 922021);
    b[i] = (float)(rand() % 922021);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_a("a", size);
    Kokkos::View<float*> d_b("b", size);
    Kokkos::View<float*> d_r("r", size);

    {
      auto ha = Kokkos::create_mirror_view(d_a);
      auto hb = Kokkos::create_mirror_view(d_b);
      for (size_t i = 0; i < size; i++) { ha(i) = a[i]; hb(i) = b[i]; }
      Kokkos::deep_copy(d_a, ha);
      Kokkos::deep_copy(d_b, hb);
    }

    // warm-up
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(size_t i) {
      d_r(i) = Kokkos::fmax(d_a(i), d_b(i));
    });
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repeat; rep++) {
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(size_t i) {
        d_r(i) = Kokkos::fmax(d_a(i), d_b(i));
      });
    }
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    auto hr = Kokkos::create_mirror_view(d_r);
    Kokkos::deep_copy(hr, d_r);
    for (size_t i = 0; i < size; i++) r[i] = hr(i);
  }
  Kokkos::finalize();

  // Verify
  bool ok = true;
  for (size_t i = 0; i < size; i++) {
    float expected = fmaxf(a[i], b[i]);
    if (fabsf(r[i] - expected) > 1e-3f) { ok = false; break; }
  }
  printf("fmax result: %s\n", ok ? "PASS" : "FAIL");

  free(a); free(b); free(r);
  return ok ? 0 : 1;
}
