#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <Kokkos_Core.hpp>
#include "code.h"

template <int STOCHASTIC>
KOKKOS_INLINE_FUNCTION
uint8_t dQuantize(const float* smem_code, const float rand_val, float x)
{
  int pivot = 127, upper_pivot = 255, lower_pivot = 0;
  float lower = -1.0f, upper = 1.0f;
  float val = smem_code[pivot];

  for (int i = 64; i > 0; i >>= 1) {
    if (x > val) {
      lower_pivot = pivot; lower = val; pivot += i;
    } else {
      upper_pivot = pivot; upper = val; pivot -= i;
    }
    val = smem_code[pivot];
  }

  if (upper_pivot == 255) upper = smem_code[upper_pivot];
  if (lower_pivot == 0)   lower = smem_code[lower_pivot];

  if (!STOCHASTIC) {
    if (x > val) {
      float midpoint = (upper + val) * 0.5f;
      return (uint8_t)((x > midpoint) ? upper_pivot : pivot);
    } else {
      float midpoint = (lower + val) * 0.5f;
      return (uint8_t)((x < midpoint) ? lower_pivot : pivot);
    }
  } else {
    if (x > val) {
      float d2u = fabsf(upper - x);
      float df  = upper - val;
      return (uint8_t)((rand_val >= d2u / df) ? upper_pivot : pivot);
    } else {
      float d2l = fabsf(lower - x);
      float df  = val - lower;
      return (uint8_t)((rand_val >= d2l / df) ? lower_pivot : pivot);
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n     = (size_t)atol(argv[1]);
  const int    repeat = atoi(argv[2]);

  float*   A   = (float*)  malloc(n * sizeof(float));
  uint8_t* out = (uint8_t*)malloc(n * sizeof(uint8_t));
  uint8_t* ref = (uint8_t*)malloc(n * sizeof(uint8_t));

  std::mt19937 gen{19937};
  std::normal_distribution<float> d{0.f, 1.f};
  for (size_t i = 0; i < n; i++) {
    A[i]   = d(gen);
    ref[i] = dQuantize<0>(code, 0.f, A[i]);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*>   d_A("A", n);
    Kokkos::View<uint8_t*> d_out("out", n);
    Kokkos::View<float*>   d_code("code", 256);

    {
      auto hA    = Kokkos::create_mirror_view(d_A);
      auto hcode = Kokkos::create_mirror_view(d_code);
      for (size_t i = 0; i < n;   i++) hA[i]    = A[i];
      for (int    i = 0; i < 256; i++) hcode[i]  = code[i];
      Kokkos::deep_copy(d_A,    hA);
      Kokkos::deep_copy(d_code, hcode);
    }

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("quant", n,
        KOKKOS_LAMBDA(size_t i) {
          d_out[i] = dQuantize<0>(d_code.data(), 0.f, d_A[i]);
        });
      Kokkos::fence();
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kQuantize kernel with block size 256: %f (us)\n",
           (time * 1e-3f) / repeat);

    {
      auto h_out = Kokkos::create_mirror_view(d_out);
      Kokkos::deep_copy(h_out, d_out);
      for (size_t i = 0; i < n; i++) out[i] = h_out[i];
    }
  }
  Kokkos::finalize();

  printf("%s\n", memcmp(out, ref, n * sizeof(uint8_t)) ? "FAIL" : "PASS");

  free(A); free(out); free(ref);
  return 0;
}
