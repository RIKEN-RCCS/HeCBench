/*
 * Reproducible Floating-point Sum (RFS).
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
float createRoundingFactor(float max, int n) {
  float delta = (max * (float)n) / (1.f - 2.f * (float)n * FLT_EPSILON);
  int exp;
  // frexpf cannot be called in device lambdas in all compilers,
  // so we use a portable integer-bit-cast trick:
  // get exponent of delta via IEEE754 representation
  // exp = ceil(log2(|delta|)) by reading the biased exponent
  unsigned int ibits;
  memcpy(&ibits, &delta, sizeof(float));
  int biased_exp = (int)((ibits >> 23) & 0xFF);
  exp = biased_exp - 126;  // unbiased exponent = biased - 127, but ceil(log2(x)) = biased-126 when x is in [0.5,1)
  // Use ldexpf to compute 2^exp
  return ldexpf(1.f, exp);
}

KOKKOS_INLINE_FUNCTION
float truncateWithRoundingFactor(float roundingFactor, float x) {
  return (roundingFactor + x) - roundingFactor;
}

void sumArray(float factor, int length, Kokkos::View<const float*> x, Kokkos::View<float*> r) {
  Kokkos::parallel_for("sumArray", length, KOKKOS_LAMBDA(int i) {
    float q = truncateWithRoundingFactor(factor, x(i));
    Kokkos::atomic_add(&r(0), q);
  });
  Kokkos::fence();
}

void sumArrays(int nArrays, int length,
               Kokkos::View<const float*> x,
               Kokkos::View<float*>       r,
               Kokkos::View<const float*> maxVal)
{
  Kokkos::parallel_for("sumArrays", nArrays, KOKKOS_LAMBDA(int i) {
    float factor = createRoundingFactor(maxVal(i), length);
    float s = 0.f;
    for (int n = length - 1; n >= 0; n--)
      s += truncateWithRoundingFactor(factor, x(i * length + n));
    r(i) = s;
  });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of arrays> <length of each array>\n", argv[0]);
    return 1;
  }
  const int nArrays = atoi(argv[1]);
  const int nElems  = atoi(argv[2]);

  float *arrays     = (float*) malloc(nArrays * nElems * sizeof(float));
  float *maxVal     = (float*) malloc(nArrays * sizeof(float));
  float *result     = (float*) malloc(nArrays * sizeof(float));
  float *factor     = (float*) malloc(nArrays * sizeof(float));
  float *result_ref = (float*) malloc(nArrays * sizeof(float));

  srand(123);
  float *arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    float mx = 0;
    for (int i = 0; i < nElems; i++) {
      arr[i] = (float)rand() / (float)RAND_MAX;
      if (rand() % 2) arr[i] = -arr[i];
      mx = fmaxf(fabsf(arr[i]), mx);
    }
    factor[n] = createRoundingFactor(mx, nElems);
    maxVal[n] = mx;
    arr += nElems;
  }

  // CPU reference (per-array sequential sum with truncation)
  arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    float f = factor[n];
    float s = 0.f;
    for (int i = nElems - 1; i >= 0; i--)
      s += truncateWithRoundingFactor(f, arr[i]);
    result_ref[n] = s;
    arr += nElems;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_arrays("d_arrays", nArrays * nElems);
    Kokkos::View<float*> d_maxVal("d_maxVal", nArrays);
    Kokkos::View<float*> d_result("d_result", nArrays);

    auto h_arrays = Kokkos::create_mirror_view(d_arrays);
    auto h_maxVal = Kokkos::create_mirror_view(d_maxVal);
    for (int i = 0; i < nArrays * nElems; i++) h_arrays(i) = arrays[i];
    for (int i = 0; i < nArrays; i++) h_maxVal(i) = maxVal[i];
    Kokkos::deep_copy(d_arrays, h_arrays);
    Kokkos::deep_copy(d_maxVal, h_maxVal);

    // Reset results
    Kokkos::parallel_for("reset", nArrays, KOKKOS_LAMBDA(int i) { d_result(i) = 0.f; });
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < nArrays; n++) {
      auto x_sub = Kokkos::subview(d_arrays, std::pair<int,int>(n * nElems, (n+1) * nElems));
      auto r_sub = Kokkos::subview(d_result, std::pair<int,int>(n, n+1));
      Kokkos::View<float*> r_view = Kokkos::subview(d_result, std::pair<int,int>(n, n+1));
      sumArray(factor[n], nElems, x_sub, r_view);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (sumArray): %f (s)\n", (time * 1e-9f) / nArrays);

    auto h_result = Kokkos::create_mirror_view(d_result);
    Kokkos::deep_copy(h_result, d_result);
    for (int i = 0; i < nArrays; i++) result[i] = h_result(i);
    bool ok = !memcmp(result_ref, result, nArrays * sizeof(float));
    printf("%s\n", ok ? "PASS" : "FAIL");

    // sumArrays kernel
    start = std::chrono::steady_clock::now();
    sumArrays(nArrays, nElems, d_arrays, d_result, d_maxVal);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Kernel execution time (sumArrays): %f (s)\n", time * 1e-9f);

    Kokkos::deep_copy(h_result, d_result);
    for (int i = 0; i < nArrays; i++) result[i] = h_result(i);
    ok = !memcmp(result_ref, result, nArrays * sizeof(float));
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(arrays); free(maxVal); free(result); free(factor); free(result_ref);
  return 0;
}
