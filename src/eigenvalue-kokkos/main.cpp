#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "reference.h"
#include "utils.cpp"

//=============================================================================
// Device-callable helper: count eigenvalues less than x
//=============================================================================
KOKKOS_INLINE_FUNCTION
unsigned int calNumEigenValuesLessThan_dev(
    float x,
    unsigned int width,
    const float *diagonal,
    const float *offDiagonal)
{
  unsigned int count = 0;
  float prev_diff = diagonal[0] - x;
  count += (prev_diff < 0) ? 1 : 0;
  for(unsigned int i = 1; i < width; i++) {
    float diff = (diagonal[i] - x)
               - (offDiagonal[i-1] * offDiagonal[i-1]) / prev_diff;
    count += (diff < 0) ? 1 : 0;
    prev_diff = diff;
  }
  return count;
}

//=============================================================================
// Kernel 1: compute number of eigenvalues in each interval
//=============================================================================
void calNumEigenValueInterval_kokkos(
    Kokkos::View<unsigned int*> d_numEigen,
    Kokkos::View<float*>        d_eigenIntervals,
    Kokkos::View<float*>        d_diagonal,
    Kokkos::View<float*>        d_offDiagonal,
    unsigned int width)
{
  Kokkos::parallel_for("calNumEigenValueInterval", (int)width,
    KOKKOS_LAMBDA(int gid) {
      unsigned int lowerId = 2 * gid;
      unsigned int upperId = lowerId + 1;
      float lowerLimit = d_eigenIntervals(lowerId);
      float upperLimit = d_eigenIntervals(upperId);
      unsigned int lower = calNumEigenValuesLessThan_dev(
          lowerLimit, width, d_diagonal.data(), d_offDiagonal.data());
      unsigned int upper = calNumEigenValuesLessThan_dev(
          upperLimit, width, d_diagonal.data(), d_offDiagonal.data());
      d_numEigen(gid) = upper - lower;
    });
  Kokkos::fence();
}

//=============================================================================
// Kernel 2: recalculate eigen intervals
//=============================================================================
void recalculateEigenIntervals_kokkos(
    Kokkos::View<float*>        d_newEigenIntervals,
    Kokkos::View<float*>        d_eigenIntervals,
    Kokkos::View<unsigned int*> d_numEigenIntervals,
    Kokkos::View<float*>        d_diagonal,
    Kokkos::View<float*>        d_offDiagonal,
    unsigned int width,
    float tolerance)
{
  Kokkos::parallel_for("recalculateEigenIntervals", (int)width,
    KOKKOS_LAMBDA(int gid) {
      unsigned int lowerId = 2 * gid;
      unsigned int upperId = lowerId + 1;
      unsigned int currentIndex = gid;

      unsigned int index = 0;
      while(currentIndex >= d_numEigenIntervals(index)) {
        currentIndex -= d_numEigenIntervals(index);
        ++index;
      }

      unsigned int lId = 2 * index;
      unsigned int uId = lId + 1;

      if(d_numEigenIntervals(index) == 1) {
        float midValue = (d_eigenIntervals(uId) + d_eigenIntervals(lId)) / 2.0f;
        float n = (float)calNumEigenValuesLessThan_dev(
            midValue, width, d_diagonal.data(), d_offDiagonal.data());
        n -= (float)calNumEigenValuesLessThan_dev(
            d_eigenIntervals(lId), width, d_diagonal.data(), d_offDiagonal.data());

        if(d_eigenIntervals(uId) - d_eigenIntervals(lId) < tolerance) {
          d_newEigenIntervals(lowerId) = d_eigenIntervals(lId);
          d_newEigenIntervals(upperId) = d_eigenIntervals(uId);
        } else if(n == 0.0f) {
          d_newEigenIntervals(lowerId) = midValue;
          d_newEigenIntervals(upperId) = d_eigenIntervals(uId);
        } else {
          d_newEigenIntervals(lowerId) = d_eigenIntervals(lId);
          d_newEigenIntervals(upperId) = midValue;
        }
      } else {
        float divisionWidth = (d_eigenIntervals(uId) - d_eigenIntervals(lId))
                            / (float)d_numEigenIntervals(index);
        d_newEigenIntervals(lowerId) = d_eigenIntervals(lId) + divisionWidth * currentIndex;
        d_newEigenIntervals(upperId) = d_newEigenIntervals(lowerId) + divisionWidth;
      }
    });
  Kokkos::fence();
}

//=============================================================================
// main
//=============================================================================
int main(int argc, char *argv[]) {
  if(argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    return 1;
  }

  int length     = atoi(argv[1]);
  int iterations = atoi(argv[2]);
  unsigned int seed = 123;
  float tolerance;

  if(isPowerOf2(length))       length = roundToPowerOf2(length);
  if(length < 256)             length = 256;

  float *diagonal      = (float*)malloc(length * sizeof(float));
  float *offDiagonal   = (float*)malloc((length-1) * sizeof(float));
  float *eigenIntervals[2];
  eigenIntervals[0] = (float*)malloc(2*length * sizeof(float));
  eigenIntervals[1] = (float*)malloc(2*length * sizeof(float));

  fillRandom<float>(diagonal,    length,   1, 0, 255, seed);
  fillRandom<float>(offDiagonal, length-1, 1, 0, 255, seed+10);

  float lowerLimit, upperLimit;
  computeGerschgorinInterval(&lowerLimit, &upperLimit, diagonal, offDiagonal, length);

  eigenIntervals[0][0] = lowerLimit;
  eigenIntervals[0][1] = upperLimit;
  for(int i = 2; i < 2*length; i++) eigenIntervals[0][i] = upperLimit;

  tolerance = 0.001f;

  unsigned int *numEigenValuesIntervalBuffer = (unsigned int*)malloc(length * sizeof(unsigned int));

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<float*>        d_diagonal("diagonal",    length);
    Kokkos::View<float*>        d_offDiag ("offDiag",     length-1);
    Kokkos::View<float*>        d_eiBuf0  ("eiBuf0",      2*length);
    Kokkos::View<float*>        d_eiBuf1  ("eiBuf1",      2*length);
    Kokkos::View<unsigned int*> d_numEigen("numEigen",    length);

    auto h_diag  = Kokkos::create_mirror_view(d_diagonal);
    auto h_offD  = Kokkos::create_mirror_view(d_offDiag);
    auto h_ei0   = Kokkos::create_mirror_view(d_eiBuf0);
    auto h_ei1   = Kokkos::create_mirror_view(d_eiBuf1);

    for(int i=0; i<length;   i++) h_diag(i) = diagonal[i];
    for(int i=0; i<length-1; i++) h_offD(i) = offDiagonal[i];
    Kokkos::deep_copy(d_diagonal, h_diag);
    Kokkos::deep_copy(d_offDiag,  h_offD);

    // Pointers that swap each iteration (like double-buffering)
    Kokkos::View<float*> *d_in  = &d_eiBuf0;
    Kokkos::View<float*> *d_out = &d_eiBuf1;
    unsigned int in = 0; // tracks which eigenIntervals[] host array is current

    // Lambda to run one full bisection pass (uploads eigenIntervals[in], iterates)
    auto runKernels = [&]() {
      // Reset: upload current eigenIntervals[in] to d_in
      auto h_in = Kokkos::create_mirror_view(*d_in);
      for(int i=0; i<2*length; i++) h_in(i) = eigenIntervals[in][i];
      Kokkos::deep_copy(*d_in, h_in);

      auto h_out = Kokkos::create_mirror_view(*d_out);
      for(int i=0; i<2*length; i++) h_out(i) = eigenIntervals[1-in][i];
      Kokkos::deep_copy(*d_out, h_out);

      in = 0;

      // Upload eigenIntervals[0] as starting buffer
      for(int i=0; i<2*length; i++) h_in(i) = eigenIntervals[0][i];
      Kokkos::deep_copy(*d_in, h_in);

      while(isComplete(eigenIntervals[in], length, tolerance)) {
        calNumEigenValueInterval_kokkos(d_numEigen, *d_in,
                                        d_diagonal, d_offDiag, (unsigned int)length);
        recalculateEigenIntervals_kokkos(*d_out, *d_in, d_numEigen,
                                         d_diagonal, d_offDiag,
                                         (unsigned int)length, tolerance);

        // Swap buffers
        std::swap(d_in, d_out);
        in = 1 - in;

        // Download current d_in (which just became d_out after swap, but we need
        // to check isComplete on it) — download the NEWLY computed intervals
        auto h_cur = Kokkos::create_mirror_view(*d_in);
        Kokkos::deep_copy(h_cur, *d_in);
        for(int i=0; i<2*length; i++) eigenIntervals[in][i] = h_cur(i);
      }
    };

    // Warmup
    for(int w=0; w<2 && iterations!=1; w++) runKernels();

    std::cout << "Executing kernel for " << iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    for(int iter=0; iter<iterations; iter++) runKernels();
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-3f;
    std::cout << "Average kernel execution time " << elapsed_us / iterations << " (us)\n";

    // in is the index of the current (final) eigenIntervals on host
    // Verify
    float *verificationEigenIntervals[2];
    verificationEigenIntervals[0] = (float*)malloc(2*length * sizeof(float));
    verificationEigenIntervals[1] = (float*)malloc(2*length * sizeof(float));

    float ll, ul;
    computeGerschgorinInterval(&ll, &ul, diagonal, offDiagonal, length);
    unsigned int verIn = 0;
    verificationEigenIntervals[verIn][0] = ll;
    verificationEigenIntervals[verIn][1] = ul;
    for(int i=2; i<2*length; i++) verificationEigenIntervals[verIn][i] = ul;

    while(isComplete(verificationEigenIntervals[verIn], length, tolerance)) {
      eigenValueCPUReference(diagonal, offDiagonal, length,
          verificationEigenIntervals[verIn],
          verificationEigenIntervals[1-verIn],
          tolerance);
      verIn = 1 - verIn;
    }

    if(compare(eigenIntervals[in], verificationEigenIntervals[verIn], 2*length))
      std::cout << "PASS\n";
    else
      std::cout << "FAIL\n";

    free(verificationEigenIntervals[0]);
    free(verificationEigenIntervals[1]);
  }
  Kokkos::finalize();

  free(diagonal);
  free(offDiagonal);
  free(eigenIntervals[0]);
  free(eigenIntervals[1]);
  free(numEigenValuesIntervalBuffer);
  return 0;
}
