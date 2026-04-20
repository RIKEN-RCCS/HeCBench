// Mersenne Twister RNG + Box-Muller - Kokkos port
// Ported from mt-omp; data files read from ../mt-omp/data/

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;

#include "../mt-omp/MT.h"
#include "../mt-omp/dci.h"
#include "../mt-omp/genmtrand.cpp"
#include "../mt-omp/MT_gold.cpp"

// -------------------------------------------------------------------------
// Box-Muller transformation (device-callable)
// -------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void BoxMullerTrans(float *u1, float *u2)
{
  const float   r = Kokkos::sqrt(-2.0f * Kokkos::log(*u1));
  const float phi = 2.0f * PI * (*u2);
  *u1 = r * Kokkos::cos(phi);
  *u2 = r * Kokkos::sin(phi);
}

// -------------------------------------------------------------------------
// Load MT parameters from file
// -------------------------------------------------------------------------
static void loadMTGPU(const char *fname,
                      unsigned int seed,
                      mt_struct_stripped *h_MT,
                      size_t size)
{
  FILE *fd = fopen(fname, "rb");
  if (!fd) { printf("Failed to open %s\n", fname); exit(-1); }
  for (unsigned int i = 0; i < size; i++)
    fread(&h_MT[i], sizeof(mt_struct_stripped), 1, fd);
  fclose(fd);
  for (unsigned int i = 0; i < size; i++)
    h_MT[i].seed = seed;
}

// -------------------------------------------------------------------------

int main(int argc, const char **argv)
{
  if (argc != 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  int numIterations = atoi(argv[1]);

  const int    seed    = 777;
  const int    nPerRng = 5860;
  const int    nRand   = MT_RNG_COUNT * nPerRng;

  printf("Initialization: load MT parameters and init host buffers...\n");

  mt_struct_stripped *h_MT =
      (mt_struct_stripped *)malloc(sizeof(mt_struct_stripped) * MT_RNG_COUNT);
  loadMTGPU("../mt-omp/data/MersenneTwister.dat", seed, h_MT, MT_RNG_COUNT);

  initMTRef("../mt-omp/data/MersenneTwister.raw");

  float *h_RandGPU = (float *)malloc(sizeof(float) * nRand);
  float *h_RandCPU = (float *)malloc(sizeof(float) * nRand);

  Kokkos::initialize(argc, const_cast<char **>(argv));
  {
    printf("Allocate device memory...\n");

    // Device views
    Kokkos::View<mt_struct_stripped *> d_MT("d_MT", MT_RNG_COUNT);
    Kokkos::View<float *>              d_Rand("d_Rand", nRand);

    // Copy MT parameters to device
    {
      auto h = Kokkos::create_mirror_view(d_MT);
      for (int i = 0; i < MT_RNG_COUNT; i++) h(i) = h_MT[i];
      Kokkos::deep_copy(d_MT, h);
    }

    printf("Call Mersenne Twister kernel... (%d iterations)\n\n", numIterations);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (int iter = 0; iter < numIterations; iter++) {

      // ---- MT generation kernel ----
      Kokkos::parallel_for(
          "mt_generate", MT_RNG_COUNT, KOKKOS_LAMBDA(int globalID) {
            int      iState, iState1, iStateM, iOut;
            unsigned mti, mti1, mtiM, x;
            unsigned mt[MT_NN], matrix_a, mask_b, mask_c;

            matrix_a = d_MT(globalID).matrix_a;
            mask_b   = d_MT(globalID).mask_b;
            mask_c   = d_MT(globalID).mask_c;

            // Initialise state
            mt[0] = d_MT(globalID).seed;
            for (iState = 1; iState < MT_NN; iState++)
              mt[iState] =
                  (1812433253U *
                       (mt[iState - 1] ^ (mt[iState - 1] >> 30)) +
                   iState) &
                  MT_WMASK;

            iState = 0;
            mti1   = mt[0];

            for (iOut = 0; iOut < nPerRng; iOut++) {
              iState1 = iState + 1;
              iStateM = iState + MT_MM;
              if (iState1 >= MT_NN) iState1 -= MT_NN;
              if (iStateM >= MT_NN) iStateM -= MT_NN;
              mti  = mti1;
              mti1 = mt[iState1];
              mtiM = mt[iStateM];

              // MT recurrence
              x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
              x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);
              mt[iState] = x;
              iState     = iState1;

              // Tempering
              x ^= (x >> MT_SHIFT0);
              x ^= (x << MT_SHIFTB) & mask_b;
              x ^= (x << MT_SHIFTC) & mask_c;
              x ^= (x >> MT_SHIFT1);

              // Store as (0,1] float
              d_Rand(globalID + iOut * MT_RNG_COUNT) =
                  ((float)x + 1.0f) / 4294967296.0f;
            }
          });
      Kokkos::fence();

      // ---- Box-Muller kernel ----
      Kokkos::parallel_for(
          "box_muller", MT_RNG_COUNT, KOKKOS_LAMBDA(int globalID) {
            for (int iOut = 0; iOut < nPerRng; iOut += 2) {
              BoxMullerTrans(
                  &d_Rand(globalID + (iOut + 0) * MT_RNG_COUNT),
                  &d_Rand(globalID + (iOut + 1) * MT_RNG_COUNT));
            }
          });
      Kokkos::fence();
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    double gpuTime = time_span.count() / (double)numIterations;
    printf("MersenneTwister, Throughput = %.4f GNumbers/s, "
           "Time = %.5f s, Size = %u Numbers, Workgroup = %d\n",
           ((double)nRand * 1.0e-9 / gpuTime), gpuTime, nRand, 128);

    printf("\nRead back results...\n");
    {
      auto h = Kokkos::create_mirror_view(d_Rand);
      Kokkos::deep_copy(h, d_Rand);
      for (int i = 0; i < nRand; i++) h_RandGPU[i] = h(i);
    }
  }
  Kokkos::finalize();

  printf("Compute CPU reference solution...\n");
  RandomRef(h_RandCPU, nPerRng, seed);
  BoxMullerRef(h_RandCPU, nPerRng);

  printf("Compare CPU and GPU results...\n");
  double sum_delta = 0, sum_ref = 0;
  for (int i = 0; i < MT_RNG_COUNT; i++)
    for (int j = 0; j < nPerRng; j++) {
      double rCPU = h_RandCPU[i * nPerRng + j];
      double rGPU = h_RandGPU[i + j * MT_RNG_COUNT];
      sum_delta += fabs(rCPU - rGPU);
      sum_ref   += fabs(rCPU);
    }
  double L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n\n", L1norm);

  free(h_MT);
  free(h_RandGPU);
  free(h_RandCPU);

  printf("%s\n", (L1norm < 1e-6) ? "PASS" : "FAIL");
  return 0;
}
