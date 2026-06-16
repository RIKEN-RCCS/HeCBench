/*
 * Portions Copyright (c) 1993-2015 NVIDIA Corporation.  All rights reserved.
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
 * Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
 *
 * Sobol Quasi-random Number Generator example
 *
 * Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
 * http://people.maths.ox.ac.uk/~gilesm/
 *
 * and C code developed by Stephen Joe, University of Waikato, New Zealand
 * and Frances Kuo, University of New South Wales, Australia
 * http://web.maths.unsw.edu.au/~fkuo/sobol/
 *
 * For theoretical background see:
 *
 * P. Bratley and B.L. Fox.
 * Implementing Sobol's quasirandom sequence generator
 * http://portal.acm.org/citation.cfm?id=42288
 * ACM Trans. on Math. Software, 14(1):88-100, 1988
 *
 * S. Joe and F. Kuo.
 * Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
 * http://portal.acm.org/citation.cfm?id=641879
 * ACM Trans. on Math. Software, 29(1):49-57, 2003
 */

#include <iostream>
#include <stdexcept>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#include "sobol.h"
#include "sobol_gold.h"
#include "sobol_primitives.h"

#define L1ERROR_TOLERANCE (1e-6)
#define k_2powneg32 2.3283064E-10F

// ---------------------------------------------------------------------------
// Host-only helpers (CPU gold reference)
// ---------------------------------------------------------------------------

// initSobolDirectionVectors and sobolCPU are compiled from sobol_gold.cpp
// (included via -I../sobol-omp and linked from the Makefile). We just
// declare them here via sobol_gold.h.

// ---------------------------------------------------------------------------
// Device helper: find-first-set (1-indexed position of lowest set bit)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int _ffs(const int x) {
  for (int i = 0; i < 32; i++)
    if ((x >> i) & 1) return (i + 1);
  return 0;
}

// ---------------------------------------------------------------------------
// GPU kernel: sobolGPU using Kokkos TeamPolicy + scratch memory
// ---------------------------------------------------------------------------
double sobolGPU(int repeat, int n_vectors, int n_dimensions,
                Kokkos::View<unsigned int*> d_dir,
                Kokkos::View<float*>        d_out)
{
  const int threadsperblock = 64;

  // Compute grid dimensions (same logic as OMP version)
  size_t dimGrid_y = n_dimensions;
  size_t dimGrid_x;

  if (n_dimensions < (4 * 24))
    dimGrid_x = 4 * 24;
  else
    dimGrid_x = 1;

  if (dimGrid_x > (unsigned int)(n_vectors / threadsperblock))
    dimGrid_x = (n_vectors + threadsperblock - 1) / threadsperblock;

  // Round dimGrid_x up to the next power of two
  unsigned int targetDimGridX = (unsigned int)dimGrid_x;
  for (dimGrid_x = 1; dimGrid_x < targetDimGridX; dimGrid_x *= 2);

  size_t numTeam = dimGrid_x * dimGrid_y;

  // Scratch memory: n_directions unsigned ints per team (level 0)
  using ScratchUInt = Kokkos::View<unsigned int*,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryUnmanaged>;
  size_t scratch_bytes = ScratchUInt::shmem_size(n_directions);

  using TeamPolicy = Kokkos::TeamPolicy<>;
  TeamPolicy policy((int)numTeam, threadsperblock);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  // Capture by value so lambdas are device-safe
  const int   nv         = n_vectors;
  const size_t dgx       = dimGrid_x;

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
    Kokkos::parallel_for("sobolGPU", policy,
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {

        const unsigned int teamX = (unsigned int)team.league_rank() % (unsigned int)dgx;
        const unsigned int teamY = (unsigned int)team.league_rank() / (unsigned int)dgx;

        // Per-team scratch for direction numbers
        ScratchUInt v(team.team_scratch(0), n_directions);

        // Load direction numbers cooperatively (first n_directions threads)
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, n_directions),
          [&](int i) {
            v(i) = d_dir(n_directions * teamY + i);
          });
        team.team_barrier();

        // Each thread computes its own independent strip of the sequence
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, threadsperblock),
          [&](int tidX) {
            const unsigned int threadSizeX = (unsigned int)threadsperblock;

            int i0     = (int)(teamX * threadSizeX) + tidX;
            int stride = (int)(dgx   * threadSizeX);

            // Gray code of the initial index
            unsigned int g = (unsigned int)i0 ^ ((unsigned int)i0 >> 1);

            // Initialise X for position i0 (Bratley & Fox eq. *)
            unsigned int X    = 0;
            unsigned int mask = 0;
            for (unsigned int k = 0; k < (unsigned int)(_ffs(stride) - 1); k++) {
              mask = -(g & 1u);
              X   ^= mask & v(k);
              g  >>= 1;
            }

            if (i0 < nv)
              d_out(nv * (int)teamY + i0) = (float)X * k_2powneg32;

            // Stride-based updates (Bratley & Fox eq. **)
            unsigned int v_log2stridem1 = v((unsigned int)(_ffs(stride) - 2));
            unsigned int v_stridemask   = (unsigned int)stride - 1u;

            for (unsigned int i = (unsigned int)i0 + (unsigned int)stride;
                 i < (unsigned int)nv;
                 i += (unsigned int)stride)
            {
              X ^= v_log2stridem1 ^
                   v((unsigned int)(_ffs(~((int)(i - (unsigned int)stride) |
                                           (int)v_stridemask)) - 1));
              d_out(nv * (int)teamY + (int)i) = (float)X * k_2powneg32;
            }
          }); // TeamThreadRange
      }); // parallel_for
  } // repeat loop
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of vectors> <number of dimensions> <repeat>\n", argv[0]);
    return 1;
  }

  int n_vectors    = atoi(argv[1]);
  int n_dimensions = atoi(argv[2]);
  int repeat       = atoi(argv[3]);

  std::cout << "Allocating CPU memory..." << std::endl;
  unsigned int *h_directions = nullptr;
  float        *h_outputCPU  = nullptr;
  float        *h_outputGPU  = nullptr;

  try {
    h_directions = new unsigned int[n_dimensions * n_directions];
    h_outputCPU  = new float[n_vectors * n_dimensions];
    h_outputGPU  = new float[n_vectors * n_dimensions];
  }
  catch (const std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::cerr << "Unable to allocate CPU memory (try fewer vectors/dimensions)\n";
    return EXIT_FAILURE;
  }

  std::cout << "Initializing direction numbers..." << std::endl;
  initSobolDirectionVectors(n_dimensions, h_directions);

  Kokkos::initialize(argc, argv);
  {
    // Allocate device views
    Kokkos::View<unsigned int*> d_directions("directions", n_dimensions * n_directions);
    Kokkos::View<float*>        d_output("output",     n_dimensions * n_vectors);

    // Copy direction numbers host → device
    {
      auto h_dir_um = Kokkos::View<unsigned int*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(h_directions,
                                                            n_dimensions * n_directions);
      Kokkos::deep_copy(d_directions, h_dir_um);
    }

    std::cout << "Executing QRNG on GPU..." << std::endl;
    double ktime = sobolGPU(repeat, n_vectors, n_dimensions, d_directions, d_output);
    std::cout << "Average kernel execution time: " << (ktime * 1e-9) / repeat << " (s)\n";

    // Copy results device → host
    {
      auto h_out_um = Kokkos::View<float*, Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>(h_outputGPU,
                                                            n_dimensions * n_vectors);
      Kokkos::deep_copy(h_out_um, d_output);
    }
  }
  Kokkos::finalize();

  std::cout << std::endl;

  // CPU reference
  std::cout << "Executing QRNG on CPU..." << std::endl;
  sobolCPU(n_vectors, n_dimensions, h_directions, h_outputCPU);

  // Check results
  std::cout << "Checking results..." << std::endl;
  float l1norm_diff = 0.0F;
  float l1norm_ref  = 0.0F;
  float l1error;

  if (n_vectors == 1) {
    for (int d = 0, v = 0; d < n_dimensions; d++) {
      float ref = h_outputCPU[d * n_vectors + v];
      l1norm_diff += fabsf(h_outputGPU[d * n_vectors + v] - ref);
      l1norm_ref  += fabsf(ref);
    }
    l1error = l1norm_diff;
    if (l1norm_ref != 0)
      std::cerr << "Error: L1-Norm of the reference is not zero (single vector)\n";
    else
      std::cout << "L1-Error: " << l1error << std::endl;
  }
  else {
    for (int d = 0; d < n_dimensions; d++) {
      for (int v = 0; v < n_vectors; v++) {
        float ref = h_outputCPU[d * n_vectors + v];
        l1norm_diff += fabsf(h_outputGPU[d * n_vectors + v] - ref);
        l1norm_ref  += fabsf(ref);
      }
    }
    l1error = l1norm_diff / l1norm_ref;
    if (l1norm_ref == 0)
      std::cerr << "Error: L1-Norm of the reference is zero\n";
    else
      std::cout << "L1-Error: " << l1error << std::endl;
  }

  std::cout << "Shutting down..." << std::endl;
  delete[] h_directions;
  delete[] h_outputCPU;
  delete[] h_outputGPU;

  if (l1error < L1ERROR_TOLERANCE)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  return 0;
}
