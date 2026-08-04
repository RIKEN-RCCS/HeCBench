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
#include "sobol_gpu.h"

#define L1ERROR_TOLERANCE (1e-6)

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
