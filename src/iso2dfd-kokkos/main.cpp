//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// ISO2DFD: the 2D-Finite-Difference-Wave Propagation,
//
// ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic
// isotropic wave equation. Kernels in this sample are implemented as 2nd order
// in space, 2nd order in time scheme without boundary conditions.
//
// Kokkos port: OMP target offloading replaced with Kokkos::parallel_for,
// raw device arrays replaced with Kokkos::View.

#include <fstream>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "iso2dfd.h"

/*
 * Host-Code
 * Utility function to display input arguments
 */
void usage(std::string programName) {
  std::cout << " Incorrect parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " n1 n2 Iterations " << std::endl << std::endl;
  std::cout << " n1 n2      : Grid sizes for the stencil " << std::endl;
  std::cout << " Iterations : No. of timesteps. " << std::endl;
}

/*
 * Host-Code
 * Function used for initialization
 */
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel,
                size_t nRows, size_t nCols) {
  std::cout << "Initializing ... " << std::endl;

  // Define source wavelet
  float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                       0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                       0.023680172, 0.005611435,  0.001823209,  -0.000720549};

  // Initialize arrays
  for (size_t i = 0; i < nRows; i++) {
    size_t offset = i * nCols;
    for (size_t k = 0; k < nCols; k++) {
      ptr_prev[offset + k] = 0.0f;
      ptr_next[offset + k] = 0.0f;
      // pre-compute squared value of sample wave velocity v*v (v = 1500 m/s)
      ptr_vel[offset + k] = 2250000.0f;
    }
  }
  // Add a source to initial wavefield as an initial condition
  for (int s = 11; s >= 0; s--) {
    for (size_t i = nRows / 2 - s; i < nRows / 2 + s; i++) {
      size_t offset = i * nCols;
      for (size_t k = nCols / 2 - s; k < nCols / 2 + s; k++) {
        ptr_prev[offset + k] = wavelet[s];
      }
    }
  }
}

/*
 * Host-Code
 * Utility function to calculate L2-norm between resulting buffer and reference
 * buffer
 */
bool within_epsilon(float* output, float* reference, const size_t dimx,
                    const size_t dimy, const unsigned int radius,
                    const float delta = 0.01f) {
  FILE* fp = fopen("./error_diff.txt", "w");
  if (!fp) fp = stderr;

  bool error = false;
  double norm2 = 0;

  for (size_t iy = 0; iy < dimy; iy++) {
    for (size_t ix = 0; ix < dimx; ix++) {
      if (ix >= radius && ix < (dimx - radius) && iy >= radius &&
          iy < (dimy - radius)) {
        float difference = fabsf(*reference - *output);
        norm2 += difference * difference;
        if (difference > delta) {
          error = true;
          fprintf(fp, " ERROR: (%zu,%zu)\t%e instead of %e (|e|=%e)\n", ix, iy,
                  *output, *reference, difference);
        }
      }
      ++output;
      ++reference;
    }
  }

  if (fp != stderr) fclose(fp);
  norm2 = sqrt(norm2);
  if (error) printf("error (Euclidean norm): %.9e\n", norm2);
  return error;
}

/*
 * Host-Code
 * CPU implementation for wavefield modeling
 * Updates wavefield for the number of iterations given in nIterations
 */
void iso_2dfd_iteration_cpu(float* next, float* prev, float* vel,
                             const float dtDIVdxy, int nRows, int nCols,
                             int nIterations) {
  for (unsigned int k = 0; k < (unsigned int)nIterations; k += 1) {
    for (size_t i = 1; i < (size_t)nRows - HALF_LENGTH; i += 1) {
      for (size_t j = 1; j < (size_t)nCols - HALF_LENGTH; j += 1) {
        size_t gid = j + (i * nCols);
        float value = 0.f;
        value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
        value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
        value *= dtDIVdxy * vel[gid];
        next[gid] = 2.f * prev[gid] - next[gid] + value;
      }
    }
    // Swap arrays
    float* swap = next;
    next = prev;
    prev = swap;
  }
}

int main(int argc, char* argv[]) {
  float* prev_base;
  float* next_base;
  float* next_cpu;
  float* vel_base;

  bool error = false;

  size_t nRows, nCols;
  unsigned int nIterations;

  // Read parameters
  try {
    nRows      = std::stoi(argv[1]);
    nCols      = std::stoi(argv[2]);
    nIterations = std::stoi(argv[3]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t nsize = nRows * nCols;

  // Allocate host arrays
  prev_base = new float[nsize];
  next_base = new float[nsize];
  next_cpu  = new float[nsize];
  vel_base  = new float[nsize];

  // Compute constant value (delta t)^2 / (delta x)^2
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  // Initialize arrays and introduce initial conditions (source)
  initialize(prev_base, next_base, vel_base, nRows, nCols);

  std::cout << "Grid Sizes: " << nRows << " " << nCols << std::endl;
  std::cout << "Iterations: " << nIterations << std::endl;
  std::cout << std::endl;

  std::cout << "Computing wavefield in device .." << std::endl;

  Kokkos::initialize(argc, argv);
  {
    using ViewType = Kokkos::View<float*>;

    // Allocate device views
    ViewType d_prev("prev", nsize);
    ViewType d_next("next", nsize);
    ViewType d_vel("vel",  nsize);

    // Create host mirrors and copy initial data to device
    auto h_prev = Kokkos::create_mirror_view(d_prev);
    auto h_next = Kokkos::create_mirror_view(d_next);
    auto h_vel  = Kokkos::create_mirror_view(d_vel);

    for (size_t i = 0; i < nsize; i++) {
      h_prev(i) = prev_base[i];
      h_next(i) = next_base[i];
      h_vel(i)  = vel_base[i];
    }

    Kokkos::deep_copy(d_prev, h_prev);
    Kokkos::deep_copy(d_next, h_next);
    Kokkos::deep_copy(d_vel,  h_vel);

    auto kstart = std::chrono::steady_clock::now();

    // Iterate over time steps, alternating next/prev each iteration
    for (unsigned int k = 0; k < nIterations; k += 1) {
      // Mirror the OMP alternation: even k writes to d_next, odd k writes to d_prev
      ViewType& cur_next = (k % 2) ? d_prev : d_next;
      ViewType& cur_prev = (k % 2) ? d_next : d_prev;

      const int nR = (int)nRows;
      const int nC = (int)nCols;

      // Only launch threads over the interior (non-halo) region
      Kokkos::parallel_for(
        "iso2dfd",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          {HALF_LENGTH, HALF_LENGTH},
          {nR - HALF_LENGTH, nC - HALF_LENGTH}),
        KOKKOS_LAMBDA(const int gidRow, const int gidCol) {
          const int gid = gidRow * nC + gidCol;
          float value = 0.f;
          value += cur_prev(gid + 1)  - 2.f * cur_prev(gid) + cur_prev(gid - 1);
          value += cur_prev(gid + nC) - 2.f * cur_prev(gid) + cur_prev(gid - nC);
          value *= dtDIVdxy * d_vel(gid);
          cur_next(gid) = 2.f * cur_prev(gid) - cur_next(gid) + value;
        });
    }

    Kokkos::fence();

    auto kend  = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    std::cout << "Total kernel execution time "
              << ktime * 1e-6f << " (ms)\n";
    std::cout << "Average kernel execution time "
              << (ktime * 1e-3f) / nIterations << " (us)\n";

    // Copy d_next back to host next_base.
    // The alternation writes to d_next on even k and d_prev on odd k —
    // matching the OMP version which maps both buffers tofrom but compares
    // next_base vs next_cpu (both last written on the same even-k iteration).
    auto h_next_out = Kokkos::create_mirror_view(d_next);
    Kokkos::deep_copy(h_next_out, d_next);
    for (size_t i = 0; i < nsize; i++) {
      next_base[i] = h_next_out(i);
    }
  }
  Kokkos::finalize();

  // Output final wavefield (computed by device) to binary file
  std::ofstream outFile;
  outFile.open("wavefield_snapshot.bin", std::ios::out | std::ios::binary);
  outFile.write(reinterpret_cast<char*>(next_base), nsize * sizeof(float));
  outFile.close();

  // Compute wavefield on CPU (for validation)
  std::cout << "Computing wavefield in CPU .." << std::endl;
  // Re-initialize arrays
  initialize(prev_base, next_cpu, vel_base, nRows, nCols);

  auto start = std::chrono::steady_clock::now();
  iso_2dfd_iteration_cpu(next_cpu, prev_base, vel_base, dtDIVdxy,
                         nRows, nCols, nIterations);
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "CPU time: " << time << " ms" << std::endl;
  std::cout << std::endl;

  // Compute error (difference between device and CPU wavefields)
  error = within_epsilon(next_base, next_cpu, nRows, nCols, HALF_LENGTH, 0.1f);

  if (error)
    std::cout << "Final wavefields from device and CPU are different: Error"
              << std::endl;
  else
    std::cout << "Final wavefields from device and CPU are equivalent: Success"
              << std::endl;

  // Output final wavefield (computed by CPU) to binary file
  outFile.open("wavefield_snapshot_cpu.bin", std::ios::out | std::ios::binary);
  outFile.write(reinterpret_cast<char*>(next_cpu), nsize * sizeof(float));
  outFile.close();

  std::cout << "Final wavefields (from device and CPU) written to disk"
            << std::endl;
  std::cout << "Finished.  " << std::endl;

  // Cleanup
  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;
  delete[] next_cpu;

  return error ? 1 : 0;
}
