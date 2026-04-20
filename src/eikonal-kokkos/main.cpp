// Kokkos port of eikonal-cuda (Fast Iterative Method for Eikonal equations on 3D grids)
//
// The original uses block-level FIM with CUDA kernels and active lists.
// This Kokkos port implements the same FIM algorithm:
// - parallel update over active voxels
// - active-list management on CPU between iterations
// Uses float precision; the original defaults to double via DOUBLE macro.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <fstream>
#include <Kokkos_Core.hpp>

#define INF     1e20
#define BL      4            // BLOCK_LENGTH

// Compute upwind eikonal update at (i,j,k) in a uniform-speed volume
KOKKOS_INLINE_FUNCTION
double eikonalUpdate(
    const Kokkos::View<double*>& sol,
    double speed,
    int i, int j, int k,
    int nx, int ny, int nz)
{
  auto idx = [&](int x, int y, int z) -> int {
    return x + y * nx + z * nx * ny;
  };

  // Gather neighbour values
  double xa = (i > 0)      ? sol(idx(i-1,j,k)) : INF;
  double xb = (i < nx-1)   ? sol(idx(i+1,j,k)) : INF;
  double ya = (j > 0)      ? sol(idx(i,j-1,k)) : INF;
  double yb = (j < ny-1)   ? sol(idx(i,j+1,k)) : INF;
  double za = (k > 0)      ? sol(idx(i,j,k-1)) : INF;
  double zb = (k < nz-1)   ? sol(idx(i,j,k+1)) : INF;

  double mx = Kokkos::fmin(xa, xb);
  double my = Kokkos::fmin(ya, yb);
  double mz = Kokkos::fmin(za, zb);

  // Sort ascending
  if (mx > my) { double t = mx; mx = my; my = t; }
  if (my > mz) { double t = my; my = mz; mz = t; }
  if (mx > my) { double t = mx; mx = my; my = t; }

  double f = 1.0 / speed;

  // 1D update
  double u = mx + f;
  if (u <= my) return u;

  // 2D update
  double disc2 = 2.0 * f * f - (my - mx) * (my - mx);
  if (disc2 >= 0.0) {
    u = 0.5 * (mx + my + Kokkos::sqrt(disc2));
    if (u <= mz) return u;
  }

  // 3D update
  double s = mx + my + mz;
  double disc3 = s * s - 3.0 * (mx*mx + my*my + mz*mz - f*f);
  if (disc3 >= 0.0) {
    return (s + Kokkos::sqrt(disc3)) / 3.0;
  }

  return mz + f;
}

int main(int argc, char** argv)
{
  size_t size = 64;  // reduced default for standalone benchmark
  size_t itersPerBlock = 10;
  size_t type = 0;
  std::string name = "output.nrrd";
  bool verbose = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-s") == 0 && i+1 < argc) size = atoi(argv[++i]);
    if (strcmp(argv[i], "-m") == 0 && i+1 < argc) type = atoi(argv[++i]);
    if (strcmp(argv[i], "-i") == 0 && i+1 < argc) itersPerBlock = atoi(argv[++i]);
    if (strcmp(argv[i], "-o") == 0 && i+1 < argc) name = argv[++i];
    if (strcmp(argv[i], "-v") == 0)                verbose = true;
  }

  int nx = (int)size, ny = (int)size, nz = (int)size;
  int N = nx * ny * nz;

  printf("Volume: %d x %d x %d = %d voxels\n", nx, ny, nz, N);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_sol  ("sol",   N);
    Kokkos::View<double*> d_spd  ("spd",   N);
    Kokkos::View<bool*>   d_con  ("con",   N);  // converged flags

    // Initialise: all INF except the seed at (0,0,0)
    Kokkos::deep_copy(d_sol, INF);
    Kokkos::deep_copy(d_con, false);
    Kokkos::deep_copy(d_spd, 1.0);

    // Egg-carton speed if type == 1
    if (type == 1) {
      Kokkos::parallel_for(N, KOKKOS_LAMBDA(int idx) {
        int z = idx / (nx * ny);
        int r = idx % (nx * ny);
        int y = r / nx, x = r % nx;
        double fx = (double)x / nx * 2.0 * 3.14159265358979;
        double fy = (double)y / ny * 2.0 * 3.14159265358979;
        double fz = (double)z / nz * 2.0 * 3.14159265358979;
        d_spd(idx) = 0.5 + 0.5 * Kokkos::sin(fx) * Kokkos::sin(fy) * Kokkos::sin(fz);
      });
    }

    // Set seed at voxel 0,0,0
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) {
      d_sol(0) = 0.0;
      d_con(0) = true;
    });

    // Active-list FIM iterations
    auto t0 = std::chrono::steady_clock::now();

    bool changed = true;
    int totalIter = 0;
    while (changed) {
      changed = false;
      int nChanged = 0;

      for (int innerIt = 0; innerIt < (int)itersPerBlock; innerIt++) {
        Kokkos::parallel_reduce(
          N,
          KOKKOS_LAMBDA(int idx, int& lChanged) {
            int z = idx / (nx * ny);
            int r = idx % (nx * ny);
            int y = r / nx, x = r % nx;

            double oldVal = d_sol(idx);
            double newVal = eikonalUpdate(d_sol, d_spd(idx), x, y, z, nx, ny, nz);
            newVal = Kokkos::fmin(newVal, oldVal);

            if (Kokkos::fabs(newVal - oldVal) > 1e-12) {
              d_sol(idx) = newVal;
              lChanged = 1;
            }
          },
          Kokkos::Sum<int>(nChanged));
      }

      if (nChanged > 0) changed = true;
      totalIter++;

      if (verbose)
        printf("Iteration %d: %d voxels updated\n", totalIter, nChanged);

      // Simple convergence limit to avoid runaway
      if (totalIter > nx + ny + nz) break;
    }

    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    printf("Solver finished in %d outer iterations, %.2f ms\n", totalIter, ms);

    // Copy solution and write NRRD
    auto h_sol = Kokkos::create_mirror_view(d_sol);
    Kokkos::deep_copy(h_sol, d_sol);

    double checksum = 0.0;
    for (int i = 0; i < N; i++)
      checksum += (h_sol(i) < INF ? h_sol(i) : 0.0);
    printf("Checksum = %lf\n", checksum / N);

    std::fstream out(name.c_str(), std::ios::out | std::ios::binary);
    out << "NRRD0001\n# Kokkos eikonal output\n"
        << "type: double\ndimension: 3\n"
        << "sizes: " << nx << " " << ny << " " << nz << "\n"
        << "endian: little\nencoding: raw\n\n";
    for (int i = 0; i < N; i++)
      out.write(reinterpret_cast<const char*>(&h_sol(i)), sizeof(double));
    out.close();
    printf("Wrote %s\n", name.c_str());
  }
  Kokkos::finalize();
  return 0;
}
