#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// D3Q19 velocity set (from dirs[] in the original CUDA kernels.h)
//
//  i  cx  cy  cz    opposite
//  0   0   0   0       0
//  1  -1   0   0      10
//  2   0  -1   0      11
//  3   0   0  -1      12
//  4  -1  -1   0      13
//  5  -1   1   0      14
//  6  -1   0  -1      15
//  7  -1   0   1      16
//  8   0  -1  -1      17
//  9   0  -1   1      18
// 10   1   0   0       1
// 11   0   1   0       2
// 12   0   0   1       3
// 13   1   1   0       4
// 14   1  -1   0       5
// 15   1   0   1       6
// 16   1   0  -1       7
// 17   0   1   1       8
// 18   0   1  -1       9
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION int d3q19_cx(int i) {
  constexpr int v[19] = {0,-1,0,0,-1,-1,-1,-1,0,0,1,0,0,1,1,1,1,0,0};
  return v[i];
}
KOKKOS_INLINE_FUNCTION int d3q19_cy(int i) {
  constexpr int v[19] = {0,0,-1,0,-1,1,0,0,-1,-1,0,1,0,1,-1,0,0,1,1};
  return v[i];
}
KOKKOS_INLINE_FUNCTION int d3q19_cz(int i) {
  constexpr int v[19] = {0,0,0,-1,0,0,-1,1,-1,1,0,0,1,0,0,1,-1,1,-1};
  return v[i];
}
// IBAR(i,19) = ((i+8)%18)+1  — opposite direction
KOKKOS_INLINE_FUNCTION int d3q19_opp(int i) {
  constexpr int v[19] = {0,10,11,12,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9};
  return v[i];
}
// Weights: w[0]=1/3, w[1..3,10..12]=1/18, w[4..9,13..18]=1/36
KOKKOS_INLINE_FUNCTION double d3q19_w(int i) {
  if (i == 0) return 1.0 / 3.0;
  if (i >= 1 && i <= 3)  return 1.0 / 18.0;
  if (i >= 4 && i <= 9)  return 1.0 / 36.0;
  if (i >= 10 && i <= 12) return 1.0 / 18.0;
  return 1.0 / 36.0; // 13..18
}

// Second-order BGK equilibrium: f_eq = w * rho * (1 + 3*cu + 4.5*cu^2 - 1.5*usq)
KOKKOS_INLINE_FUNCTION
double d3q19_feq(int i, double rho, double ux, double uy, double uz) {
  double cu  = d3q19_cx(i) * ux + d3q19_cy(i) * uy + d3q19_cz(i) * uz;
  double usq = ux * ux + uy * uy + uz * uz;
  return d3q19_w(i) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
}

// Macroscopic quantities from populations
KOKKOS_INLINE_FUNCTION
void macroscopic(const double f[19],
                 double& rho, double& ux, double& uy, double& uz) {
  double xm = f[1]+f[4]+f[5]+f[6]+f[7];
  double xp = f[10]+f[13]+f[14]+f[15]+f[16];
  double ym = f[2]+f[4]+f[8]+f[9]+f[14];
  double yp = f[5]+f[11]+f[13]+f[17]+f[18];
  double zm = f[3]+f[6]+f[8]+f[16]+f[18];
  double zp = f[7]+f[9]+f[12]+f[15]+f[17];
  rho = xm + xp + f[0]+f[2]+f[3]+f[8]+f[9]+f[11]+f[12]+f[17]+f[18];
  double inv = 1.0 / rho;
  ux = (xp - xm) * inv;
  uy = (yp - ym) * inv;
  uz = (zp - zm) * inv;
}

// Population array layout: f[dir * nz*ny*nx  +  z*ny*nx  +  y*nx  +  x]
// (Structure of Arrays, matching IDF macro in the CUDA code)
KOKKOS_INLINE_FUNCTION
int idf(int x, int y, int z, int i, int nx, int ny, int nz) {
  return i * (nz * ny * nx) + z * (ny * nx) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Collision-streaming kernel
// ---------------------------------------------------------------------------
// For each fluid cell (interior: 1 <= x <= nx-2, 1 <= y <= ny-2, 1 <= z <= nz-2):
//   Pull populations from upstream neighbours (streaming).
//   If upstream cell is a wall: apply bounce-back.
//   If upstream cell is the z=0 moving lid: add velocity correction.
//   Compute macroscopic quantities and BGK equilibrium.
//   Write post-collision populations to f1.
//
// Moving lid: z=0 moves in +x direction with velocity ulb.
// Bottom wall: z=nz-1 is stationary.
// x, y faces: stationary walls.
// ---------------------------------------------------------------------------
void collide_and_stream(
    Kokkos::View<double*> f0,
    Kokkos::View<double*> f1,
    int nx, int ny, int nz,
    double omega, double ulb)
{
  Kokkos::parallel_for(
    "lbm_step",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1},{nz-1,ny-1,nx-1}),
    KOKKOS_LAMBDA(int z, int y, int x) {
      double fin[19];

      for (int i = 0; i < 19; i++) {
        int xs = x - d3q19_cx(i);
        int ys = y - d3q19_cy(i);
        int zs = z - d3q19_cz(i);

        // Determine if source cell is a wall
        bool is_wall = (xs < 1 || xs > nx-2 ||
                        ys < 1 || ys > ny-2 ||
                        zs < 1 || zs > nz-2);

        if (!is_wall) {
          // Regular streaming: pull population i from upstream cell
          fin[i] = f0(idf(xs, ys, zs, i, nx, ny, nz));
        } else {
          // Bounce-back: reflect using opposite direction from current cell
          fin[i] = f0(idf(x, y, z, d3q19_opp(i), nx, ny, nz));

          // Moving-lid correction (z=0 lid moves in +x with speed ulb)
          // Only applies when the wall is z=0 (cz[i]==1 for cell at z==1)
          if (zs == 0) {
            int cxi = d3q19_cx(i);
            // Non-zero corrections only for directions 7 (cx=-1,cz=1)
            // and 15 (cx=1,cz=1): fin[i] += ±2*w[i]*3*ulb
            if      (cxi == -1) fin[i] -= 2.0 * (1.0/36.0) * 3.0 * ulb;
            else if (cxi ==  1) fin[i] += 2.0 * (1.0/36.0) * 3.0 * ulb;
          }
        }
      }

      // Macroscopic quantities
      double rho, ux, uy, uz;
      macroscopic(fin, rho, ux, uy, uz);

      // BGK collision and write to f1
      for (int i = 0; i < 19; i++) {
        double feq = d3q19_feq(i, rho, ux, uy, uz);
        f1(idf(x, y, z, i, nx, ny, nz)) =
          (1.0 - omega) * fin[i] + omega * feq;
      }
    });
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 5) {
      printf("Usage: %s <nx> <ny> <nz> <iterations>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int nx         = atoi(argv[1]);
    const int ny         = atoi(argv[2]);
    const int nz         = atoi(argv[3]);
    const int iterations = atoi(argv[4]);
    const int nl         = nx * ny * nz;

    // Physical parameters (lid-driven cavity at Re=100, like the CUDA code)
    const double ulb   = 0.02;
    const double nu    = ulb * (ny - 2) / 100.0;
    const double omega = 1.0 / (3.0 * nu + 0.5);
    printf("nx=%d ny=%d nz=%d  omega=%.6f\n", nx, ny, nz, omega);

    // Population arrays (SoA)
    Kokkos::View<double*> f0("f0", 19 * nl);
    Kokkos::View<double*> f1("f1", 19 * nl);

    // Initialise all populations with equilibrium at rest (rho=1, u=0)
    Kokkos::parallel_for("init", nl, KOKKOS_LAMBDA(int flat) {
      int x = flat % nx;
      int y = (flat / nx) % ny;
      int z = flat / (nx * ny);
      for (int i = 0; i < 19; i++)
        f0(idf(x, y, z, i, nx, ny, nz)) = d3q19_feq(i, 1.0, 0.0, 0.0, 0.0);
    });
    Kokkos::fence();

    // Warm-up (100 iterations, not timed)
    const int warmup = 100;
    for (int t = 0; t < warmup; t++) {
      collide_and_stream(f0, f1, nx, ny, nz, omega, ulb);
      Kokkos::fence();
      Kokkos::deep_copy(f0, f1);
    }
    printf("Warmup (%d iters) done.\n", warmup);

    // Timed benchmark
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; t++) {
      collide_and_stream(f0, f1, nx, ny, nz, omega, ulb);
      Kokkos::fence();
      // Swap f0 ↔ f1 (double buffering)
      Kokkos::deep_copy(f0, f1);
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double mlups   = ((double)nl * iterations / elapsed) / 1.0e6;
    printf("Time: %.3f s  MLUPS: %.2f\n", elapsed, mlups);

    // Sanity check: sum of all populations should be conserved (≈ nl)
    double pop_sum = 0.0;
    Kokkos::parallel_reduce("sum", nl, KOKKOS_LAMBDA(int flat, double& acc) {
      int x = flat % nx;
      int y = (flat / nx) % ny;
      int z = flat / (nx * ny);
      for (int i = 0; i < 19; i++)
        acc += f1(idf(x, y, z, i, nx, ny, nz));
    }, pop_sum);
    Kokkos::fence();
    printf("Population sum: %.6f  (expected ≈ %d)\n", pop_sum, nl);
  }
  Kokkos::finalize();
  return 0;
}
