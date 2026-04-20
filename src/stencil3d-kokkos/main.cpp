#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// 2D block size
#define BSIZE 16
// Tile size in the x direction
#define XTILE 20

typedef float Real;

using exec_space  = Kokkos::DefaultExecutionSpace;
using mem_space   = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;
using ScratchView  = Kokkos::View<Real*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// sm_flat layout: 4 slots of BSIZE*BSIZE each.
// Slot 0,1,2: ping-pong for previous/current/next x-slices.
// Slot 3: temporary for y/z charge communication.
#define SM(s,y,z) sm_flat[(s)*BSIZE*BSIZE + (y)*BSIZE + (z)]

void stencil3d(
    Kokkos::View<const Real*> d_psi,
    Kokkos::View<Real*>       d_npsi,
    Kokkos::View<const Real*> d_sigma,  // size 9*vol: [sigmaX|sigmaY|sigmaZ], each 3*vol
    int bdimx, int bdimy, int bdimz,
    int nx, int ny, int nz)
{
  const int vol = nx * ny * nz;
  const int scratch_size = ScratchView::shmem_size(4 * BSIZE * BSIZE);

  Kokkos::parallel_for("stencil3d",
    Kokkos::TeamPolicy<exec_space>(bdimz * bdimy * bdimx, BSIZE * BSIZE)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<exec_space>::member_type& team) {

      ScratchView sm_flat(team.team_scratch(0), 4 * BSIZE * BSIZE);

      const int tid        = team.team_rank();
      const int tjj        = tid / BSIZE;
      const int tkk        = tid % BSIZE;
      const int blockIdx_x = team.league_rank() % bdimx;
      const int blockIdx_y = (team.league_rank() / bdimx) % bdimy;
      const int blockIdx_z = team.league_rank() / (bdimx * bdimy);
      const int gridDim_x  = bdimx;
      const int gridDim_y  = bdimy;
      const int gridDim_z  = bdimz;

      // Base offsets into the global arrays for this block
      const int base_x = XTILE    * blockIdx_x;
      const int base_y = (BSIZE-2) * blockIdx_y;
      const int base_z = (BSIZE-2) * blockIdx_z;

      // Global array accessors (shifted by block base)
      // psi(x,y,z) uses index: (base_z+z) + nz*((base_y+y) + ny*(base_x+x))
      auto psi_r = [&](int x, int y, int z) -> Real {
        return d_psi[(base_z + z) + nz * ((base_y + y) + ny * (base_x + x))];
      };
      auto npsi_w = [&](int x, int y, int z) -> Real& {
        return d_npsi[(base_z + z) + nz * ((base_y + y) + ny * (base_x + x))];
      };
      // sigmaX(x,y,z,dir) = d_sigma[base + dir*vol],  layout: z + nz*(y + ny*(x + nx*dir))
      // With block shift: d_sigma[(base_z+z) + nz*((base_y+y) + ny*(base_x+x + nx*dir))]
      auto sigX = [&](int x, int y, int z, int dir) -> Real {
        return d_sigma[0*vol + (base_z+z) + nz*((base_y+y) + ny*(base_x+x + nx*dir))];
      };
      auto sigY = [&](int x, int y, int z, int dir) -> Real {
        return d_sigma[3*vol + (base_z+z) + nz*((base_y+y) + ny*(base_x+x + nx*dir))];
      };
      auto sigZ = [&](int x, int y, int z, int dir) -> Real {
        return d_sigma[6*vol + (base_z+z) + nz*((base_y+y) + ny*(base_x+x + nx*dir))];
      };

      int nLast_x = XTILE+1,    nLast_y = (BSIZE-1), nLast_z = (BSIZE-1);
      if (blockIdx_x == gridDim_x-1) nLast_x = nx-2 - XTILE*(blockIdx_x) + 1;
      if (blockIdx_y == gridDim_y-1) nLast_y = ny-2 - (BSIZE-2)*blockIdx_y + 1;
      if (blockIdx_z == gridDim_z-1) nLast_z = nz-2 - (BSIZE-2)*blockIdx_z + 1;

      // previous, current, next, and temp slot indices
      int pii, cii, nii, tii;
      Real xcharge, ycharge, zcharge, dV = 0.f;

      if (tjj <= nLast_y && tkk <= nLast_z) {
        pii = 0; cii = 1; nii = 2;
        SM(cii, tjj, tkk) = psi_r(0, tjj, tkk);
        SM(nii, tjj, tkk) = psi_r(1, tjj, tkk);
      }
      team.team_barrier();

      // Compute initial x-face contribution (x=1, between x=0 and x=1)
      if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y)) {
        Real xd = -SM(cii,tjj,tkk) + SM(nii,tjj,tkk);
        Real yd = (-SM(cii,-1+tjj,tkk) + SM(cii,1+tjj,tkk)
                   -SM(nii,-1+tjj,tkk) + SM(nii,1+tjj,tkk)) / 4.f;
        Real zd = (-SM(cii,tjj,-1+tkk) + SM(cii,tjj,1+tkk)
                   -SM(nii,tjj,-1+tkk) + SM(nii,tjj,1+tkk)) / 4.f;
        dV -= sigX(1,tjj,tkk,0)*xd + sigX(1,tjj,tkk,1)*yd + sigX(1,tjj,tkk,2)*zd;
      }

      if (tjj <= nLast_y && tkk <= nLast_z) {
        tii = pii; pii = cii; cii = nii; nii = tii;
      }

      for (int ii = 1; ii < nLast_x; ii++) {
        if (tjj <= nLast_y && tkk <= nLast_z)
          SM(nii, tjj, tkk) = psi_r(ii+1, tjj, tkk);
        team.team_barrier();

        // y face current
        if ((tkk>0) && (tkk<nLast_z) && (tjj<nLast_y)) {
          Real xd = (-SM(pii,tjj,tkk) - SM(pii,1+tjj,tkk)
                     +SM(nii,tjj,tkk) + SM(nii,1+tjj,tkk)) / 4.f;
          Real yd = -SM(cii,tjj,tkk) + SM(cii,1+tjj,tkk);
          Real zd = (-SM(cii,tjj,-1+tkk) + SM(cii,tjj,1+tkk)
                     -SM(cii,1+tjj,-1+tkk) + SM(cii,1+tjj,1+tkk)) / 4.f;
          ycharge = sigY(ii,tjj+1,tkk,0)*xd + sigY(ii,tjj+1,tkk,1)*yd + sigY(ii,tjj+1,tkk,2)*zd;
          dV += ycharge;
          SM(3, tjj, tkk) = ycharge;
        }
        team.team_barrier();

        if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
          dV -= SM(3, tjj-1, tkk);
        team.team_barrier();

        // z face current
        if ((tkk<nLast_z) && (tjj>0) && (tjj<nLast_y)) {
          Real xd = (-SM(pii,tjj,tkk) - SM(pii,tjj,1+tkk)
                     +SM(nii,tjj,tkk) + SM(nii,tjj,1+tkk)) / 4.f;
          Real yd = (-SM(cii,-1+tjj,tkk) - SM(cii,-1+tjj,1+tkk)
                     +SM(cii,1+tjj,tkk) + SM(cii,1+tjj,1+tkk)) / 4.f;
          Real zd = -SM(cii,tjj,tkk) + SM(cii,tjj,1+tkk);
          zcharge = sigZ(ii,tjj,tkk+1,0)*xd + sigZ(ii,tjj,tkk+1,1)*yd + sigZ(ii,tjj,tkk+1,2)*zd;
          dV += zcharge;
          SM(3, tjj, tkk) = zcharge;
        }
        team.team_barrier();

        if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y))
          dV -= SM(3, tjj, tkk-1);
        team.team_barrier();

        // x face current — store result and pass charge to next cell
        if ((tkk>0) && (tkk<nLast_z) && (tjj>0) && (tjj<nLast_y)) {
          Real xd = -SM(cii,tjj,tkk) + SM(nii,tjj,tkk);
          Real yd = (-SM(cii,-1+tjj,tkk) + SM(cii,1+tjj,tkk)
                     -SM(nii,-1+tjj,tkk) + SM(nii,1+tjj,tkk)) / 4.f;
          Real zd = (-SM(cii,tjj,-1+tkk) + SM(cii,tjj,1+tkk)
                     -SM(nii,tjj,-1+tkk) + SM(nii,tjj,1+tkk)) / 4.f;
          xcharge = sigX(ii+1,tjj,tkk,0)*xd + sigX(ii+1,tjj,tkk,1)*yd + sigX(ii+1,tjj,tkk,2)*zd;
          dV += xcharge;
          npsi_w(ii, tjj, tkk) = dV;
          dV = -xcharge;
        }
        team.team_barrier();

        if (tjj <= nLast_y && tkk <= nLast_z) {
          tii = pii; pii = cii; cii = nii; nii = tii;
        }
      }
    });
}

#undef SM

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <grid dimension> <repeat>\n", argv[0]);
    return 1;
  }
  const int size   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int nx = size, ny = size, nz = size;
  const int vol = nx * ny * nz;
  printf("Grid dimension: nx=%d ny=%d nz=%d\n", nx, ny, nz);

  Real *h_Vm    = (Real*)malloc(sizeof(Real) * vol);
  Real *h_sigma = (Real*)malloc(sizeof(Real) * vol * 9);
  Real *h_dVm   = (Real*)malloc(sizeof(Real) * vol);

  for (int ii = 0; ii < nx; ii++)
    for (int jj = 0; jj < ny; jj++)
      for (int kk = 0; kk < nz; kk++)
        h_Vm[kk + nz*(jj + ny*ii)] = (ii*(ny*nz) + jj*nz + kk) % 19;

  for (int i = 0; i < vol*9; i++) h_sigma[i] = (Real)(i % 19);
  memset(h_dVm, 0, sizeof(Real) * vol);

  int bdimz = (nz-2)/(BSIZE-2) + ((nz-2)%(BSIZE-2)==0 ? 0 : 1);
  int bdimy = (ny-2)/(BSIZE-2) + ((ny-2)%(BSIZE-2)==0 ? 0 : 1);
  int bdimx = (nx-2)/XTILE     + ((nx-2)%XTILE==0     ? 0 : 1);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<Real*, mem_space> d_psi  ("psi",   vol);
    Kokkos::View<Real*, mem_space> d_npsi ("npsi",  vol);
    Kokkos::View<Real*, mem_space> d_sigma("sigma", vol * 9);

    // Host-to-device transfers
    {
      auto hv = Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_Vm, vol);
      Kokkos::deep_copy(d_psi, hv);
    }
    {
      auto hv = Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_sigma, vol*9);
      Kokkos::deep_copy(d_sigma, hv);
    }
    Kokkos::deep_copy(d_npsi, Real(0));

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stencil3d(d_psi, d_npsi, d_sigma, bdimx, bdimy, bdimz, nx, ny, nz);

    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    // Device-to-host
    {
      auto hv = Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h_dVm, vol);
      Kokkos::deep_copy(hv, d_npsi);
    }
  }
  Kokkos::finalize();

#ifdef DUMP
  for (int ii = 0; ii < nx; ii++)
    for (int jj = 0; jj < ny; jj++)
      for (int kk = 0; kk < nz; kk++)
        printf("dVm (%d,%d,%d)=%e\n", ii, jj, kk, h_dVm[kk+nz*(jj+ny*ii)]);
#endif

  free(h_sigma);
  free(h_Vm);
  free(h_dVm);
  return 0;
}
