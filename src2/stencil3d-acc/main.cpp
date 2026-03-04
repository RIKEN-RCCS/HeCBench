#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include <openacc.h>

#define BSIZE 16
#define XTILE 20

typedef float Real;

void stencil3d(
    const Real*__restrict d_psi,
          Real*__restrict d_npsi,
    const Real*__restrict d_sigmaX,
    const Real*__restrict d_sigmaY,
    const Real*__restrict d_sigmaZ,
    int bdimx, int bdimy, int bdimz,
    int nx, int ny, int nz)
{
  const int total_teams = bdimz * bdimy * bdimx;

  #pragma acc parallel loop gang num_gangs(total_teams) vector_length(BSIZE*BSIZE) \
          present(d_psi[0:nx*ny*nz], d_npsi[0:nx*ny*nz], \
                  d_sigmaX[0:3*nx*ny*nz], d_sigmaY[0:3*nx*ny*nz], d_sigmaZ[0:3*nx*ny*nz])
  for (int team = 0; team < total_teams; ++team)
  {
    
	  Real sm_psi[4][BSIZE][BSIZE];
    Real dV_lane[BSIZE][BSIZE];

    const int blockIdx_x = team % bdimx;
    const int blockIdx_y = (team / bdimx) % bdimy;
    const int blockIdx_z = team / (bdimx * bdimy);

    const int gridDim_x = bdimx;
    const int gridDim_y = bdimy;
    const int gridDim_z = bdimz;

    const int base_x = XTILE * blockIdx_x;
    const int base_y = (BSIZE - 2) * blockIdx_y;
    const int base_z = (BSIZE - 2) * blockIdx_z;

    const int base_off = base_z + nz * (base_y + ny * base_x);

    const Real* __restrict psi_ptr   = d_psi   + base_off;
          Real* __restrict npsi_ptr  = d_npsi  + base_off;
    const Real* __restrict sigX_ptr  = d_sigmaX + base_off;
    const Real* __restrict sigY_ptr  = d_sigmaY + base_off;
    const Real* __restrict sigZ_ptr  = d_sigmaZ + base_off;

    // Local macros (x,y,z are LOCAL coordinates inside the shifted tile)
    #define PSI(x,y,z)   psi_ptr[ (z) + nz * ( (y) + ny * (x) ) ]
    #define NPSI(x,y,z)  npsi_ptr[ (z) + nz * ( (y) + ny * (x) ) ]

    // sigma arrays have an extra (dir) dimension stored as x + nx*dir (same as original)
    #define SIGX(x,y,z,dir) sigX_ptr[ (z) + nz * ( (y) + ny * ( (x) + nx * (dir) ) ) ]
    #define SIGY(x,y,z,dir) sigY_ptr[ (z) + nz * ( (y) + ny * ( (x) + nx * (dir) ) ) ]
    #define SIGZ(x,y,z,dir) sigZ_ptr[ (z) + nz * ( (y) + ny * ( (x) + nx * (dir) ) ) ]

    int nLast_x = XTILE + 1;
    int nLast_y = (BSIZE - 1);
    int nLast_z = (BSIZE - 1);

    if (blockIdx_x == gridDim_x - 1) nLast_x = (nx - 2) - XTILE * blockIdx_x + 1;
    if (blockIdx_y == gridDim_y - 1) nLast_y = (ny - 2) - (BSIZE - 2) * blockIdx_y + 1;
    if (blockIdx_z == gridDim_z - 1) nLast_z = (nz - 2) - (BSIZE - 2) * blockIdx_z + 1;

    // Shared (per-gang) indices for the sliding window planes.
    int pii = 0, cii = 1, nii = 2, tii = 0;

    #pragma acc loop vector
    for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
      const int tjj = tid / BSIZE;
      const int tkk = tid % BSIZE;

      // Initialize per-lane accumulator (thread-private in the OpenMP version)
      dV_lane[tjj][tkk] = (Real)0;

      if (tjj <= nLast_y && tkk <= nLast_z) {
        sm_psi[cii][tjj][tkk] = PSI(0, tjj, tkk);
        sm_psi[nii][tjj][tkk] = PSI(1, tjj, tkk);
      }
    }

    #pragma acc loop vector
    for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
      const int tjj = tid / BSIZE;
      const int tkk = tid % BSIZE;

      if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
        const Real V1 = sm_psi[cii][tjj][tkk];
        const Real V2 = sm_psi[nii][tjj][tkk];

        const Real xd = -V1 + V2;
        const Real yd = (-sm_psi[cii][tjj-1][tkk] + sm_psi[cii][tjj+1][tkk]
                         -sm_psi[nii][tjj-1][tkk] + sm_psi[nii][tjj+1][tkk]) / (Real)4;
        const Real zd = (-sm_psi[cii][tjj][tkk-1] + sm_psi[cii][tjj][tkk+1]
                         -sm_psi[nii][tjj][tkk-1] + sm_psi[nii][tjj][tkk+1]) / (Real)4;

        dV_lane[tjj][tkk] -= SIGX(1, tjj, tkk, 0) * xd
                           + SIGX(1, tjj, tkk, 1) * yd
                           + SIGX(1, tjj, tkk, 2) * zd;
      }
    }

    #pragma acc loop vector
    for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
      if (tid == 0) {
        tii = pii; pii = cii; cii = nii; nii = tii;
      }
    }

    for (int ii = 1; ii < nLast_x; ++ii)
    {
      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        const int tjj = tid / BSIZE;
        const int tkk = tid % BSIZE;
        if (tjj <= nLast_y && tkk <= nLast_z) {
          sm_psi[nii][tjj][tkk] = PSI(ii + 1, tjj, tkk);
        }
      }

      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        const int tjj = tid / BSIZE;
        const int tkk = tid % BSIZE;

        if ((tkk > 0) && (tkk < nLast_z) && (tjj < nLast_y)) {
          const Real V0  = sm_psi[pii][tjj][tkk];
          const Real V0y = sm_psi[pii][tjj+1][tkk];
          const Real V2  = sm_psi[nii][tjj][tkk];
          const Real V2y = sm_psi[nii][tjj+1][tkk];

          const Real xd = (-V0 - V0y + V2 + V2y) / (Real)4;

          const Real V1  = sm_psi[cii][tjj][tkk];
          const Real V1y = sm_psi[cii][tjj+1][tkk];
          const Real yd  = -V1 + V1y;

          const Real zd = (-sm_psi[cii][tjj][tkk-1] + sm_psi[cii][tjj][tkk+1]
                           -sm_psi[cii][tjj+1][tkk-1] + sm_psi[cii][tjj+1][tkk+1]) / (Real)4;

          const Real ycharge = SIGY(ii, tjj + 1, tkk, 0) * xd
                             + SIGY(ii, tjj + 1, tkk, 1) * yd
                             + SIGY(ii, tjj + 1, tkk, 2) * zd;

          dV_lane[tjj][tkk] += ycharge;
          sm_psi[3][tjj][tkk] = ycharge;
        }
      }

      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        const int tjj = tid / BSIZE;
        const int tkk = tid % BSIZE;

        if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
          dV_lane[tjj][tkk] -= sm_psi[3][tjj - 1][tkk];
        }
      }

      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        const int tjj = tid / BSIZE;
        const int tkk = tid % BSIZE;

        if ((tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
          const Real V0  = sm_psi[pii][tjj][tkk];
          const Real V0z = sm_psi[pii][tjj][tkk+1];
          const Real V2  = sm_psi[nii][tjj][tkk];
          const Real V2z = sm_psi[nii][tjj][tkk+1];

          const Real xd = (-V0 - V0z + V2 + V2z) / (Real)4;

          const Real yd = (-sm_psi[cii][tjj-1][tkk] - sm_psi[cii][tjj-1][tkk+1]
                           +sm_psi[cii][tjj+1][tkk] + sm_psi[cii][tjj+1][tkk+1]) / (Real)4;

          const Real V1  = sm_psi[cii][tjj][tkk];
          const Real V1z = sm_psi[cii][tjj][tkk+1];
          const Real zd  = -V1 + V1z;

          const Real zcharge = SIGZ(ii, tjj, tkk + 1, 0) * xd
                             + SIGZ(ii, tjj, tkk + 1, 1) * yd
                             + SIGZ(ii, tjj, tkk + 1, 2) * zd;

          dV_lane[tjj][tkk] += zcharge;
          sm_psi[3][tjj][tkk] = zcharge;
        }
      }

      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        const int tjj = tid / BSIZE;
        const int tkk = tid % BSIZE;

        if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
          dV_lane[tjj][tkk] -= sm_psi[3][tjj][tkk - 1];
        }
      }

      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        const int tjj = tid / BSIZE;
        const int tkk = tid % BSIZE;

        if ((tkk > 0) && (tkk < nLast_z) && (tjj > 0) && (tjj < nLast_y)) {
          const Real V1 = sm_psi[cii][tjj][tkk];
          const Real V2 = sm_psi[nii][tjj][tkk];

          const Real xd = -V1 + V2;
          const Real yd = (-sm_psi[cii][tjj-1][tkk] + sm_psi[cii][tjj+1][tkk]
                           -sm_psi[nii][tjj-1][tkk] + sm_psi[nii][tjj+1][tkk]) / (Real)4;
          const Real zd = (-sm_psi[cii][tjj][tkk-1] + sm_psi[cii][tjj][tkk+1]
                           -sm_psi[nii][tjj][tkk-1] + sm_psi[nii][tjj][tkk+1]) / (Real)4;

          const Real xcharge = SIGX(ii + 1, tjj, tkk, 0) * xd
                             + SIGX(ii + 1, tjj, tkk, 1) * yd
                             + SIGX(ii + 1, tjj, tkk, 2) * zd;

          dV_lane[tjj][tkk] += xcharge;
          NPSI(ii, tjj, tkk) = dV_lane[tjj][tkk];
          dV_lane[tjj][tkk] = -xcharge;
        }
      }

      #pragma acc loop vector
      for (int tid = 0; tid < BSIZE*BSIZE; ++tid) {
        if (tid == 0) {
          tii = pii; pii = cii; cii = nii; nii = tii;
        }
      }
    }

    #undef PSI
    #undef NPSI
    #undef SIGX
    #undef SIGY
    #undef SIGZ
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <grid dimension> <repeat>\n", argv[0]);
    return 1;
  }
  const int size   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int nx  = size;
  const int ny  = size;
  const int nz  = size;
  const int vol = nx * ny * nz;

  printf("Grid dimension: nx=%d ny=%d nz=%d\n", nx, ny, nz);

  // allocate and initialize Vm
  Real *h_Vm = (Real*)malloc(sizeof(Real) * vol);

  #define h_Vm_at(x,y,z) h_Vm[ (z) + nz * ( (y) + ny * (x) ) ]

  for (int ii = 0; ii < nx; ii++)
    for (int jj = 0; jj < ny; jj++)
      for (int kk = 0; kk < nz; kk++)
        h_Vm_at(ii, jj, kk) = (ii * (ny * nz) + jj * nz + kk) % 19;

  // allocate and initialize sigma
  Real *h_sigma = (Real*)malloc(sizeof(Real) * vol * 9);
  for (int i = 0; i < vol * 9; i++) h_sigma[i] = i % 19;

  // reset dVm
  Real *h_dVm = (Real*)malloc(sizeof(Real) * vol);
  memset(h_dVm, 0, sizeof(Real) * vol);

  // determine block sizes
  int bdimz = (nz - 2) / (BSIZE - 2) + (((nz - 2) % (BSIZE - 2) == 0) ? 0 : 1);
  int bdimy = (ny - 2) / (BSIZE - 2) + (((ny - 2) % (BSIZE - 2) == 0) ? 0 : 1);
  int bdimx = (nx - 2) / XTILE       + (((nx - 2) % XTILE == 0) ? 0 : 1);

  #pragma acc data copyin(h_Vm[0:vol], h_sigma[0:vol*9]) copy(h_dVm[0:vol])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stencil3d(h_Vm, h_dVm,
                h_sigma,            // sigmaX (dir 0..2)
                h_sigma + 3 * vol,  // sigmaY
                h_sigma + 6 * vol,  // sigmaZ
                bdimx, bdimy, bdimz, nx, ny, nz);

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

#ifdef DUMP
  for (int ii = 0; ii < nx; ii++)
    for (int jj = 0; jj < ny; jj++)
      for (int kk = 0; kk < nz; kk++)
        printf("dVm (%d,%d,%d)=%e\n", ii, jj, kk, h_dVm[kk + nz * (jj + ny * ii)]);
#endif

  free(h_sigma);
  free(h_Vm);
  free(h_dVm);

  return 0;
}
