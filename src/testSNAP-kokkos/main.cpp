// TestSNAP SNAP force kernel benchmark – Kokkos port
// Ported from testSNAP-omp by replacing OpenMP target offload with Kokkos.
//
// Original copyright (2019) Sandia Corporation – Zero Clause BSD License.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

// snap.h provides COMPLEX, SNA_BINDICES, index macros, and declarations.
#include "../testSNAP-omp/snap.h"

// Reference data
#if REFDATA_TWOJ == 14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ == 8
#include "refdata_2J8_W.h"
#elif REFDATA_TWOJ == 4
#include "refdata_2J4_W.h"
#else
#include "refdata_2J2_W.h"
#endif

int nsteps = 1;

// ============================================================================
// Host-only utility functions (copied verbatim from utils.cpp)
// ============================================================================

static const double nfac_table[] = {
  1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800,
  479001600, 6227020800, 87178291200, 1307674368000, 20922789888000,
  355687428096000, 6.402373705728e+15, 1.21645100408832e+17,
  2.43290200817664e+18, 5.10909421717094e+19, 1.12400072777761e+21,
  2.5852016738885e+22, 6.20448401733239e+23, 1.5511210043331e+25,
  4.03291461126606e+26, 1.08888694504184e+28, 3.04888344611714e+29,
  8.8417619937397e+30, 2.65252859812191e+32, 8.22283865417792e+33,
  2.63130836933694e+35, 8.68331761881189e+36, 2.95232799039604e+38,
  1.03331479663861e+40, 3.71993326789901e+41, 1.37637530912263e+43,
  5.23022617466601e+44, 2.03978820811974e+46,
};

double factorial(int n) {
  if (n < 0 || n > nmaxfactorial) { exit(1); }
  return nfac_table[n];
}

double deltacg(int j1, int j2, int j) {
  double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return sqrt(factorial((j1 + j2 - j) / 2) *
              factorial((j1 - j2 + j) / 2) *
              factorial((-j1 + j2 + j) / 2) / sfaccg);
}

int compute_ncoeff(int twojmax) {
  int ncount = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) ncount++;
  return ncount;
}

inline double elapsedTime(timeval start_time, timeval end_time) {
  return ((end_time.tv_sec - start_time.tv_sec) +
          1e-6 * (end_time.tv_usec - start_time.tv_usec));
}

void options(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
      printf("TestSNAP 1.0 (stand-alone SNAP force kernel)\n\n");
      printf("-ns, --nsteps <val>: set the number of force calls (default 1)\n");
      exit(0);
    } else if ((strcmp(argv[i], "-ns") == 0) ||
               (strcmp(argv[i], "--nsteps") == 0)) {
      nsteps = atoi(argv[++i]);
    } else {
      printf("ERROR: Unknown command line argument: %s\n", argv[i]);
      exit(1);
    }
  }
}

// ============================================================================
// Device-callable functions (annotated with KOKKOS_INLINE_FUNCTION)
// Originally wrapped with #pragma omp declare target in utils.cpp
// ============================================================================

KOKKOS_INLINE_FUNCTION
double compute_sfac(double r, double rcut, int switch_flag) {
  if (switch_flag == 0) return 1.0;
  if (switch_flag == 1) {
    if (r <= rmin0) return 1.0;
    else if (r > rcut) return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
double compute_dsfac(double r, double rcut, int switch_flag) {
  if (switch_flag == 0) return 0.0;
  if (switch_flag == 1) {
    if (r <= rmin0) return 0.0;
    else if (r > rcut) return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return -0.5 * sin((r - rmin0) * rcutfac) * rcutfac;
    }
  }
  return 0.0;
}

KOKKOS_INLINE_FUNCTION
void compute_duarray(
    const int natom, const int nbor,
    const int num_atoms, const int num_nbor,
    const int twojmax, const int idxdu_max,
    const int jdimpq, const int switch_flag,
    const double x, const double y, const double z,
    const double z0, const double r, const double dz0dr,
    const double wj_in, const double rcut,
    const double* rootpqarray,
    const COMPLEX* ulist, COMPLEX* dulist)
{
  double r0inv, dr0invdr;
  double a_r, a_i, b_r, b_i;
  double da_r[3], da_i[3], db_r[3], db_i[3];
  double dz0[3], dr0inv[3];
  double rootpq;
  int jju, jjup, jjdu, jjdup;

  double rinv  = 1.0 / r;
  double ux    = x * rinv;
  double uy    = y * rinv;
  double uz    = z * rinv;

  r0inv  = 1.0 / sqrt(r * r + z0 * z0);
  a_r    = z0 * r0inv;
  a_i    = -z  * r0inv;
  b_r    =  y  * r0inv;
  b_i    = -x  * r0inv;

  dr0invdr = -r0inv * r0inv * r0inv * (r + z0 * dz0dr);

  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;

  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  for (int k = 0; k < 3; k++) {
    da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
    da_i[k] = -z * dr0inv[k];
  }
  da_i[2] += -r0inv;

  for (int k = 0; k < 3; k++) {
    db_r[k] = y * dr0inv[k];
    db_i[k] = -x * dr0inv[k];
  }
  db_i[0] += -r0inv;
  db_r[1] +=  r0inv;

  for (int k = 0; k < 3; ++k)
    dulist[DULIST_INDEX(natom, nbor, 0, k)] = { 0.0, 0.0 };

  jju  = 1;
  jjdu = 1;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++) {
      for (int k = 0; k < 3; ++k)
        dulist[DULIST_INDEX(natom, nbor, jjdu, k)] = { 0.0, 0.0 };
      jju  += deljju;
      jjdu += deljju;
    }
    int ncolhalf = deljju / 2;
    jju  += deljju * ncolhalf;
  }

  jju   = 1; jjdu  = 1;
  jjup  = 0; jjdup = 0;
  for (int j = 1; j <= twojmax; j++) {
    int deljju  = j + 1;
    int deljjup = j;

    for (int mb = 0; 2 * mb < j; mb++) {
      for (int ma = 0; ma < j; ma++) {
        double up_r = ulist[ULIST_INDEX(natom, nbor, jjup)].re;
        double up_i = ulist[ULIST_INDEX(natom, nbor, jjup)].im;

        rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, j - mb)];
        for (int k = 0; k < 3; k++) {
          dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re +=
            rootpq * (da_r[k] * up_r + da_i[k] * up_i +
                      a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re +
                      a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
          dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im +=
            rootpq * (da_r[k] * up_i - da_i[k] * up_r +
                      a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im -
                      a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
        }

        rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, j - mb)];
        for (int k = 0; k < 3; k++) {
          dulist[DULIST_INDEX(natom, nbor, jjdu + 1, k)].re =
            -rootpq * (db_r[k] * up_r + db_i[k] * up_i +
                       b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re +
                       b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
          dulist[DULIST_INDEX(natom, nbor, jjdu + 1, k)].im =
            -rootpq * (db_r[k] * up_i - db_i[k] * up_r +
                       b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im -
                       b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
        }

        if (2 * (mb + 1) == j) {
          rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, mb + 1)];
          for (int k = 0; k < 3; k++) {
            dulist[DULIST_INDEX(natom, nbor, jjdu + deljju, k)].re +=
              rootpq * (db_r[k] * up_r - db_i[k] * up_i +
                        b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re -
                        b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
            dulist[DULIST_INDEX(natom, nbor, jjdu + deljju, k)].im +=
              rootpq * (db_r[k] * up_i + db_i[k] * up_r +
                        b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im +
                        b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
          }

          rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, mb + 1)];
          for (int k = 0; k < 3; k++) {
            dulist[DULIST_INDEX(natom, nbor, jjdu + 1 + deljju, k)].re =
              rootpq * (da_r[k] * up_r - da_i[k] * up_i +
                        a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re -
                        a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
            dulist[DULIST_INDEX(natom, nbor, jjdu + 1 + deljju, k)].im =
              rootpq * (da_r[k] * up_i + da_i[k] * up_r +
                        a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im +
                        a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
          }
        }

        jju++; jjup++; jjdu++; jjdup++;
      }
      jju++; jjdu++;
    }

    if (j % 2 == 0) { jju += deljju; jjdu += deljju; }
    int ncolhalf  = deljju  / 2;
    int ncolhalfp = deljjup / 2;
    jju   += deljju  * ncolhalf;
    jjup  += deljjup * ncolhalfp;
  }

  double sfac  = compute_sfac (r, rcut, switch_flag) * wj_in;
  double dsfac = compute_dsfac(r, rcut, switch_flag) * wj_in;

  jju  = 0; jjdu = 0;
  for (int j = 0; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].re =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].re * ux +
          sfac  * dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].re;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].im =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].im * ux +
          sfac  * dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].im;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].re =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].re * uy +
          sfac  * dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].re;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].im =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].im * uy +
          sfac  * dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].im;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].re =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].re * uz +
          sfac  * dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].re;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].im =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].im * uz +
          sfac  * dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].im;
        jju++; jjdu++;
      }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    options(argc, argv);

    const int switch_flag = 1;

    double elapsed_ui = 0.0, elapsed_yi = 0.0,
           elapsed_duidrj = 0.0, elapsed_deidrj = 0.0;

    const int ninside  = refdata.ninside;
    const int ncoeff   = refdata.ncoeff;
    const int nlocal   = refdata.nlocal;
    const int nghost   = refdata.nghost;
    const int ntotal   = nlocal + nghost;
    const int twojmax  = refdata.twojmax;
    const double rcutfac = refdata.rcutfac;

    const double wself    = 1.0;
    const int num_atoms   = nlocal;
    const int num_nbor    = ninside;

    double* coeffi = (double*) malloc(sizeof(double) * (ncoeff + 1));
    for (int icoeff = 0; icoeff < ncoeff + 1; icoeff++)
      coeffi[icoeff] = refdata.coeff[icoeff];
    double* beta = coeffi + 1;

    const int jdim = twojmax + 1;

    // ---- Build index lists (identical to original) -------------------------

    int* idxcg_block = (int*) malloc(sizeof(int) * jdim * jdim * jdim);
    int idxcg_count  = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
          idxcg_block[j1 + j2 * jdim + jdim * jdim * j] = idxcg_count;
          for (int m1 = 0; m1 <= j1; m1++)
            for (int m2 = 0; m2 <= j2; m2++)
              idxcg_count++;
        }
    const int idxcg_max = idxcg_count;

    int* idxu_block = (int*) malloc(sizeof(int) * jdim);
    int  idxu_count = 0;
    for (int j = 0; j <= twojmax; j++) {
      idxu_block[j] = idxu_count;
      for (int mb = 0; mb <= j; mb++)
        for (int ma = 0; ma <= j; ma++)
          idxu_count++;
    }
    const int idxu_max = idxu_count;

    int* ulist_parity = (int*) malloc(sizeof(int) * idxu_max);
    idxu_count = 0;
    for (int j = 0; j <= twojmax; j++) {
      int mbpar = 1;
      for (int mb = 0; mb <= j; mb++) {
        int mapar = mbpar;
        for (int ma = 0; ma <= j; ma++) {
          ulist_parity[idxu_count] = mapar;
          mapar = -mapar;
          idxu_count++;
        }
        mbpar = -mbpar;
      }
    }

    int* idxdu_block = (int*) malloc(sizeof(int) * jdim);
    int  idxdu_count = 0;
    for (int j = 0; j <= twojmax; j++) {
      idxdu_block[j] = idxdu_count;
      for (int mb = 0; 2 * mb <= j; mb++)
        for (int ma = 0; ma <= j; ma++)
          idxdu_count++;
    }
    const int idxdu_max = idxdu_count;

    int idxb_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
          if (j >= j1) idxb_count++;
    const int idxb_max = idxb_count;
    SNA_BINDICES* idxb = (SNA_BINDICES*) malloc(sizeof(SNA_BINDICES) * idxb_max);

    idxb_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
          if (j >= j1) {
            idxb[idxb_count].j1 = j1;
            idxb[idxb_count].j2 = j2;
            idxb[idxb_count].j  = j;
            idxb_count++;
          }

    int* idxb_block = (int*) malloc(sizeof(int) * jdim * jdim * jdim);
    idxb_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
          if (j < j1) continue;
          idxb_block[j1 * jdim * jdim + j2 * jdim + j] = idxb_count++;
        }

    int idxz_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
          for (int mb = 0; 2 * mb <= j; mb++)
            for (int ma = 0; ma <= j; ma++)
              idxz_count++;
    const int idxz_max = idxz_count;

    int*    idxz      = (int*)    malloc(sizeof(int)    * idxz_max * 9);
    double* idxzbeta  = (double*) malloc(sizeof(double) * idxz_max);
    int*    idxz_block= (int*)    malloc(sizeof(int)    * jdim * jdim * jdim);

    idxz_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
          idxz_block[j1 * jdim * jdim + j2 * jdim + j] = idxz_count;

          double betaj;
          if (j >= j1) {
            const int jjb = idxb_block[j1 * jdim * jdim + j2 * jdim + j];
            if (j1 == j) betaj = (j2 == j) ? 3 * beta[jjb] : 2 * beta[jjb];
            else         betaj = beta[jjb];
          } else if (j >= j2) {
            const int jjb = idxb_block[j * jdim * jdim + j2 * jdim + j1];
            betaj = (j2 == j) ? 2 * beta[jjb] * (j1 + 1) / (j + 1.0)
                               :     beta[jjb] * (j1 + 1) / (j + 1.0);
          } else {
            const int jjb = idxb_block[j2 * jdim * jdim + j * jdim + j1];
            betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
          }

          for (int mb = 0; 2 * mb <= j; mb++)
            for (int ma = 0; ma <= j; ma++) {
              idxz[IDXZ_INDEX(idxz_count, 0)] = j1;
              idxz[IDXZ_INDEX(idxz_count, 1)] = j2;
              idxz[IDXZ_INDEX(idxz_count, 2)] = j;

              int ma1min = MAX(0, (2 * ma - j - j2 + j1) / 2);
              idxz[IDXZ_INDEX(idxz_count, 3)] = ma1min;
              idxz[IDXZ_INDEX(idxz_count, 4)] = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
              idxz[IDXZ_INDEX(idxz_count, 5)] = MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;

              int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
              idxz[IDXZ_INDEX(idxz_count, 6)] = mb1min;
              idxz[IDXZ_INDEX(idxz_count, 7)] = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
              idxz[IDXZ_INDEX(idxz_count, 8)] = MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;

              idxzbeta[idxz_count] = betaj;
              idxz_count++;
            }
        }

    if (compute_ncoeff(twojmax) != ncoeff) {
      printf("ERROR: ncoeff from SNA does not match reference data\n");
      Kokkos::finalize();
      return 1;
    }

    double* rij    = (double*) malloc(sizeof(double) * num_atoms * num_nbor * 3);
    double* inside = (double*) malloc(sizeof(double) * num_atoms * num_nbor);
    double* wj     = (double*) malloc(sizeof(double) * num_atoms * num_nbor);
    double* rcutij = (double*) malloc(sizeof(double) * num_atoms * num_nbor);

    const int jdimpq = twojmax + 2;
    double* rootpqarray = (double*) malloc(sizeof(double) * jdimpq * jdimpq);
    double* cglist      = (double*) malloc(sizeof(double) * idxcg_max);
    double* dedr        = (double*) malloc(sizeof(double) * num_atoms * num_nbor * 3);

    COMPLEX* ulist    = (COMPLEX*) malloc(sizeof(COMPLEX) * num_atoms * num_nbor * idxu_max);
    COMPLEX* ylist    = (COMPLEX*) malloc(sizeof(COMPLEX) * num_atoms * idxdu_max);
    COMPLEX* ulisttot = (COMPLEX*) malloc(sizeof(COMPLEX) * num_atoms * idxu_max);
    COMPLEX* dulist   = (COMPLEX*) malloc(sizeof(COMPLEX) * num_atoms * num_nbor * 3 * idxdu_max);

    // Init rootpqarray
    for (int p = 1; p <= twojmax; p++)
      for (int q = 1; q <= twojmax; q++)
        rootpqarray[ROOTPQ_INDEX(p, q)] = sqrt((double)p / q);

    // Init Clebsch-Gordan coefficients
    idxcg_count = 0;
    for (int j1 = 0; j1 <= twojmax; j1++)
      for (int j2 = 0; j2 <= j1; j2++)
        for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
          for (int m1 = 0; m1 <= j1; m1++) {
            int aa2 = 2 * m1 - j1;
            for (int m2 = 0; m2 <= j2; m2++) {
              int bb2 = 2 * m2 - j2;
              int m   = (aa2 + bb2 + j) / 2;
              if (m < 0 || m > j) { cglist[idxcg_count++] = 0.0; continue; }

              double sum = 0.0;
              for (int z = MAX(0, MAX(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
                   z <= MIN((j1 + j2 - j) / 2,
                             MIN((j1 - aa2) / 2, (j2 + bb2) / 2)); z++) {
                int ifac = z % 2 ? -1 : 1;
                sum += ifac / (factorial(z) *
                               factorial((j1 + j2 - j) / 2 - z) *
                               factorial((j1 - aa2) / 2 - z) *
                               factorial((j2 + bb2) / 2 - z) *
                               factorial((j - j2 + aa2) / 2 + z) *
                               factorial((j - j1 - bb2) / 2 + z));
              }
              int cc2 = 2 * m - j;
              double dcg     = deltacg(j1, j2, j);
              double sfaccg  = sqrt(
                factorial((j1 + aa2) / 2) * factorial((j1 - aa2) / 2) *
                factorial((j2 + bb2) / 2) * factorial((j2 - bb2) / 2) *
                factorial((j + cc2) / 2)  * factorial((j - cc2) / 2)  *
                (j + 1));
              cglist[idxcg_count++] = sum * dcg * sfaccg;
            }
          }
        }

    double* f = (double*) malloc(sizeof(double) * ntotal * 3);

    // ---- Allocate Kokkos device Views for offloaded arrays -----------------

    using exec_space  = Kokkos::DefaultExecutionSpace;
    using host_mirror = Kokkos::DefaultHostExecutionSpace;

    // Views for arrays that stay resident on device across kernels
    Kokkos::View<int*,    exec_space> d_idxu_block   ("idxu_block",    jdim);
    Kokkos::View<int*,    exec_space> d_ulist_parity  ("ulist_parity",  idxu_max);
    Kokkos::View<double*, exec_space> d_rootpqarray   ("rootpqarray",   jdimpq * jdimpq);
    Kokkos::View<int*,    exec_space> d_idxz          ("idxz",          idxz_max * 9);
    Kokkos::View<double*, exec_space> d_idxzbeta      ("idxzbeta",      idxz_max);
    Kokkos::View<int*,    exec_space> d_idxcg_block   ("idxcg_block",   jdim * jdim * jdim);
    Kokkos::View<int*,    exec_space> d_idxdu_block   ("idxdu_block",   jdim);
    Kokkos::View<double*, exec_space> d_cglist        ("cglist",        idxcg_max);
    Kokkos::View<COMPLEX*,exec_space> d_ulist         ("ulist",         (size_t)num_atoms * num_nbor * idxu_max);
    Kokkos::View<COMPLEX*,exec_space> d_dulist        ("dulist",        (size_t)num_atoms * num_nbor * 3 * idxdu_max);
    Kokkos::View<double*, exec_space> d_dedr          ("dedr",          (size_t)num_atoms * num_nbor * 3);
    Kokkos::View<COMPLEX*,exec_space> d_ulisttot      ("ulisttot",      (size_t)num_atoms * idxu_max);
    Kokkos::View<COMPLEX*,exec_space> d_ylist         ("ylist",         (size_t)num_atoms * idxdu_max);
    Kokkos::View<double*, exec_space> d_rij           ("rij",           (size_t)num_atoms * num_nbor * 3);
    Kokkos::View<double*, exec_space> d_rcutij        ("rcutij",        (size_t)num_atoms * num_nbor);
    Kokkos::View<double*, exec_space> d_wj            ("wj",            (size_t)num_atoms * num_nbor);

    // Copy static arrays to device
    {
      auto copy = [](auto& dv, const auto* src) {
        auto hv = Kokkos::create_mirror_view(dv);
        for (int i = 0; i < (int)dv.extent(0); i++) hv(i) = src[i];
        Kokkos::deep_copy(dv, hv);
      };
      copy(d_idxu_block,  idxu_block);
      copy(d_ulist_parity,ulist_parity);
      copy(d_rootpqarray, rootpqarray);
      copy(d_idxz,        idxz);
      copy(d_idxzbeta,    idxzbeta);
      copy(d_idxcg_block, idxcg_block);
      copy(d_idxdu_block, idxdu_block);
      copy(d_cglist,      cglist);
    }

    // Initialize ulist/dulist/dedr host values to zero on device
    Kokkos::deep_copy(d_ulist,  COMPLEX{0.0, 0.0});
    Kokkos::deep_copy(d_dulist, COMPLEX{0.0, 0.0});
    Kokkos::deep_copy(d_dedr,   0.0);

    // ---- Main step loop -----------------------------------------------------

    double sumsqferr = 0.0;

    auto begin = myclock::now();
    for (int istep = 0; istep < nsteps; istep++) {

      // Reset force array on host
      for (int j = 0; j < ntotal * 3; j++) f[j] = 0.0;

      // Fill rij, inside, wj, rcutij from reference data
      int jt = 0, jjt = 0;
      for (int natom = 0; natom < num_atoms; natom++) {
        for (int nbor = 0; nbor < num_nbor; nbor++) {
          rij   [ULIST_INDEX(natom, nbor, 0)] = refdata.rij[jt++];
          rij   [ULIST_INDEX(natom, nbor, 1)] = refdata.rij[jt++];
          rij   [ULIST_INDEX(natom, nbor, 2)] = refdata.rij[jt++];
          inside[INDEX_2D(natom, nbor)]        = refdata.jlist[jjt++];
          wj    [INDEX_2D(natom, nbor)]        = 1.0;
          rcutij[INDEX_2D(natom, nbor)]        = rcutfac;
        }
      }

      // Copy per-step data to device
      {
        auto hv_rij   = Kokkos::create_mirror_view(d_rij);
        auto hv_rcutij= Kokkos::create_mirror_view(d_rcutij);
        auto hv_wj    = Kokkos::create_mirror_view(d_wj);
        for (int i = 0; i < num_atoms * num_nbor * 3; i++) hv_rij(i)    = rij[i];
        for (int i = 0; i < num_atoms * num_nbor;     i++) hv_rcutij(i) = rcutij[i];
        for (int i = 0; i < num_atoms * num_nbor;     i++) hv_wj(i)     = wj[i];
        Kokkos::deep_copy(d_rij,    hv_rij);
        Kokkos::deep_copy(d_rcutij, hv_rcutij);
        Kokkos::deep_copy(d_wj,     hv_wj);
      }

      // ---- Get raw device pointers for use in kernels ---------------------
      int*     p_idxu_block  = d_idxu_block.data();
      int*     p_ulist_parity= d_ulist_parity.data();
      double*  p_rootpqarray = d_rootpqarray.data();
      int*     p_idxz        = d_idxz.data();
      double*  p_idxzbeta    = d_idxzbeta.data();
      int*     p_idxcg_block = d_idxcg_block.data();
      int*     p_idxdu_block = d_idxdu_block.data();
      double*  p_cglist      = d_cglist.data();
      COMPLEX* p_ulist       = d_ulist.data();
      COMPLEX* p_dulist      = d_dulist.data();
      double*  p_dedr        = d_dedr.data();
      COMPLEX* p_ulisttot    = d_ulisttot.data();
      COMPLEX* p_ylist       = d_ylist.data();
      double*  p_rij         = d_rij.data();
      double*  p_rcutij      = d_rcutij.data();
      double*  p_wj          = d_wj.data();

      // ==== compute_ui =====================================================
      using tp_start = std::chrono::system_clock;
      auto start_tp = tp_start::now();

      // Zero ulisttot
      Kokkos::parallel_for("ui_zero_tot",
        Kokkos::RangePolicy<>(0, num_atoms * idxu_max),
        KOKKOS_LAMBDA(int i) { p_ulisttot[i] = {0.0, 0.0}; });

      // Set diagonal elements
      Kokkos::parallel_for("ui_diag",
        Kokkos::RangePolicy<>(0, num_atoms),
        KOKKOS_LAMBDA(int natom) {
          for (int j = 0; j <= twojmax; j++) {
            int jju = p_idxu_block[j];
            for (int ma = 0; ma <= j; ma++) {
              p_ulisttot[INDEX_2D(natom, jju)] = {wself, 0.0};
              jju += j + 2;
            }
          }
        });

      // Main ulist accumulation (collapse(2) over nbor, natom)
      Kokkos::parallel_for("ui_accumulate",
        Kokkos::RangePolicy<>(0, num_nbor * num_atoms),
        KOKKOS_LAMBDA(int idx) {
          int nbor  = idx / num_atoms;
          int natom = idx % num_atoms;

          double x   = p_rij[ULIST_INDEX(natom, nbor, 0)];
          double y   = p_rij[ULIST_INDEX(natom, nbor, 1)];
          double z   = p_rij[ULIST_INDEX(natom, nbor, 2)];
          double rsq = x * x + y * y + z * z;
          double r   = sqrt(rsq);

          double theta0 = (r - rmin0) * rfac0 * MY_PI /
                          (p_rcutij[INDEX_2D(natom, nbor)] - rmin0);
          double z0 = r / tan(theta0);

          double r0inv = 1.0 / sqrt(r * r + z0 * z0);
          double a_r = r0inv * z0;
          double a_i = -r0inv * z;
          double b_r = r0inv * y;
          double b_i = -r0inv * x;

          double sfac = compute_sfac(r, p_rcutij[INDEX_2D(natom, nbor)], switch_flag)
                        * p_wj[INDEX_2D(natom, nbor)];

          // Initialise first entry
          p_ulist[ULIST_INDEX(natom, nbor, 0)].re = 1.0;
          p_ulist[ULIST_INDEX(natom, nbor, 0)].im = 0.0;

          // Zero top-row entries
          int jju = 1;
          for (int j = 1; j <= twojmax; j++) {
            int deljju = j + 1;
            for (int mb = 0; 2 * mb <= j; mb++) {
              p_ulist[ULIST_INDEX(natom, nbor, jju)].re = 0.0;
              p_ulist[ULIST_INDEX(natom, nbor, jju)].im = 0.0;
              jju += deljju;
            }
            int ncolhalf = deljju / 2;
            jju += deljju * ncolhalf;
          }

          jju = 1; int jjup = 0;
          for (int j = 1; j <= twojmax; j++) {
            int deljju  = j + 1;
            int deljjup = j;
            int mb_max  = (j + 1) / 2;
            int ma_max  = j;
            int m_max   = ma_max * mb_max;

            for (int m_iter = 0; m_iter < m_max; ++m_iter) {
              int mb = m_iter / ma_max;
              int ma = m_iter % ma_max;

              double up_r = p_ulist[ULIST_INDEX(natom, nbor, jjup)].re;
              double up_i = p_ulist[ULIST_INDEX(natom, nbor, jjup)].im;
              double rootpq;

              rootpq = p_rootpqarray[ROOTPQ_INDEX(j - ma, j - mb)];
              p_ulist[ULIST_INDEX(natom, nbor, jju)].re +=
                rootpq * (a_r * up_r + a_i * up_i);
              p_ulist[ULIST_INDEX(natom, nbor, jju)].im +=
                rootpq * (a_r * up_i - a_i * up_r);

              rootpq = p_rootpqarray[ROOTPQ_INDEX(ma + 1, j - mb)];
              p_ulist[ULIST_INDEX(natom, nbor, jju + 1)].re =
                -rootpq * (b_r * up_r + b_i * up_i);
              p_ulist[ULIST_INDEX(natom, nbor, jju + 1)].im =
                -rootpq * (b_r * up_i - b_i * up_r);

              if (2 * (mb + 1) == j) {
                rootpq = p_rootpqarray[ROOTPQ_INDEX(j - ma, mb + 1)];
                p_ulist[ULIST_INDEX(natom, nbor, jju + deljju)].re +=
                  rootpq * (b_r * up_r - b_i * up_i);
                p_ulist[ULIST_INDEX(natom, nbor, jju + deljju)].im +=
                  rootpq * (b_r * up_i + b_i * up_r);

                rootpq = p_rootpqarray[ROOTPQ_INDEX(ma + 1, mb + 1)];
                p_ulist[ULIST_INDEX(natom, nbor, jju + deljju + 1)].re =
                  rootpq * (a_r * up_r - a_i * up_i);
                p_ulist[ULIST_INDEX(natom, nbor, jju + deljju + 1)].im =
                  rootpq * (a_r * up_i + a_i * up_r);
              }

              jju++; jjup++;
              if (ma == ma_max - 1) jju++;
            }

            // Copy left side to right with inversion symmetry
            int jjui  = p_idxu_block[j];
            int jjuip = jjui + (j + 1) * (j + 1) - 1;
            for (int mb = 0; 2 * mb < j; mb++) {
              for (int ma = 0; ma <= j; ma++) {
                p_ulist[ULIST_INDEX(natom, nbor, jjuip)].re =
                  p_ulist_parity[jjui] * p_ulist[ULIST_INDEX(natom, nbor, jjui)].re;
                p_ulist[ULIST_INDEX(natom, nbor, jjuip)].im =
                  p_ulist_parity[jjui] * (-p_ulist[ULIST_INDEX(natom, nbor, jjui)].im);
                jjui++; jjuip--;
              }
            }

            if (j % 2 == 0) jju += deljju;
            int ncolhalf  = deljju  / 2;
            int ncolhalfp = deljjup / 2;
            jju  += deljju  * ncolhalf;
            jjup += deljjup * ncolhalfp;
          }

          // Accumulate into ulisttot (atomic)
          sfac = compute_sfac(r, p_rcutij[INDEX_2D(natom, nbor)], switch_flag)
                 * p_wj[INDEX_2D(natom, nbor)];
          for (int j = 0; j <= twojmax; j++) {
            int jju_acc = p_idxu_block[j];
            for (int mb = 0; mb <= j; mb++)
              for (int ma = 0; ma <= j; ma++) {
                Kokkos::atomic_add(
                  &p_ulisttot[INDEX_2D(natom, jju_acc)].re,
                  sfac * p_ulist[ULIST_INDEX(natom, nbor, jju_acc)].re);
                Kokkos::atomic_add(
                  &p_ulisttot[INDEX_2D(natom, jju_acc)].im,
                  sfac * p_ulist[ULIST_INDEX(natom, nbor, jju_acc)].im);
                jju_acc++;
              }
          }
        });
      Kokkos::fence();
      {
        auto end_tp = tp_start::now();
        elapsed_ui += std::chrono::duration<double>(end_tp - start_tp).count();
      }

      // ==== compute_yi =====================================================
      start_tp = tp_start::now();

      Kokkos::parallel_for("yi_zero",
        Kokkos::RangePolicy<>(0, num_atoms * idxdu_max),
        KOKKOS_LAMBDA(int i) { p_ylist[i] = {0.0, 0.0}; });

      Kokkos::parallel_for("yi_accumulate",
        Kokkos::RangePolicy<>(0, idxz_max * num_atoms),
        KOKKOS_LAMBDA(int idx) {
          int jjz   = idx / num_atoms;
          int natom = idx % num_atoms;

          const int j1     = p_idxz[IDXZ_INDEX(jjz, 0)];
          const int j2     = p_idxz[IDXZ_INDEX(jjz, 1)];
          const int j      = p_idxz[IDXZ_INDEX(jjz, 2)];
          const int ma1min = p_idxz[IDXZ_INDEX(jjz, 3)];
          const int ma2max = p_idxz[IDXZ_INDEX(jjz, 4)];
          const int na     = p_idxz[IDXZ_INDEX(jjz, 5)];
          const int mb1min = p_idxz[IDXZ_INDEX(jjz, 6)];
          const int mb2max = p_idxz[IDXZ_INDEX(jjz, 7)];
          const int nb     = p_idxz[IDXZ_INDEX(jjz, 8)];

          const double betaj = p_idxzbeta[jjz];
          const double* cgblock =
            p_cglist + p_idxcg_block[j1 + jdim * j2 + jdim * jdim * j];

          int mb = (2 * (mb1min + mb2max) - j1 - j2 + j) / 2;
          int ma = (2 * (ma1min + ma2max) - j1 - j2 + j) / 2;
          const int jjdu = p_idxdu_block[j] + (j + 1) * mb + ma;

          int jju1 = p_idxu_block[j1] + (j1 + 1) * mb1min;
          int jju2 = p_idxu_block[j2] + (j2 + 1) * mb2max;
          int icgb = mb1min * (j2 + 1) + mb2max;

          double ztmp_r = 0.0, ztmp_i = 0.0;

          for (int ib = 0; ib < nb; ib++) {
            double suma1_r = 0.0, suma1_i = 0.0;
            int ma1  = ma1min, ma2 = ma2max;
            int icga = ma1min * (j2 + 1) + ma2max;

            for (int ia = 0; ia < na; ia++) {
              suma1_r += cgblock[icga] *
                (p_ulisttot[INDEX_2D(natom, jju1 + ma1)].re *
                 p_ulisttot[INDEX_2D(natom, jju2 + ma2)].re -
                 p_ulisttot[INDEX_2D(natom, jju1 + ma1)].im *
                 p_ulisttot[INDEX_2D(natom, jju2 + ma2)].im);
              suma1_i += cgblock[icga] *
                (p_ulisttot[INDEX_2D(natom, jju1 + ma1)].re *
                 p_ulisttot[INDEX_2D(natom, jju2 + ma2)].im +
                 p_ulisttot[INDEX_2D(natom, jju1 + ma1)].im *
                 p_ulisttot[INDEX_2D(natom, jju2 + ma2)].re);
              ma1++; ma2--; icga += j2;
            }

            ztmp_r += cgblock[icgb] * suma1_r;
            ztmp_i += cgblock[icgb] * suma1_i;
            jju1 += j1 + 1;
            jju2 -= j2 + 1;
            icgb += j2;
          }

          Kokkos::atomic_add(&p_ylist[INDEX_2D(natom, jjdu)].re, betaj * ztmp_r);
          Kokkos::atomic_add(&p_ylist[INDEX_2D(natom, jjdu)].im, betaj * ztmp_i);
        });
      Kokkos::fence();
      {
        auto end_tp = tp_start::now();
        elapsed_yi += std::chrono::duration<double>(end_tp - start_tp).count();
      }

      // ==== compute_duidrj =================================================
      start_tp = tp_start::now();

      Kokkos::parallel_for("duidrj",
        Kokkos::RangePolicy<>(0, num_nbor * num_atoms),
        KOKKOS_LAMBDA(int idx) {
          int nbor  = idx / num_atoms;
          int natom = idx % num_atoms;

          double wj_in = p_wj   [INDEX_2D(natom, nbor)];
          double rcut  = p_rcutij[INDEX_2D(natom, nbor)];
          double x = p_rij[ULIST_INDEX(natom, nbor, 0)];
          double y = p_rij[ULIST_INDEX(natom, nbor, 1)];
          double z = p_rij[ULIST_INDEX(natom, nbor, 2)];
          double rsq = x * x + y * y + z * z;
          double r   = sqrt(rsq);
          double rscale0 = rfac0 * MY_PI / (rcut - rmin0);
          double theta0  = (r - rmin0) * rscale0;
          double cs  = cos(theta0);
          double sn  = sin(theta0);
          double z0  = r * cs / sn;
          double dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;

          compute_duarray(natom, nbor, num_atoms, num_nbor,
                          twojmax, idxdu_max, jdimpq, switch_flag,
                          x, y, z, z0, r, dz0dr, wj_in, rcut,
                          p_rootpqarray, p_ulist, p_dulist);
        });
      Kokkos::fence();
      {
        auto end_tp = tp_start::now();
        elapsed_duidrj += std::chrono::duration<double>(end_tp - start_tp).count();
      }

      // ==== compute_deidrj =================================================
      start_tp = tp_start::now();

      Kokkos::parallel_for("deidrj",
        Kokkos::RangePolicy<>(0, num_nbor * num_atoms),
        KOKKOS_LAMBDA(int idx) {
          int nbor  = idx / num_atoms;
          int natom = idx % num_atoms;

          for (int k = 0; k < 3; k++)
            p_dedr[ULIST_INDEX(natom, nbor, k)] = 0.0;

          for (int j = 0; j <= twojmax; j++) {
            int jjdu = p_idxdu_block[j];
            for (int mb = 0; 2 * mb < j; mb++)
              for (int ma = 0; ma <= j; ma++) {
                double yr = p_ylist[INDEX_2D(natom, jjdu)].re;
                double yi = p_ylist[INDEX_2D(natom, jjdu)].im;
                for (int k = 0; k < 3; k++)
                  p_dedr[ULIST_INDEX(natom, nbor, k)] +=
                    p_dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * yr +
                    p_dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * yi;
                jjdu++;
              }

            if (j % 2 == 0) {
              int mb = j / 2;
              for (int ma = 0; ma < mb; ma++) {
                double yr = p_ylist[INDEX_2D(natom, jjdu)].re;
                double yi = p_ylist[INDEX_2D(natom, jjdu)].im;
                for (int k = 0; k < 3; k++)
                  p_dedr[ULIST_INDEX(natom, nbor, k)] +=
                    p_dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * yr +
                    p_dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * yi;
                jjdu++;
              }
              double yr = p_ylist[INDEX_2D(natom, jjdu)].re;
              double yi = p_ylist[INDEX_2D(natom, jjdu)].im;
              for (int k = 0; k < 3; k++)
                p_dedr[ULIST_INDEX(natom, nbor, k)] +=
                  (p_dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re * yr +
                   p_dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im * yi) * 0.5;
              jjdu++;
            }
          }

          for (int k = 0; k < 3; k++)
            p_dedr[ULIST_INDEX(natom, nbor, k)] *= 2.0;
        });
      Kokkos::fence();
      {
        auto end_tp = tp_start::now();
        elapsed_deidrj += std::chrono::duration<double>(end_tp - start_tp).count();
      }

      // Copy dedr back to host
      {
        auto hv = Kokkos::create_mirror_view(d_dedr);
        Kokkos::deep_copy(hv, d_dedr);
        for (int i = 0; i < num_atoms * num_nbor * 3; i++) dedr[i] = hv(i);
      }

      // Compute forces (host)
      for (int natom = 0; natom < num_atoms; natom++) {
        for (int nbor = 0; nbor < num_nbor; nbor++) {
          int j = (int)inside[INDEX_2D(natom, nbor)];
          f[F_INDEX(natom, 0)] += dedr[ULIST_INDEX(natom, nbor, 0)];
          f[F_INDEX(natom, 1)] += dedr[ULIST_INDEX(natom, nbor, 1)];
          f[F_INDEX(natom, 2)] += dedr[ULIST_INDEX(natom, nbor, 2)];
          f[F_INDEX(j,     0)] -= dedr[ULIST_INDEX(natom, nbor, 0)];
          f[F_INDEX(j,     1)] -= dedr[ULIST_INDEX(natom, nbor, 1)];
          f[F_INDEX(j,     2)] -= dedr[ULIST_INDEX(natom, nbor, 2)];
        }
      }

      // Compute error
      jt = 0;
      for (int j = 0; j < ntotal; j++) {
        double ferrx = f[F_INDEX(j, 0)] - refdata.fj[jt++];
        double ferry = f[F_INDEX(j, 1)] - refdata.fj[jt++];
        double ferrz = f[F_INDEX(j, 2)] - refdata.fj[jt++];
        sumsqferr += ferrx * ferrx + ferry * ferry + ferrz * ferrz;
      }
    } // end step loop

    auto stop = myclock::now();
    myduration elapsed_total = stop - begin;
    double duration = elapsed_total.count();

    printf("-----------------------\n");
    printf("Summary of TestSNAP run\n");
    printf("-----------------------\n");
    printf("natoms = %d \n",       nlocal);
    printf("nghostatoms = %d \n",  nghost);
    printf("nsteps = %d \n",       nsteps);
    printf("nneighs = %d \n",      ninside);
    printf("twojmax = %d \n",      twojmax);
    printf("duration = %g [sec]\n",duration);

    double ktime = elapsed_ui + elapsed_yi + elapsed_duidrj + elapsed_deidrj;
    printf("step time = %g [msec/step]\n", 1000.0 * duration / nsteps);
    printf("\n Individual kernel timings for each step\n");
    printf("   compute_ui = %g [msec/step]\n",     1000.0 * elapsed_ui     / nsteps);
    printf("   compute_yi = %g [msec/step]\n",     1000.0 * elapsed_yi     / nsteps);
    printf("   compute_duidrj = %g [msec/step]\n", 1000.0 * elapsed_duidrj / nsteps);
    printf("   compute_deidrj = %g [msec/step]\n", 1000.0 * elapsed_deidrj / nsteps);
    printf("   Total kernel time = %g [msec/step]\n", 1000.0 * ktime / nsteps);
    printf("   Percentage of step time = %g%%\n\n", ktime / duration * 100.0);
    printf("grind time = %g [msec/atom-step]\n",
           1000.0 * duration / (nlocal * nsteps));
    printf("RMS |Fj| deviation %g [eV/A]\n",
           sqrt(sumsqferr / (ntotal * nsteps)));

    // Free host allocations
    free(coeffi); free(idxcg_block); free(idxu_block); free(ulist_parity);
    free(idxdu_block); free(idxb); free(idxb_block); free(idxz);
    free(idxzbeta); free(idxz_block); free(rij); free(inside);
    free(wj); free(rcutij); free(rootpqarray); free(cglist);
    free(dedr); free(ulist); free(ylist); free(ulisttot); free(dulist); free(f);
  }
  Kokkos::finalize();
  return 0;
}
