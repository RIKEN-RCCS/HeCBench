//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////
//// Kokkos port: replaces OpenMP target offloading with Kokkos parallel constructs.
////////////////////////////////////////////////////////////////////////////////

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <vector>
#include <tuple>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define float_sw4 double

// ─── Sarray class (from utils.cpp, no omp dependency) ────────────────────

class Sarray {
  public:
    Sarray() {}
    ~Sarray();
    Sarray(int nc, int ibeg, int iend, int jbeg, int jend, int kbeg, int kend);
    std::string fill(std::istringstream& iss);
    void init();
    float_sw4 norm();
    std::tuple<float_sw4,float_sw4> minmax();
    int m_nc, m_ni, m_nj, m_nk;
    int m_ib, m_ie, m_jb, m_je, m_kb, m_ke;
    ssize_t m_base;
    size_t m_offi, m_offj, m_offk, m_offc, m_npts;
    float_sw4* m_data;
    size_t size;
    int g;
};

std::string Sarray::fill(std::istringstream& iss) {
  std::string name;
  if (!(iss >> name >> g >> m_nc >> m_ni >> m_nj >> m_nk >> m_ib >> m_ie >>
        m_jb >> m_je >> m_kb >> m_ke >> m_base >> m_offi >> m_offj >> m_offk >>
        m_offc >> m_npts))
    return "Break";
#ifdef VERBOSE
  std::cout << name << " " << m_npts << "\n";
#endif
  size = m_nc * m_ni * m_nj * m_nk * sizeof(float_sw4);

  float_sw4* ptr = (float_sw4*) malloc (size);
  if (ptr == nullptr) {
    std::cerr << "malloc failed (size:" << size << " bytes)\n";
    abort();
  }

#ifdef VERBOSE
  std::cout << "Allocated " << size << " bytes " << name << "[" << g << "]\n";
#endif
  m_data = ptr;
  return name;
}

Sarray::~Sarray() {
#ifdef VERBOSE
  std::cout << "Free " << size << " bytes\n";
#endif
  free(m_data);
}

void Sarray::init() {

  const float_sw4 dx = 0.001;
  int nc = m_nc;
  int offi = nc;
  int offj = nc*m_ni;
  int offk = nc*m_ni*m_nj;

  for (int i = 0; i < m_ni; i++)
    for (int j = 0; j < m_nj; j++)
      for (int k = 0; k < m_nk; k++)
        for (int c = 0; c < nc; c++) {
          int indx = c + i * offi + j * offj + k * offk;
          float_sw4 x = i*dx;
          float_sw4 y = j*dx;
          float_sw4 z = k*dx;
          float_sw4 f = sin(x)*sin(y)*sin(z);
          m_data[indx]=f;
        }

}
float_sw4 Sarray::norm() {
  float_sw4 ret = 0.0;
  for (size_t i = 0; i < size / 8; i++) ret += m_data[i] * m_data[i];
  return ret;
}

std::tuple<float_sw4,float_sw4> Sarray::minmax(){
  float_sw4 min = std::numeric_limits<float_sw4>::max();
  float_sw4 max = std::numeric_limits<float_sw4>::min();
  for (size_t i = 0; i < size / 8; i++) {
    min=std::min(min,m_data[i]);
    max=std::max(max,m_data[i]);
  }
  return std::make_tuple(min,max);
}




// ─── Array-indexing macros (from curvilinear4sg.h) ───────────────────────

#define ni      (ilast - ifirst + 1)
#define nij     (ni * (jlast - jfirst + 1))
#define nijk    (nij * (klast - kfirst + 1))
#define base    (-(ifirst + ni * jfirst + nij * kfirst))
#define base3   (base - nijk)
#define base4   (base - nijk)
#define ifirst0 (ifirst)
#define jfirst0 (jfirst)

#define mu(i, j, k)     a_mu[base + (i) + ni * (j) + nij * (k)]
#define la(i, j, k)     a_lambda[base + (i) + ni * (j) + nij * (k)]
#define jac(i, j, k)    a_jac[base + (i) + ni * (j) + nij * (k)]
#define u(c, i, j, k)   a_u[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define lu(c, i, j, k)  a_lu[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define met(c, i, j, k) a_met[base4 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define strx(i)         a_strx[(i) - ifirst0]
#define stry(j)         a_stry[(j) - jfirst0]
#define acof(i, j, k)   a_acof[(i-1) + 6*(j-1) + 48*(k-1)]
#define bope(i, j)      a_bope[(i-1) + 6*(j-1)]
#define ghcof(i)        a_ghcof[(i)-1]
#define acof_no_gp(i,j,k) a_acof_no_gp[(i-1)+6*(j-1)+48*(k-1)]
#define ghcof_no_gp(i)  a_ghcof_no_gp[(i)-1]

#define i6  ((float_sw4)(1.0 / 6))
#define tf  ((float_sw4)(0.75))
#define c1  ((float_sw4)(2.0 / 3))
#define c2  ((float_sw4)(-1.0 / 12))

// ─── curvilinear4sg_ci – Kokkos implementation ───────────────────────────

void curvilinear4sg_ci(
    int ifirst, int ilast,
    int jfirst, int jlast,
    int kfirst, int klast,
    float_sw4* d_u,
    float_sw4* d_mu,
    float_sw4* d_lambda,
    float_sw4* d_met,
    float_sw4* d_jac,
    float_sw4* d_lu,
    int* onesided,
    float_sw4* d_cof,
    float_sw4* d_str,
    int nk, char op)
{
  float_sw4 a1 = 0, sgn = 1;
  if (op == '=') { a1 = 0; sgn =  1; }
  else if (op == '+') { a1 = 1; sgn =  1; }
  else if (op == '-') { a1 = 1; sgn = -1; }

  int kstart = kfirst + 2;
  int kend   = klast  - 2;
  if (onesided[5] == 1) kend = nk - 6;

  // Pre-compute pointer offsets (same as OMP version)
  float_sw4* d_acof        = d_cof + 6;
  float_sw4* d_bope        = d_cof + 6 + 384 + 24;
  float_sw4* d_ghcof       = d_cof + 6 + 384 + 24 + 48;
  float_sw4* d_acof_no_gp  = d_cof + 6 + 384 + 24 + 48 + 6;
  float_sw4* d_ghcof_no_gp = d_cof + 6 + 384 + 24 + 48 + 6 + 384;
  float_sw4* d_strx        = d_str;
  float_sw4* d_stry        = d_str + (ilast - ifirst + 1);

  using policy3 = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

  // ── kernel1: bottom boundary (onesided[4] == 1) ──────────────────────
  if (onesided[4] == 1) {
    int s0 = ifirst + 2, e0 = ilast - 1;
    int s1 = jfirst + 2, e1 = jlast - 1;
    int s2 = 1,          e2 = 7;   // K(1, 6+1)

    auto a_u        = d_u;
    auto a_mu       = d_mu;
    auto a_lambda   = d_lambda;
    auto a_met      = d_met;
    auto a_jac      = d_jac;
    auto a_lu       = d_lu;
    auto a_acof     = d_acof;
    auto a_bope     = d_bope;
    auto a_ghcof    = d_ghcof;
    auto a_acof_no_gp  = d_acof_no_gp;
    auto a_ghcof_no_gp = d_ghcof_no_gp;
    auto a_strx     = d_strx;
    auto a_stry     = d_stry;

    Kokkos::parallel_for("kernel1",
      policy3({s2, s1, s0}, {e2, e1, e0}),
      KOKKOS_LAMBDA(int k, int j, int i) {
    // 5 ops
    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
    // float_sw4 ijac = 1 / jac(i, j, k);
    float_sw4 istry = 1 / (stry(j));
    float_sw4 istrx = 1 / (strx(i));
    float_sw4 istrxy = istry * istrx;
    // ijac*=strx(i) * stry(j);

    float_sw4 r1 = 0, r2 = 0, r3 = 0;

    // pp derivative (u) (u-eq)
    // 53 ops, tot=58
    float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
      met(1, i - 2, j, k) * met(1, i - 2, j, k) *
      strx(i - 2);
    float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
      met(1, i - 1, j, k) * met(1, i - 1, j, k) *
      strx(i - 1);
    float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * strx(i);
    float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
      met(1, i + 1, j, k) * met(1, i + 1, j, k) *
      strx(i + 1);
    float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
      met(1, i + 2, j, k) * met(1, i + 2, j, k) *
      strx(i + 2);

    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

    r1 = r1 + i6 *
      (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
       mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
       mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
       mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
      istry;

    // qq derivative (u) (u-eq)
    // 43 ops, tot=101
    cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
      met(1, i, j - 2, k) * stry(j - 2);
    cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
      met(1, i, j - 1, k) * stry(j - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
    cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
      met(1, i, j + 1, k) * stry(j + 1);
    cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
      met(1, i, j + 2, k) * stry(j + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r1 = r1 + i6 *
      (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
       mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
       mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
       mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
      istrx;

    // pp derivative (v) (v-eq)
    // 43 ops, tot=144
    cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
      met(1, i - 2, j, k) * strx(i - 2);
    cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
      met(1, i - 1, j, k) * strx(i - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
      met(1, i + 1, j, k) * strx(i + 1);
    cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
      met(1, i + 2, j, k) * strx(i + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 = r2 + i6 *
      (mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +
       mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +
       mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +
       mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *
      istry;

    // qq derivative (v) (v-eq)
    // 53 ops, tot=197
    cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
      met(1, i, j - 2, k) * met(1, i, j - 2, k) * stry(j - 2);
    cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
      met(1, i, j - 1, k) * met(1, i, j - 1, k) * stry(j - 1);
    cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * stry(j);
    cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
      met(1, i, j + 1, k) * met(1, i, j + 1, k) * stry(j + 1);
    cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
      met(1, i, j + 2, k) * met(1, i, j + 2, k) * stry(j + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 = r2 + i6 *
      (mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +
       mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +
       mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +
       mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *
      istrx;

    // pp derivative (w) (w-eq)
    // 43 ops, tot=240
    cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
      met(1, i - 2, j, k) * strx(i - 2);
    cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
      met(1, i - 1, j, k) * strx(i - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
      met(1, i + 1, j, k) * strx(i + 1);
    cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
      met(1, i + 2, j, k) * strx(i + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r3 = r3 + i6 *
      (mux1 * (u(3, i - 2, j, k) - u(3, i, j, k)) +
       mux2 * (u(3, i - 1, j, k) - u(3, i, j, k)) +
       mux3 * (u(3, i + 1, j, k) - u(3, i, j, k)) +
       mux4 * (u(3, i + 2, j, k) - u(3, i, j, k))) *
      istry;

    // qq derivative (w) (w-eq)
    // 43 ops, tot=283
    cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
      met(1, i, j - 2, k) * stry(j - 2);
    cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
      met(1, i, j - 1, k) * stry(j - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
    cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
      met(1, i, j + 1, k) * stry(j + 1);
    cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
      met(1, i, j + 2, k) * stry(j + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r3 = r3 + i6 *
      (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +
       mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +
       mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +
       mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *
      istrx;

    // All rr-derivatives at once
    // averaging the coefficient
    // 54*8*8+25*8 = 3656 ops, tot=3939
    float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;
    //#pragma unroll 1 // slowdown due to register spills
    for (int q = 1; q <= 8; q++) {
      mucofu2 = 0;
      mucofuv = 0;
      mucofuw = 0;
      mucofvw = 0;
      mucofv2 = 0;
      mucofw2 = 0;
      //#pragma unroll 1 // slowdown due to register spills
      for (int m = 1; m <= 8; m++) {
        mucofu2 += acof(k, q, m) *
          ((2 * mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
           strx(i) * met(2, i, j, m) * strx(i) +
           mu(i, j, m) * (met(3, i, j, m) * stry(j) *
             met(3, i, j, m) * stry(j) +
             met(4, i, j, m) * met(4, i, j, m)));
        mucofv2 += acof(k, q, m) *
          ((2 * mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *
           stry(j) * met(3, i, j, m) * stry(j) +
           mu(i, j, m) * (met(2, i, j, m) * strx(i) *
             met(2, i, j, m) * strx(i) +
             met(4, i, j, m) * met(4, i, j, m)));
        mucofw2 += acof(k, q, m) *
          ((2 * mu(i, j, m) + la(i, j, m)) * met(4, i, j, m) *
           met(4, i, j, m) +
           mu(i, j, m) * (met(2, i, j, m) * strx(i) *
             met(2, i, j, m) * strx(i) +
             met(3, i, j, m) * stry(j) *
             met(3, i, j, m) * stry(j)));
        mucofuv += acof(k, q, m) * (mu(i, j, m) + la(i, j, m)) *
          met(2, i, j, m) * met(3, i, j, m);
        mucofuw += acof(k, q, m) * (mu(i, j, m) + la(i, j, m)) *
          met(2, i, j, m) * met(4, i, j, m);
        mucofvw += acof(k, q, m) * (mu(i, j, m) + la(i, j, m)) *
          met(3, i, j, m) * met(4, i, j, m);
      }

      // Computing the second derivative,
      r1 += istrxy * mucofu2 * u(1, i, j, q) + mucofuv * u(2, i, j, q) +
        istry * mucofuw * u(3, i, j, q);
      r2 += mucofuv * u(1, i, j, q) + istrxy * mucofv2 * u(2, i, j, q) +
        istrx * mucofvw * u(3, i, j, q);
      r3 += istry * mucofuw * u(1, i, j, q) +
        istrx * mucofvw * u(2, i, j, q) +
        istrxy * mucofw2 * u(3, i, j, q);
    }

    // Ghost point values, only nonzero for k=1.
    // 72 ops., tot=4011
    mucofu2 =
      ghcof(k) * ((2 * mu(i, j, 1) + la(i, j, 1)) * met(2, i, j, 1) *
          strx(i) * met(2, i, j, 1) * strx(i) +
          mu(i, j, 1) * (met(3, i, j, 1) * stry(j) *
            met(3, i, j, 1) * stry(j) +
            met(4, i, j, 1) * met(4, i, j, 1)));
    mucofv2 =
      ghcof(k) * ((2 * mu(i, j, 1) + la(i, j, 1)) * met(3, i, j, 1) *
          stry(j) * met(3, i, j, 1) * stry(j) +
          mu(i, j, 1) * (met(2, i, j, 1) * strx(i) *
            met(2, i, j, 1) * strx(i) +
            met(4, i, j, 1) * met(4, i, j, 1)));
    mucofw2 =
      ghcof(k) *
      ((2 * mu(i, j, 1) + la(i, j, 1)) * met(4, i, j, 1) *
       met(4, i, j, 1) +
       mu(i, j, 1) *
       (met(2, i, j, 1) * strx(i) * met(2, i, j, 1) * strx(i) +
        met(3, i, j, 1) * stry(j) * met(3, i, j, 1) * stry(j)));
    mucofuv = ghcof(k) * (mu(i, j, 1) + la(i, j, 1)) * met(2, i, j, 1) *
      met(3, i, j, 1);
    mucofuw = ghcof(k) * (mu(i, j, 1) + la(i, j, 1)) * met(2, i, j, 1) *
      met(4, i, j, 1);
    mucofvw = ghcof(k) * (mu(i, j, 1) + la(i, j, 1)) * met(3, i, j, 1) *
      met(4, i, j, 1);
    r1 += istrxy * mucofu2 * u(1, i, j, 0) + mucofuv * u(2, i, j, 0) +
      istry * mucofuw * u(3, i, j, 0);
    r2 += mucofuv * u(1, i, j, 0) + istrxy * mucofv2 * u(2, i, j, 0) +
      istrx * mucofvw * u(3, i, j, 0);
    r3 += istry * mucofuw * u(1, i, j, 0) +
      istrx * mucofvw * u(2, i, j, 0) +
      istrxy * mucofw2 * u(3, i, j, 0);

    // pq-derivatives (u-eq)
    // 38 ops., tot=4049
    r1 +=
      c2 *
      (mu(i, j + 2, k) * met(1, i, j + 2, k) *
       met(1, i, j + 2, k) *
       (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
        c1 *
        (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
       mu(i, j - 2, k) * met(1, i, j - 2, k) *
       met(1, i, j - 2, k) *
       (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
        c1 * (u(2, i + 1, j - 2, k) -
          u(2, i - 1, j - 2, k)))) +
      c1 *
      (mu(i, j + 1, k) * met(1, i, j + 1, k) *
       met(1, i, j + 1, k) *
       (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
        c1 *
        (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
       mu(i, j - 1, k) * met(1, i, j - 1, k) *
       met(1, i, j - 1, k) *
       (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
        c1 *
        (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

    // qp-derivatives (u-eq)
    // 38 ops. tot=4087
    r1 +=
      c2 *
      (la(i + 2, j, k) * met(1, i + 2, j, k) *
       met(1, i + 2, j, k) *
       (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
        c1 *
        (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
       la(i - 2, j, k) * met(1, i - 2, j, k) *
       met(1, i - 2, j, k) *
       (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
        c1 * (u(2, i - 2, j + 1, k) -
          u(2, i - 2, j - 1, k)))) +
      c1 *
      (la(i + 1, j, k) * met(1, i + 1, j, k) *
       met(1, i + 1, j, k) *
       (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
        c1 *
        (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
       la(i - 1, j, k) * met(1, i - 1, j, k) *
       met(1, i - 1, j, k) *
       (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
        c1 *
        (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

    // pq-derivatives (v-eq)
    // 38 ops. , tot=4125
    r2 +=
      c2 *
      (la(i, j + 2, k) * met(1, i, j + 2, k) *
       met(1, i, j + 2, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +
        c1 *
        (u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -
       la(i, j - 2, k) * met(1, i, j - 2, k) *
       met(1, i, j - 2, k) *
       (c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i + 1, j - 2, k) -
          u(1, i - 1, j - 2, k)))) +
      c1 *
      (la(i, j + 1, k) * met(1, i, j + 1, k) *
       met(1, i, j + 1, k) *
       (c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +
        c1 *
        (u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -
       la(i, j - 1, k) * met(1, i, j - 1, k) *
       met(1, i, j - 1, k) *
       (c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +
        c1 *
        (u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));

    //* qp-derivatives (v-eq)
    // 38 ops., tot=4163
    r2 +=
      c2 *
      (mu(i + 2, j, k) * met(1, i + 2, j, k) *
       met(1, i + 2, j, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +
        c1 *
        (u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -
       mu(i - 2, j, k) * met(1, i - 2, j, k) *
       met(1, i - 2, j, k) *
       (c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i - 2, j + 1, k) -
          u(1, i - 2, j - 1, k)))) +
      c1 *
      (mu(i + 1, j, k) * met(1, i + 1, j, k) *
       met(1, i + 1, j, k) *
       (c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +
        c1 *
        (u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -
       mu(i - 1, j, k) * met(1, i - 1, j, k) *
       met(1, i - 1, j, k) *
       (c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +
        c1 *
        (u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));

    // rp - derivatives
    // 24*8 = 192 ops, tot=4355
    float_sw4 dudrm2 = 0, dudrm1 = 0, dudrp1 = 0, dudrp2 = 0;
    float_sw4 dvdrm2 = 0, dvdrm1 = 0, dvdrp1 = 0, dvdrp2 = 0;
    float_sw4 dwdrm2 = 0, dwdrm1 = 0, dwdrp1 = 0, dwdrp2 = 0;
    //#pragma unroll 1
    for (int q = 1; q <= 8; q++) {
      dudrm2 += bope(k, q) * u(1, i - 2, j, q);
      dvdrm2 += bope(k, q) * u(2, i - 2, j, q);
      dwdrm2 += bope(k, q) * u(3, i - 2, j, q);
      dudrm1 += bope(k, q) * u(1, i - 1, j, q);
      dvdrm1 += bope(k, q) * u(2, i - 1, j, q);
      dwdrm1 += bope(k, q) * u(3, i - 1, j, q);
      dudrp2 += bope(k, q) * u(1, i + 2, j, q);
      dvdrp2 += bope(k, q) * u(2, i + 2, j, q);
      dwdrp2 += bope(k, q) * u(3, i + 2, j, q);
      dudrp1 += bope(k, q) * u(1, i + 1, j, q);
      dvdrp1 += bope(k, q) * u(2, i + 1, j, q);
      dwdrp1 += bope(k, q) * u(3, i + 1, j, q);
    }

    // rp derivatives (u-eq)
    // 67 ops, tot=4422
    r1 += (c2 * ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
          met(2, i + 2, j, k) * met(1, i + 2, j, k) *
          strx(i + 2) * dudrp2 +
          la(i + 2, j, k) * met(3, i + 2, j, k) *
          met(1, i + 2, j, k) * dvdrp2 * stry(j) +
          la(i + 2, j, k) * met(4, i + 2, j, k) *
          met(1, i + 2, j, k) * dwdrp2 -
          ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
           met(2, i - 2, j, k) * met(1, i - 2, j, k) *
           strx(i - 2) * dudrm2 +
           la(i - 2, j, k) * met(3, i - 2, j, k) *
           met(1, i - 2, j, k) * dvdrm2 * stry(j) +
           la(i - 2, j, k) * met(4, i - 2, j, k) *
           met(1, i - 2, j, k) * dwdrm2)) +
        c1 * ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
          met(2, i + 1, j, k) * met(1, i + 1, j, k) *
          strx(i + 1) * dudrp1 +
          la(i + 1, j, k) * met(3, i + 1, j, k) *
          met(1, i + 1, j, k) * dvdrp1 * stry(j) +
          la(i + 1, j, k) * met(4, i + 1, j, k) *
          met(1, i + 1, j, k) * dwdrp1 -
          ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
           met(2, i - 1, j, k) * met(1, i - 1, j, k) *
           strx(i - 1) * dudrm1 +
           la(i - 1, j, k) * met(3, i - 1, j, k) *
           met(1, i - 1, j, k) * dvdrm1 * stry(j) +
           la(i - 1, j, k) * met(4, i - 1, j, k) *
           met(1, i - 1, j, k) * dwdrm1))) *
           istry;

    // rp derivatives (v-eq)
    // 42 ops, tot=4464
    r2 +=
      c2 *
      (mu(i + 2, j, k) * met(3, i + 2, j, k) *
       met(1, i + 2, j, k) * dudrp2 +
       mu(i + 2, j, k) * met(2, i + 2, j, k) *
       met(1, i + 2, j, k) * dvdrp2 * strx(i + 2) * istry -
       (mu(i - 2, j, k) * met(3, i - 2, j, k) *
        met(1, i - 2, j, k) * dudrm2 +
        mu(i - 2, j, k) * met(2, i - 2, j, k) *
        met(1, i - 2, j, k) * dvdrm2 * strx(i - 2) * istry)) +
      c1 * (mu(i + 1, j, k) * met(3, i + 1, j, k) *
          met(1, i + 1, j, k) * dudrp1 +
          mu(i + 1, j, k) * met(2, i + 1, j, k) *
          met(1, i + 1, j, k) * dvdrp1 * strx(i + 1) * istry -
          (mu(i - 1, j, k) * met(3, i - 1, j, k) *
           met(1, i - 1, j, k) * dudrm1 +
           mu(i - 1, j, k) * met(2, i - 1, j, k) *
           met(1, i - 1, j, k) * dvdrm1 * strx(i - 1) * istry));

    // rp derivatives (w-eq)
    // 38 ops, tot=4502
    r3 += istry *
      (c2 * (mu(i + 2, j, k) * met(4, i + 2, j, k) *
             met(1, i + 2, j, k) * dudrp2 +
             mu(i + 2, j, k) * met(2, i + 2, j, k) *
             met(1, i + 2, j, k) * dwdrp2 * strx(i + 2) -
             (mu(i - 2, j, k) * met(4, i - 2, j, k) *
        met(1, i - 2, j, k) * dudrm2 +
        mu(i - 2, j, k) * met(2, i - 2, j, k) *
        met(1, i - 2, j, k) * dwdrm2 * strx(i - 2))) +
       c1 * (mu(i + 1, j, k) * met(4, i + 1, j, k) *
         met(1, i + 1, j, k) * dudrp1 +
         mu(i + 1, j, k) * met(2, i + 1, j, k) *
         met(1, i + 1, j, k) * dwdrp1 * strx(i + 1) -
         (mu(i - 1, j, k) * met(4, i - 1, j, k) *
          met(1, i - 1, j, k) * dudrm1 +
          mu(i - 1, j, k) * met(2, i - 1, j, k) *
          met(1, i - 1, j, k) * dwdrm1 * strx(i - 1))));

    // rq - derivatives
    // 24*8 = 192 ops , tot=4694

    dudrm2 = 0;
    dudrm1 = 0;
    dudrp1 = 0;
    dudrp2 = 0;
    dvdrm2 = 0;
    dvdrm1 = 0;
    dvdrp1 = 0;
    dvdrp2 = 0;
    dwdrm2 = 0;
    dwdrm1 = 0;
    dwdrp1 = 0;
    dwdrp2 = 0;
    //#pragma unroll 1
    for (int q = 1; q <= 8; q++) {
      dudrm2 += bope(k, q) * u(1, i, j - 2, q);
      dvdrm2 += bope(k, q) * u(2, i, j - 2, q);
      dwdrm2 += bope(k, q) * u(3, i, j - 2, q);
      dudrm1 += bope(k, q) * u(1, i, j - 1, q);
      dvdrm1 += bope(k, q) * u(2, i, j - 1, q);
      dwdrm1 += bope(k, q) * u(3, i, j - 1, q);
      dudrp2 += bope(k, q) * u(1, i, j + 2, q);
      dvdrp2 += bope(k, q) * u(2, i, j + 2, q);
      dwdrp2 += bope(k, q) * u(3, i, j + 2, q);
      dudrp1 += bope(k, q) * u(1, i, j + 1, q);
      dvdrp1 += bope(k, q) * u(2, i, j + 1, q);
      dwdrp1 += bope(k, q) * u(3, i, j + 1, q);
    }

    // rq derivatives (u-eq)
    // 42 ops, tot=4736
    r1 +=
      c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
          met(1, i, j + 2, k) * dudrp2 * stry(j + 2) * istrx +
          mu(i, j + 2, k) * met(2, i, j + 2, k) *
          met(1, i, j + 2, k) * dvdrp2 -
          (mu(i, j - 2, k) * met(3, i, j - 2, k) *
           met(1, i, j - 2, k) * dudrm2 * stry(j - 2) * istrx +
           mu(i, j - 2, k) * met(2, i, j - 2, k) *
           met(1, i, j - 2, k) * dvdrm2)) +
      c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
          met(1, i, j + 1, k) * dudrp1 * stry(j + 1) * istrx +
          mu(i, j + 1, k) * met(2, i, j + 1, k) *
          met(1, i, j + 1, k) * dvdrp1 -
          (mu(i, j - 1, k) * met(3, i, j - 1, k) *
           met(1, i, j - 1, k) * dudrm1 * stry(j - 1) * istrx +
           mu(i, j - 1, k) * met(2, i, j - 1, k) *
           met(1, i, j - 1, k) * dvdrm1));

    // rq derivatives (v-eq)
    // 70 ops, tot=4806
    r2 += c2 * (la(i, j + 2, k) * met(2, i, j + 2, k) *
        met(1, i, j + 2, k) * dudrp2 +
        (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
        met(3, i, j + 2, k) * met(1, i, j + 2, k) * dvdrp2 *
        stry(j + 2) * istrx +
        la(i, j + 2, k) * met(4, i, j + 2, k) *
        met(1, i, j + 2, k) * dwdrp2 * istrx -
        (la(i, j - 2, k) * met(2, i, j - 2, k) *
         met(1, i, j - 2, k) * dudrm2 +
         (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
         met(3, i, j - 2, k) * met(1, i, j - 2, k) *
         dvdrm2 * stry(j - 2) * istrx +
         la(i, j - 2, k) * met(4, i, j - 2, k) *
         met(1, i, j - 2, k) * dwdrm2 * istrx)) +
      c1 * (la(i, j + 1, k) * met(2, i, j + 1, k) *
          met(1, i, j + 1, k) * dudrp1 +
          (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
          met(3, i, j + 1, k) * met(1, i, j + 1, k) * dvdrp1 *
          stry(j + 1) * istrx +
          la(i, j + 1, k) * met(4, i, j + 1, k) *
          met(1, i, j + 1, k) * dwdrp1 * istrx -
          (la(i, j - 1, k) * met(2, i, j - 1, k) *
           met(1, i, j - 1, k) * dudrm1 +
           (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
           met(3, i, j - 1, k) * met(1, i, j - 1, k) *
           dvdrm1 * stry(j - 1) * istrx +
           la(i, j - 1, k) * met(4, i, j - 1, k) *
           met(1, i, j - 1, k) * dwdrm1 * istrx));

    // rq derivatives (w-eq)
    // 39 ops, tot=4845
    r3 += (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
          met(1, i, j + 2, k) * dwdrp2 * stry(j + 2) +
          mu(i, j + 2, k) * met(4, i, j + 2, k) *
          met(1, i, j + 2, k) * dvdrp2 -
          (mu(i, j - 2, k) * met(3, i, j - 2, k) *
           met(1, i, j - 2, k) * dwdrm2 * stry(j - 2) +
           mu(i, j - 2, k) * met(4, i, j - 2, k) *
           met(1, i, j - 2, k) * dvdrm2)) +
        c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
          met(1, i, j + 1, k) * dwdrp1 * stry(j + 1) +
          mu(i, j + 1, k) * met(4, i, j + 1, k) *
          met(1, i, j + 1, k) * dvdrp1 -
          (mu(i, j - 1, k) * met(3, i, j - 1, k) *
           met(1, i, j - 1, k) * dwdrm1 * stry(j - 1) +
           mu(i, j - 1, k) * met(4, i, j - 1, k) *
           met(1, i, j - 1, k) * dvdrm1))) *
      istrx;

    // pr and qr derivatives at once
    // in loop: 8*(53+53+43) = 1192 ops, tot=6037
    //#pragma unroll 1
    for (int q = 1; q <= 8; q++) {
      // (u-eq)
      // 53 ops
      r1 += bope(k, q) *
        (
         // pr
         (2 * mu(i, j, q) + la(i, j, q)) * met(2, i, j, q) *
         met(1, i, j, q) *
         (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
          c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
         strx(i) * istry +
         mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
          c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) +
         mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
          c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
         istry
         // qr
         + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
          c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) *
         stry(j) * istrx +
         la(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
          c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))));

      // (v-eq)
      // 53 ops
      r2 += bope(k, q) *
        (
         // pr
         la(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
          c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) +
         mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
          c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) *
         strx(i) * istry
         // qr
         + mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
          c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) +
         (2 * mu(i, j, q) + la(i, j, q)) * met(3, i, j, q) *
         met(1, i, j, q) *
         (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
          c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
         stry(j) * istrx +
         mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
          c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
         istrx);

      // (w-eq)
      // 43 ops
      r3 += bope(k, q) *
        (
         // pr
         la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
          c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
         istry +
         mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
          c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
         strx(i) * istry
         // qr
         + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
          c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
         stry(j) * istrx +
         la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
          c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
         istrx);
    }

    // 12 ops, tot=6049
    lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
    lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;
    lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;
      });
    Kokkos::fence();
  }

  // ── kernel2: interior u-equation (was CPU in OMP, now GPU) ───────────
  {
    int s0 = ifirst + 2, e0 = ilast - 1;
    int s1 = jfirst + 2, e1 = jlast - 1;
    int s2 = kstart,     e2 = kend + 1;

    auto a_u        = d_u;
    auto a_mu       = d_mu;
    auto a_lambda   = d_lambda;
    auto a_met      = d_met;
    auto a_jac      = d_jac;
    auto a_lu       = d_lu;
    auto a_acof     = d_acof;
    auto a_bope     = d_bope;
    auto a_ghcof    = d_ghcof;
    auto a_acof_no_gp  = d_acof_no_gp;
    auto a_ghcof_no_gp = d_ghcof_no_gp;
    auto a_strx     = d_strx;
    auto a_stry     = d_stry;

    Kokkos::parallel_for("kernel2",
      policy3({s2, s1, s0}, {e2, e1, e0}),
      KOKKOS_LAMBDA(int k, int j, int i) {
    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
    float_sw4 istry = 1 / (stry(j));
    float_sw4 istrx = 1 / (strx(i));
    float_sw4 istrxy = istry * istrx;

    float_sw4 r1 = 0;

    // pp derivative (u)
    // 53 ops, tot=58
    float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
      met(1, i - 2, j, k) * met(1, i - 2, j, k) *
      strx(i - 2);
    float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
      met(1, i - 1, j, k) * met(1, i - 1, j, k) *
      strx(i - 1);
    float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * strx(i);
    float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
      met(1, i + 1, j, k) * met(1, i + 1, j, k) *
      strx(i + 1);
    float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
      met(1, i + 2, j, k) * met(1, i + 2, j, k) *
      strx(i + 2);
    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

    r1 += i6 *
      (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
       mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
       mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
       mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
      istry;
    // qq derivative (u)
    // 43 ops, tot=101
    {
      float_sw4 cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
        met(1, i, j - 2, k) * stry(j - 2);
      float_sw4 cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
        met(1, i, j - 1, k) * stry(j - 1);
      float_sw4 cof3 =
        (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
      float_sw4 cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
        met(1, i, j + 1, k) * stry(j + 1);
      float_sw4 cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
        met(1, i, j + 2, k) * stry(j + 2);
      float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
      float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
      float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
      float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

      r1 += i6 *
        (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
         mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
         mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
         mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
        istrx;
    }
    // rr derivative (u)
    // 5*11+14+14=83 ops, tot=184
    {
      float_sw4 cof1 =
        (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
        strx(i) * met(2, i, j, k - 2) * strx(i) +
        mu(i, j, k - 2) * (met(3, i, j, k - 2) * stry(j) *
            met(3, i, j, k - 2) * stry(j) +
            met(4, i, j, k - 2) * met(4, i, j, k - 2));
      float_sw4 cof2 =
        (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
        strx(i) * met(2, i, j, k - 1) * strx(i) +
        mu(i, j, k - 1) * (met(3, i, j, k - 1) * stry(j) *
            met(3, i, j, k - 1) * stry(j) +
            met(4, i, j, k - 1) * met(4, i, j, k - 1));
      float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) *
        strx(i) * met(2, i, j, k) * strx(i) +
        mu(i, j, k) * (met(3, i, j, k) * stry(j) *
            met(3, i, j, k) * stry(j) +
            met(4, i, j, k) * met(4, i, j, k));
      float_sw4 cof4 =
        (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
        strx(i) * met(2, i, j, k + 1) * strx(i) +
        mu(i, j, k + 1) * (met(3, i, j, k + 1) * stry(j) *
            met(3, i, j, k + 1) * stry(j) +
            met(4, i, j, k + 1) * met(4, i, j, k + 1));
      float_sw4 cof5 =
        (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
        strx(i) * met(2, i, j, k + 2) * strx(i) +
        mu(i, j, k + 2) * (met(3, i, j, k + 2) * stry(j) *
            met(3, i, j, k + 2) * stry(j) +
            met(4, i, j, k + 2) * met(4, i, j, k + 2));

      float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
      float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
      float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
      float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

      r1 += i6 *
        (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
         mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
         mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
         mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *
        istrxy;
    }
    // rr derivative (v)
    // 42 ops, tot=226
    cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
      met(3, i, j, k - 2);
    cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
      met(3, i, j, k - 1);
    cof3 =
      (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);
    cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
      met(3, i, j, k + 1);
    cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
      met(3, i, j, k + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r1 += i6 * (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +
        mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +
        mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +
        mux4 * (u(2, i, j, k + 2) - u(2, i, j, k)));

    // rr derivative (w)
    // 43 ops, tot=269
    cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
      met(4, i, j, k - 2);
    cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
      met(4, i, j, k - 1);
    cof3 =
      (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);
    cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
      met(4, i, j, k + 1);
    cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
      met(4, i, j, k + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r1 += i6 *
      (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
       mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
       mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
       mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
      istry;

    // pq-derivatives
    // 38 ops, tot=307
    r1 +=
      c2 *
      (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
        c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
       mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
       (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
        c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +
      c1 *
      (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
        c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
       mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
       (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
        c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

    // qp-derivatives
    // 38 ops, tot=345
    r1 +=
      c2 *
      (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
        c1 * (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
       la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
       (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
        c1 * (u(2, i - 2, j + 1, k) - u(2, i - 2, j - 1, k)))) +
      c1 *
      (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
        c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
       la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
       (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
        c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

    // pr-derivatives
    // 130 ops., tot=475
    r1 +=
      c2 *
      ((2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
       met(2, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
        c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *
       strx(i) * istry +
       mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +
        c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) +
       mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +
        c1 * (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *
       istry -
       ((2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
        met(2, i, j, k - 2) * met(1, i, j, k - 2) *
        (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
         c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) *
        strx(i) * istry +
        mu(i, j, k - 2) * met(3, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +
         c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) +
        mu(i, j, k - 2) * met(4, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +
         c1 * (u(3, i + 1, j, k - 2) - u(3, i - 1, j, k - 2))) *
        istry)) +
        c1 *
        ((2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
         met(2, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
          c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *
         strx(i) * istry +
         mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +
          c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) +
         mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +
          c1 * (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *
         istry -
         ((2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
          met(2, i, j, k - 1) * met(1, i, j, k - 1) *
          (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
           c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) *
          strx(i) * istry +
          mu(i, j, k - 1) * met(3, i, j, k - 1) *
          met(1, i, j, k - 1) *
          (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +
           c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) +
          mu(i, j, k - 1) * met(4, i, j, k - 1) *
          met(1, i, j, k - 1) *
          (c2 * (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +
           c1 * (u(3, i + 1, j, k - 1) - u(3, i - 1, j, k - 1))) *
          istry));

    // rp derivatives
    // 130 ops, tot=605
    r1 +=
      (c2 *
       ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
        met(2, i + 2, j, k) * met(1, i + 2, j, k) *
        (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +
         c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) *
        strx(i + 2) +
        la(i + 2, j, k) * met(3, i + 2, j, k) *
        met(1, i + 2, j, k) *
        (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +
         c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *
        stry(j) +
        la(i + 2, j, k) * met(4, i + 2, j, k) *
        met(1, i + 2, j, k) *
        (c2 * (u(3, i + 2, j, k + 2) - u(3, i + 2, j, k - 2)) +
         c1 * (u(3, i + 2, j, k + 1) - u(3, i + 2, j, k - 1))) -
        ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
         met(2, i - 2, j, k) * met(1, i - 2, j, k) *
         (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +
          c1 *
          (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) *
         strx(i - 2) +
         la(i - 2, j, k) * met(3, i - 2, j, k) *
         met(1, i - 2, j, k) *
         (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +
          c1 *
          (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *
         stry(j) +
         la(i - 2, j, k) * met(4, i - 2, j, k) *
         met(1, i - 2, j, k) *
         (c2 * (u(3, i - 2, j, k + 2) - u(3, i - 2, j, k - 2)) +
          c1 * (u(3, i - 2, j, k + 1) -
            u(3, i - 2, j, k - 1))))) +
            c1 *
            ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
             met(2, i + 1, j, k) * met(1, i + 1, j, k) *
             (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +
              c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) *
             strx(i + 1) +
             la(i + 1, j, k) * met(3, i + 1, j, k) *
             met(1, i + 1, j, k) *
             (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +
              c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *
             stry(j) +
             la(i + 1, j, k) * met(4, i + 1, j, k) *
             met(1, i + 1, j, k) *
             (c2 * (u(3, i + 1, j, k + 2) - u(3, i + 1, j, k - 2)) +
              c1 * (u(3, i + 1, j, k + 1) - u(3, i + 1, j, k - 1))) -
             ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
              met(2, i - 1, j, k) * met(1, i - 1, j, k) *
              (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +
               c1 *
               (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) *
              strx(i - 1) +
              la(i - 1, j, k) * met(3, i - 1, j, k) *
              met(1, i - 1, j, k) *
              (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +
               c1 *
               (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *
              stry(j) +
              la(i - 1, j, k) * met(4, i - 1, j, k) *
              met(1, i - 1, j, k) *
              (c2 * (u(3, i - 1, j, k + 2) - u(3, i - 1, j, k - 2)) +
               c1 * (u(3, i - 1, j, k + 1) -
                 u(3, i - 1, j, k - 1)))))) *
                 istry;

    // qr derivatives
    // 82 ops, tot=687
    r1 +=
      c2 *
      (mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +
        c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) *
       stry(j) * istrx +
       la(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
        c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) -
       (mu(i, j, k - 2) * met(3, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
         c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) *
        stry(j) * istrx +
        la(i, j, k - 2) * met(2, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
         c1 * (u(2, i, j + 1, k - 2) -
           u(2, i, j - 1, k - 2))))) +
      c1 *
      (mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
       (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) *
       stry(j) * istrx +
       la(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
       (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) -
       (mu(i, j, k - 1) * met(3, i, j, k - 1) *
        met(1, i, j, k - 1) *
        (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
         c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) *
        stry(j) * istrx +
        la(i, j, k - 1) * met(2, i, j, k - 1) *
        met(1, i, j, k - 1) *
        (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
         c1 *
         (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1)))));

    // rq derivatives
    // 82 ops, tot=769
    r1 +=
      c2 *
      (mu(i, j + 2, k) * met(3, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
        c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) *
       stry(j + 2) * istrx +
       mu(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +
        c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) -
       (mu(i, j - 2, k) * met(3, i, j - 2, k) *
        met(1, i, j - 2, k) *
        (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
         c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) *
        stry(j - 2) * istrx +
        mu(i, j - 2, k) * met(2, i, j - 2, k) *
        met(1, i, j - 2, k) *
        (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +
         c1 * (u(2, i, j - 2, k + 1) -
           u(2, i, j - 2, k - 1))))) +
      c1 *
      (mu(i, j + 1, k) * met(3, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) *
       stry(j + 1) * istrx +
       mu(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +
        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) -
       (mu(i, j - 1, k) * met(3, i, j - 1, k) *
        met(1, i, j - 1, k) *
        (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
         c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) *
        stry(j - 1) * istrx +
        mu(i, j - 1, k) * met(2, i, j - 1, k) *
        met(1, i, j - 1, k) *
        (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +
         c1 *
         (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1)))));

    // 4 ops, tot=773
    lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
      });
    Kokkos::fence();
  }

  // ── kernel3: interior v-equation ─────────────────────────────────────
  {
    int s0 = ifirst + 2, e0 = ilast - 1;
    int s1 = jfirst + 2, e1 = jlast - 1;
    int s2 = kstart,     e2 = kend + 1;

    auto a_u        = d_u;
    auto a_mu       = d_mu;
    auto a_lambda   = d_lambda;
    auto a_met      = d_met;
    auto a_jac      = d_jac;
    auto a_lu       = d_lu;
    auto a_acof     = d_acof;
    auto a_bope     = d_bope;
    auto a_ghcof    = d_ghcof;
    auto a_acof_no_gp  = d_acof_no_gp;
    auto a_ghcof_no_gp = d_ghcof_no_gp;
    auto a_strx     = d_strx;
    auto a_stry     = d_stry;

    Kokkos::parallel_for("kernel3",
      policy3({s2, s1, s0}, {e2, e1, e0}),
      KOKKOS_LAMBDA(int k, int j, int i) {
    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
    float_sw4 istry = 1 / (stry(j));
    float_sw4 istrx = 1 / (strx(i));
    float_sw4 istrxy = istry * istrx;

    float_sw4 r2 = 0;
    // v-equation

    //      r1 = 0;
    // pp derivative (v)
    // 43 ops, tot=816
    float_sw4 cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
      met(1, i - 2, j, k) * strx(i - 2);
    float_sw4 cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
      met(1, i - 1, j, k) * strx(i - 1);
    float_sw4 cof3 =
      (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    float_sw4 cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
      met(1, i + 1, j, k) * strx(i + 1);
    float_sw4 cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
      met(1, i + 2, j, k) * strx(i + 2);

    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +
       mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +
       mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +
       mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *
      istry;

    // qq derivative (v)
    // 53 ops, tot=869
    cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) * met(1, i, j - 2, k) *
      met(1, i, j - 2, k) * stry(j - 2);
    cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) * met(1, i, j - 1, k) *
      met(1, i, j - 1, k) * stry(j - 1);
    cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * stry(j);
    cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) * met(1, i, j + 1, k) *
      met(1, i, j + 1, k) * stry(j + 1);
    cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) * met(1, i, j + 2, k) *
      met(1, i, j + 2, k) * stry(j + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +
       mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +
       mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +
       mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *
      istrx;

    // rr derivative (u)
    // 42 ops, tot=911
    cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
      met(3, i, j, k - 2);
    cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
      met(3, i, j, k - 1);
    cof3 =
      (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);
    cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
      met(3, i, j, k + 1);
    cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
      met(3, i, j, k + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 * (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
        mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
        mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
        mux4 * (u(1, i, j, k + 2) - u(1, i, j, k)));

    // rr derivative (v)
    // 83 ops, tot=994
    cof1 = (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(3, i, j, k - 2) *
      stry(j) * met(3, i, j, k - 2) * stry(j) +
      mu(i, j, k - 2) * (met(2, i, j, k - 2) * strx(i) *
          met(2, i, j, k - 2) * strx(i) +
          met(4, i, j, k - 2) * met(4, i, j, k - 2));
    cof2 = (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(3, i, j, k - 1) *
      stry(j) * met(3, i, j, k - 1) * stry(j) +
      mu(i, j, k - 1) * (met(2, i, j, k - 1) * strx(i) *
          met(2, i, j, k - 1) * strx(i) +
          met(4, i, j, k - 1) * met(4, i, j, k - 1));
    cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(3, i, j, k) * stry(j) *
      met(3, i, j, k) * stry(j) +
      mu(i, j, k) *
      (met(2, i, j, k) * strx(i) * met(2, i, j, k) * strx(i) +
       met(4, i, j, k) * met(4, i, j, k));
    cof4 = (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(3, i, j, k + 1) *
      stry(j) * met(3, i, j, k + 1) * stry(j) +
      mu(i, j, k + 1) * (met(2, i, j, k + 1) * strx(i) *
          met(2, i, j, k + 1) * strx(i) +
          met(4, i, j, k + 1) * met(4, i, j, k + 1));
    cof5 = (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(3, i, j, k + 2) *
      stry(j) * met(3, i, j, k + 2) * stry(j) +
      mu(i, j, k + 2) * (met(2, i, j, k + 2) * strx(i) *
          met(2, i, j, k + 2) * strx(i) +
          met(4, i, j, k + 2) * met(4, i, j, k + 2));

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +
       mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +
       mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +
       mux4 * (u(2, i, j, k + 2) - u(2, i, j, k))) *
      istrxy;

    // rr derivative (w)
    // 43 ops, tot=1037
    cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(3, i, j, k - 2) *
      met(4, i, j, k - 2);
    cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(3, i, j, k - 1) *
      met(4, i, j, k - 1);
    cof3 =
      (mu(i, j, k) + la(i, j, k)) * met(3, i, j, k) * met(4, i, j, k);
    cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(3, i, j, k + 1) *
      met(4, i, j, k + 1);
    cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(3, i, j, k + 2) *
      met(4, i, j, k + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 += i6 *
      (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
       mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
       mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
       mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
      istrx;

    // pq-derivatives
    // 38 ops, tot=1075
    r2 +=
      c2 *
      (la(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +
        c1 * (u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -
       la(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
       (c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i + 1, j - 2, k) - u(1, i - 1, j - 2, k)))) +
      c1 *
      (la(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +
        c1 * (u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -
       la(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
       (c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +
        c1 * (u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));

    // qp-derivatives
    // 38 ops, tot=1113
    r2 +=
      c2 *
      (mu(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +
        c1 * (u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -
       mu(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
       (c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i - 2, j + 1, k) - u(1, i - 2, j - 1, k)))) +
      c1 *
      (mu(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +
        c1 * (u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -
       mu(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
       (c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +
        c1 * (u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));

    // pr-derivatives
    // 82 ops, tot=1195
    r2 +=
      c2 *
      ((la(i, j, k + 2)) * met(3, i, j, k + 2) *
       met(1, i, j, k + 2) *
       (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
        c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) +
       mu(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +
        c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) *
       strx(i) * istry -
       ((la(i, j, k - 2)) * met(3, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
         c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) +
        mu(i, j, k - 2) * met(2, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +
         c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) *
        strx(i) * istry)) +
      c1 *
      ((la(i, j, k + 1)) * met(3, i, j, k + 1) *
       met(1, i, j, k + 1) *
       (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
        c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) +
       mu(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
       (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +
        c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) *
       strx(i) * istry -
       (la(i, j, k - 1) * met(3, i, j, k - 1) *
        met(1, i, j, k - 1) *
        (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
         c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) +
        mu(i, j, k - 1) * met(2, i, j, k - 1) *
        met(1, i, j, k - 1) *
        (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +
         c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) *
        strx(i) * istry));

    // rp derivatives
    // 82 ops, tot=1277
    r2 +=
      c2 *
      ((mu(i + 2, j, k)) * met(3, i + 2, j, k) *
       met(1, i + 2, j, k) *
       (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +
        c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) +
       mu(i + 2, j, k) * met(2, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +
        c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *
       strx(i + 2) * istry -
       (mu(i - 2, j, k) * met(3, i - 2, j, k) *
        met(1, i - 2, j, k) *
        (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +
         c1 * (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) +
        mu(i - 2, j, k) * met(2, i - 2, j, k) *
        met(1, i - 2, j, k) *
        (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +
         c1 * (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *
        strx(i - 2) * istry)) +
      c1 *
      ((mu(i + 1, j, k)) * met(3, i + 1, j, k) *
       met(1, i + 1, j, k) *
       (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +
        c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) +
       mu(i + 1, j, k) * met(2, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +
        c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *
       strx(i + 1) * istry -
       (mu(i - 1, j, k) * met(3, i - 1, j, k) *
        met(1, i - 1, j, k) *
        (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +
         c1 * (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) +
        mu(i - 1, j, k) * met(2, i - 1, j, k) *
        met(1, i - 1, j, k) *
        (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +
         c1 * (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *
        strx(i - 1) * istry));

    // qr derivatives
    // 130 ops, tot=1407
    r2 +=
      c2 *
      (mu(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +
        c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) +
       (2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
       met(3, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
        c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) *
       stry(j) * istrx +
       mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *
       (c2 * (u(3, i, j + 2, k + 2) - u(3, i, j - 2, k + 2)) +
        c1 * (u(3, i, j + 1, k + 2) - u(3, i, j - 1, k + 2))) *
       istrx -
       (mu(i, j, k - 2) * met(2, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
         c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) +
        (2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
        met(3, i, j, k - 2) * met(1, i, j, k - 2) *
        (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
         c1 * (u(2, i, j + 1, k - 2) - u(2, i, j - 1, k - 2))) *
        stry(j) * istrx +
        mu(i, j, k - 2) * met(4, i, j, k - 2) *
        met(1, i, j, k - 2) *
        (c2 * (u(3, i, j + 2, k - 2) - u(3, i, j - 2, k - 2)) +
         c1 * (u(3, i, j + 1, k - 2) - u(3, i, j - 1, k - 2))) *
        istrx)) +
        c1 *
        (mu(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
          c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) +
         (2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
         met(3, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
          c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) *
         stry(j) * istrx +
         mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *
         (c2 * (u(3, i, j + 2, k + 1) - u(3, i, j - 2, k + 1)) +
          c1 * (u(3, i, j + 1, k + 1) - u(3, i, j - 1, k + 1))) *
         istrx -
         (mu(i, j, k - 1) * met(2, i, j, k - 1) *
          met(1, i, j, k - 1) *
          (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
           c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) +
          (2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
          met(3, i, j, k - 1) * met(1, i, j, k - 1) *
          (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
           c1 * (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1))) *
          stry(j) * istrx +
          mu(i, j, k - 1) * met(4, i, j, k - 1) *
          met(1, i, j, k - 1) *
          (c2 * (u(3, i, j + 2, k - 1) - u(3, i, j - 2, k - 1)) +
           c1 * (u(3, i, j + 1, k - 1) - u(3, i, j - 1, k - 1))) *
          istrx));

    // rq derivatives
    // 130 ops, tot=1537
    r2 +=
      c2 *
      (la(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
        c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) +
       (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
       met(3, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +
        c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) *
       stry(j + 2) * istrx +
       la(i, j + 2, k) * met(4, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(3, i, j + 2, k + 2) - u(3, i, j + 2, k - 2)) +
        c1 * (u(3, i, j + 2, k + 1) - u(3, i, j + 2, k - 1))) *
       istrx -
       (la(i, j - 2, k) * met(2, i, j - 2, k) *
        met(1, i, j - 2, k) *
        (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
         c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) +
        (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
        met(3, i, j - 2, k) * met(1, i, j - 2, k) *
        (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +
         c1 * (u(2, i, j - 2, k + 1) - u(2, i, j - 2, k - 1))) *
        stry(j - 2) * istrx +
        la(i, j - 2, k) * met(4, i, j - 2, k) *
        met(1, i, j - 2, k) *
        (c2 * (u(3, i, j - 2, k + 2) - u(3, i, j - 2, k - 2)) +
         c1 * (u(3, i, j - 2, k + 1) - u(3, i, j - 2, k - 1))) *
        istrx)) +
        c1 *
        (la(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
         (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
          c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) +
         (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
         met(3, i, j + 1, k) * met(1, i, j + 1, k) *
         (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +
          c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) *
         stry(j + 1) * istrx +
         la(i, j + 1, k) * met(4, i, j + 1, k) * met(1, i, j + 1, k) *
         (c2 * (u(3, i, j + 1, k + 2) - u(3, i, j + 1, k - 2)) +
          c1 * (u(3, i, j + 1, k + 1) - u(3, i, j + 1, k - 1))) *
         istrx -
         (la(i, j - 1, k) * met(2, i, j - 1, k) *
          met(1, i, j - 1, k) *
          (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
           c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) +
          (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
          met(3, i, j - 1, k) * met(1, i, j - 1, k) *
          (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +
           c1 * (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1))) *
          stry(j - 1) * istrx +
          la(i, j - 1, k) * met(4, i, j - 1, k) *
          met(1, i, j - 1, k) *
          (c2 * (u(3, i, j - 1, k + 2) - u(3, i, j - 1, k - 2)) +
           c1 * (u(3, i, j - 1, k + 1) - u(3, i, j - 1, k - 1))) *
          istrx));

    // 4 ops, tot=1541
    lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;
      });
    Kokkos::fence();
  }

  // ── kernel4: interior w-equation ─────────────────────────────────────
  {
    int s0 = ifirst + 2, e0 = ilast - 1;
    int s1 = jfirst + 2, e1 = jlast - 1;
    int s2 = kstart,     e2 = kend + 1;

    auto a_u        = d_u;
    auto a_mu       = d_mu;
    auto a_lambda   = d_lambda;
    auto a_met      = d_met;
    auto a_jac      = d_jac;
    auto a_lu       = d_lu;
    auto a_acof     = d_acof;
    auto a_bope     = d_bope;
    auto a_ghcof    = d_ghcof;
    auto a_acof_no_gp  = d_acof_no_gp;
    auto a_ghcof_no_gp = d_ghcof_no_gp;
    auto a_strx     = d_strx;
    auto a_stry     = d_stry;

    Kokkos::parallel_for("kernel4",
      policy3({s2, s1, s0}, {e2, e1, e0}),
      KOKKOS_LAMBDA(int k, int j, int i) {
    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
    float_sw4 istry = 1 / (stry(j));
    float_sw4 istrx = 1 / (strx(i));
    float_sw4 istrxy = istry * istrx;

    float_sw4 r3 = 0.0;

    // w-equation

    //      r1 = 0;
    // pp derivative (w)
    // 43 ops, tot=1580
    float_sw4 cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) *
      met(1, i - 2, j, k) * strx(i - 2);
    float_sw4 cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) *
      met(1, i - 1, j, k) * strx(i - 1);
    float_sw4 cof3 =
      (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    float_sw4 cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) *
      met(1, i + 1, j, k) * strx(i + 1);
    float_sw4 cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) *
      met(1, i + 2, j, k) * strx(i + 2);

    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

    r3 += i6 *
      (mux1 * (u(3, i - 2, j, k) - u(3, i, j, k)) +
       mux2 * (u(3, i - 1, j, k) - u(3, i, j, k)) +
       mux3 * (u(3, i + 1, j, k) - u(3, i, j, k)) +
       mux4 * (u(3, i + 2, j, k) - u(3, i, j, k))) *
      istry;

    // qq derivative (w)
    // 43 ops, tot=1623
    {
      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;
      cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
        met(1, i, j - 2, k) * stry(j - 2);
      cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
        met(1, i, j - 1, k) * stry(j - 1);
      cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
      cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
        met(1, i, j + 1, k) * stry(j + 1);
      cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
        met(1, i, j + 2, k) * stry(j + 2);
      mux1 = cof2 - tf * (cof3 + cof1);
      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
      mux4 = cof4 - tf * (cof3 + cof5);

      r3 += i6 *
        (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +
         mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +
         mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +
         mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *
        istrx;
    }
    // rr derivative (u)
    // 43 ops, tot=1666
    {
      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;
      cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
        met(4, i, j, k - 2);
      cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
        met(4, i, j, k - 1);
      cof3 =
        (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);
      cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
        met(4, i, j, k + 1);
      cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
        met(4, i, j, k + 2);

      mux1 = cof2 - tf * (cof3 + cof1);
      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
      mux4 = cof4 - tf * (cof3 + cof5);

      r3 += i6 *
        (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
         mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
         mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
         mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *
        istry;
    }
    // rr derivative (v)
    // 43 ops, tot=1709
    {
      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;
      cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(3, i, j, k - 2) *
        met(4, i, j, k - 2);
      cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(3, i, j, k - 1) *
        met(4, i, j, k - 1);
      cof3 =
        (mu(i, j, k) + la(i, j, k)) * met(3, i, j, k) * met(4, i, j, k);
      cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(3, i, j, k + 1) *
        met(4, i, j, k + 1);
      cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(3, i, j, k + 2) *
        met(4, i, j, k + 2);

      mux1 = cof2 - tf * (cof3 + cof1);
      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
      mux4 = cof4 - tf * (cof3 + cof5);

      r3 += i6 *
        (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +
         mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +
         mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +
         mux4 * (u(2, i, j, k + 2) - u(2, i, j, k))) *
        istrx;
    }

    // rr derivative (w)
    // 83 ops, tot=1792
    {
      float_sw4 cof1, cof2, cof3, cof4, cof5, mux1, mux3, mux4;
      cof1 = (2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
        met(4, i, j, k - 2) * met(4, i, j, k - 2) +
        mu(i, j, k - 2) * (met(2, i, j, k - 2) * strx(i) *
            met(2, i, j, k - 2) * strx(i) +
            met(3, i, j, k - 2) * stry(j) *
            met(3, i, j, k - 2) * stry(j));
      cof2 = (2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
        met(4, i, j, k - 1) * met(4, i, j, k - 1) +
        mu(i, j, k - 1) * (met(2, i, j, k - 1) * strx(i) *
            met(2, i, j, k - 1) * strx(i) +
            met(3, i, j, k - 1) * stry(j) *
            met(3, i, j, k - 1) * stry(j));
      cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(4, i, j, k) *
        met(4, i, j, k) +
        mu(i, j, k) *
        (met(2, i, j, k) * strx(i) * met(2, i, j, k) * strx(i) +
         met(3, i, j, k) * stry(j) * met(3, i, j, k) * stry(j));
      cof4 = (2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
        met(4, i, j, k + 1) * met(4, i, j, k + 1) +
        mu(i, j, k + 1) * (met(2, i, j, k + 1) * strx(i) *
            met(2, i, j, k + 1) * strx(i) +
            met(3, i, j, k + 1) * stry(j) *
            met(3, i, j, k + 1) * stry(j));
      cof5 = (2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
        met(4, i, j, k + 2) * met(4, i, j, k + 2) +
        mu(i, j, k + 2) * (met(2, i, j, k + 2) * strx(i) *
            met(2, i, j, k + 2) * strx(i) +
            met(3, i, j, k + 2) * stry(j) *
            met(3, i, j, k + 2) * stry(j));
      mux1 = cof2 - tf * (cof3 + cof1);
      mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
      mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
      mux4 = cof4 - tf * (cof3 + cof5);

      r3 +=
        i6 *
        (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
         mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
         mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
         mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
        istrxy
        // pr-derivatives
        // 86 ops, tot=1878
        // r1 +=
        +
        c2 *
        ((la(i, j, k + 2)) * met(4, i, j, k + 2) *
         met(1, i, j, k + 2) *
         (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
          c1 *
          (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *
         istry +
         mu(i, j, k + 2) * met(2, i, j, k + 2) *
         met(1, i, j, k + 2) *
         (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +
          c1 *
          (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *
         strx(i) * istry -
         ((la(i, j, k - 2)) * met(4, i, j, k - 2) *
          met(1, i, j, k - 2) *
          (c2 *
           (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
           c1 * (u(1, i + 1, j, k - 2) -
             u(1, i - 1, j, k - 2))) *
          istry +
          mu(i, j, k - 2) * met(2, i, j, k - 2) *
          met(1, i, j, k - 2) *
          (c2 *
           (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +
           c1 * (u(3, i + 1, j, k - 2) -
             u(3, i - 1, j, k - 2))) *
          strx(i) * istry)) +
          c1 *
          ((la(i, j, k + 1)) * met(4, i, j, k + 1) *
           met(1, i, j, k + 1) *
           (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
            c1 *
            (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *
           istry +
           mu(i, j, k + 1) * met(2, i, j, k + 1) *
           met(1, i, j, k + 1) *
           (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +
            c1 *
            (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *
           strx(i) * istry -
           (la(i, j, k - 1) * met(4, i, j, k - 1) *
            met(1, i, j, k - 1) *
            (c2 *
             (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
             c1 * (u(1, i + 1, j, k - 1) -
               u(1, i - 1, j, k - 1))) *
            istry +
            mu(i, j, k - 1) * met(2, i, j, k - 1) *
            met(1, i, j, k - 1) *
            (c2 *
             (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +
             c1 * (u(3, i + 1, j, k - 1) -
               u(3, i - 1, j, k - 1))) *
            strx(i) * istry))
            // rp derivatives
            // 79 ops, tot=1957
            //   r1 +=
            + istry * (c2 * ((mu(i + 2, j, k)) * met(4, i + 2, j, k) *
                  met(1, i + 2, j, k) *
                  (c2 * (u(1, i + 2, j, k + 2) -
                   u(1, i + 2, j, k - 2)) +
                   c1 * (u(1, i + 2, j, k + 1) -
                     u(1, i + 2, j, k - 1))) +
                  mu(i + 2, j, k) * met(2, i + 2, j, k) *
                  met(1, i + 2, j, k) *
                  (c2 * (u(3, i + 2, j, k + 2) -
                   u(3, i + 2, j, k - 2)) +
                   c1 * (u(3, i + 2, j, k + 1) -
                     u(3, i + 2, j, k - 1))) *
                  strx(i + 2) -
                  (mu(i - 2, j, k) * met(4, i - 2, j, k) *
                   met(1, i - 2, j, k) *
                   (c2 * (u(1, i - 2, j, k + 2) -
                    u(1, i - 2, j, k - 2)) +
                    c1 * (u(1, i - 2, j, k + 1) -
                      u(1, i - 2, j, k - 1))) +
                   mu(i - 2, j, k) * met(2, i - 2, j, k) *
                   met(1, i - 2, j, k) *
                   (c2 * (u(3, i - 2, j, k + 2) -
                    u(3, i - 2, j, k - 2)) +
                    c1 * (u(3, i - 2, j, k + 1) -
                      u(3, i - 2, j, k - 1))) *
                   strx(i - 2))) +
                   c1 * ((mu(i + 1, j, k)) * met(4, i + 1, j, k) *
                       met(1, i + 1, j, k) *
                       (c2 * (u(1, i + 1, j, k + 2) -
                        u(1, i + 1, j, k - 2)) +
                        c1 * (u(1, i + 1, j, k + 1) -
                          u(1, i + 1, j, k - 1))) +
                       mu(i + 1, j, k) * met(2, i + 1, j, k) *
                       met(1, i + 1, j, k) *
                       (c2 * (u(3, i + 1, j, k + 2) -
                        u(3, i + 1, j, k - 2)) +
                        c1 * (u(3, i + 1, j, k + 1) -
                          u(3, i + 1, j, k - 1))) *
                       strx(i + 1) -
                       (mu(i - 1, j, k) * met(4, i - 1, j, k) *
                        met(1, i - 1, j, k) *
                        (c2 * (u(1, i - 1, j, k + 2) -
                         u(1, i - 1, j, k - 2)) +
                         c1 * (u(1, i - 1, j, k + 1) -
                           u(1, i - 1, j, k - 1))) +
                        mu(i - 1, j, k) * met(2, i - 1, j, k) *
                        met(1, i - 1, j, k) *
                        (c2 * (u(3, i - 1, j, k + 2) -
                         u(3, i - 1, j, k - 2)) +
                         c1 * (u(3, i - 1, j, k + 1) -
                           u(3, i - 1, j, k - 1))) *
                        strx(i - 1))))
                        // qr derivatives
                        // 86 ops, tot=2043
                        //     r1 +=
                        +
                        c2 *
                        (mu(i, j, k + 2) * met(3, i, j, k + 2) *
                         met(1, i, j, k + 2) *
                         (c2 * (u(3, i, j + 2, k + 2) - u(3, i, j - 2, k + 2)) +
                    c1 *
                    (u(3, i, j + 1, k + 2) - u(3, i, j - 1, k + 2))) *
                         stry(j) * istrx +
                         la(i, j, k + 2) * met(4, i, j, k + 2) *
                         met(1, i, j, k + 2) *
                         (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
                    c1 *
                    (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) *
                         istrx -
                         (mu(i, j, k - 2) * met(3, i, j, k - 2) *
                    met(1, i, j, k - 2) *
                    (c2 *
                     (u(3, i, j + 2, k - 2) - u(3, i, j - 2, k - 2)) +
                     c1 * (u(3, i, j + 1, k - 2) -
                       u(3, i, j - 1, k - 2))) *
                    stry(j) * istrx +
                    la(i, j, k - 2) * met(4, i, j, k - 2) *
                    met(1, i, j, k - 2) *
                    (c2 *
                     (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
                     c1 * (u(2, i, j + 1, k - 2) -
                       u(2, i, j - 1, k - 2))) *
                    istrx)) +
                    c1 *
                    (mu(i, j, k + 1) * met(3, i, j, k + 1) *
                     met(1, i, j, k + 1) *
                     (c2 * (u(3, i, j + 2, k + 1) - u(3, i, j - 2, k + 1)) +
                      c1 *
                      (u(3, i, j + 1, k + 1) - u(3, i, j - 1, k + 1))) *
                     stry(j) * istrx +
                     la(i, j, k + 1) * met(4, i, j, k + 1) *
                     met(1, i, j, k + 1) *
                     (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
                      c1 *
                      (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) *
                     istrx -
                     (mu(i, j, k - 1) * met(3, i, j, k - 1) *
                      met(1, i, j, k - 1) *
                      (c2 *
                       (u(3, i, j + 2, k - 1) - u(3, i, j - 2, k - 1)) +
                       c1 * (u(3, i, j + 1, k - 1) -
                         u(3, i, j - 1, k - 1))) *
                      stry(j) * istrx +
                      la(i, j, k - 1) * met(4, i, j, k - 1) *
                      met(1, i, j, k - 1) *
                      (c2 *
                       (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
                       c1 * (u(2, i, j + 1, k - 1) -
                         u(2, i, j - 1, k - 1))) *
                      istrx))
                      // rq derivatives
                      //  79 ops, tot=2122
                      //  r1 +=
                      + istrx * (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
                            met(1, i, j + 2, k) *
                            (c2 * (u(3, i, j + 2, k + 2) -
                             u(3, i, j + 2, k - 2)) +
                             c1 * (u(3, i, j + 2, k + 1) -
                               u(3, i, j + 2, k - 1))) *
                            stry(j + 2) +
                            mu(i, j + 2, k) * met(4, i, j + 2, k) *
                            met(1, i, j + 2, k) *
                            (c2 * (u(2, i, j + 2, k + 2) -
                             u(2, i, j + 2, k - 2)) +
                             c1 * (u(2, i, j + 2, k + 1) -
                               u(2, i, j + 2, k - 1))) -
                            (mu(i, j - 2, k) * met(3, i, j - 2, k) *
                             met(1, i, j - 2, k) *
                             (c2 * (u(3, i, j - 2, k + 2) -
                              u(3, i, j - 2, k - 2)) +
                              c1 * (u(3, i, j - 2, k + 1) -
                                u(3, i, j - 2, k - 1))) *
                             stry(j - 2) +
                             mu(i, j - 2, k) * met(4, i, j - 2, k) *
                             met(1, i, j - 2, k) *
                             (c2 * (u(2, i, j - 2, k + 2) -
                              u(2, i, j - 2, k - 2)) +
                              c1 * (u(2, i, j - 2, k + 1) -
                                u(2, i, j - 2, k - 1))))) +
                                c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
                                    met(1, i, j + 1, k) *
                                    (c2 * (u(3, i, j + 1, k + 2) -
                                     u(3, i, j + 1, k - 2)) +
                                     c1 * (u(3, i, j + 1, k + 1) -
                                       u(3, i, j + 1, k - 1))) *
                                    stry(j + 1) +
                                    mu(i, j + 1, k) * met(4, i, j + 1, k) *
                                    met(1, i, j + 1, k) *
                                    (c2 * (u(2, i, j + 1, k + 2) -
                                     u(2, i, j + 1, k - 2)) +
                                     c1 * (u(2, i, j + 1, k + 1) -
                                       u(2, i, j + 1, k - 1))) -
                                    (mu(i, j - 1, k) * met(3, i, j - 1, k) *
                                     met(1, i, j - 1, k) *
                                     (c2 * (u(3, i, j - 1, k + 2) -
                                      u(3, i, j - 1, k - 2)) +
                                      c1 * (u(3, i, j - 1, k + 1) -
                                        u(3, i, j - 1, k - 1))) *
                                     stry(j - 1) +
                                     mu(i, j - 1, k) * met(4, i, j - 1, k) *
                                     met(1, i, j - 1, k) *
                                     (c2 * (u(2, i, j - 1, k + 2) -
                                      u(2, i, j - 1, k - 2)) +
                                      c1 * (u(2, i, j - 1, k + 1) -
                                        u(2, i, j - 1, k - 1))))));
    }

    // 4 ops, tot=2126
    lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;
      });
    Kokkos::fence();
  }

  // ── kernel5: top boundary (onesided[5] == 1) ─────────────────────────
  if (onesided[5] == 1) {
    int s0 = ifirst + 2, e0 = ilast - 1;
    int s1 = jfirst + 2, e1 = jlast - 1;
    int s2 = nk - 5,     e2 = nk + 1;  // K(nk-5, nk+1)

    auto a_u        = d_u;
    auto a_mu       = d_mu;
    auto a_lambda   = d_lambda;
    auto a_met      = d_met;
    auto a_jac      = d_jac;
    auto a_lu       = d_lu;
    auto a_acof     = d_acof;
    auto a_bope     = d_bope;
    auto a_ghcof    = d_ghcof;
    auto a_acof_no_gp  = d_acof_no_gp;
    auto a_ghcof_no_gp = d_ghcof_no_gp;
    auto a_strx     = d_strx;
    auto a_stry     = d_stry;

    Kokkos::parallel_for("kernel5",
      policy3({s2, s1, s0}, {e2, e1, e0}),
      KOKKOS_LAMBDA(int k, int j, int i) {
    // 5 ops
    float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
    float_sw4 istry = 1 / (stry(j));
    float_sw4 istrx = 1 / (strx(i));
    float_sw4 istrxy = istry * istrx;

    float_sw4 r1 = 0, r2 = 0, r3 = 0;

    // pp derivative (u) (u-eq)
    // 53 ops, tot=58
    float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
      met(1, i - 2, j, k) * met(1, i - 2, j, k) *
      strx(i - 2);
    float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
      met(1, i - 1, j, k) * met(1, i - 1, j, k) *
      strx(i - 1);
    float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * strx(i);
    float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
      met(1, i + 1, j, k) * met(1, i + 1, j, k) *
      strx(i + 1);
    float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
      met(1, i + 2, j, k) * met(1, i + 2, j, k) *
      strx(i + 2);

    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

    r1 = r1 + i6 *
      (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
       mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
       mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
       mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
      istry;

    // qq derivative (u) (u-eq)
    // 43 ops, tot=101
    cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
      stry(j - 2);
    cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
      stry(j - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
    cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
      stry(j + 1);
    cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
      stry(j + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r1 = r1 + i6 *
      (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
       mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
       mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
       mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
      istrx;

    // pp derivative (v) (v-eq)
    // 43 ops, tot=144
    cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
      strx(i - 2);
    cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
      strx(i - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
      strx(i + 1);
    cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
      strx(i + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 = r2 + i6 *
      (mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +
       mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +
       mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +
       mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *
      istry;

    // qq derivative (v) (v-eq)
    // 53 ops, tot=197
    cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) * met(1, i, j - 2, k) *
      met(1, i, j - 2, k) * stry(j - 2);
    cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) * met(1, i, j - 1, k) *
      met(1, i, j - 1, k) * stry(j - 1);
    cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
      met(1, i, j, k) * stry(j);
    cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) * met(1, i, j + 1, k) *
      met(1, i, j + 1, k) * stry(j + 1);
    cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) * met(1, i, j + 2, k) *
      met(1, i, j + 2, k) * stry(j + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r2 = r2 + i6 *
      (mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +
       mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +
       mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +
       mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *
      istrx;

    // pp derivative (w) (w-eq)
    // 43 ops, tot=240
    cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
      strx(i - 2);
    cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
      strx(i - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
    cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
      strx(i + 1);
    cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
      strx(i + 2);

    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r3 = r3 + i6 *
      (mux1 * (u(3, i - 2, j, k) - u(3, i, j, k)) +
       mux2 * (u(3, i - 1, j, k) - u(3, i, j, k)) +
       mux3 * (u(3, i + 1, j, k) - u(3, i, j, k)) +
       mux4 * (u(3, i + 2, j, k) - u(3, i, j, k))) *
      istry;

    // qq derivative (w) (w-eq)
    // 43 ops, tot=283
    cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
      stry(j - 2);
    cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
      stry(j - 1);
    cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
    cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
      stry(j + 1);
    cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
      stry(j + 2);
    mux1 = cof2 - tf * (cof3 + cof1);
    mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
    mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
    mux4 = cof4 - tf * (cof3 + cof5);

    r3 = r3 + i6 *
      (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +
       mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +
       mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +
       mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *
      istrx;

    // All rr-derivatives at once
    // averaging the coefficient
    // 54*8*8+25*8 = 3656 ops, tot=3939
    float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;
    //#pragma unroll 8
    for (int q = nk - 7; q <= nk; q++) {
      mucofu2 = 0;
      mucofuv = 0;
      mucofuw = 0;
      mucofvw = 0;
      mucofv2 = 0;
      mucofw2 = 0;
      //#pragma unroll 8
      for (int m = nk - 7; m <= nk; m++) {
        mucofu2 += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
          ((2 * mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
           strx(i) * met(2, i, j, m) * strx(i) +
           mu(i, j, m) * (met(3, i, j, m) * stry(j) *
             met(3, i, j, m) * stry(j) +
             met(4, i, j, m) * met(4, i, j, m)));
        mucofv2 += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
          ((2 * mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *
           stry(j) * met(3, i, j, m) * stry(j) +
           mu(i, j, m) * (met(2, i, j, m) * strx(i) *
             met(2, i, j, m) * strx(i) +
             met(4, i, j, m) * met(4, i, j, m)));
        mucofw2 +=
          acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
          ((2 * mu(i, j, m) + la(i, j, m)) * met(4, i, j, m) *
           met(4, i, j, m) +
           mu(i, j, m) *
           (met(2, i, j, m) * strx(i) * met(2, i, j, m) * strx(i) +
            met(3, i, j, m) * stry(j) * met(3, i, j, m) * stry(j)));
        mucofuv += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
          (mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
          met(3, i, j, m);
        mucofuw += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
          (mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
          met(4, i, j, m);
        mucofvw += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
          (mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *
          met(4, i, j, m);
      }

      // Computing the second derivative,
      r1 += istrxy * mucofu2 * u(1, i, j, q) + mucofuv * u(2, i, j, q) +
        istry * mucofuw * u(3, i, j, q);
      r2 += mucofuv * u(1, i, j, q) + istrxy * mucofv2 * u(2, i, j, q) +
        istrx * mucofvw * u(3, i, j, q);
      r3 += istry * mucofuw * u(1, i, j, q) +
        istrx * mucofvw * u(2, i, j, q) +
        istrxy * mucofw2 * u(3, i, j, q);
    }

    // Ghost point values, only nonzero for k=nk.
    // 72 ops., tot=4011
    mucofu2 = ghcof_no_gp(nk - k + 1) *
      ((2 * mu(i, j, nk) + la(i, j, nk)) * met(2, i, j, nk) *
       strx(i) * met(2, i, j, nk) * strx(i) +
       mu(i, j, nk) * (met(3, i, j, nk) * stry(j) *
         met(3, i, j, nk) * stry(j) +
         met(4, i, j, nk) * met(4, i, j, nk)));
    mucofv2 = ghcof_no_gp(nk - k + 1) *
      ((2 * mu(i, j, nk) + la(i, j, nk)) * met(3, i, j, nk) *
       stry(j) * met(3, i, j, nk) * stry(j) +
       mu(i, j, nk) * (met(2, i, j, nk) * strx(i) *
         met(2, i, j, nk) * strx(i) +
         met(4, i, j, nk) * met(4, i, j, nk)));
    mucofw2 =
      ghcof_no_gp(nk - k + 1) *
      ((2 * mu(i, j, nk) + la(i, j, nk)) * met(4, i, j, nk) *
       met(4, i, j, nk) +
       mu(i, j, nk) *
       (met(2, i, j, nk) * strx(i) * met(2, i, j, nk) * strx(i) +
        met(3, i, j, nk) * stry(j) * met(3, i, j, nk) * stry(j)));
    mucofuv = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *
      met(2, i, j, nk) * met(3, i, j, nk);
    mucofuw = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *
      met(2, i, j, nk) * met(4, i, j, nk);
    mucofvw = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *
      met(3, i, j, nk) * met(4, i, j, nk);
    r1 += istrxy * mucofu2 * u(1, i, j, nk + 1) +
      mucofuv * u(2, i, j, nk + 1) +
      istry * mucofuw * u(3, i, j, nk + 1);
    r2 += mucofuv * u(1, i, j, nk + 1) +
      istrxy * mucofv2 * u(2, i, j, nk + 1) +
      istrx * mucofvw * u(3, i, j, nk + 1);
    r3 += istry * mucofuw * u(1, i, j, nk + 1) +
      istrx * mucofvw * u(2, i, j, nk + 1) +
      istrxy * mucofw2 * u(3, i, j, nk + 1);

    // pq-derivatives (u-eq)
    // 38 ops., tot=4049
    r1 +=
      c2 *
      (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
        c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
       mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
       (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
        c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +
      c1 *
      (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
        c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
       mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
       (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
        c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

    // qp-derivatives (u-eq)
    // 38 ops. tot=4087
    r1 +=
      c2 *
      (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
        c1 * (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
       la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
       (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
        c1 * (u(2, i - 2, j + 1, k) - u(2, i - 2, j - 1, k)))) +
      c1 *
      (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
        c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
       la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
       (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
        c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

    // pq-derivatives (v-eq)
    // 38 ops. , tot=4125
    r2 +=
      c2 *
      (la(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +
        c1 * (u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -
       la(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
       (c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i + 1, j - 2, k) - u(1, i - 1, j - 2, k)))) +
      c1 *
      (la(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
       (c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +
        c1 * (u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -
       la(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
       (c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +
        c1 * (u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));

    //* qp-derivatives (v-eq)
    // 38 ops., tot=4163
    r2 +=
      c2 *
      (mu(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
       (c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +
        c1 * (u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -
       mu(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
       (c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +
        c1 * (u(1, i - 2, j + 1, k) - u(1, i - 2, j - 1, k)))) +
      c1 *
      (mu(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
       (c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +
        c1 * (u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -
       mu(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
       (c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +
        c1 * (u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));

    // rp - derivatives
    // 24*8 = 192 ops, tot=4355
    float_sw4 dudrm2 = 0, dudrm1 = 0, dudrp1 = 0, dudrp2 = 0;
    float_sw4 dvdrm2 = 0, dvdrm1 = 0, dvdrp1 = 0, dvdrp2 = 0;
    float_sw4 dwdrm2 = 0, dwdrm1 = 0, dwdrp1 = 0, dwdrp2 = 0;
    //#pragma unroll 8
    for (int q = nk - 7; q <= nk; q++) {
      dudrm2 -= bope(nk - k + 1, nk - q + 1) * u(1, i - 2, j, q);
      dvdrm2 -= bope(nk - k + 1, nk - q + 1) * u(2, i - 2, j, q);
      dwdrm2 -= bope(nk - k + 1, nk - q + 1) * u(3, i - 2, j, q);
      dudrm1 -= bope(nk - k + 1, nk - q + 1) * u(1, i - 1, j, q);
      dvdrm1 -= bope(nk - k + 1, nk - q + 1) * u(2, i - 1, j, q);
      dwdrm1 -= bope(nk - k + 1, nk - q + 1) * u(3, i - 1, j, q);
      dudrp2 -= bope(nk - k + 1, nk - q + 1) * u(1, i + 2, j, q);
      dvdrp2 -= bope(nk - k + 1, nk - q + 1) * u(2, i + 2, j, q);
      dwdrp2 -= bope(nk - k + 1, nk - q + 1) * u(3, i + 2, j, q);
      dudrp1 -= bope(nk - k + 1, nk - q + 1) * u(1, i + 1, j, q);
      dvdrp1 -= bope(nk - k + 1, nk - q + 1) * u(2, i + 1, j, q);
      dwdrp1 -= bope(nk - k + 1, nk - q + 1) * u(3, i + 1, j, q);
    }

    // rp derivatives (u-eq)
    // 67 ops, tot=4422
    r1 += (c2 * ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
          met(2, i + 2, j, k) * met(1, i + 2, j, k) *
          strx(i + 2) * dudrp2 +
          la(i + 2, j, k) * met(3, i + 2, j, k) *
          met(1, i + 2, j, k) * dvdrp2 * stry(j) +
          la(i + 2, j, k) * met(4, i + 2, j, k) *
          met(1, i + 2, j, k) * dwdrp2 -
          ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
           met(2, i - 2, j, k) * met(1, i - 2, j, k) *
           strx(i - 2) * dudrm2 +
           la(i - 2, j, k) * met(3, i - 2, j, k) *
           met(1, i - 2, j, k) * dvdrm2 * stry(j) +
           la(i - 2, j, k) * met(4, i - 2, j, k) *
           met(1, i - 2, j, k) * dwdrm2)) +
        c1 * ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
          met(2, i + 1, j, k) * met(1, i + 1, j, k) *
          strx(i + 1) * dudrp1 +
          la(i + 1, j, k) * met(3, i + 1, j, k) *
          met(1, i + 1, j, k) * dvdrp1 * stry(j) +
          la(i + 1, j, k) * met(4, i + 1, j, k) *
          met(1, i + 1, j, k) * dwdrp1 -
          ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
           met(2, i - 1, j, k) * met(1, i - 1, j, k) *
           strx(i - 1) * dudrm1 +
           la(i - 1, j, k) * met(3, i - 1, j, k) *
           met(1, i - 1, j, k) * dvdrm1 * stry(j) +
           la(i - 1, j, k) * met(4, i - 1, j, k) *
           met(1, i - 1, j, k) * dwdrm1))) *
           istry;

    // rp derivatives (v-eq)
    // 42 ops, tot=4464
    r2 +=
      c2 * (mu(i + 2, j, k) * met(3, i + 2, j, k) *
          met(1, i + 2, j, k) * dudrp2 +
          mu(i + 2, j, k) * met(2, i + 2, j, k) *
          met(1, i + 2, j, k) * dvdrp2 * strx(i + 2) * istry -
          (mu(i - 2, j, k) * met(3, i - 2, j, k) *
           met(1, i - 2, j, k) * dudrm2 +
           mu(i - 2, j, k) * met(2, i - 2, j, k) *
           met(1, i - 2, j, k) * dvdrm2 * strx(i - 2) * istry)) +
      c1 * (mu(i + 1, j, k) * met(3, i + 1, j, k) *
          met(1, i + 1, j, k) * dudrp1 +
          mu(i + 1, j, k) * met(2, i + 1, j, k) *
          met(1, i + 1, j, k) * dvdrp1 * strx(i + 1) * istry -
          (mu(i - 1, j, k) * met(3, i - 1, j, k) *
           met(1, i - 1, j, k) * dudrm1 +
           mu(i - 1, j, k) * met(2, i - 1, j, k) *
           met(1, i - 1, j, k) * dvdrm1 * strx(i - 1) * istry));

    // rp derivatives (w-eq)
    // 38 ops, tot=4502
    r3 +=
      istry * (c2 * (mu(i + 2, j, k) * met(4, i + 2, j, k) *
            met(1, i + 2, j, k) * dudrp2 +
            mu(i + 2, j, k) * met(2, i + 2, j, k) *
            met(1, i + 2, j, k) * dwdrp2 * strx(i + 2) -
            (mu(i - 2, j, k) * met(4, i - 2, j, k) *
             met(1, i - 2, j, k) * dudrm2 +
             mu(i - 2, j, k) * met(2, i - 2, j, k) *
             met(1, i - 2, j, k) * dwdrm2 * strx(i - 2))) +
          c1 * (mu(i + 1, j, k) * met(4, i + 1, j, k) *
            met(1, i + 1, j, k) * dudrp1 +
            mu(i + 1, j, k) * met(2, i + 1, j, k) *
            met(1, i + 1, j, k) * dwdrp1 * strx(i + 1) -
            (mu(i - 1, j, k) * met(4, i - 1, j, k) *
             met(1, i - 1, j, k) * dudrm1 +
             mu(i - 1, j, k) * met(2, i - 1, j, k) *
             met(1, i - 1, j, k) * dwdrm1 * strx(i - 1))));

    // rq - derivatives
    // 24*8 = 192 ops , tot=4694

    dudrm2 = 0;
    dudrm1 = 0;
    dudrp1 = 0;
    dudrp2 = 0;
    dvdrm2 = 0;
    dvdrm1 = 0;
    dvdrp1 = 0;
    dvdrp2 = 0;
    dwdrm2 = 0;
    dwdrm1 = 0;
    dwdrp1 = 0;
    dwdrp2 = 0;
    //#pragma unroll 8
    for (int q = nk - 7; q <= nk; q++) {
      dudrm2 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j - 2, q);
      dvdrm2 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j - 2, q);
      dwdrm2 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j - 2, q);
      dudrm1 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j - 1, q);
      dvdrm1 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j - 1, q);
      dwdrm1 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j - 1, q);
      dudrp2 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j + 2, q);
      dvdrp2 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j + 2, q);
      dwdrp2 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j + 2, q);
      dudrp1 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j + 1, q);
      dvdrp1 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j + 1, q);
      dwdrp1 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j + 1, q);
    }

    // rq derivatives (u-eq)
    // 42 ops, tot=4736
    r1 += c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
        met(1, i, j + 2, k) * dudrp2 * stry(j + 2) * istrx +
        mu(i, j + 2, k) * met(2, i, j + 2, k) *
        met(1, i, j + 2, k) * dvdrp2 -
        (mu(i, j - 2, k) * met(3, i, j - 2, k) *
         met(1, i, j - 2, k) * dudrm2 * stry(j - 2) * istrx +
         mu(i, j - 2, k) * met(2, i, j - 2, k) *
         met(1, i, j - 2, k) * dvdrm2)) +
      c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
          met(1, i, j + 1, k) * dudrp1 * stry(j + 1) * istrx +
          mu(i, j + 1, k) * met(2, i, j + 1, k) *
          met(1, i, j + 1, k) * dvdrp1 -
          (mu(i, j - 1, k) * met(3, i, j - 1, k) *
           met(1, i, j - 1, k) * dudrm1 * stry(j - 1) * istrx +
           mu(i, j - 1, k) * met(2, i, j - 1, k) *
           met(1, i, j - 1, k) * dvdrm1));

    // rq derivatives (v-eq)
    // 70 ops, tot=4806
    r2 += c2 * (la(i, j + 2, k) * met(2, i, j + 2, k) *
        met(1, i, j + 2, k) * dudrp2 +
        (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
        met(3, i, j + 2, k) * met(1, i, j + 2, k) * dvdrp2 *
        stry(j + 2) * istrx +
        la(i, j + 2, k) * met(4, i, j + 2, k) *
        met(1, i, j + 2, k) * dwdrp2 * istrx -
        (la(i, j - 2, k) * met(2, i, j - 2, k) *
         met(1, i, j - 2, k) * dudrm2 +
         (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
         met(3, i, j - 2, k) * met(1, i, j - 2, k) * dvdrm2 *
         stry(j - 2) * istrx +
         la(i, j - 2, k) * met(4, i, j - 2, k) *
         met(1, i, j - 2, k) * dwdrm2 * istrx)) +
      c1 * (la(i, j + 1, k) * met(2, i, j + 1, k) *
          met(1, i, j + 1, k) * dudrp1 +
          (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
          met(3, i, j + 1, k) * met(1, i, j + 1, k) * dvdrp1 *
          stry(j + 1) * istrx +
          la(i, j + 1, k) * met(4, i, j + 1, k) *
          met(1, i, j + 1, k) * dwdrp1 * istrx -
          (la(i, j - 1, k) * met(2, i, j - 1, k) *
           met(1, i, j - 1, k) * dudrm1 +
           (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
           met(3, i, j - 1, k) * met(1, i, j - 1, k) * dvdrm1 *
           stry(j - 1) * istrx +
           la(i, j - 1, k) * met(4, i, j - 1, k) *
           met(1, i, j - 1, k) * dwdrm1 * istrx));

    // rq derivatives (w-eq)
    // 39 ops, tot=4845
    r3 += (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
          met(1, i, j + 2, k) * dwdrp2 * stry(j + 2) +
          mu(i, j + 2, k) * met(4, i, j + 2, k) *
          met(1, i, j + 2, k) * dvdrp2 -
          (mu(i, j - 2, k) * met(3, i, j - 2, k) *
           met(1, i, j - 2, k) * dwdrm2 * stry(j - 2) +
           mu(i, j - 2, k) * met(4, i, j - 2, k) *
           met(1, i, j - 2, k) * dvdrm2)) +
        c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
          met(1, i, j + 1, k) * dwdrp1 * stry(j + 1) +
          mu(i, j + 1, k) * met(4, i, j + 1, k) *
          met(1, i, j + 1, k) * dvdrp1 -
          (mu(i, j - 1, k) * met(3, i, j - 1, k) *
           met(1, i, j - 1, k) * dwdrm1 * stry(j - 1) +
           mu(i, j - 1, k) * met(4, i, j - 1, k) *
           met(1, i, j - 1, k) * dvdrm1))) *
      istrx;

    // pr and qr derivatives at once
    // in loop: 8*(53+53+43) = 1192 ops, tot=6037
    //#pragma unroll 8
    for (int q = nk - 7; q <= nk; q++) {
      // (u-eq)
      // 53 ops
      r1 -= bope(nk - k + 1, nk - q + 1) *
        (
         // pr
         (2 * mu(i, j, q) + la(i, j, q)) * met(2, i, j, q) *
         met(1, i, j, q) *
         (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
          c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
         strx(i) * istry +
         mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
          c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) +
         mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
          c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
         istry
         // qr
         + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
          c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) *
         stry(j) * istrx +
         la(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
          c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))));

      // (v-eq)
      // 53 ops
      r2 -= bope(nk - k + 1, nk - q + 1) *
        (
         // pr
         la(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
          c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) +
         mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
          c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) *
         strx(i) * istry
         // qr
         + mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
          c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) +
         (2 * mu(i, j, q) + la(i, j, q)) * met(3, i, j, q) *
         met(1, i, j, q) *
         (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
          c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
         stry(j) * istrx +
         mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
          c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
         istrx);

      // (w-eq)
      // 43 ops
      r3 -= bope(nk - k + 1, nk - q + 1) *
        (
         // pr
         la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
          c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
         istry +
         mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
          c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
         strx(i) * istry
         // qr
         + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
         (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
          c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
         stry(j) * istrx +
         la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
         (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
          c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
         istrx);
    }

    // 12 ops, tot=6049
    lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
    lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;
    lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;
      });
    Kokkos::fence();
  }
}

main // ─── ───────────────────

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to file> <repeat>\n";
    return 1;
  }

  std::ifstream iff;
  iff.open(argv[1]);

  const int repeat = atoi(argv[2]);

  std::map<std::string, Sarray*> arrays[10];
  std::vector<int*> onesided;
  std::string line;
  int lc = 0;
  std::cout << "Reading from file " << argv[1] << "\n";
  while (std::getline(iff, line)) {
    std::istringstream iss(line);
    int* optr = new int[14];
    const int N = 16;
    if ((lc % N) == 0) {
      if (!(iss >> optr[0] >> optr[1] >> optr[2] >> optr[3] >> optr[4] >>
            optr[5] >> optr[6] >> optr[7] >> optr[8] >> optr[9] >> optr[10] >>
            optr[11] >> optr[12] >> optr[13])) {
        std::cerr << "Error reading data on line " << lc + 1 << "\n";
        break;
      }
      onesided.push_back(optr);
    } else {
      Sarray* s = new Sarray();
      auto name = s->fill(iss);
      if (name == "Break") {
        std::cerr << "Error reading Sarray data on line " << lc + 1 << "\n";
        break;
      } else {
        arrays[lc / N][name] = s;
      }
    }
    lc++;
  }

  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i])
      x.second->init();

  int cof_size = (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6);
  float_sw4 *cof_h = (float_sw4*) malloc(sizeof(float_sw4) * cof_size);
  for (int i = 0; i < cof_size; i++) cof_h[i] = i / 1000.0;

  float_sw4 exact_norm[2] = {2.2502232733796421194, 202.0512747393526638};

  Kokkos::initialize(argc, argv);
  {
    // Allocate device View for cof (shared across iterations)
    Kokkos::View<float_sw4*> cof_d("cof_d", cof_size);
    {
      auto cof_host = Kokkos::View<float_sw4*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(cof_h, cof_size);
      Kokkos::deep_copy(cof_d, cof_host);
    }

    for (int i = 0; i < 2; i++) {
      int* optr = onesided[i];

      auto& arr_alpha  = *arrays[i]["a_AlphaVE_0"];
      auto& arr_mua    = *arrays[i]["mMuVE_0"];
      auto& arr_lambda = *arrays[i]["mLambdaVE_0"];
      auto& arr_met    = *arrays[i]["mMetric"];
      auto& arr_jac    = *arrays[i]["mJ"];
      auto& arr_uacc   = *arrays[i]["a_Uacc"];

      int alpha_size  = arr_alpha.m_nc  * arr_alpha.m_ni  * arr_alpha.m_nj  * arr_alpha.m_nk;
      int mua_size    = arr_mua.m_nc    * arr_mua.m_ni    * arr_mua.m_nj    * arr_mua.m_nk;
      int lambda_size = arr_lambda.m_nc * arr_lambda.m_ni * arr_lambda.m_nj * arr_lambda.m_nk;
      int met_size    = arr_met.m_nc    * arr_met.m_ni    * arr_met.m_ni    * arr_met.m_nk;
      int jac_size    = arr_jac.m_nc    * arr_jac.m_ni    * arr_jac.m_nj    * arr_jac.m_nk;
      int uacc_size   = arr_uacc.m_nc   * arr_uacc.m_ni   * arr_uacc.m_nj   * arr_uacc.m_nk;

      // Use actual size from m_npts
      int alpha_sz  = (int)arr_alpha.m_npts;
      int mua_sz    = (int)arr_mua.m_npts;
      int lambda_sz = (int)arr_lambda.m_npts;
      int met_sz    = (int)arr_met.m_npts;
      int jac_sz    = (int)arr_jac.m_npts;
      int uacc_sz   = (int)arr_uacc.m_npts;

      // Allocate device views
      Kokkos::View<float_sw4*> alpha_d ("alpha_d",  alpha_sz);
      Kokkos::View<float_sw4*> mua_d   ("mua_d",    mua_sz);
      Kokkos::View<float_sw4*> lambda_d("lambda_d", lambda_sz);
      Kokkos::View<float_sw4*> met_d   ("met_d",    met_sz);
      Kokkos::View<float_sw4*> jac_d   ("jac_d",    jac_sz);
      Kokkos::View<float_sw4*> uacc_d  ("uacc_d",   uacc_sz);

      // H2D copies
      Kokkos::deep_copy(alpha_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_alpha.m_data, alpha_sz));
      Kokkos::deep_copy(mua_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_mua.m_data, mua_sz));
      Kokkos::deep_copy(lambda_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_lambda.m_data, lambda_sz));
      Kokkos::deep_copy(met_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_met.m_data, met_sz));
      Kokkos::deep_copy(jac_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_jac.m_data, jac_sz));

      int nkg = optr[12];
      char op = '-';

      int sg_str_size = (optr[7] - optr[6] + optr[9] - optr[8] + 2);
      float_sw4* sg_str = (float_sw4*) malloc(sg_str_size * sizeof(float_sw4));
      for (int n = 0; n < sg_str_size; n++) sg_str[n] = n / 1000.0;

      Kokkos::View<float_sw4*> sg_str_d("sg_str_d", sg_str_size);
      Kokkos::deep_copy(sg_str_d,
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(sg_str, sg_str_size));

      double time = 0.0;

      for (int p = 0; p < repeat; p++) {
        // Reset uacc on device each iteration
        Kokkos::deep_copy(uacc_d,
          Kokkos::View<float_sw4*, Kokkos::HostSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_uacc.m_data, uacc_sz));

        auto start = std::chrono::high_resolution_clock::now();

        curvilinear4sg_ci(
          optr[6], optr[7], optr[8], optr[9], optr[10], optr[11],
          alpha_d.data(),  // d_u (the "alpha" acceleration array in this benchmark)
          mua_d.data(),
          lambda_d.data(),
          met_d.data(),
          jac_d.data(),
          uacc_d.data(),
          optr,
          cof_d.data(),
          sg_str_d.data(),
          nkg, op);

        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }

      std::cout << "\nAverage execution time of sw4ck kernels: "
                << (time * 1e-6f) / repeat << " milliseconds\n\n";

      // D2H copy of uacc result
      Kokkos::deep_copy(
        Kokkos::View<float_sw4*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(arr_uacc.m_data, uacc_sz),
        uacc_d);

      float_sw4 norm = arr_uacc.norm();
      float_sw4 err = (norm - exact_norm[i]) / exact_norm[i] * 100;
      std::cout << "Error = " << err << " %\n";

      free(sg_str);
      delete optr;
    }

    free(cof_h);
  }
  Kokkos::finalize();

  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i])
      delete x.second;

  return 0;
}
