/*
 * Kokkos port of S3D combustion benchmark.
 * Original: s3d-omp/
 *
 * Translation:
 *   #pragma omp target teams distribute parallel for  →  Kokkos::parallel_for(RangePolicy)
 *   #pragma omp target teams num_teams(N) thread_limit(T) { #pragma omp parallel { body }}
 *        → Kokkos::parallel_for(RangePolicy(0,n), ...) since each (team,thread) maps to one i
 *
 * The kernel .h files use index variable 'i' and pointer-named variables
 * (T, RF, RB, C, A, EG, RKLOW, WDOT, Y, P, molwt) via macros like:
 *   RF(q) = RF[(q-1)*n + i]   (macro idx2)
 * Inside each lambda, local pointer aliases re-use those same names so
 * the macros expand correctly.
 *
 * All template helper functions (DIV, EXP, LOG, POW, MAX, MIN, polyx) get
 * KOKKOS_INLINE_FUNCTION so they work on both host and device.
 */

#include <Kokkos_Core.hpp>
#include <cassert>
#include <chrono>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// ============================================================
// S3D.h equivalents with KOKKOS_INLINE_FUNCTION
// ============================================================
#define BLOCK_SIZE   64
#define BLOCK_SIZE2  (2*BLOCK_SIZE)
#define RESTRICT __restrict__

#if 1 // REPLACE_DIV_WITH_RCP
template <class T1, class T2>
KOKKOS_INLINE_FUNCTION T1 DIV(T1 x, T2 y) { return x * (T1(1.0) / y); }
#else
template <class T1, class T2>
KOKKOS_INLINE_FUNCTION T1 DIV(T1 x, T2 y) { return x / y; }
#endif

template<class T> KOKKOS_INLINE_FUNCTION T POW (T x, T y);
template<> KOKKOS_INLINE_FUNCTION double POW<double>(double x, double y) { return pow(x,y); }
template<> KOKKOS_INLINE_FUNCTION float  POW<float> (float  x, float  y) { return powf(x,y); }

template<class T> KOKKOS_INLINE_FUNCTION T EXP(T x);
template<> KOKKOS_INLINE_FUNCTION double EXP<double>(double x) { return exp(x); }
template<> KOKKOS_INLINE_FUNCTION float  EXP<float> (float  x) { return expf(x); }

template<class T> KOKKOS_INLINE_FUNCTION T EXP10(T x);
template<> KOKKOS_INLINE_FUNCTION double EXP10<double>(double x) { return pow(10.0,x); }
template<> KOKKOS_INLINE_FUNCTION float  EXP10<float> (float  x) { return powf(10.0f,x); }

template<class T> KOKKOS_INLINE_FUNCTION T EXP2(T x);
template<> KOKKOS_INLINE_FUNCTION double EXP2<double>(double x) { return exp2(x); }
template<> KOKKOS_INLINE_FUNCTION float  EXP2<float> (float  x) { return exp2f(x); }

template<class T> KOKKOS_INLINE_FUNCTION T MAX(T a, T b);
template<> KOKKOS_INLINE_FUNCTION double MAX<double>(double a, double b) { return fmax(a,b); }
template<> KOKKOS_INLINE_FUNCTION float  MAX<float> (float  a, float  b) { return fmaxf(a,b); }

template<class T> KOKKOS_INLINE_FUNCTION T MIN(T a, T b);
template<> KOKKOS_INLINE_FUNCTION double MIN<double>(double a, double b) { return fmin(a,b); }
template<> KOKKOS_INLINE_FUNCTION float  MIN<float> (float  a, float  b) { return fminf(a,b); }

template<class T> KOKKOS_INLINE_FUNCTION T LOG(T x);
template<> KOKKOS_INLINE_FUNCTION double LOG<double>(double x) { return log(x); }
template<> KOKKOS_INLINE_FUNCTION float  LOG<float> (float  x) { return logf(x); }

template<class T> KOKKOS_INLINE_FUNCTION T LOG10(T x);
template<> KOKKOS_INLINE_FUNCTION double LOG10<double>(double x) { return log10(x); }
template<> KOKKOS_INLINE_FUNCTION float  LOG10<float> (float  x) { return log10f(x); }

template <class t1, class t2, class t3, class t4, class t5>
KOKKOS_INLINE_FUNCTION t1 polyx(t1 x, t2 c0, t3 c1, t4 c2, t5 c3)
{
    return (((c3 * x + c2) * x + c1) * x + c0) * x;
}

// Index macros (n is captured by value in each lambda)
#define N_GP        n
#define idx2(p,z)   (p[(((z)-1)*(N_GP)) + i])
#define idx(x, y)   ((x)[(y)-1])

#define C(q)     idx2(C, q)
#define Y(q)     idx2(Y, q)
#define RF(q)    idx2(RF, q)
#define EG(q)    idx2(EG, q)
#define RB(q)    idx2(RB, q)
#define RKLOW(q) idx2(RKLOW, q)
#define ROP(q)   idx(ROP, q)
#define ROP2(a)  (RF(a) - RB(a))
#define WDOT(q)  idx2(WDOT, q)
#define A_DIM    (11)
#define A(b, c)  idx2(A, (((b)*A_DIM)+c))

#define C_SIZE      (22)
#define RF_SIZE    (206)
#define RB_SIZE    (206)
#define WDOT_SIZE   (22)
#define RKLOW_SIZE  (21)
#define Y_SIZE      (22)
#define A_SIZE     (A_DIM * A_DIM)
#define EG_SIZE     (32)

// ============================================================
// RunTest: allocate views, run all kernels, verify
// ============================================================
template <class real>
void RunTest(const std::string &testName, int sizeClass, unsigned int passes)
{
  int n = sizeClass * sizeClass * sizeClass;

  // Host allocations
  real *host_t     = (real *)malloc(n * sizeof(real));
  real *host_p     = (real *)malloc(n * sizeof(real));
  real *host_y     = (real *)malloc(Y_SIZE * n * sizeof(real));
  real *host_molwt = (real *)malloc(WDOT_SIZE * sizeof(real));
  real *host_WDOT  = (real *)malloc(WDOT_SIZE * n * sizeof(real));

  real rateconv = 1.0, tconv = 1.0, pconv = 1.0;

  for (int i = 0; i < n; i++) { host_p[i] = 1.0132e6; host_t[i] = 1000.0; }
  for (int i = 0; i < WDOT_SIZE; i++) host_molwt[i] = 1;
  for (int j = 0; j < Y_SIZE; j++)
    for (int i = 0; i < n; i++) {
      host_y[j*n+i] = 0.0;
      if (j == 14) host_y[j*n+i] = 0.064;
      if (j ==  3) host_y[j*n+i] = 0.218;
      if (j == 21) host_y[j*n+i] = 0.718;
    }

  // Device views
  Kokkos::View<real*> d_T    ("T",     n);
  Kokkos::View<real*> d_P    ("P",     n);
  Kokkos::View<real*> d_Y    ("Y",     (size_t)Y_SIZE * n);
  Kokkos::View<real*> d_molwt("molwt", WDOT_SIZE);
  Kokkos::View<real*> d_RF   ("RF",    (size_t)RF_SIZE * n);
  Kokkos::View<real*> d_RB   ("RB",    (size_t)RB_SIZE * n);
  Kokkos::View<real*> d_RKLOW("RKLOW", (size_t)RKLOW_SIZE * n);
  Kokkos::View<real*> d_C    ("C",     (size_t)C_SIZE * n);
  Kokkos::View<real*> d_A    ("A",     (size_t)A_SIZE * n);
  Kokkos::View<real*> d_EG   ("EG",    (size_t)EG_SIZE * n);
  Kokkos::View<real*> d_WDOT ("WDOT",  (size_t)WDOT_SIZE * n);

  // Copy inputs
  {
    auto hm_T    = Kokkos::create_mirror_view(d_T);
    auto hm_P    = Kokkos::create_mirror_view(d_P);
    auto hm_Y    = Kokkos::create_mirror_view(d_Y);
    auto hm_molwt= Kokkos::create_mirror_view(d_molwt);
    for (int i = 0; i < n; i++) { hm_T[i] = host_t[i]; hm_P[i] = host_p[i]; }
    for (int j = 0; j < Y_SIZE; j++)
      for (int i = 0; i < n; i++) hm_Y[j*n+i] = host_y[j*n+i];
    for (int i = 0; i < WDOT_SIZE; i++) hm_molwt[i] = host_molwt[i];
    Kokkos::deep_copy(d_T,     hm_T);
    Kokkos::deep_copy(d_P,     hm_P);
    Kokkos::deep_copy(d_Y,     hm_Y);
    Kokkos::deep_copy(d_molwt, hm_molwt);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned int pass = 0; pass < passes; pass++) {

    // ratt kernel
    Kokkos::parallel_for("ratt", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      #include "../s3d-omp/ratt.h"
    });

    // rdsmh kernel
    Kokkos::parallel_for("rdsmh", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *EG = d_EG.data();
      #include "../s3d-omp/rdsmh.h"
    });

    // gr_base kernel
    Kokkos::parallel_for("gr_base", n, KOKKOS_LAMBDA(int i) {
      real *P = d_P.data(); real *T = d_T.data();
      real *Y = d_Y.data(); real *C = d_C.data();
      #include "../s3d-omp/gr_base.h"
    });

    // ratt2 kernel
    Kokkos::parallel_for("ratt2", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt2.h"
    });

    // ratt3 kernel
    Kokkos::parallel_for("ratt3", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt3.h"
    });

    // ratt4 kernel
    Kokkos::parallel_for("ratt4", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt4.h"
    });

    // ratt5 kernel
    Kokkos::parallel_for("ratt5", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt5.h"
    });

    // ratt6 kernel
    Kokkos::parallel_for("ratt6", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt6.h"
    });

    // ratt7 kernel
    Kokkos::parallel_for("ratt7", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt7.h"
    });

    // ratt8 kernel
    Kokkos::parallel_for("ratt8", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt8.h"
    });

    // ratt9 kernel
    Kokkos::parallel_for("ratt9", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RF = d_RF.data();
      real *RB = d_RB.data(); real *EG = d_EG.data();
      #include "../s3d-omp/ratt9.h"
    });

    // ratt10 kernel
    Kokkos::parallel_for("ratt10", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *RKLOW = d_RKLOW.data();
      #include "../s3d-omp/ratt10.h"
    });

    // ratx kernel (uses BLOCK_SIZE threads → flat parallel_for)
    Kokkos::parallel_for("ratx", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *C = d_C.data();
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *RKLOW = d_RKLOW.data();
      #include "../s3d-omp/ratx.h"
    });

    // ratxb kernel
    Kokkos::parallel_for("ratxb", n, KOKKOS_LAMBDA(int i) {
      real *T = d_T.data(); real *C = d_C.data();
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *RKLOW = d_RKLOW.data();
      #include "../s3d-omp/ratxb.h"
    });

    // ratx2: OMP uses num_teams(n/thrds2), each team+thread → one i
    Kokkos::parallel_for("ratx2", n, KOKKOS_LAMBDA(int i) {
      real *C = d_C.data(); real *RF = d_RF.data(); real *RB = d_RB.data();
      #include "../s3d-omp/ratx2.h"
    });

    // ratx4
    Kokkos::parallel_for("ratx4", n, KOKKOS_LAMBDA(int i) {
      real *C = d_C.data(); real *RF = d_RF.data(); real *RB = d_RB.data();
      #include "../s3d-omp/ratx4.h"
    });

    // qssa
    Kokkos::parallel_for("qssa", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data(); real *A = d_A.data();
      #include "../s3d-omp/qssa.h"
    });

    // qssab
    Kokkos::parallel_for("qssab", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data(); real *A = d_A.data();
      #include "../s3d-omp/qssab.h"
    });

    // qssa2
    Kokkos::parallel_for("qssa2", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data(); real *A = d_A.data();
      #include "../s3d-omp/qssa2.h"
    });

    // rdwdot kernels: rateconv captured from outer scope by [=] lambda
    Kokkos::parallel_for("rdwdot", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot.h"
    });

    Kokkos::parallel_for("rdwdot2", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot2.h"
    });

    Kokkos::parallel_for("rdwdot3", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot3.h"
    });

    Kokkos::parallel_for("rdwdot6", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot6.h"
    });

    Kokkos::parallel_for("rdwdot7", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot7.h"
    });

    Kokkos::parallel_for("rdwdot8", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot8.h"
    });

    Kokkos::parallel_for("rdwdot9", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot9.h"
    });

    Kokkos::parallel_for("rdwdot10", n, KOKKOS_LAMBDA(int i) {
      real *RF = d_RF.data(); real *RB = d_RB.data();
      real *WDOT = d_WDOT.data(); real *molwt = d_molwt.data();
      #include "../s3d-omp/rdwdot10.h"
    });

    Kokkos::fence();
  } // passes

  auto end  = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\n[%s] Average kernel time: %.3f us\n",
         testName.c_str(), (time * 1e-3) / passes);

  // Copy WDOT back and print
  {
    auto hm_WDOT = Kokkos::create_mirror_view(d_WDOT);
    Kokkos::deep_copy(hm_WDOT, d_WDOT);
    for (int k = 0; k < WDOT_SIZE; k++) {
      printf("% 23.16E ", hm_WDOT[k * n]);
      if (k % 3 == 2) printf("\n");
    }
    printf("\n");
    for (int k = 0; k < WDOT_SIZE; k++) host_WDOT[k * n] = hm_WDOT[k * n];
  }

  free(host_t); free(host_p); free(host_y); free(host_molwt); free(host_WDOT);
}

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[])
{
  int  sizeClass = 16;   // default cube side (16^3 = 4096 points, size=2)
  int  passes    = 1;

  // Simple arg parsing:  ./main <size_index 1-4>  <passes>
  if (argc >= 2) {
    int sz = atoi(argv[1]);
    int probSizes[4] = {8, 16, 32, 64};
    if (sz >= 1 && sz <= 4) sizeClass = probSizes[sz - 1];
  }
  if (argc >= 3) passes = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    RunTest<float>("S3D-SP", sizeClass, (unsigned int)passes);
    auto t2 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() * 1e-9;
    printf("Total time (SP): %.6f s\n", total);
  }
  Kokkos::finalize();
  return 0;
}
