#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <Kokkos_Core.hpp>

#define FLOAT float

#define NTHR_PER_BLK 256
#define NBLOCK (56*4)
#define Npoint (NBLOCK*NTHR_PER_BLK)
#define Neq 100000
#define Ngen_per_block 5000

#define DELTA 2.f
#define FOUR  4.f
#define TWO   2.f
#define ONE   1.f
#define HALF  0.5f
#define ZERO  0.f

KOKKOS_INLINE_FUNCTION float EXP_F(float x)  { return Kokkos::exp(x); }
KOKKOS_INLINE_FUNCTION float SQRT_F(float x) { return Kokkos::sqrt(x); }

KOKKOS_INLINE_FUNCTION
float LCG_random(unsigned int* seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return (float)(*seed) / (float)m;
}

KOKKOS_INLINE_FUNCTION
void LCG_random_init(unsigned int* seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
}

KOKKOS_INLINE_FUNCTION
void compute_distances(float x1, float y1, float z1,
                       float x2, float y2, float z2,
                       float& r1, float& r2, float& r12) {
  r1  = SQRT_F(x1*x1 + y1*y1 + z1*z1);
  r2  = SQRT_F(x2*x2 + y2*y2 + z2*z2);
  float xx = x1-x2, yy = y1-y2, zz = z1-z2;
  r12 = SQRT_F(xx*xx + yy*yy + zz*zz);
}

KOKKOS_INLINE_FUNCTION
float wave_function(float x1, float y1, float z1,
                    float x2, float y2, float z2) {
  float r1, r2, r12;
  compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
  return (ONE + HALF*r12) * EXP_F(-TWO*(r1 + r2));
}

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = typename ExecSpace::memory_space;
using FView     = Kokkos::View<FLOAT*, MemSpace>;
using UIView    = Kokkos::View<unsigned int*, MemSpace>;

void initran(int npoint, unsigned int seed, UIView states) {
  Kokkos::parallel_for("initran",
    Kokkos::RangePolicy<ExecSpace>(0, npoint),
    KOKKOS_LAMBDA(int i) {
      states(i) = seed ^ (unsigned)i;
      LCG_random_init(&states(i));
    });
  Kokkos::fence();
}

void initialize(int npoint, FView x1, FView y1, FView z1,
                FView x2, FView y2, FView z2, FView psi, UIView states) {
  Kokkos::parallel_for("init",
    Kokkos::RangePolicy<ExecSpace>(0, npoint),
    KOKKOS_LAMBDA(int i) {
      x1(i) = (LCG_random(&states(i)) - HALF)*FOUR;
      y1(i) = (LCG_random(&states(i)) - HALF)*FOUR;
      z1(i) = (LCG_random(&states(i)) - HALF)*FOUR;
      x2(i) = (LCG_random(&states(i)) - HALF)*FOUR;
      y2(i) = (LCG_random(&states(i)) - HALF)*FOUR;
      z2(i) = (LCG_random(&states(i)) - HALF)*FOUR;
      psi(i) = wave_function(x1(i),y1(i),z1(i),x2(i),y2(i),z2(i));
    });
  Kokkos::fence();
}

void zero_stats(int npoint, FView stats) {
  Kokkos::parallel_for("zero_stats",
    Kokkos::RangePolicy<ExecSpace>(0, npoint),
    KOKKOS_LAMBDA(int i) {
      stats(0*npoint+i) = ZERO;
      stats(1*npoint+i) = ZERO;
      stats(2*npoint+i) = ZERO;
      stats(3*npoint+i) = ZERO;
    });
  Kokkos::fence();
}

void propagate(int npoint, int nstep,
               FView X1, FView Y1, FView Z1,
               FView X2, FView Y2, FView Z2,
               FView P, FView stats, UIView states) {
  Kokkos::parallel_for("propagate",
    Kokkos::RangePolicy<ExecSpace>(0, npoint),
    KOKKOS_LAMBDA(int i) {
      FLOAT x1 = X1(i), y1 = Y1(i), z1 = Z1(i);
      FLOAT x2 = X2(i), y2 = Y2(i), z2 = Z2(i);
      FLOAT p  = P(i);
      for (int step = 0; step < nstep; step++) {
        FLOAT x1n = x1 + (LCG_random(&states(i))-HALF)*DELTA;
        FLOAT y1n = y1 + (LCG_random(&states(i))-HALF)*DELTA;
        FLOAT z1n = z1 + (LCG_random(&states(i))-HALF)*DELTA;
        FLOAT x2n = x2 + (LCG_random(&states(i))-HALF)*DELTA;
        FLOAT y2n = y2 + (LCG_random(&states(i))-HALF)*DELTA;
        FLOAT z2n = z2 + (LCG_random(&states(i))-HALF)*DELTA;
        FLOAT pnew = wave_function(x1n,y1n,z1n,x2n,y2n,z2n);
        if (pnew*pnew > p*p*LCG_random(&states(i))) {
          stats(3*npoint+i)++;
          p = pnew; x1=x1n; y1=y1n; z1=z1n; x2=x2n; y2=y2n; z2=z2n;
        }
        FLOAT r1, r2, r12;
        compute_distances(x1,y1,z1,x2,y2,z2,r1,r2,r12);
        stats(0*npoint+i) += r1;
        stats(1*npoint+i) += r2;
        stats(2*npoint+i) += r12;
      }
      X1(i)=x1; Y1(i)=y1; Z1(i)=z1;
      X2(i)=x2; Y2(i)=y2; Z2(i)=z2;
      P(i)=p;
    });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <number of blocks to sample>\n", argv[0]);
    return 1;
  }
  const int Nsample = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    FView  x1    ("x1",    Npoint);
    FView  y1    ("y1",    Npoint);
    FView  z1    ("z1",    Npoint);
    FView  x2    ("x2",    Npoint);
    FView  y2    ("y2",    Npoint);
    FView  z2    ("z2",    Npoint);
    FView  psi   ("psi",   Npoint);
    FView  stats ("stats", 4*Npoint);
    UIView ranst ("ranst", Npoint);

    initran(Npoint, 5551212, ranst);
    initialize(Npoint, x1,y1,z1,x2,y2,z2,psi,ranst);
    zero_stats(Npoint, stats);
    propagate(Npoint, Neq, x1,y1,z1,x2,y2,z2,psi,stats,ranst);

    double r1_tot=0,r1_sq=0,r2_tot=0,r2_sq=0,r12_tot=0,r12_sq=0,naccept=0;
    double time = 0.0;

    for (int sample = 0; sample < Nsample; sample++) {
      auto start = std::chrono::steady_clock::now();

      zero_stats(Npoint, stats);
      propagate(Npoint, Ngen_per_block, x1,y1,z1,x2,y2,z2,psi,stats,ranst);

      // Reduce stats on device
      FLOAT sum_r1=0, sum_r2=0, sum_r12=0, sum_acc=0;
      Kokkos::parallel_reduce("sum_r1",
        Kokkos::RangePolicy<ExecSpace>(0, Npoint),
        KOKKOS_LAMBDA(int i, FLOAT& v){ v += stats(0*Npoint+i); }, sum_r1);
      Kokkos::parallel_reduce("sum_r2",
        Kokkos::RangePolicy<ExecSpace>(0, Npoint),
        KOKKOS_LAMBDA(int i, FLOAT& v){ v += stats(1*Npoint+i); }, sum_r2);
      Kokkos::parallel_reduce("sum_r12",
        Kokkos::RangePolicy<ExecSpace>(0, Npoint),
        KOKKOS_LAMBDA(int i, FLOAT& v){ v += stats(2*Npoint+i); }, sum_r12);
      Kokkos::parallel_reduce("sum_acc",
        Kokkos::RangePolicy<ExecSpace>(0, Npoint),
        KOKKOS_LAMBDA(int i, FLOAT& v){ v += stats(3*Npoint+i); }, sum_acc);

      Kokkos::fence();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      naccept += sum_acc;
      float r1  = sum_r1  / ((FLOAT)Ngen_per_block * Npoint);
      float r2  = sum_r2  / ((FLOAT)Ngen_per_block * Npoint);
      float r12 = sum_r12 / ((FLOAT)Ngen_per_block * Npoint);

#ifdef DEBUG
      printf(" block %6d  %.6f  %.6f  %.6f\n", sample, r1, r2, r12);
#endif
      r1_tot  += r1;  r1_sq  += (double)r1*r1;
      r2_tot  += r2;  r2_sq  += (double)r2*r2;
      r12_tot += r12; r12_sq += (double)r12*r12;
    }

    r1_tot /= Nsample; r1_sq  /= Nsample;
    r2_tot /= Nsample; r2_sq  /= Nsample;
    r12_tot/= Nsample; r12_sq /= Nsample;

    double r1s  = sqrt((r1_sq  - r1_tot *r1_tot ) / Nsample);
    double r2s  = sqrt((r2_sq  - r2_tot *r2_tot ) / Nsample);
    double r12s = sqrt((r12_sq - r12_tot*r12_tot) / Nsample);

    printf(" <r1>  = %.6f +- %.6f\n", r1_tot,  r1s);
    printf(" <r2>  = %.6f +- %.6f\n", r2_tot,  r2s);
    printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);
    printf(" acceptance ratio=%.1f%%\n",
      100.0*naccept/double(Npoint)/double(Ngen_per_block)/double(Nsample));
    printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / Nsample);
  }
  Kokkos::finalize();
  return 0;
}
