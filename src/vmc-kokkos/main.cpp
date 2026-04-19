#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <Kokkos_Core.hpp>

#define FLOAT float

// Number of threads per block
#define NTHR_PER_BLK 256
// Number of blocks
#define NBLOCK (56*4)
// No. of independent samples
#define Npoint (NBLOCK*NTHR_PER_BLK)
// No. of generations to equilibrate
#define Neq 100000
// No. of generations per block
#define Ngen_per_block 5000

// Explicitly typed constants
#define DELTA 2.f
#define FOUR  4.f
#define TWO   2.f
#define ONE   1.f
#define HALF  0.5f
#define ZERO  0.f

KOKKOS_INLINE_FUNCTION float EXP(float x)  { return expf(x); }
KOKKOS_INLINE_FUNCTION double EXP(double x) { return exp(x); }
KOKKOS_INLINE_FUNCTION float SQRT(float x)  { return sqrtf(x); }
KOKKOS_INLINE_FUNCTION double SQRT(double x) { return sqrt(x); }

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
void compute_distances(const FLOAT x1, const FLOAT y1, const FLOAT z1,
                       const FLOAT x2, const FLOAT y2, const FLOAT z2,
                       FLOAT& r1, FLOAT& r2, FLOAT& r12)
{
  r1 = SQRT(x1*x1 + y1*y1 + z1*z1);
  r2 = SQRT(x2*x2 + y2*y2 + z2*z2);
  FLOAT xx = x1 - x2;
  FLOAT yy = y1 - y2;
  FLOAT zz = z1 - z2;
  r12 = SQRT(xx*xx + yy*yy + zz*zz);
}

KOKKOS_INLINE_FUNCTION
FLOAT wave_function(const FLOAT x1, const FLOAT y1, const FLOAT z1,
                    const FLOAT x2, const FLOAT y2, const FLOAT z2)
{
  FLOAT r1, r2, r12;
  compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
  return (ONE + HALF*r12) * EXP(-TWO*(r1 + r2));
}

void propagate(const int npoint, const int nstep,
               Kokkos::View<FLOAT*> X1, Kokkos::View<FLOAT*> Y1, Kokkos::View<FLOAT*> Z1,
               Kokkos::View<FLOAT*> X2, Kokkos::View<FLOAT*> Y2, Kokkos::View<FLOAT*> Z2,
               Kokkos::View<FLOAT*> P,  Kokkos::View<FLOAT*> stats,
               Kokkos::View<unsigned int*> states)
{
  Kokkos::parallel_for("propagate", npoint, KOKKOS_LAMBDA(int i) {
    FLOAT x1 = X1(i), y1 = Y1(i), z1 = Z1(i);
    FLOAT x2 = X2(i), y2 = Y2(i), z2 = Z2(i);
    FLOAT p  = P(i);

    for (int step = 0; step < nstep; step++) {
      FLOAT x1new = x1 + (LCG_random(&states(i)) - HALF) * DELTA;
      FLOAT y1new = y1 + (LCG_random(&states(i)) - HALF) * DELTA;
      FLOAT z1new = z1 + (LCG_random(&states(i)) - HALF) * DELTA;
      FLOAT x2new = x2 + (LCG_random(&states(i)) - HALF) * DELTA;
      FLOAT y2new = y2 + (LCG_random(&states(i)) - HALF) * DELTA;
      FLOAT z2new = z2 + (LCG_random(&states(i)) - HALF) * DELTA;
      FLOAT pnew  = wave_function(x1new, y1new, z1new, x2new, y2new, z2new);

      if (pnew*pnew > p*p * LCG_random(&states(i))) {
        stats(3*npoint + i)++;
        p  = pnew;
        x1 = x1new; y1 = y1new; z1 = z1new;
        x2 = x2new; y2 = y2new; z2 = z2new;
      }

      FLOAT r1, r2, r12;
      compute_distances(x1, y1, z1, x2, y2, z2, r1, r2, r12);
      stats(0*npoint + i) += r1;
      stats(1*npoint + i) += r2;
      stats(2*npoint + i) += r12;
    }

    X1(i) = x1; Y1(i) = y1; Z1(i) = z1;
    X2(i) = x2; Y2(i) = y2; Z2(i) = z2;
    P(i)  = p;
  });
  Kokkos::fence();
}

void initran(const int npoint, unsigned int seed,
             Kokkos::View<unsigned int*> states)
{
  Kokkos::parallel_for("initran", npoint, KOKKOS_LAMBDA(int i) {
    states(i) = seed ^ (unsigned int)i;
    LCG_random_init(&states(i));
  });
  Kokkos::fence();
}

void initialize(const int npoint,
                Kokkos::View<FLOAT*> x1, Kokkos::View<FLOAT*> y1, Kokkos::View<FLOAT*> z1,
                Kokkos::View<FLOAT*> x2, Kokkos::View<FLOAT*> y2, Kokkos::View<FLOAT*> z2,
                Kokkos::View<FLOAT*> psi,
                Kokkos::View<unsigned int*> states)
{
  Kokkos::parallel_for("initialize", npoint, KOKKOS_LAMBDA(int i) {
    x1(i) = (LCG_random(&states(i)) - HALF) * FOUR;
    y1(i) = (LCG_random(&states(i)) - HALF) * FOUR;
    z1(i) = (LCG_random(&states(i)) - HALF) * FOUR;
    x2(i) = (LCG_random(&states(i)) - HALF) * FOUR;
    y2(i) = (LCG_random(&states(i)) - HALF) * FOUR;
    z2(i) = (LCG_random(&states(i)) - HALF) * FOUR;
    psi(i) = wave_function(x1(i), y1(i), z1(i), x2(i), y2(i), z2(i));
  });
  Kokkos::fence();
}

void zero_stats(const int npoint, Kokkos::View<FLOAT*> stats) {
  Kokkos::parallel_for("zero_stats", npoint, KOKKOS_LAMBDA(int i) {
    stats(0*npoint + i) = ZERO;
    stats(1*npoint + i) = ZERO;
    stats(2*npoint + i) = ZERO;
    stats(3*npoint + i) = ZERO;
  });
  Kokkos::fence();
}

// Sum data[data_off + 0 .. n-1] into blocks of size `threads`.
// Block sums are written to blocksums[bsums_off + 0 .. teams-1].
void SumWithinBlocks(const int n, const int threads,
                     Kokkos::View<FLOAT*> data,      int data_off,
                     Kokkos::View<FLOAT*> blocksums, int bsums_off)
{
  const int teams = n / threads;
  Kokkos::parallel_for("SumWithinBlocks", teams, KOKKOS_LAMBDA(int block) {
    FLOAT sum = ZERO;
    // Replicate the original cycling pattern: each thread covers the full
    // stride, cycling while index < n.
    const int nthread = teams * threads;
    for (int gid = block * threads; gid < n; gid += nthread)
      sum += data(data_off + gid);
    blocksums(bsums_off + block) = sum;
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
    Kokkos::View<FLOAT*>        d_x1("x1", Npoint);
    Kokkos::View<FLOAT*>        d_y1("y1", Npoint);
    Kokkos::View<FLOAT*>        d_z1("z1", Npoint);
    Kokkos::View<FLOAT*>        d_x2("x2", Npoint);
    Kokkos::View<FLOAT*>        d_y2("y2", Npoint);
    Kokkos::View<FLOAT*>        d_z2("z2", Npoint);
    Kokkos::View<FLOAT*>        d_psi("psi", Npoint);
    Kokkos::View<FLOAT*>        d_stats("stats", 4 * Npoint);
    Kokkos::View<FLOAT*>        d_statsum("statsum", 4);
    Kokkos::View<FLOAT*>        d_blocksums("blocksums", NBLOCK);
    Kokkos::View<unsigned int*> d_states("states", Npoint);

    initran(Npoint, 5551212u, d_states);

    initialize(Npoint, d_x1, d_y1, d_z1, d_x2, d_y2, d_z2, d_psi, d_states);

    zero_stats(Npoint, d_stats);

    // Equilibrate
    propagate(Npoint, Neq, d_x1, d_y1, d_z1, d_x2, d_y2, d_z2,
              d_psi, d_stats, d_states);

    double r1_tot  = ZERO, r1_sq_tot  = ZERO;
    double r2_tot  = ZERO, r2_sq_tot  = ZERO;
    double r12_tot = ZERO, r12_sq_tot = ZERO;
    double naccept = ZERO;

    double time = 0.0;

    auto h_statsum = Kokkos::create_mirror_view(d_statsum);

    for (int sample = 0; sample < Nsample; sample++) {
      auto start = std::chrono::steady_clock::now();

      zero_stats(Npoint, d_stats);

      propagate(Npoint, Ngen_per_block, d_x1, d_y1, d_z1, d_x2, d_y2, d_z2,
                d_psi, d_stats, d_states);

      for (int what = 0; what < 4; what++) {
        SumWithinBlocks(Npoint, NTHR_PER_BLK, d_stats,     what * Npoint, d_blocksums, 0);
        SumWithinBlocks(NBLOCK, NBLOCK,        d_blocksums, 0,             d_statsum,   what);
      }

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      Kokkos::deep_copy(h_statsum, d_statsum);

      struct { FLOAT r1, r2, r12, accept; } s;
      s.r1     = h_statsum(0);
      s.r2     = h_statsum(1);
      s.r12    = h_statsum(2);
      s.accept = h_statsum(3);

      naccept += s.accept;
      s.r1  /= (FLOAT)(Ngen_per_block * Npoint);
      s.r2  /= (FLOAT)(Ngen_per_block * Npoint);
      s.r12 /= (FLOAT)(Ngen_per_block * Npoint);

#ifdef DEBUG
      printf(" block %6d  %.6f  %.6f  %.6f\n", sample, s.r1, s.r2, s.r12);
#endif

      r1_tot  += s.r1;   r1_sq_tot  += s.r1  * s.r1;
      r2_tot  += s.r2;   r2_sq_tot  += s.r2  * s.r2;
      r12_tot += s.r12;  r12_sq_tot += s.r12 * s.r12;
    }

    r1_tot  /= Nsample; r1_sq_tot  /= Nsample;
    r2_tot  /= Nsample; r2_sq_tot  /= Nsample;
    r12_tot /= Nsample; r12_sq_tot /= Nsample;

    double r1s  = sqrt((r1_sq_tot  - r1_tot  * r1_tot)  / Nsample);
    double r2s  = sqrt((r2_sq_tot  - r2_tot  * r2_tot)  / Nsample);
    double r12s = sqrt((r12_sq_tot - r12_tot * r12_tot) / Nsample);

    printf(" <r1>  = %.6f +- %.6f\n", r1_tot,  r1s);
    printf(" <r2>  = %.6f +- %.6f\n", r2_tot,  r2s);
    printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);

    printf(" acceptance ratio=%.1f%%\n",
           100.0 * naccept / (double)Npoint / (double)Ngen_per_block / (double)Nsample);

    printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / Nsample);
  }
  Kokkos::finalize();
  return 0;
}
