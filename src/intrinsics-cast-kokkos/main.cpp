#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ── Portable bit-reinterpret (union trick, works in CUDA/HIP/host) ──────────

template <typename To, typename From>
KOKKOS_INLINE_FUNCTION To bit_reinterpret(From f)
{
  static_assert(sizeof(To) == sizeof(From), "bit_reinterpret: size mismatch");
  union { From from; To to; } u;
  u.from = f;
  return u.to;
}

// Upper / lower 32-bit halves of a double's bit pattern
KOKKOS_INLINE_FUNCTION int double2hiint(double x)
{
  return (int)(bit_reinterpret<long long>(x) >> 32);
}

KOKKOS_INLINE_FUNCTION int double2loint(double x)
{
  return (int)(bit_reinterpret<long long>(x) & 0xFFFFFFFFLL);
}

// Reconstruct a double from (hi, lo) int pair (matches __hiloint2double)
KOKKOS_INLINE_FUNCTION double hiloint2double(int hi, int lo)
{
  long long v = ((long long)(unsigned int)hi << 32) | (unsigned int)lo;
  return bit_reinterpret<double>(v);
}

// ── Kernel 1: double[] -> long long[]  ───────────────────────────────────────
// Replaces CUDA type-conversion intrinsics with standard C++ casts.
// Rounding modes: rd=floor, rn=round, ru=ceil, rz=truncate.

KOKKOS_INLINE_FUNCTION long long cast1(double x)
{
  int           r1 = 0;
  unsigned int  r2 = 0;
  long long     r3 = 0;
  unsigned long long r4 = 0;

  // ── int results (r1) ─────────────────────────────────────────
  r1 ^= double2hiint(x);                           // __double2hiint
  r1 ^= double2loint(x);                           // __double2loint

  r1 ^= (int)floor(x);                             // __double2int_rd
  r1 ^= (int)round(x);                             // __double2int_rn
  r1 ^= (int)ceil(x);                              // __double2int_ru
  r1 ^= (int)x;                                    // __double2int_rz

  r1 ^= (int)floorf((float)x);                     // __float2int_rd
  r1 ^= (int)roundf((float)x);                     // __float2int_rn
  r1 ^= (int)ceilf((float)x);                      // __float2int_ru
  r1 ^= (int)(float)x;                             // __float2int_rz

  r1 ^= bit_reinterpret<int>((float)x);            // __float_as_int

  // ── unsigned int results (r2) ────────────────────────────────
  r2 ^= (unsigned int)(unsigned long long)floor(x);  // __double2uint_rd
  r2 ^= (unsigned int)(unsigned long long)round(x);  // __double2uint_rn
  r2 ^= (unsigned int)(unsigned long long)ceil(x);   // __double2uint_ru
  r2 ^= (unsigned int)(unsigned long long)x;         // __double2uint_rz

  r2 ^= (unsigned int)(unsigned long long)floorf((float)x);  // __float2uint_rd
  r2 ^= (unsigned int)(unsigned long long)roundf((float)x);  // __float2uint_rn
  r2 ^= (unsigned int)(unsigned long long)ceilf((float)x);   // __float2uint_ru
  r2 ^= (unsigned int)(unsigned long long)(float)x;          // __float2uint_rz

  r2 ^= bit_reinterpret<unsigned int>((float)x);    // __float_as_uint

  // ── long long results (r3) ───────────────────────────────────
  r3 ^= (long long)floor(x);                        // __double2ll_rd
  r3 ^= (long long)round(x);                        // __double2ll_rn
  r3 ^= (long long)ceil(x);                         // __double2ll_ru
  r3 ^= (long long)x;                               // __double2ll_rz

  r3 ^= (long long)floorf((float)x);                // __float2ll_rd
  r3 ^= (long long)roundf((float)x);                // __float2ll_rn
  r3 ^= (long long)ceilf((float)x);                 // __float2ll_ru
  r3 ^= (long long)(float)x;                        // __float2ll_rz

  r3 ^= bit_reinterpret<long long>(x);              // __double_as_longlong

  // ── unsigned long long results (r4) ──────────────────────────
  r4 ^= (unsigned long long)floor(x);               // __double2ull_rd
  r4 ^= (unsigned long long)round(x);               // __double2ull_rn
  r4 ^= (unsigned long long)ceil(x);                // __double2ull_ru
  r4 ^= (unsigned long long)x;                      // __double2ull_rz

  r4 ^= (unsigned long long)floorf((float)x);       // __float2ull_rd
  r4 ^= (unsigned long long)roundf((float)x);       // __float2ull_rn
  r4 ^= (unsigned long long)ceilf((float)x);        // __float2ull_ru
  r4 ^= (unsigned long long)(float)x;               // __float2ull_rz

  // Combine: (r1+r2) promotes to unsigned int; (r3+r4) promotes to ull
  return (long long)(
    (unsigned long long)((unsigned int)r1 + r2) +
    (unsigned long long)r3 + r4);
}

// ── Kernel 2: long long[] -> long long[]  ────────────────────────────────────
// int/uint/ll/ull -> float/double conversions; all rounding modes map to
// the same standard C++ cast (hardware always rounds to nearest for int->fp).

KOKKOS_INLINE_FUNCTION long long cast2(long long x)
{
  float  r1 = 0.f;
  double r2 = 0.0;

  // __hiloint2double(hi, lo) where hi=x>>32, lo=x  →  bit-cast ll to double
  r1 += (float)hiloint2double((int)(x >> 32), (int)x);  // __hiloint2double

  r1 += (float)(int)x;            // __int2float_rd
  r1 += (float)(int)x;            // __int2float_rn
  r1 += (float)(int)x;            // __int2float_ru
  r1 += (float)(int)x;            // __int2float_rz

  r1 += (float)(unsigned int)x;   // __uint2float_rd
  r1 += (float)(unsigned int)x;   // __uint2float_rn
  r1 += (float)(unsigned int)x;   // __uint2float_ru
  r1 += (float)(unsigned int)x;   // __uint2float_rz

  r1 += bit_reinterpret<float>((int)x);           // __int_as_float
  r1 += bit_reinterpret<float>((unsigned int)x);  // __uint_as_float

  r1 += (float)(long long)x;      // __ll2float_rd
  r1 += (float)(long long)x;      // __ll2float_rn
  r1 += (float)(long long)x;      // __ll2float_ru
  r1 += (float)(long long)x;      // __ll2float_rz

  r1 += (float)(unsigned long long)x;  // __ull2float_rd
  r1 += (float)(unsigned long long)x;  // __ull2float_rn
  r1 += (float)(unsigned long long)x;  // __ull2float_ru
  r1 += (float)(unsigned long long)x;  // __ull2float_rz

  r2 += (double)(int)x;            // __int2double_rn
  r2 += (double)(unsigned int)x;   // __uint2double_rn

  r2 += (double)(long long)x;      // __ll2double_rd
  r2 += (double)(long long)x;      // __ll2double_rn
  r2 += (double)(long long)x;      // __ll2double_ru
  r2 += (double)(long long)x;      // __ll2double_rz

  r2 += (double)(unsigned long long)x;  // __ull2double_rd
  r2 += (double)(unsigned long long)x;  // __ull2double_rn
  r2 += (double)(unsigned long long)x;  // __ull2double_ru
  r2 += (double)(unsigned long long)x;  // __ull2double_rz

  r2 += bit_reinterpret<double>(x);     // __longlong_as_double

  return bit_reinterpret<long long>((double)r1 + r2); // __double_as_longlong
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  double    *h_in1  = new double[n];
  long long *h_out1 = new long long[n];
  long long *h_in2  = new long long[n];
  long long *h_out2 = new long long[n];

  // Initialise (matching CUDA original; 1-based to preserve same values)
  for (int i = 1; i <= n; i++) {
    h_in1[i - 1] = 22.44 / i;
    h_in2[i - 1] = (long long)0x403670A3D70A3D71LL;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double *>    d_in1 ("in1",  n);
    Kokkos::View<long long *> d_out1("out1", n);
    Kokkos::View<long long *> d_in2 ("in2",  n);
    Kokkos::View<long long *> d_out2("out2", n);

    {
      auto m_in1 = Kokkos::create_mirror_view(d_in1);
      auto m_in2 = Kokkos::create_mirror_view(d_in2);
      for (int i = 0; i < n; i++) { m_in1(i) = h_in1[i]; m_in2(i) = h_in2[i]; }
      Kokkos::deep_copy(d_in1, m_in1);
      Kokkos::deep_copy(d_in2, m_in2);
    }

    // ── cast1 benchmark ──────────────────────────────────────────
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++)
      Kokkos::parallel_for("cast1", n, KOKKOS_LAMBDA(int i) {
        d_out1(i) = cast1(d_in1(i));
      });

    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time of the cast intrinsics kernel (from FP): %f (us)\n",
           (std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
            * 1e-3f) / repeat);

    {
      auto m_out1 = Kokkos::create_mirror_view(d_out1);
      Kokkos::deep_copy(m_out1, d_out1);
      long long checksum = 0;
      for (int i = 0; i < n; i++) checksum ^= m_out1(i);
      printf("Checksum = %llx\n", checksum);
    }

    // ── cast2 benchmark ──────────────────────────────────────────
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++)
      Kokkos::parallel_for("cast2", n, KOKKOS_LAMBDA(int i) {
        d_out2(i) = cast2(d_in2(i));
      });

    Kokkos::fence();
    t1 = std::chrono::steady_clock::now();
    printf("Average execution time of the cast intrinsics kernel (to FP): %f (us)\n",
           (std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
            * 1e-3f) / repeat);

    {
      auto m_out2 = Kokkos::create_mirror_view(d_out2);
      Kokkos::deep_copy(m_out2, d_out2);
      long long checksum = 0;
      for (int i = 0; i < n; i++) checksum ^= m_out2(i);
      printf("Checksum = %llx\n", checksum);
    }
  }
  Kokkos::finalize();

  delete[] h_in1;
  delete[] h_out1;
  delete[] h_in2;
  delete[] h_out2;
  return 0;
}
