/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

///////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generator + Moro's Inverse CND
// Kokkos port
///////////////////////////////////////////////////////////////////////////////

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

typedef long long int INT64;

#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

// -----------------------------------------------------------------------
// CPU reference: table generation (from reference.cpp)
// -----------------------------------------------------------------------
static INT64 cjn[63][QRNG_DIMENSIONS];

static int GeneratePolynomials(int buffer[QRNG_DIMENSIONS], bool primitive) {
  int i, j, n, p1, p2, l;
  int e_p1, e_p2, e_b;

  for (n = 1, buffer[0] = 0x2, p2 = 0, l = 0; n < QRNG_DIMENSIONS; ++n) {
    for (p1 = buffer[n - 1] + 1; ; ++p1) {
      for (e_p1 = 30; (p1 & (1 << e_p1)) == 0; --e_p1) {}
      for (i = 0; i < n; ++i) {
        for (e_b = e_p1; (buffer[i] & (1 << e_b)) == 0; --e_b) {}
        for (p2 = (buffer[i] << ((e_p2 = e_p1) - e_b)) ^ p1;
             p2 >= buffer[i];
             p2 = (buffer[i] << (e_p2 - e_b)) ^ p2) {
          for (; (p2 & (1 << e_p2)) == 0; --e_p2) {}
        }
        if (p2 == 0) break;
      }
      if (p2 != 0) {
        e_p2 = 0;
        if (primitive) {
          j = ~(0xffffffff << (e_p1 + 1));
          e_b = (1 << e_p1) | 0x1;
          for (p2 = e_b, e_p2 = (1 << e_p1) - 2; e_p2 > 0; --e_p2) {
            p2 <<= 1;
            i = p2 & p1;
            i = (i & 0x55555555) + ((i >> 1) & 0x55555555);
            i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
            i = (i & 0x07070707) + ((i >> 4) & 0x07070707);
            p2 |= (i % 255) & 1;
            if ((p2 & j) == e_b) break;
          }
        }
        if (e_p2 == 0) {
          buffer[n] = p1;
          l += e_p1;
          break;
        }
      }
    }
  }
  return l + 1;
}

static void GenerateCJ() {
  int buffer[QRNG_DIMENSIONS];
  int *polynomials;
  int n, p1, l, e_p1;

  l = GeneratePolynomials(buffer, false);
  polynomials = new int[l + 2 * QRNG_DIMENSIONS + 1];
  for (n = 0, l = 0; n < QRNG_DIMENSIONS; ++n) {
    for (p1 = buffer[n], e_p1 = 30; (p1 & (1 << e_p1)) == 0; --e_p1) {}
    polynomials[l++] = 1;
    for (--e_p1; e_p1 >= 0; --e_p1)
      polynomials[l++] = (p1 >> e_p1) & 1;
    polynomials[l++] = -1;
  }
  polynomials[l] = -1;

  int *p = polynomials, e, d;
  int b_arr[1024], *b, m;
  int v_arr[1024], *v;
  int t_arr[1024], *t;
  int i, j, u, m1, ip, it;

  for (d = 0; p[0] != -1; p += e + 2) {
    for (i = 0; i < 63; ++i) cjn[i][d] = 0;
    for (e = 0; p[e + 1] != -1; ++e) {}
    (b = b_arr + 1023)[m = 0] = 1;
    v = v_arr + 1023 - (63 + e - 2);

    for (j = 63 - 1, u = e; j >= 0; --j, ++u) {
      if (u == e) {
        u = 0;
        for (i = 0, t = t_arr + 1023 - (m1 = m); i <= m; ++i) t[i] = b[i];
        b = b_arr + 1023 - (m += e);
        for (i = 0; i <= m; ++i) {
          b[i] = 0;
          for (ip = e - (m - i), it = m1; ip <= e && it >= 0; ++ip, --it)
            if (ip >= 0) b[i] ^= p[ip] & t[it];
        }
        for (i = 0; i < m1; ++i) v[i] = 0;
        for (; i < m; ++i) v[i] = 1;
        for (; i <= 63 + e - 2; ++i) {
          v[i] = 0;
          for (it = 1; it <= m; ++it) v[i] ^= v[i - it] & b[it];
        }
      }
      for (i = 0; i < 63; i++)
        cjn[i][d] |= (INT64)v[i + u] << j;
    }
    ++d;
  }
  delete[] polynomials;
}

void initQuasirandomGenerator(unsigned int *table) {
  GenerateCJ();
  for (int dim = 0; dim < QRNG_DIMENSIONS; dim++)
    for (int bit = 0; bit < QRNG_RESOLUTION; bit++)
      table[dim * QRNG_RESOLUTION + bit] = (int)((cjn[bit][dim] >> 32) & 0x7FFFFFFF);
}

double getQuasirandomValue63(INT64 i, int dim) {
  const double INT63_SCALE = (1.0 / (double)0x8000000000000001ULL);
  INT64 result = 0;
  for (int bit = 0; bit < 63; bit++, i >>= 1)
    if (i & 1) result ^= cjn[bit][dim];
  return (double)(result + 1) * INT63_SCALE;
}

double MoroInvCNDcpu(unsigned int x) {
  const double a1 =  2.50662823884,   a2 = -18.61500062529;
  const double a3 = 41.39119773534,   a4 = -25.44106049637;
  const double b1 = -8.4735109309,    b2 =  23.08336743743;
  const double b3 = -21.06224101826,  b4 =   3.13082909833;
  const double c1 =  0.337475482272615, c2 = 0.976169019091719;
  const double c3 =  0.160797971491821, c4 = 2.76438810333863E-02;
  const double c5 =  3.8405729373609E-03, c6 = 3.951896511919E-04;
  const double c7 =  3.21767881768E-05, c8 = 2.888167364E-07;
  const double c9 =  3.960315187E-07;
  double z;
  bool negate = false;
  if (x >= 0x80000000UL) { x = 0xffffffffUL - x; negate = true; }
  const double x1 = 1.0 / static_cast<double>(0xffffffffUL);
  const double x2 = x1 / 2.0;
  double p1 = x * x1 + x2;
  double p2 = p1 - 0.5;
  if (p2 > -0.42)
    z = p2 * (((a4 * p2*p2 + a3) * p2*p2 + a2) * p2*p2 + a1) /
        ((((b4 * p2*p2 + b3) * p2*p2 + b2) * p2*p2 + b1) * p2*p2 + 1.0);
  else {
    z = log(-log(p1));
    z = -(c1 + z*(c2 + z*(c3 + z*(c4 + z*(c5 + z*(c6 + z*(c7 + z*(c8 + z*c9))))))));
  }
  return negate ? -z : z;
}

// -----------------------------------------------------------------------
// GPU device function: Moro's Inverse CND
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION float MoroInvCNDgpu(unsigned int x) {
  const float a1 =  2.50662823884f,  a2 = -18.61500062529f;
  const float a3 = 41.39119773534f,  a4 = -25.44106049637f;
  const float b1 = -8.4735109309f,   b2 =  23.08336743743f;
  const float b3 = -21.06224101826f, b4 =   3.13082909833f;
  const float c1 = 0.337475482272615f, c2 = 0.976169019091719f;
  const float c3 = 0.160797971491821f, c4 = 2.76438810333863E-02f;
  const float c5 = 3.8405729373609E-03f, c6 = 3.951896511919E-04f;
  const float c7 = 3.21767881768E-05f,   c8 = 2.888167364E-07f;
  const float c9 = 3.960315187E-07f;

  float z;
  bool negate = false;
  if (x >= 0x80000000UL) { x = 0xffffffffUL - x; negate = true; }
  const float x1 = 1.0f / (float)0xffffffffUL;
  const float x2 = x1 / 2.0f;
  float p1 = x * x1 + x2;
  float p2 = p1 - 0.5f;
  if (p2 > -0.42f) {
    z = p2 * p2;
    z = p2 * (((a4 * z + a3) * z + a2) * z + a1) /
        ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
  } else {
    z = logf(-logf(p1));
    z = -(c1 + z*(c2 + z*(c3 + z*(c4 + z*(c5 + z*(c6 + z*(c7 + z*(c8 + z*c9))))))));
  }
  return negate ? -z : z;
}

const unsigned int N = 1048576;

int main(int argc, const char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, const_cast<char **>(argv));
  {
    unsigned int dim, pos;
    double delta, ref, sumDelta, sumRef, L1norm;
    unsigned int table[QRNG_DIMENSIONS * QRNG_RESOLUTION];
    bool bPassFlag = false;

    float *output = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

    printf("Initializing QRNG tables...\n");
    initQuasirandomGenerator(table);
    printf(">>>Launch QuasirandomGenerator kernel...\n\n");

    // Copy table to device
    using UintView  = Kokkos::View<unsigned int*>;
    using FloatView = Kokkos::View<float*>;

    UintView  d_table("table",  QRNG_DIMENSIONS * QRNG_RESOLUTION);
    FloatView d_output("output", QRNG_DIMENSIONS * N);

    auto hm_table = Kokkos::create_mirror_view(d_table);
    for (int i = 0; i < QRNG_DIMENSIONS * QRNG_RESOLUTION; i++)
      hm_table(i) = table[i];
    Kokkos::deep_copy(d_table, hm_table);

    const unsigned int seed = 0;
    const int qrng_res = QRNG_RESOLUTION;
    const int qrng_dim = QRNG_DIMENSIONS;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("qrng",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {(long long)N, (long long)qrng_dim}),
        KOKKOS_LAMBDA(const long long pos, const long long y) {
          unsigned int result = 0;
          unsigned int data = seed + (unsigned int)pos;
          for (int bit = 0; bit < qrng_res; bit++, data >>= 1)
            if (data & 1) result ^= d_table[(int)y * qrng_res + bit];
          d_output[(int)y * N + (int)pos] = (float)(result + 1) * INT_SCALE;
        });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (qrng): %f (us)\n", (t * 1e-3f) / repeat);

    printf("\nRead back results...\n");
    auto hm_output = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(hm_output, d_output);
    for (int i = 0; i < QRNG_DIMENSIONS * (int)N; i++) output[i] = hm_output(i);

    printf("Comparing to the CPU results...\n\n");
    sumDelta = 0; sumRef = 0;
    for (dim = 0; dim < QRNG_DIMENSIONS; dim++) {
      for (pos = 0; pos < N; pos++) {
        ref       = getQuasirandomValue63(pos, dim);
        delta     = (double)output[dim * N + pos] - ref;
        sumDelta += fabs(delta);
        sumRef   += fabs(ref);
      }
    }
    L1norm = sumDelta / sumRef;
    printf("  L1 norm: %E\n", L1norm);
    printf("  ckQuasirandomGenerator deviations %s Allowable Tolerance\n\n\n",
           (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
    bPassFlag = (L1norm < 1e-6);

    // InverseCND kernel
    printf(">>>Launch InverseCND kernel...\n\n");
    const unsigned int pathN    = QRNG_DIMENSIONS * N;
    const unsigned int distance = ((unsigned int)-1) / (pathN + 1);

    start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("icnd", Kokkos::RangePolicy<>(0, (int)pathN),
        KOKKOS_LAMBDA(const int pos) {
          unsigned int d = (unsigned int)(pos + 1) * distance;
          d_output[pos] = MoroInvCNDgpu(d);
        });
      Kokkos::fence();
    }

    end = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (icnd): %f (us)\n", (t * 1e-3f) / repeat);

    printf("\nRead back results...\n");
    Kokkos::deep_copy(hm_output, d_output);
    for (int i = 0; i < QRNG_DIMENSIONS * (int)N; i++) output[i] = hm_output(i);

    printf("Comparing to the CPU results...\n\n");
    sumDelta = 0; sumRef = 0;
    for (pos = 0; pos < QRNG_DIMENSIONS * N; pos++) {
      unsigned int d = (pos + 1) * distance;
      ref       = MoroInvCNDcpu(d);
      delta     = (double)output[pos] - ref;
      sumDelta += fabs(delta);
      sumRef   += fabs(ref);
    }
    L1norm = sumDelta / sumRef;
    printf("  L1 norm: %E\n", L1norm);
    printf("  ckInverseCNDGPU deviations %s Allowable Tolerance\n\n\n",
           (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
    bPassFlag &= (L1norm < 1e-6);

    if (bPassFlag) printf("PASS\n"); else printf("FAIL\n");

    free(output);
  }
  Kokkos::finalize();
  return 0;
}
