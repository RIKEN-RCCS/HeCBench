/*
 *  Extension to the interleaving example in CUDA Programming by Shane Cook
 *  Ported to Kokkos
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUM_ELEMENTS 4096
#define COUNT        4096

typedef struct {
  unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
  unsigned int s8, s9, sa, sb, sc, sd, se, sf;
} INTERLEAVED_T;

typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct {
  ARRAY_MEMBER_T s0, s1, s2, s3, s4, s5, s6, s7;
  ARRAY_MEMBER_T s8, s9, sa, sb, sc, sd, se, sf;
} NON_INTERLEAVED_T;

// ---------- init & verify (inlined from util.cpp) ----------

void initialize(INTERLEAVED_ARRAY_T &interleaved_src,
                INTERLEAVED_ARRAY_T &interleaved_dst,
                NON_INTERLEAVED_T   &non_interleaved_src,
                NON_INTERLEAVED_T   &non_interleaved_dst,
                const int n)
{
  for (int i = 0; i < n; i++) {
    interleaved_src[i].s0 = non_interleaved_src.s0[i] = rand() % 16;
    interleaved_src[i].s1 = non_interleaved_src.s1[i] = rand() % 16;
    interleaved_src[i].s2 = non_interleaved_src.s2[i] = rand() % 16;
    interleaved_src[i].s3 = non_interleaved_src.s3[i] = rand() % 16;
    interleaved_src[i].s4 = non_interleaved_src.s4[i] = rand() % 16;
    interleaved_src[i].s5 = non_interleaved_src.s5[i] = rand() % 16;
    interleaved_src[i].s6 = non_interleaved_src.s6[i] = rand() % 16;
    interleaved_src[i].s7 = non_interleaved_src.s7[i] = rand() % 16;
    interleaved_src[i].s8 = non_interleaved_src.s8[i] = rand() % 16;
    interleaved_src[i].s9 = non_interleaved_src.s9[i] = rand() % 16;
    interleaved_src[i].sa = non_interleaved_src.sa[i] = rand() % 16;
    interleaved_src[i].sb = non_interleaved_src.sb[i] = rand() % 16;
    interleaved_src[i].sc = non_interleaved_src.sc[i] = rand() % 16;
    interleaved_src[i].sd = non_interleaved_src.sd[i] = rand() % 16;
    interleaved_src[i].se = non_interleaved_src.se[i] = rand() % 16;
    interleaved_src[i].sf = non_interleaved_src.sf[i] = rand() % 16;
    interleaved_dst[i].s0 = non_interleaved_dst.s0[i] = 0;
    interleaved_dst[i].s1 = non_interleaved_dst.s1[i] = 0;
    interleaved_dst[i].s2 = non_interleaved_dst.s2[i] = 0;
    interleaved_dst[i].s3 = non_interleaved_dst.s3[i] = 0;
    interleaved_dst[i].s4 = non_interleaved_dst.s4[i] = 0;
    interleaved_dst[i].s5 = non_interleaved_dst.s5[i] = 0;
    interleaved_dst[i].s6 = non_interleaved_dst.s6[i] = 0;
    interleaved_dst[i].s7 = non_interleaved_dst.s7[i] = 0;
    interleaved_dst[i].s8 = non_interleaved_dst.s8[i] = 0;
    interleaved_dst[i].s9 = non_interleaved_dst.s9[i] = 0;
    interleaved_dst[i].sa = non_interleaved_dst.sa[i] = 0;
    interleaved_dst[i].sb = non_interleaved_dst.sb[i] = 0;
    interleaved_dst[i].sc = non_interleaved_dst.sc[i] = 0;
    interleaved_dst[i].sd = non_interleaved_dst.sd[i] = 0;
    interleaved_dst[i].se = non_interleaved_dst.se[i] = 0;
    interleaved_dst[i].sf = non_interleaved_dst.sf[i] = 0;
  }
}

void verify(INTERLEAVED_ARRAY_T &interleaved_dst,
            NON_INTERLEAVED_T   &non_interleaved_dst,
            const int n)
{
  for (int i = 0; i < n; i++) {
    assert(interleaved_dst[i].s0 == non_interleaved_dst.s0[i]);
    assert(interleaved_dst[i].s1 == non_interleaved_dst.s1[i]);
    assert(interleaved_dst[i].s2 == non_interleaved_dst.s2[i]);
    assert(interleaved_dst[i].s3 == non_interleaved_dst.s3[i]);
    assert(interleaved_dst[i].s4 == non_interleaved_dst.s4[i]);
    assert(interleaved_dst[i].s5 == non_interleaved_dst.s5[i]);
    assert(interleaved_dst[i].s6 == non_interleaved_dst.s6[i]);
    assert(interleaved_dst[i].s7 == non_interleaved_dst.s7[i]);
    assert(interleaved_dst[i].s8 == non_interleaved_dst.s8[i]);
    assert(interleaved_dst[i].s9 == non_interleaved_dst.s9[i]);
    assert(interleaved_dst[i].sa == non_interleaved_dst.sa[i]);
    assert(interleaved_dst[i].sb == non_interleaved_dst.sb[i]);
    assert(interleaved_dst[i].sc == non_interleaved_dst.sc[i]);
    assert(interleaved_dst[i].sd == non_interleaved_dst.sd[i]);
    assert(interleaved_dst[i].se == non_interleaved_dst.se[i]);
    assert(interleaved_dst[i].sf == non_interleaved_dst.sf[i]);
  }
}

// ---------- Kokkos test functions ----------

void add_test_interleaved(INTERLEAVED_ARRAY_T h_dst,
                          const INTERLEAVED_ARRAY_T h_src,
                          const int repeat)
{
  Kokkos::View<INTERLEAVED_T *> d_src("interleaved_src", NUM_ELEMENTS);
  Kokkos::View<INTERLEAVED_T *> d_dst("interleaved_dst", NUM_ELEMENTS);

  {
    auto m_src = Kokkos::create_mirror_view(d_src);
    auto m_dst = Kokkos::create_mirror_view(d_dst);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
      m_src(i) = h_src[i];
      m_dst(i) = h_dst[i];
    }
    Kokkos::deep_copy(d_src, m_src);
    Kokkos::deep_copy(d_dst, m_dst);
  }

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for(
        "add_interleaved",
        Kokkos::RangePolicy<>(0, NUM_ELEMENTS),
        KOKKOS_LAMBDA(int tid) {
          for (unsigned int i = 0; i < COUNT; i++) {
            d_dst(tid).s0 += d_src(tid).s0;
            d_dst(tid).s1 += d_src(tid).s1;
            d_dst(tid).s2 += d_src(tid).s2;
            d_dst(tid).s3 += d_src(tid).s3;
            d_dst(tid).s4 += d_src(tid).s4;
            d_dst(tid).s5 += d_src(tid).s5;
            d_dst(tid).s6 += d_src(tid).s6;
            d_dst(tid).s7 += d_src(tid).s7;
            d_dst(tid).s8 += d_src(tid).s8;
            d_dst(tid).s9 += d_src(tid).s9;
            d_dst(tid).sa += d_src(tid).sa;
            d_dst(tid).sb += d_src(tid).sb;
            d_dst(tid).sc += d_src(tid).sc;
            d_dst(tid).sd += d_src(tid).sd;
            d_dst(tid).se += d_src(tid).se;
            d_dst(tid).sf += d_src(tid).sf;
          }
        });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (interleaved) execution time %f (s)\n",
         (time * 1e-9f) / repeat);

  auto m_dst = Kokkos::create_mirror_view(d_dst);
  Kokkos::deep_copy(m_dst, d_dst);
  for (int i = 0; i < NUM_ELEMENTS; i++)
    h_dst[i] = m_dst(i);
}

void add_test_non_interleaved(NON_INTERLEAVED_T *h_dst,
                              const NON_INTERLEAVED_T *h_src,
                              const int repeat)
{
  Kokkos::View<NON_INTERLEAVED_T *> d_src("non_interleaved_src", 1);
  Kokkos::View<NON_INTERLEAVED_T *> d_dst("non_interleaved_dst", 1);

  {
    auto m_src = Kokkos::create_mirror_view(d_src);
    auto m_dst = Kokkos::create_mirror_view(d_dst);
    m_src(0) = *h_src;
    m_dst(0) = *h_dst;
    Kokkos::deep_copy(d_src, m_src);
    Kokkos::deep_copy(d_dst, m_dst);
  }

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for(
        "add_non_interleaved",
        Kokkos::RangePolicy<>(0, NUM_ELEMENTS),
        KOKKOS_LAMBDA(int tid) {
          for (unsigned int i = 0; i < COUNT; i++) {
            d_dst(0).s0[tid] += d_src(0).s0[tid];
            d_dst(0).s1[tid] += d_src(0).s1[tid];
            d_dst(0).s2[tid] += d_src(0).s2[tid];
            d_dst(0).s3[tid] += d_src(0).s3[tid];
            d_dst(0).s4[tid] += d_src(0).s4[tid];
            d_dst(0).s5[tid] += d_src(0).s5[tid];
            d_dst(0).s6[tid] += d_src(0).s6[tid];
            d_dst(0).s7[tid] += d_src(0).s7[tid];
            d_dst(0).s8[tid] += d_src(0).s8[tid];
            d_dst(0).s9[tid] += d_src(0).s9[tid];
            d_dst(0).sa[tid] += d_src(0).sa[tid];
            d_dst(0).sb[tid] += d_src(0).sb[tid];
            d_dst(0).sc[tid] += d_src(0).sc[tid];
            d_dst(0).sd[tid] += d_src(0).sd[tid];
            d_dst(0).se[tid] += d_src(0).se[tid];
            d_dst(0).sf[tid] += d_src(0).sf[tid];
          }
        });
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (non-interleaved) execution time %f (s)\n",
         (time * 1e-9f) / repeat);

  auto m_dst = Kokkos::create_mirror_view(d_dst);
  Kokkos::deep_copy(m_dst, d_dst);
  *h_dst = m_dst(0);
}

// ---------- main ----------

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  static NON_INTERLEAVED_T non_interleaved_src, non_interleaved_dst;
  static INTERLEAVED_ARRAY_T interleaved_src, interleaved_dst;

  initialize(interleaved_src, interleaved_dst,
             non_interleaved_src, non_interleaved_dst, NUM_ELEMENTS);

  Kokkos::initialize(argc, argv);
  {
    add_test_non_interleaved(&non_interleaved_dst, &non_interleaved_src, repeat);
    add_test_interleaved(interleaved_dst, interleaved_src, repeat);
  }
  Kokkos::finalize();

  verify(interleaved_dst, non_interleaved_dst, NUM_ELEMENTS);
  printf("PASS\n");
  return 0;
}
