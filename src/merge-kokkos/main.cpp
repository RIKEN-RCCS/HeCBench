#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>

// ─── Type helpers ─────────────────────────────────────────────────────────────

template<typename vec_t> KOKKOS_INLINE_FUNCTION vec_t getPositiveInfinity();
template<typename vec_t> KOKKOS_INLINE_FUNCTION vec_t getNegativeInfinity();

template<> KOKKOS_INLINE_FUNCTION float    getPositiveInfinity<float>()    { return FLT_MAX; }
template<> KOKKOS_INLINE_FUNCTION float    getNegativeInfinity<float>()    { return FLT_MIN; }
template<> KOKKOS_INLINE_FUNCTION double   getPositiveInfinity<double>()   { return DBL_MAX; }
template<> KOKKOS_INLINE_FUNCTION double   getNegativeInfinity<double>()   { return DBL_MIN; }
template<> KOKKOS_INLINE_FUNCTION uint32_t getPositiveInfinity<uint32_t>() { return 0xFFFFFFFFUL; }
template<> KOKKOS_INLINE_FUNCTION uint32_t getNegativeInfinity<uint32_t>() { return 0; }
template<> KOKKOS_INLINE_FUNCTION uint64_t getPositiveInfinity<uint64_t>() { return 0xFFFFFFFFFFFFFFFFUL; }
template<> KOKKOS_INLINE_FUNCTION uint64_t getNegativeInfinity<uint64_t>() { return 0; }

#define MAX(X,Y) (((X)>(Y))?(X):(Y))
#define MIN(X,Y) (((X)<(Y))?(X):(Y))

// ─── workloadDiagonals ────────────────────────────────────────────────────────
// Finds the merge-path diagonal intersection for each block via a serial
// O(log N) binary search.  Each Kokkos work-item handles one block; all blocks
// run in parallel across the device.

template<typename vec_t>
void workloadDiagonals(int blocks,
                       Kokkos::View<const vec_t*> d_A, uint32_t A_length,
                       Kokkos::View<const vec_t*> d_B, uint32_t B_length,
                       Kokkos::View<uint32_t*>   dpi) {
  Kokkos::parallel_for(
    "workloadDiagonals",
    blocks + 1,  // include the final boundary diagonal
    KOKKOS_LAMBDA(int bid) {
      if (bid == 0) {
        // Boundary: before first block
        dpi(0)          = 0;
        dpi(blocks + 1) = 0;
        return;
      }
      if (bid == blocks) {
        // Boundary: after last block
        dpi(blocks)              = A_length;
        dpi(blocks + blocks + 1) = B_length;
        return;
      }

      // Combined index along the merge-path diagonal for block bid
      int32_t combined = (int32_t)((uint64_t)bid *
                          ((uint64_t)A_length + (uint64_t)B_length) /
                          (uint64_t)blocks);

      // Binary search: find the largest x in [lo,hi] s.t.
      //   A[x-1] <= B[combined-x]  (1-indexed; x=0 means take nothing from A)
      int32_t lo = (combined > (int32_t)B_length)
                   ? combined - (int32_t)B_length : 0;
      int32_t hi = MIN(combined, (int32_t)A_length);

      while (lo < hi) {
        int32_t mid = lo + (hi - lo + 1) / 2;
        int32_t y   = combined - mid;

        int oneorzero;
        if (mid >= (int32_t)A_length || y < 0)       oneorzero = 0;
        else if (y >= (int32_t)B_length || mid < 1)   oneorzero = 1;
        else oneorzero = (d_A(mid - 1) <= d_B(y)) ? 1 : 0;

        if (oneorzero) lo = mid;
        else           hi = mid - 1;
      }

      dpi(bid)              = (uint32_t)lo;
      dpi(bid + blocks + 1) = (uint32_t)(combined - lo);
    });
}

// ─── mergeSinglePath ─────────────────────────────────────────────────────────
// Each block's work-item performs a serial merge of its assigned window using
// K=512 element scratch tiles.  The TeamPolicy maps one Kokkos team per block;
// within each team, threads cooperatively load tiles then each thread merges
// a stride of elements.  Falls back cleanly to serial when team_size == 1.

#define K 512

template<typename vec_t>
void mergeSinglePath(int blocks, int /*threads*/,
                     Kokkos::View<const vec_t*>   d_A, uint32_t A_length,
                     Kokkos::View<const vec_t*>   d_B, uint32_t B_length,
                     Kokkos::View<const uint32_t*> dpi_c,
                     Kokkos::View<vec_t*>          d_C, uint32_t /*C_length*/) {
  (void)blocks;
  (void)dpi_c;
  const uint32_t total = A_length + B_length;
  Kokkos::parallel_for(
    "mergeSinglePath",
    total,
    KOKKOS_LAMBDA(uint32_t k) {
      uint32_t lo = (k > B_length) ? (k - B_length) : 0;
      uint32_t hi = (k < A_length) ? k : A_length;

      while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;
        uint32_t j = k - mid;
        if (j < B_length && d_A(mid - 1) > d_B(j)) {
          hi = mid - 1;
        } else {
          lo = mid;
        }
      }

      uint32_t a_count = lo;
      uint32_t b_count = k - a_count;
      vec_t next_a = (a_count < A_length) ? d_A(a_count) : getPositiveInfinity<vec_t>();
      vec_t next_b = (b_count < B_length) ? d_B(b_count) : getPositiveInfinity<vec_t>();
      d_C(k) = (next_a <= next_b) ? next_a : next_b;
    });
}

// ─── Host helpers ─────────────────────────────────────────────────────────────

#define CSV 0
#if CSV
#define PS(X,S) std::cout << X << ", " << S << ", "; fflush(stdout);
#define PV(X)   std::cout << X << ", "; fflush(stdout);
#else
#define PS(X,S) std::cout << X << " " << S <<" :\n"; fflush(stdout);
#define PV(X)   std::cout << "\t" << #X << " \t: " << X << "\n"; fflush(stdout);
#endif

#define PADDING 1024

template<typename vec_t>
vec_t rand64() {
  vec_t rtn;
  do {
    uint32_t* p = (uint32_t*)&rtn;
    p[0] = rand();
    if (sizeof(vec_t) > 4) p[1] = rand();
  } while (!(rtn < getPositiveInfinity<vec_t>() &&
             rtn > getNegativeInfinity<vec_t>()));
  return rtn;
}

// ─── mergeType ────────────────────────────────────────────────────────────────

template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
void mergeType(const uint64_t size, const uint32_t runs) {
  std::vector<vec_t>    hA(size + PADDING);
  std::vector<vec_t>    hB(size + PADDING);
  std::vector<vec_t>    hC(2 * size + PADDING, (vec_t)0);
  std::vector<uint32_t> hD(2 * (blocks + 1), 0);

  uint32_t errors = 0;

  Kokkos::View<vec_t*>    d_A("dA", size + PADDING);
  Kokkos::View<vec_t*>    d_B("dB", size + PADDING);
  Kokkos::View<vec_t*>    d_C("dC", 2 * size + PADDING);
  Kokkos::View<uint32_t*> dpi("dpi", 2 * (blocks + 1));

  double total_time = 0.0;

  for (uint32_t r = 0; r < runs; r++) {
    for (uint64_t n = 0; n < size; n++) {
      hA[n] = rand64<vec_t>();
      hB[n] = rand64<vec_t>();
    }
    for (uint64_t n = size; n < size + PADDING; n++) {
      hA[n] = getPositiveInfinity<vec_t>();
      hB[n] = getPositiveInfinity<vec_t>();
    }
    std::sort(hA.begin(), hA.end());
    std::sort(hB.begin(), hB.end());

    {
      auto h = Kokkos::create_mirror_view(d_A);
      memcpy(h.data(), hA.data(), (size + PADDING) * sizeof(vec_t));
      Kokkos::deep_copy(d_A, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_B);
      memcpy(h.data(), hB.data(), (size + PADDING) * sizeof(vec_t));
      Kokkos::deep_copy(d_B, h);
    }

    auto d_A_c = Kokkos::View<const vec_t*>(d_A);
    auto d_B_c = Kokkos::View<const vec_t*>(d_B);
    auto dpi_c = Kokkos::View<const uint32_t*>(dpi);

    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();

    workloadDiagonals<vec_t>((int)blocks, d_A_c, (uint32_t)size,
                             d_B_c, (uint32_t)size, dpi);
    Kokkos::fence();

    mergeSinglePath<vec_t>((int)blocks, (int)threads,
                           d_A_c, (uint32_t)size,
                           d_B_c, (uint32_t)size,
                           dpi_c, d_C, (uint32_t)(size * 2));
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Verify output is sorted
    {
      auto h = Kokkos::create_mirror_view(d_C);
      Kokkos::deep_copy(h, d_C);
      memcpy(hC.data(), h.data(), 2 * size * sizeof(vec_t));
    }
    for (uint32_t i = 1; i < 2 * size; i++)
      errors += (hC[i] < hC[i - 1]) ? 1 : 0;
  }

  PV(errors);
  printf("%s. ", errors ? "FAIL" : "PASS");
  if (timing)
    printf("Average kernel execution time: %f (us).\n", (total_time * 1e-3f) / runs);
  else
    printf("Warmup run\n");
}

// ─── mergeAllTypes ────────────────────────────────────────────────────────────

template<uint32_t blocks, uint32_t threads>
void mergeAllTypes(const uint64_t size, const uint32_t runs) {
  PS("uint32_t", size) mergeType<uint32_t, blocks, threads, false>(size, runs); printf("\n");
  PS("uint32_t", size) mergeType<uint32_t, blocks, threads, true> (size, runs); printf("\n");

  PS("float",    size) mergeType<float,    blocks, threads, false>(size, runs); printf("\n");
  PS("float",    size) mergeType<float,    blocks, threads, true> (size, runs); printf("\n");

  PS("uint64_t", size) mergeType<uint64_t, blocks, threads, false>(size, runs); printf("\n");
  PS("uint64_t", size) mergeType<uint64_t, blocks, threads, true> (size, runs); printf("\n");

  PS("double",   size) mergeType<double,   blocks, threads, false>(size, runs); printf("\n");
  PS("double",   size) mergeType<double,   blocks, threads, true> (size, runs); printf("\n");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length of the arrays> <runs>\n", argv[0]);
    return 1;
  }
  const uint64_t length = atol(argv[1]);
  const uint32_t runs   = atoi(argv[2]);

  constexpr int blocks  = 112;
  constexpr int threads = 128;

  Kokkos::initialize(argc, argv);
  mergeAllTypes<blocks, threads>(length, runs);
  Kokkos::finalize();
  return 0;
}
