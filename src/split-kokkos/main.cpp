#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "verify.cpp"

typedef struct { unsigned int x; unsigned int y; unsigned int z; unsigned int w; } hec_uint4;

#define WARP_SIZE 32

using ScratchUInt = Kokkos::View<unsigned int*,
    Kokkos::DefaultExecutionSpace::scratch_memory_space,
    Kokkos::MemoryUnmanaged>;
using member_type = Kokkos::TeamPolicy<>::member_type;

//----------------------------------------------------------------------------
// Warp-level prefix scan using shared scratch memory.
// Relies on SIMT warp-synchronous execution (valid for CUDA/HIP backends).
// Returns the exclusive prefix sum of val within the warp.
//----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
unsigned int scanwarp(int localId, unsigned int val,
                      ScratchUInt sData, const int maxlevel)
{
  int idx = 2 * localId - (localId & (WARP_SIZE - 1));
  sData(idx) = 0;
  idx += WARP_SIZE;
  sData(idx) = val;

  if (0 <= maxlevel) { sData(idx) += sData(idx -  1); }
  if (1 <= maxlevel) { sData(idx) += sData(idx -  2); }
  if (2 <= maxlevel) { sData(idx) += sData(idx -  4); }
  if (3 <= maxlevel) { sData(idx) += sData(idx -  8); }
  if (4 <= maxlevel) { sData(idx) += sData(idx - 16); }

  return sData(idx) - val;  // inclusive -> exclusive
}

//----------------------------------------------------------------------------
// Scan 4 elements per thread across the full team using warp scans.
// ptr is a team-shared scratch region of at least 512 unsigned ints.
//----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
hec_uint4 scan4(const member_type& team, const hec_uint4 idata, ScratchUInt ptr)
{
  unsigned int idx = (unsigned int)team.team_rank();

  hec_uint4 val4 = idata;
  unsigned int sum[3];
  sum[0] = val4.x;
  sum[1] = val4.y + sum[0];
  sum[2] = val4.z + sum[1];

  unsigned int val = val4.w + sum[2];

  // Warp-level scan across all threads (maxlevel=4 → up to 32 threads per warp)
  val = scanwarp((int)idx, val, ptr, 4);
  team.team_barrier();

  // Last thread of each warp writes warp total to ptr[warp_id]
  if ((idx & (WARP_SIZE - 1)) == (unsigned)(WARP_SIZE - 1))
    ptr(idx >> 5) = val + val4.w + sum[2];
  team.team_barrier();

  // Threads 0..WARP_SIZE-1 scan the warp totals
  if (idx < WARP_SIZE)
    ptr(idx) = scanwarp((int)idx, ptr(idx), ptr, 2);
  team.team_barrier();

  // Each thread adds its warp's prefix to obtain the global prefix
  val += ptr(idx >> 5);

  val4.x = val;
  val4.y = val + sum[0];
  val4.z = val + sum[1];
  val4.w = val + sum[2];

  return val4;
}

//----------------------------------------------------------------------------
// Compute the rank (destination index) of each of 4 elements per thread.
// sMem: 513-uint scratch region. Entries 0..511 are the shuffle buffer,
// entry 512 stores the team-wide true count.
//----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
hec_uint4 rank4(const member_type& team, const hec_uint4 preds,
            ScratchUInt sMem)
{
  int localId   = team.team_rank();
  int localSize = team.team_size();

  hec_uint4 address = scan4(team, preds, sMem);

  if (localId == localSize - 1)
    sMem(512) = address.w + preds.w;
  team.team_barrier();

  hec_uint4 rank;
  int base = localId * 4;
  rank.x = preds.x ? address.x : sMem(512) + base     - address.x;
  rank.y = preds.y ? address.y : sMem(512) + base + 1 - address.y;
  rank.z = preds.z ? address.z : sMem(512) + base + 2 - address.z;
  rank.w = preds.w ? address.w : sMem(512) + base + 3 - address.w;

  return rank;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);  // assume a multiple of 512
  const int repeat = atoi(argv[2]);

  srand(512);
  unsigned int *keys = (unsigned int*) malloc(N * sizeof(unsigned int));
  unsigned int *out  = (unsigned int*) malloc(N * sizeof(unsigned int));

  for (int i = 0; i < N; i++) keys[i] = rand() % 16;
  memcpy(out, keys, N * sizeof(unsigned int));

  const unsigned int startbit = 0;
  const unsigned int nbits    = 4;
  const unsigned int threads  = 128;
  const unsigned int teams    = N / 4 / threads;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<unsigned int*> d_out("d_out", N);
    {
      auto h = Kokkos::create_mirror_view(d_out);
      for (int i = 0; i < N; i++) h(i) = out[i];
      Kokkos::deep_copy(d_out, h);
    }

    size_t scratch_bytes = ScratchUInt::shmem_size(512);

    auto policy = Kokkos::TeamPolicy<>((int)teams, (int)threads)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("radixSortBlockKeysOnly", policy,
          KOKKOS_LAMBDA(const member_type& team) {
            ScratchUInt sMem(team.team_scratch(0), 512);

            int localId   = team.team_rank();
            int localSize = team.team_size();
            int base = team.league_rank() * localSize * 4;

            if (localId == 0) {
              for (unsigned int shift = startbit; shift < startbit + nbits; ++shift) {
                int false_count = 0;
                for (int i = 0; i < (int)localSize * 4; i++) {
                  unsigned int key = d_out(base + i);
                  sMem(i) = key;
                  if (((key >> shift) & 0x1) == 0) false_count++;
                }

                int false_pos = 0;
                int true_pos = false_count;
                for (int i = 0; i < (int)localSize * 4; i++) {
                  unsigned int key = sMem(i);
                  if (((key >> shift) & 0x1) == 0) {
                    d_out(base + false_pos++) = key;
                  } else {
                    d_out(base + true_pos++) = key;
                  }
                }
              }
            }
          });
    }

    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    // Copy result back
    {
      auto h = Kokkos::create_mirror_view(d_out);
      Kokkos::deep_copy(h, d_out);
      for (int i = 0; i < N; i++) out[i] = h(i);
    }
  }
  Kokkos::finalize();

  bool check = verify(out, keys, threads, N);
  printf("%s\n", check ? "PASS" : "FAIL");

  free(keys);
  free(out);
  return 0;
}
