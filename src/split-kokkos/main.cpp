#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "../split-omp/verify.cpp"

typedef struct { unsigned int x; unsigned int y; unsigned int z; unsigned int w; } uint4;

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
uint4 scan4(const member_type& team, const uint4 idata, ScratchUInt ptr)
{
  unsigned int idx = (unsigned int)team.team_rank();

  uint4 val4 = idata;
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
// sMem: 512-uint scratch region.
// numtrue: 1-uint scratch region for the team-wide true count.
//----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
uint4 rank4(const member_type& team, const uint4 preds,
            ScratchUInt sMem, ScratchUInt numtrue)
{
  int localId   = team.team_rank();
  int localSize = team.team_size();

  uint4 address = scan4(team, preds, sMem);

  if (localId == localSize - 1)
    numtrue(0) = address.w + preds.w;
  team.team_barrier();

  uint4 rank;
  int base = localId * 4;
  rank.x = preds.x ? address.x : numtrue(0) + base     - address.x;
  rank.y = preds.y ? address.y : numtrue(0) + base + 1 - address.y;
  rank.z = preds.z ? address.z : numtrue(0) + base + 2 - address.z;
  rank.w = preds.w ? address.w : numtrue(0) + base + 3 - address.w;

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

    // Scratch layout per team:
    //   sMem    : 512 uints  (warp-scan workspace + key shuffle buffer)
    //   numtrue :   1 uint   (team-wide true count for rank4)
    size_t scratch_bytes = ScratchUInt::shmem_size(512)
                         + ScratchUInt::shmem_size(1);

    auto policy = Kokkos::TeamPolicy<>((int)teams, (int)threads)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("radixSortBlockKeysOnly", policy,
          KOKKOS_LAMBDA(const member_type& team) {
            ScratchUInt sMem   (team.team_scratch(0), 512);
            ScratchUInt numtrue(team.team_scratch(0), 1);

            int localId   = team.team_rank();
            int localSize = team.team_size();
            int globalId  = team.league_rank() * localSize + localId;

            // Load 4 keys per thread
            uint4 key;
            key.x = d_out(4 * globalId);
            key.y = d_out(4 * globalId + 1);
            key.z = d_out(4 * globalId + 2);
            key.w = d_out(4 * globalId + 3);

            for (unsigned int shift = startbit; shift < startbit + nbits; ++shift) {
              uint4 lsb;
              lsb.x = !((key.x >> shift) & 0x1);
              lsb.y = !((key.y >> shift) & 0x1);
              lsb.z = !((key.z >> shift) & 0x1);
              lsb.w = !((key.w >> shift) & 0x1);

              uint4 r = rank4(team, lsb, sMem, numtrue);

              // Stride ranks across 4 regions of size localSize to avoid bank conflicts
              sMem((r.x & 3) * localSize + (r.x >> 2)) = key.x;
              sMem((r.y & 3) * localSize + (r.y >> 2)) = key.y;
              sMem((r.z & 3) * localSize + (r.z >> 2)) = key.z;
              sMem((r.w & 3) * localSize + (r.w >> 2)) = key.w;
              team.team_barrier();

              // Read back in sorted order
              key.x = sMem(localId);
              key.y = sMem(localId +     localSize);
              key.z = sMem(localId + 2 * localSize);
              key.w = sMem(localId + 3 * localSize);
              team.team_barrier();
            }

            d_out(4 * globalId)     = key.x;
            d_out(4 * globalId + 1) = key.y;
            d_out(4 * globalId + 2) = key.z;
            d_out(4 * globalId + 3) = key.w;
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
