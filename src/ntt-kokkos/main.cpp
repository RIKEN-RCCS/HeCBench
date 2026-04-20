/*
 * Kokkos port of the NTT (Number Theoretic Transform) intt_3_64k_modcrt kernel.
 *
 * The original OMP kernel uses 64 threads per team with team-shared memory.
 * For the Kokkos OpenMP (CPU) backend, team_size > 1 is not supported, so
 * this port uses league-level parallelism (one team per block) and a local
 * 512-element buffer.  Each "team" (single thread) processes all 64 virtual
 * thread lanes sequentially; the two phases are separated by the natural
 * sequential ordering that replaces the OMP barrier.
 *
 * On a GPU backend (CUDA/HIP) the TeamPolicy + scratch-memory variant
 * should be used instead; that variant is left in comments below for reference.
 *
 * Args: <repeat>
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include "modP.h"

// Each logical block processes 512 src elements → 512 dst elements.
// 64 virtual threads per block, 8 elements each.
static const int THREADS_PER_BLOCK = 64;
static const int BUF_SIZE          = 512;   // == THREADS_PER_BLOCK * 8

void intt_3_64k_modcrt(
    const uint32          numBlocks,
    Kokkos::View<uint32*> d_dst,
    Kokkos::View<uint64*> d_src)
{
  // One Kokkos "team" = one logical block.  team_size=1 for OpenMP backend.
  Kokkos::parallel_for("intt_3_64k",
    Kokkos::RangePolicy<>(0, (int)numBlocks),
    KOKKOS_LAMBDA(int bidx) {

      // Team-local shared buffer (512 uint64).
      // On CPU this lives on the call stack.
      uint64 buffer[BUF_SIZE];

      // ---- Phase 1: load, NTT8, twiddle, write to buffer ----
      for (int tidx = 0; tidx < THREADS_PER_BLOCK; tidx++) {
        uint64 samples[8];
        uint32 fmem = (bidx << 9) | ((tidx & 0x3E) << 3) | (tidx & 0x1);
        uint32 tbuf = tidx << 3;

        for (int i = 0; i < 8; i++)
          samples[i] = d_src(fmem | (i << 1));

        ntt8(samples);

        for (int i = 0; i < 8; i++)
          buffer[tbuf | i] = _ls_modP(samples[i], ((tidx & 0x1) << 2) * i * 3);
      }

      // ---- Phase 2 (after implicit barrier): butterfly and write to dst ----
      for (int tidx = 0; tidx < THREADS_PER_BLOCK; tidx++) {
        uint64 samples[8], s8[8];
        uint32 fbuf = ((tidx & 0x38) << 3) | (tidx & 0x7);
        uint32 tmem = (bidx << 9) | ((tidx & 0x38) << 3) | (tidx & 0x7);

        for (int i = 0; i < 8; i++)
          samples[i] = buffer[fbuf | (i << 3)];

        for (int i = 0; i < 4; i++) {
          s8[2 * i]     = _add_modP(samples[2 * i],     samples[2 * i + 1]);
          s8[2 * i + 1] = _sub_modP(samples[2 * i],     samples[2 * i + 1]);
        }

        for (int i = 0; i < 8; i++) {
          uint32 out_idx =
              (((tmem | (i << 3)) & 0xf) << 12) | ((tmem | (i << 3)) >> 4);
          d_dst(out_idx) =
              (uint32)(_mul_modP(s8[i], 18446462594437939201UL, valP));
        }
      }
    });
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int nttLen = 64 * 1024;

  // Host input/output
  std::vector<uint64_t> h_ntt(nttLen);
  std::vector<uint32_t> h_res(nttLen, 0);

  srand(123);
  for (int i = 0; i < nttLen; i++) {
    uint64 hi = rand(), lo = rand();
    h_ntt[i] = (hi << 32) | lo;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<uint64*> d_src("ntt_src", nttLen);
    Kokkos::View<uint32*> d_dst("ntt_dst", nttLen);

    {
      auto hv = Kokkos::create_mirror_view(d_src);
      for (int i = 0; i < nttLen; i++) hv(i) = h_ntt[i];
      Kokkos::deep_copy(d_src, hv);
    }

    const uint32 numTeams = nttLen / 512;  // = 128

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      intt_3_64k_modcrt(numTeams, d_dst, d_src);
    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    auto hv = Kokkos::create_mirror_view(d_dst);
    Kokkos::deep_copy(hv, d_dst);
    for (int i = 0; i < nttLen; i++) h_res[i] = hv(i);
  }
  Kokkos::finalize();

  uint64 checksum = 0;
  for (int i = 0; i < nttLen; i++) checksum += h_res[i];
  printf("Checksum: %lu\n", checksum);

  return 0;
}
