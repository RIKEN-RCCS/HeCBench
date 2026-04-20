/*
GFC code: A GPU-based compressor for arrays of double-precision
floating-point values.

Kokkos port.
*/

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <cstring>

#define ull unsigned long long
#define MAX (64*1024*1024)
#define WARPSIZE 32
#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

// ---- Device function: count leading zeros ----
KOKKOS_INLINE_FUNCTION
int clzll_device(ull num) {
  int count = 0;
  while (!(num & 0x1000000000000000ULL)) {
    count++;
    num <<= 1;
  }
  return count;
}

// ---- Compression kernel ----
// Uses TeamPolicy: nTeams teams, nThreads threads each.
// Shared ibufs[32*(3*WARPSIZE/2)] stored in team scratch.
void CompressionKernel(
    const int nTeams,
    const int nThreads,
    const int dimensionalityd,
    const ull*  cbufd,
    char*       dbufd,
    const int*  cutd,
    int*        offd)
{
  using team_policy_t = Kokkos::TeamPolicy<>;
  using team_member_t = team_policy_t::member_type;
  using ScratchSpace  = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView   = Kokkos::View<int*, ScratchSpace, Kokkos::MemoryUnmanaged>;

  // ibufs size: 32 * (3 * WARPSIZE / 2) = 1536 ints per team
  const int ibuf_size = 32 * (3 * WARPSIZE / 2);
  const size_t scratch_bytes = ibuf_size * sizeof(int);

  Kokkos::parallel_for(
    "CompressionKernel",
    team_policy_t(nTeams, nThreads)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchView ibufs(team.team_scratch(0), ibuf_size);

      int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
      ull diff, prev;

      int lid = team.team_rank();

      lane   = lid & 31;
      iindex = (lid / WARPSIZE) * (3 * WARPSIZE / 2) + lane;
      ibufs[iindex] = 0;
      iindex += WARPSIZE / 2;
      lastidx = ((lid / WARPSIZE) + 1) * (3 * WARPSIZE / 2) - 1;
      warp   = (lid + team.league_rank() * nThreads) / WARPSIZE;
      offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

      start = 0;
      if (warp > 0) start = cutd[warp - 1];
      term  = cutd[warp];
      off   = ((start + 1) / 2 * 17);

      prev = 0;
      for (int i = start + lane; i < term; i += WARPSIZE) {
        diff  = cbufd[i] - prev;
        code  = (int)((diff >> 60) & 8);
        if (code != 0) diff = (ull)(-(long long)diff);

        bcount = 8 - (clzll_device(diff) >> 3);
        if (bcount == 2) bcount = 3;

        // Prefix sum (6 steps for WARPSIZE=32)
        ibufs[iindex] = bcount;
        team.team_barrier();
        ibufs[iindex] += ibufs[iindex - 1];
        team.team_barrier();
        ibufs[iindex] += ibufs[iindex - 2];
        team.team_barrier();
        ibufs[iindex] += ibufs[iindex - 4];
        team.team_barrier();
        ibufs[iindex] += ibufs[iindex - 8];
        team.team_barrier();
        ibufs[iindex] += ibufs[iindex - 16];
        team.team_barrier();

        beg = off + (WARPSIZE / 2) + ibufs[iindex - 1];
        end = beg + bcount;
        for (; beg < end; beg++) {
          dbufd[beg] = (char)(diff & 0xFF);
          diff >>= 8;
        }

        if (bcount >= 3) bcount--;
        tmp = ibufs[lastidx];
        code |= bcount;
        ibufs[iindex] = code;
        team.team_barrier();

        if ((lane & 1) != 0) {
          dbufd[off + (lane >> 1)] = (char)(ibufs[iindex - 1] | (code << 4));
        }
        off += tmp + (WARPSIZE / 2);

        if (i + offset < term)
          prev = cbufd[i + offset];
      }

      if (lane == 31) offd[warp] = off;
    }
  );
}

// ---- Compress driver ----
static void Compress(int blocks, int warpsperblock, int repeat, int dimensionality)
{
  FILE* fp = fopen("input.bin", "wb");
  if (!fp) { fprintf(stderr, "Failed to open input.bin for write.\n"); return; }
  for (int i = 0; i < MAX; i++) { double t = (double)i; fwrite(&t, 8, 1, fp); }
  fclose(fp);

  fp = fopen("input.bin", "rb");
  if (!fp) { fprintf(stderr, "Failed to open input.bin for read.\n"); return; }

  ull* cbuf = (ull*)malloc(sizeof(ull) * MAX);
  if (!cbuf) { fprintf(stderr, "cannot allocate cbuf\n"); fclose(fp); return; }

  int doubles = (int)fread(cbuf, 8, MAX, fp);
  fclose(fp);
  if (doubles != MAX) {
    fprintf(stderr, "Error reading input.bin.\n");
    free(cbuf); return;
  }

  const int num_warps = blocks * warpsperblock;
  char* dbuf = (char*)malloc(sizeof(char) * ((MAX + 1) / 2 * 17));
  int*  cut  = (int*)malloc(sizeof(int) * num_warps);
  int*  off  = (int*)malloc(sizeof(int) * num_warps);
  if (!dbuf || !cut || !off) { fprintf(stderr, "alloc failed\n"); free(cbuf); return; }

  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  int per = (doubles + num_warps - 1) / num_warps;
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < num_warps; i++) {
    curr += per;
    cut[i] = min(curr, doubles);
    if (cut[i] - before > 0) d = cut[i] - before;
    before = cut[i];
  }

  if (d <= WARPSIZE) {
    for (int i = doubles - padding; i < doubles; i++) cbuf[i] = 0;
  } else {
    for (int i = doubles - padding; i < doubles; i++)
      cbuf[i] = cbuf[(i & -WARPSIZE) - (dimensionality - i % dimensionality)];
  }

  // Allocate device views
  Kokkos::View<ull*>  d_cbuf("cbuf", doubles);
  Kokkos::View<char*> d_dbuf("dbuf", (doubles+1)/2*17);
  Kokkos::View<int*>  d_cut("cut",  num_warps);
  Kokkos::View<int*>  d_off("off",  num_warps);

  {
    auto h_cbuf = Kokkos::create_mirror_view(d_cbuf);
    for (int i = 0; i < doubles; i++) h_cbuf(i) = cbuf[i];
    Kokkos::deep_copy(d_cbuf, h_cbuf);
  }
  {
    auto h_cut = Kokkos::create_mirror_view(d_cut);
    for (int i = 0; i < num_warps; i++) h_cut(i) = cut[i];
    Kokkos::deep_copy(d_cut, h_cut);
  }

  auto t_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    CompressionKernel(blocks, WARPSIZE * warpsperblock,
                      dimensionality,
                      d_cbuf.data(), d_dbuf.data(),
                      d_cut.data(),  d_off.data());
    Kokkos::fence();
  }
  auto t_end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
  fprintf(stderr, "Average compression kernel execution time %f (s)\n",
          (elapsed * 1e-9) / repeat);

  // Copy offsets back
  {
    auto h_off = Kokkos::create_mirror_view(d_off);
    Kokkos::deep_copy(h_off, d_off);
    for (int i = 0; i < num_warps; i++) off[i] = h_off(i);
  }

  fp = fopen("output.bin", "wb");
  if (!fp) { fprintf(stderr, "Failed to open output.bin.\n"); goto cleanup; }

  {
    int doublecnt = doubles - padding;
    fwrite(&blocks, 1, 1, fp);
    fwrite(&warpsperblock, 1, 1, fp);
    fwrite(&dimensionality, 1, 1, fp);
    fwrite(&doublecnt, 4, 1, fp);

    for (int i = 0; i < num_warps; i++) {
      int start_i = (i > 0) ? cut[i-1] : 0;
      off[i] -= ((start_i + 1) / 2 * 17);
      fwrite(&off[i], 4, 1, fp);
    }

    auto h_dbuf = Kokkos::create_mirror_view(d_dbuf);
    Kokkos::deep_copy(h_dbuf, d_dbuf);
    for (int i = 0; i < num_warps; i++) {
      int start_i = (i > 0) ? cut[i-1] : 0;
      int byte_off = ((start_i + 1) / 2 * 17);
      // copy to host buffer
      for (int k = 0; k < off[i]; k++) dbuf[byte_off + k] = h_dbuf(byte_off + k);
      fwrite(&dbuf[byte_off], 1, off[i], fp);
    }
    fclose(fp);

    fp = fopen("input.bin", "rb");
    fseek(fp, 0, SEEK_END); long input_sz = ftell(fp); fclose(fp);
    fp = fopen("output.bin", "rb");
    fseek(fp, 0, SEEK_END); long output_sz = ftell(fp); fclose(fp);
    fprintf(stderr, "Compression ratio = %lf\n", 1.0 * input_sz / output_sz);
  }

cleanup:
  free(cbuf); free(dbuf); free(cut); free(off);
}

static void VerifySystemParameters() {
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));
  int val = 1;
  assert(1 == *((char*)&val));
  if ((WARPSIZE <= 0) || ((WARPSIZE & (WARPSIZE-1)) != 0)) {
    fprintf(stderr, "Warp size must be > 0 and power of 2\n");
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  fprintf(stderr, "GPU FP Compressor v2.2 (Kokkos port)\n");
  VerifySystemParameters();

  if ((4 == argc) || (5 == argc)) {
    int blocks          = atoi(argv[1]);
    int warpsperblock   = atoi(argv[2]);
    int repeat          = atoi(argv[3]);
    int dimensionality  = (5 == argc) ? atoi(argv[4]) : 1;

    assert((0 < blocks)         && (blocks < 256));
    assert((0 < warpsperblock)  && (warpsperblock < 256));
    assert((0 < dimensionality) && (dimensionality <= WARPSIZE));

    Kokkos::initialize(argc, argv);
    Compress(blocks, warpsperblock, repeat, dimensionality);
    Kokkos::finalize();
  } else {
    fprintf(stderr, "usage: compress <blocks> <warps/block> <repeat> [dimensionality]\n");
  }
  return 0;
}
