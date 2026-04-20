#include "utils.h"
#include <chrono>
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Phase 1 – scan: each team (≡ CUDA block) sweeps its assigned input chunks
// and writes one min and one max value per team into the scratch arrays.
// ---------------------------------------------------------------------------
static void bitPackConfigScanLaunch(
    Kokkos::View<int32_t*> minValue,
    Kokkos::View<int32_t*> maxValue,
    Kokkos::View<int32_t*> in,
    size_t const n)
{
  using exec_space   = Kokkos::DefaultExecutionSpace;
  using scratch_sp   = exec_space::scratch_memory_space;
  using scratch_view = Kokkos::View<int32_t*, scratch_sp, Kokkos::MemoryUnmanaged>;

  const int numBlocks =
      static_cast<int>(roundUpDiv(n, static_cast<size_t>(BLOCK_SIZE)));
  const int gridBlocks = Kokkos::min(BLOCK_WIDTH, numBlocks);

  // Two BLOCK_SIZE arrays of int32_t in shared memory (min + max buffers).
  const int scratch_size = scratch_view::shmem_size(BLOCK_SIZE) * 2;

  Kokkos::parallel_for(
      "bitPackConfigScan",
      Kokkos::TeamPolicy<>(gridBlocks, BLOCK_SIZE)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        scratch_view minBuffer(team.team_scratch(0), BLOCK_SIZE);
        scratch_view maxBuffer(team.team_scratch(0), BLOCK_SIZE);

        const int blockIdx_x  = team.league_rank();
        const int threadIdx_x = team.team_rank();

        int32_t localMin = 0;
        int32_t localMax = 0;
        int lastThread   = 0;

        // Grid-stride loop: each team accumulates multiple input chunks.
        for (int block = blockIdx_x; block < numBlocks;
             block += team.league_size()) {
          const int blockOffset = BLOCK_SIZE * block;
          const int blockEnd =
              Kokkos::min(static_cast<int>(n) - blockOffset, BLOCK_SIZE);
          lastThread = Kokkos::max(lastThread, blockEnd);

          if (threadIdx_x < blockEnd) {
            const int32_t val = in[blockOffset + threadIdx_x];
            if (block == blockIdx_x) {
              localMin = val;
              localMax = val;
            } else {
              localMin = Kokkos::min(val, localMin);
              localMax = Kokkos::max(val, localMax);
            }
          }
        }

        minBuffer[threadIdx_x] = localMin;
        maxBuffer[threadIdx_x] = localMax;
        team.team_barrier();

        // Tree reduction: lower half of threads handles minBuffer,
        // upper half handles maxBuffer (mirrors the CUDA reduceMinAndMax).
        for (int d = BLOCK_SIZE / 2; d > 0; d >>= 1) {
          if (threadIdx_x < BLOCK_SIZE / 2) {
            const int idx = threadIdx_x;
            if (idx < d && idx + d < lastThread)
              minBuffer[idx] = Kokkos::min(minBuffer[idx], minBuffer[d + idx]);
          } else {
            const int idx = threadIdx_x - (BLOCK_SIZE / 2);
            if (idx < d && idx + d < lastThread)
              maxBuffer[idx] = Kokkos::max(maxBuffer[idx], maxBuffer[d + idx]);
          }
          team.team_barrier();
        }

        if (threadIdx_x == 0) {
          minValue[blockIdx_x] = minBuffer[0];
          maxValue[blockIdx_x] = maxBuffer[0];
        }
      });
}

// ---------------------------------------------------------------------------
// Phase 2 – finalize: a single team reduces the per-block min/max arrays
// produced by the scan, then computes the required bit-width.
// ---------------------------------------------------------------------------
static void bitPackConfigFinalizeLaunch(
    Kokkos::View<int32_t*>       inMin,
    Kokkos::View<int32_t*>       inMax,
    Kokkos::View<unsigned char*> numBitsPtr,
    Kokkos::View<int32_t*>       minValOutPtr,
    size_t const n)
{
  using exec_space   = Kokkos::DefaultExecutionSpace;
  using scratch_sp   = exec_space::scratch_memory_space;
  using scratch_view = Kokkos::View<int32_t*, scratch_sp, Kokkos::MemoryUnmanaged>;

  // Number of values written by the scan kernel.
  const int numScanBlocks = static_cast<int>(Kokkos::min(
      roundUpDiv(n, static_cast<size_t>(BLOCK_SIZE)),
      static_cast<size_t>(BLOCK_WIDTH)));

  const int scratch_size = scratch_view::shmem_size(BLOCK_SIZE) * 2;

  Kokkos::parallel_for(
      "bitPackConfigFinalize",
      Kokkos::TeamPolicy<>(1, BLOCK_SIZE)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        scratch_view minBuffer(team.team_scratch(0), BLOCK_SIZE);
        scratch_view maxBuffer(team.team_scratch(0), BLOCK_SIZE);

        const int threadIdx_x = team.team_rank();
        const int blockEnd    = numScanBlocks;

        // readMinAndMax: each thread covers [threadIdx_x, blockEnd) striding
        // by BLOCK_SIZE, accumulating into a single shared-memory slot.
        if (threadIdx_x < blockEnd) {
          int32_t localMin = inMin[threadIdx_x];
          int32_t localMax = inMax[threadIdx_x];
          for (int i = threadIdx_x + BLOCK_SIZE;
               i < BLOCK_WIDTH && i < blockEnd; i += BLOCK_SIZE) {
            localMin = Kokkos::min(inMin[i], localMin);
            localMax = Kokkos::max(inMax[i], localMax);
          }
          minBuffer[threadIdx_x] = localMin;
          maxBuffer[threadIdx_x] = localMax;
        }
        team.team_barrier();

        // reduceMinAndMax over the occupied portion of the shared buffer.
        const int reduceEnd = Kokkos::min(BLOCK_SIZE, blockEnd);
        for (int d = BLOCK_SIZE / 2; d > 0; d >>= 1) {
          if (threadIdx_x < BLOCK_SIZE / 2) {
            const int idx = threadIdx_x;
            if (idx < d && idx + d < reduceEnd)
              minBuffer[idx] = Kokkos::min(minBuffer[idx], minBuffer[d + idx]);
          } else {
            const int idx = threadIdx_x - (BLOCK_SIZE / 2);
            if (idx < d && idx + d < reduceEnd)
              maxBuffer[idx] = Kokkos::max(maxBuffer[idx], maxBuffer[d + idx]);
          }
          team.team_barrier();
        }

        if (threadIdx_x == 0) {
          minValOutPtr(0) = minBuffer[0];
          const uint32_t range =
              static_cast<uint32_t>(maxBuffer[0])
              - static_cast<uint32_t>(minBuffer[0]);
          numBitsPtr(0) =
              static_cast<unsigned char>(32 - countLeadingZeros32(range));
        }
      });
}

// ---------------------------------------------------------------------------
// Phase 3 – pack: each thread writes one OUTPUT word by gathering input bits.
// ---------------------------------------------------------------------------
static void bitPackLaunch(
    Kokkos::View<unsigned char*> numBitsPtr,
    Kokkos::View<int32_t*>       valueOffsetPtr,
    Kokkos::View<uint32_t*>      outPtr,
    Kokkos::View<int32_t*>       in,
    size_t const n)
{
  static_assert(
      BLOCK_SIZE % (sizeof(uint32_t) * 8U) == 0,
      "BLOCK_SIZE must be a multiple of output word size in bits.");

  using exec_space   = Kokkos::DefaultExecutionSpace;
  using scratch_sp   = exec_space::scratch_memory_space;
  using scratch_view = Kokkos::View<uint32_t*, scratch_sp, Kokkos::MemoryUnmanaged>;

  const int numBlocks =
      static_cast<int>(roundUpDiv(n, static_cast<size_t>(BLOCK_SIZE)));
  const int gridBlocks = std::min(4096, numBlocks);

  const int scratch_size = scratch_view::shmem_size(BLOCK_SIZE);

  Kokkos::parallel_for(
      "bitPack",
      Kokkos::TeamPolicy<>(gridBlocks, BLOCK_SIZE)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        scratch_view inBuffer(team.team_scratch(0), BLOCK_SIZE);

        const int blockId     = team.league_rank();
        const int threadIdx_x = team.team_rank();

        // Read the packed config scalars once per team.
        const int     numBits     = static_cast<int>(numBitsPtr(0));
        const int32_t valueOffset = valueOffsetPtr(0);

        // Grid-stride over output blocks.
        for (int blockI = blockId; blockI < numBlocks;
             blockI += team.league_size()) {

          const int    outputIdx = threadIdx_x + blockI * BLOCK_SIZE;
          const size_t bitStart  = static_cast<size_t>(outputIdx) * 32U;
          const size_t bitEnd    = bitStart + 32U;

          const int startIdx = static_cast<int>(
              Kokkos::min(bitStart / static_cast<size_t>(numBits), n));
          const int endIdx = static_cast<int>(
              Kokkos::min(roundUpDiv(bitEnd, static_cast<size_t>(numBits)), n));

          const size_t blockStartBit =
              static_cast<size_t>(blockI) * BLOCK_SIZE * 32U;
          const size_t blockEndBit =
              static_cast<size_t>(blockI + 1) * BLOCK_SIZE * 32U;

          const int blockStartIdx = static_cast<int>(Kokkos::min(
              roundDownTo(
                  blockStartBit / static_cast<size_t>(numBits),
                  static_cast<size_t>(BLOCK_SIZE)),
              n));
          const int blockEndIdx = static_cast<int>(Kokkos::min(
              roundUpTo(
                  roundUpDiv(blockEndBit, static_cast<size_t>(numBits)),
                  static_cast<size_t>(BLOCK_SIZE)),
              n));

          uint32_t val = 0;
          for (int bufferStart = blockStartIdx; bufferStart < blockEndIdx;
               bufferStart += BLOCK_SIZE) {

            team.team_barrier();

            const int inputIdx = bufferStart + threadIdx_x;
            if (inputIdx < static_cast<int>(n))
              inBuffer[threadIdx_x] =
                  static_cast<uint32_t>(in[inputIdx] - valueOffset);

            team.team_barrier();

            const int currentStartIdx = Kokkos::max(startIdx, bufferStart);
            const int currentEndIdx =
                Kokkos::min(endIdx, bufferStart + BLOCK_SIZE);

            for (int idx = currentStartIdx; idx < currentEndIdx; ++idx) {
              uint32_t bits     = inBuffer[idx - bufferStart];
              const int offset  = static_cast<int>(
                  static_cast<int64_t>(idx) * numBits
                  - static_cast<int64_t>(bitStart));
              if (offset > 0)
                bits <<= offset;
              else
                bits >>= -offset;
              val |= bits;
            }
          }

          if (startIdx < static_cast<int>(n))
            outPtr[outputIdx] = val;
        }
      });
}

// ---------------------------------------------------------------------------
// compress – runs all three phases 1000 times and reports total kernel time.
// Handles only int32_t input (NVCOMP_TYPE_INT).
// ---------------------------------------------------------------------------
void compress(
    Kokkos::View<int32_t*>       minScratch,
    Kokkos::View<int32_t*>       maxScratch,
    Kokkos::View<uint32_t*>      output,
    Kokkos::View<int32_t*>       input,
    size_t const n,
    Kokkos::View<int32_t*>       minValueDevice,
    Kokkos::View<unsigned char*> numBitsDevice)
{
  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < 1000; ++i) {
    bitPackConfigScanLaunch(minScratch, maxScratch, input, n);
    bitPackConfigFinalizeLaunch(
        minScratch, maxScratch, numBitsDevice, minValueDevice, n);
    bitPackLaunch(numBitsDevice, minValueDevice, output, input, n);
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
  printf("Total kernel execution time (1000 iterations) = %f (s)\n",
         time * 1e-9f);
}
