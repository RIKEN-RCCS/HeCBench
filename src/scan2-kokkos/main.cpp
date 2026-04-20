#include <chrono>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "scan.h"

// Hierarchical prefix scan over a large array using Kokkos.
// The algorithm is a two-level (or multi-level) scan mirroring the
// block-scan / prefix-sum-on-block-sums / block-add structure of the
// original OMP version.

// Block-wise exclusive scan; also writes per-block sums into sumBuffer.
void bScan(const int blockSize,
           const int len,
           Kokkos::View<float*> input,
           Kokkos::View<float*> output,
           Kokkos::View<float*> sumBuffer)
{
  int nblocks = len / blockSize;
  Kokkos::parallel_for("bScan",
    Kokkos::TeamPolicy<>(nblocks, blockSize / 2),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      int bid  = team.league_rank();
      int tid  = team.team_rank();
      int tsz  = team.team_size(); // blockSize/2

      // scratch array of size blockSize
      using ScratchView = Kokkos::View<float*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                       Kokkos::MemoryUnmanaged>;
      ScratchView block(team.team_scratch(0), blockSize);

      int gid = bid * tsz + tid;
      block(2 * tid)     = input(2 * gid);
      block(2 * tid + 1) = input(2 * gid + 1);
      team.team_barrier();

      float cache0 = block(0);
      float cache1 = cache0 + block(1);

      for (int stride = 1; stride < blockSize; stride *= 2) {
        if (2 * tid >= stride) {
          cache0 = block(2 * tid - stride) + block(2 * tid);
          cache1 = block(2 * tid + 1 - stride) + block(2 * tid + 1);
        }
        team.team_barrier();
        block(2 * tid)     = cache0;
        block(2 * tid + 1) = cache1;
        team.team_barrier();
      }

      sumBuffer(bid) = block(blockSize - 1);

      if (tid == 0) {
        output(2 * gid)     = 0;
        output(2 * gid + 1) = block(2 * tid);
      } else {
        output(2 * gid)     = block(2 * tid - 1);
        output(2 * gid + 1) = block(2 * tid);
      }
    },
    Kokkos::TeamPolicy<>::scratch_size(0, sizeof(float) * blockSize));
}

// Scan over a small array (the block sums) using a single team.
void pScan(const int blockSize,
           const int len,
           Kokkos::View<float*> input,
           Kokkos::View<float*> output)
{
  int tsz = len / 2;
  Kokkos::parallel_for("pScan",
    Kokkos::TeamPolicy<>(1, tsz),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      int tid = team.team_rank();
      int gid = tid;

      using ScratchView = Kokkos::View<float*, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                       Kokkos::MemoryUnmanaged>;
      ScratchView block(team.team_scratch(0), len);

      block(2 * tid)     = input(2 * gid);
      block(2 * tid + 1) = input(2 * gid + 1);
      team.team_barrier();

      float cache0 = block(0);
      float cache1 = cache0 + block(1);

      for (int stride = 1; stride < blockSize; stride *= 2) {
        if (2 * tid >= stride) {
          cache0 = block(2 * tid - stride) + block(2 * tid);
          cache1 = block(2 * tid + 1 - stride) + block(2 * tid + 1);
        }
        team.team_barrier();
        block(2 * tid)     = cache0;
        block(2 * tid + 1) = cache1;
        team.team_barrier();
      }

      if (tid == 0) {
        output(2 * gid)     = 0;
        output(2 * gid + 1) = block(2 * tid);
      } else {
        output(2 * gid)     = block(2 * tid - 1);
        output(2 * gid + 1) = block(2 * tid);
      }
    },
    Kokkos::TeamPolicy<>::scratch_size(0, sizeof(float) * len));
}

// Add per-block prefix sums back into block outputs.
void bAddition(const int blockSize,
               const int len,
               Kokkos::View<float*> input,
               Kokkos::View<float*> output)
{
  int nblocks = len / blockSize;
  Kokkos::parallel_for("bAddition",
    Kokkos::TeamPolicy<>(nblocks, blockSize),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      int bid = team.league_rank();
      int tid = team.team_rank();
      int gid = bid * blockSize + tid;

      float value = input(bid);
      output(gid) += value;
    });
}

void scanLargeArraysCPUReference(float* output, float* input, unsigned int length)
{
  output[0] = 0;
  for (unsigned int i = 1; i < length; ++i)
    output[i] = input[i - 1] + output[i - 1];
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length> <block size>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length     = atoi(argv[2]);
  int blockSize  = atoi(argv[3]);

  if (iterations < 1) {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if (!isPowerOf2(length)) length = roundToPowerOf2(length);

  if ((length / blockSize > GROUP_SIZE) && (((length) & (length - 1)) != 0)) {
    std::cout << "Invalid length: " << length << std::endl;
    return -1;
  }

  unsigned int sizeBytes = length * sizeof(float);
  float* inputBuffer = (float*)malloc(sizeBytes);
  fillRandom<float>(inputBuffer, length, 1, 0, 255);

  blockSize = (blockSize < length / 2) ? blockSize : length / 2;

  float t = std::log((float)length) / std::log((float)blockSize);
  unsigned int pass = (unsigned int)t;
  if (std::fabs(t - (float)pass) < 1e-7) pass--;

  int outputBufferSize = 0;
  int* outputBufferSizeOffset = (int*)malloc(sizeof(int) * pass);
  for (unsigned int i = 0; i < pass; i++) {
    outputBufferSizeOffset[i] = outputBufferSize;
    outputBufferSize += (int)(length / std::pow((float)blockSize, (float)i));
  }

  int blockSumBufferSize = 0;
  int* blockSumBufferSizeOffset = (int*)malloc(sizeof(int) * pass);
  for (unsigned int i = 0; i < pass; i++) {
    blockSumBufferSizeOffset[i] = blockSumBufferSize;
    blockSumBufferSize += (int)(length / std::pow((float)blockSize, (float)(i + 1)));
  }

  int tempLength = (int)(length / std::pow((float)blockSize, (float)pass));

  float* outputBuffer  = (float*)malloc(sizeof(float) * outputBufferSize);
  float* blockSumBuffer= (float*)malloc(sizeof(float) * blockSumBufferSize);
  float* tempBuffer    = (float*)malloc(sizeof(float) * tempLength);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_input("input",        length);
    Kokkos::View<float*> d_output("output",      outputBufferSize);
    Kokkos::View<float*> d_blockSum("blockSum",  blockSumBufferSize);
    Kokkos::View<float*> d_temp("temp",          tempLength);

    {
      auto h_in = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(inputBuffer, length);
      Kokkos::deep_copy(d_input, h_in);
    }

    std::cout << "Executing kernel for " << iterations << " iterations\n";
    std::cout << "-------------------------------------------\n";

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < iterations; n++) {
      // Helper subview lambdas
      auto out_sub = [&](int i) {
        return Kokkos::subview(d_output, Kokkos::pair<int,int>(outputBufferSizeOffset[i],
                                                               outputBufferSize));
      };
      auto bsum_sub = [&](int i) {
        return Kokkos::subview(d_blockSum, Kokkos::pair<int,int>(blockSumBufferSizeOffset[i],
                                                                  blockSumBufferSize));
      };

      bScan(blockSize, length, d_input, out_sub(0), bsum_sub(0));

      for (int i = 1; i < (int)pass; i++) {
        int size = (int)(length / std::pow((float)blockSize, (float)i));
        bScan(blockSize, size, bsum_sub(i - 1), out_sub(i), bsum_sub(i));
      }

      pScan(blockSize, tempLength, bsum_sub(pass - 1), d_temp);

      bAddition(blockSize, (int)(length / std::pow((float)blockSize, (float)(pass - 1))),
                d_temp, out_sub(pass - 1));

      for (int i = (int)pass - 1; i > 0; i--) {
        bAddition(blockSize, (int)(length / std::pow((float)blockSize, (float)(i - 1))),
                  out_sub(i), out_sub(i - 1));
      }
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average execution time of scan kernels: " << time * 1e-3f / iterations
              << " (us)\n";

    // Copy first block of output back
    int copy_len = (pass == 1) ? outputBufferSize : outputBufferSizeOffset[1];
    {
      auto h_out = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(outputBuffer, copy_len);
      auto d_out_sub = Kokkos::subview(d_output, Kokkos::pair<int,int>(0, copy_len));
      Kokkos::deep_copy(h_out, d_out_sub);
    }
  }
  Kokkos::finalize();

  // Verification
  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);
  scanLargeArraysCPUReference(verificationOutput, inputBuffer, length);

  if (compare<float>(outputBuffer, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  free(verificationOutput);
  free(inputBuffer);
  free(tempBuffer);
  free(blockSumBuffer);
  free(blockSumBufferSizeOffset);
  free(outputBuffer);
  free(outputBufferSizeOffset);
  return 0;
}
