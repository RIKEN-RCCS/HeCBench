#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <limits>
#include <algorithm>

// Run the full bit-packing pipeline on the device for a single size.
static void runBitPackingOnGPU(
    int32_t const* inputHost,
    void*          outputHost,
    int const      numBitsMax,
    size_t const   n,
    int*           numBitsOut,
    int32_t*       minValOut)
{
  // ---- device views -------------------------------------------------------
  Kokkos::View<int32_t*> input("input", n);
  {
    auto h = Kokkos::create_mirror_view(input);
    for (size_t i = 0; i < n; ++i) h(i) = inputHost[i];
    Kokkos::deep_copy(input, h);
  }

  const size_t packedSizeBytes =
      (((static_cast<size_t>(numBitsMax) * n) / 64U) + 1U) * 8U;
  const size_t packedSizeWords =
      (packedSizeBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);

  // Kokkos::View zero-initialises by default – no explicit memset needed.
  Kokkos::View<uint32_t*>      output("output",        packedSizeWords);
  Kokkos::View<int32_t*>       minValueDevice("minVal", 1);
  Kokkos::View<unsigned char*> numBitsDevice("numBits", 1);

  // Scratch arrays: one entry per scan block, capped at BLOCK_WIDTH.
  const size_t scratchN = std::min(
      static_cast<size_t>(BLOCK_WIDTH),
      roundUpDiv(n, static_cast<size_t>(BLOCK_SIZE)));
  Kokkos::View<int32_t*> minScratch("minScratch", scratchN);
  Kokkos::View<int32_t*> maxScratch("maxScratch", scratchN);

  // ---- run ----------------------------------------------------------------
  compress(minScratch, maxScratch, output, input, n,
           minValueDevice, numBitsDevice);

  // ---- copy results back to host ------------------------------------------
  {
    auto h = Kokkos::create_mirror_view(minValueDevice);
    Kokkos::deep_copy(h, minValueDevice);
    *minValOut = h(0);
  }
  {
    auto h = Kokkos::create_mirror_view(numBitsDevice);
    Kokkos::deep_copy(h, numBitsDevice);
    *numBitsOut = static_cast<int>(h(0));
  }
  {
    const size_t copyBytes =
        std::min(packedSizeBytes, n * sizeof(int32_t));
    const size_t copyWords =
        (copyBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    auto h = Kokkos::create_mirror_view(output);
    Kokkos::deep_copy(h, output);
    std::memcpy(outputHost, h.data(), copyWords * sizeof(uint32_t));
  }
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    const int offset  = 87231;
    const int numBits = 13;

    std::vector<size_t> const sizes{2, 123, 3411, 83621, 872163, 100000001};

    using T = int32_t;

    std::vector<T> source(sizes.back());
    std::srand(0);
    for (T& v : source)
      v = std::abs(static_cast<T>(std::rand())) % std::numeric_limits<T>::max();

    const size_t numBytes = sizes.back() * sizeof(T);
    T*    inputHost  = static_cast<T*>(aligned_alloc(1024, numBytes));
    void* outputHost = aligned_alloc(1024, numBytes);

    for (size_t const n : sizes) {
      for (size_t i = 0; i < n; ++i)
        inputHost[i] =
            (source[i] & ((1U << numBits) - 1)) + offset;

      T   minValue;
      int numBitsAct;

      printf("Size = %10zu\n", n);
      auto start = std::chrono::steady_clock::now();

      runBitPackingOnGPU(
          inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end - start).count();
      printf("Device offload time = %f (s)\n", time * 1e-9f);

      assert(numBitsAct <= numBits);

      // Unpack and verify a sample of the packed output.
      std::vector<T> unpackedHost;
      unpackedHost.reserve(n);
      for (size_t i = 0; i < n; ++i)
        unpackedHost.emplace_back(unpackBytes(
            outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));

      assert(unpackedHost.size() == n);

      bool ok = true;
      const size_t numSamples =
          static_cast<size_t>(std::sqrt(static_cast<double>(n))) + 1;
      for (size_t i = 0; i < numSamples; ++i) {
        const size_t idx = static_cast<uint32_t>(source[i]) % n;
        if (unpackedHost[idx] != inputHost[idx]) {
          ok = false;
          break;
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    free(inputHost);
    free(outputHost);
  }
  Kokkos::finalize();
  return 0;
}
