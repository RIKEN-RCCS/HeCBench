#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include "utils.h"

#define BLOCK_SIZE   2048   // bytes per data block
#define WINDOW_SIZE  32     // sliding-window size (in elements)

typedef uint32_t INPUT_TYPE;

// Number of INPUT_TYPE elements per data block
constexpr int BLOCK_ELEMS = BLOCK_SIZE / (int)sizeof(INPUT_TYPE);  // = 512

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;

int main(int argc, char *argv[])
{
    std::string inputFileName;
    int opt, repeat = 1;

    while ((opt = getopt(argc, argv, "i:n:h")) != -1) {
        switch (opt) {
        case 'i': inputFileName = optarg;      break;
        case 'n': repeat = atoi(optarg);       break;
        case 'h':
            printf("Usage: ./main -i <inputfile> -n <repeat>\n");
            return 0;
        }
    }

    if (inputFileName.empty()) {
        printf("Usage: ./main -i <inputfile> -n <repeat>\n");
        return 1;
    }

    Kokkos::initialize(argc, argv);
    {
        INPUT_TYPE *hostArray = io::read_binary_to_new_array<INPUT_TYPE>(inputFileName);
        uint32_t fileSize   = (uint32_t)io::FileSize(inputFileName);

        uint32_t paddingSize  = fileSize % BLOCK_SIZE == 0 ? 0
                                                           : BLOCK_SIZE - fileSize % BLOCK_SIZE;
        uint32_t datatypeSize = (fileSize + paddingSize) / sizeof(INPUT_TYPE);
        uint32_t numOfBlocks  = datatypeSize * sizeof(INPUT_TYPE) / BLOCK_SIZE;

        // Allocate device views
        Kokkos::View<INPUT_TYPE*, MemSpace> deviceArray("deviceArray", datatypeSize);
        Kokkos::View<INPUT_TYPE*, MemSpace> deviceOutput("deviceOutput", datatypeSize);
        Kokkos::View<uint32_t*, MemSpace>   flagArrSize("flagArrSize", numOfBlocks + 1);
        Kokkos::View<uint32_t*, MemSpace>   flagArrOff("flagArrOff",   numOfBlocks + 1);
        Kokkos::View<uint32_t*, MemSpace>   compDataSize("compDataSize", numOfBlocks + 1);
        Kokkos::View<uint32_t*, MemSpace>   compDataOff("compDataOff",   numOfBlocks + 1);
        // tmp staging buffers (sized in bytes)
        Kokkos::View<uint8_t*, MemSpace>    tmpFlagArr("tmpFlagArr",   datatypeSize / 8);
        Kokkos::View<uint8_t*, MemSpace>    tmpCompData("tmpCompData",  (size_t)datatypeSize * sizeof(INPUT_TYPE));
        Kokkos::View<uint8_t*, MemSpace>    flagArr("flagArr",         datatypeSize / 8);
        Kokkos::View<uint8_t*, MemSpace>    compData("compData",       (size_t)datatypeSize * sizeof(INPUT_TYPE));

        // Copy input; zero-pad the trailing bytes
        {
            auto h = Kokkos::create_mirror_view(deviceArray);
            memset(h.data(), 0, datatypeSize * sizeof(INPUT_TYPE));
            memcpy(h.data(), hostArray, fileSize);
            Kokkos::deep_copy(deviceArray, h);
        }

        int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

        // ----------------------------------------------------------------
        // Compression loop (one work-item per block, fully sequential within
        // each block so no scratch memory / TeamPolicy is needed)
        // ----------------------------------------------------------------
        auto compStart = std::chrono::steady_clock::now();

        for (int iter = 0; iter < repeat; iter++) {
            Kokkos::deep_copy(flagArrSize,  (uint32_t)0);
            Kokkos::deep_copy(compDataSize, (uint32_t)0);
            Kokkos::deep_copy(tmpFlagArr,   (uint8_t)0);
            Kokkos::deep_copy(tmpCompData,  (uint8_t)0);

            // --- Kernel I: match-finding + encode (one thread per block) ---
            Kokkos::parallel_for("compressKernelI",
                Kokkos::RangePolicy<ExecSpace>(0, (int)numOfBlocks),
                KOKKOS_LAMBDA(int blk) {
                    uint8_t lenBuf[BLOCK_ELEMS];
                    uint8_t offBuf[BLOCK_ELEMS];

                    // Phase 1: find best match for each element in the block
                    for (int pos = 0; pos < BLOCK_ELEMS; pos++) {
                        int winStart = (pos - WINDOW_SIZE) < 0 ? 0 : pos - WINDOW_SIZE;
                        int winPtr   = winStart, bufPtr = pos;
                        uint8_t maxLen = 0, maxOff = 0, len = 0, off = 0;

                        while (winPtr < pos && bufPtr < BLOCK_ELEMS) {
                            if (deviceArray[blk * BLOCK_ELEMS + bufPtr] ==
                                deviceArray[blk * BLOCK_ELEMS + winPtr]) {
                                if (off == 0) off = (uint8_t)(bufPtr - winPtr);
                                len++; bufPtr++;
                            } else {
                                if (len > maxLen) { maxLen = len; maxOff = off; }
                                len = 0; off = 0; bufPtr = pos;
                            }
                            winPtr++;
                        }
                        if (len > maxLen) { maxLen = len; maxOff = off; }
                        lenBuf[pos] = maxLen;
                        offBuf[pos] = maxOff;
                    }

                    // Phase 2: encode and write to tmp buffers in one pass
                    uint32_t outOff  = 0;
                    uint8_t  flagPos = 0x01, byteFlag = 0;
                    uint32_t fc      = 0;
                    size_t   tmpBase = (size_t)blk * BLOCK_ELEMS * sizeof(INPUT_TYPE);
                    size_t   fBase   = (size_t)blk * (BLOCK_ELEMS / 8);

                    int ei = 0;
                    while (ei < BLOCK_ELEMS) {
                        if (lenBuf[ei] < (uint8_t)minEncodeLength) {
                            const uint8_t *src =
                                (const uint8_t *)&deviceArray[blk * BLOCK_ELEMS + ei];
                            for (int b = 0; b < (int)sizeof(INPUT_TYPE); b++)
                                tmpCompData[tmpBase + outOff + b] = src[b];
                            outOff += (uint32_t)sizeof(INPUT_TYPE);
                            ei++;
                        } else {
                            tmpCompData[tmpBase + outOff]     = lenBuf[ei];
                            tmpCompData[tmpBase + outOff + 1] = offBuf[ei];
                            outOff += 2;
                            ei     += lenBuf[ei];
                            byteFlag |= flagPos;
                        }
                        if (flagPos == 0x80) {
                            tmpFlagArr[fBase + fc++] = byteFlag;
                            flagPos = 0x01; byteFlag = 0;
                            continue;
                        }
                        flagPos <<= 1;
                    }
                    if (flagPos != 0x01)
                        tmpFlagArr[fBase + fc++] = byteFlag;

                    compDataSize[blk] = outOff;
                    flagArrSize [blk] = fc;
                }
            );
            Kokkos::fence();

            // --- Exclusive prefix-sums of per-block sizes ---
            Kokkos::parallel_scan("flagScan", (int)(numOfBlocks + 1),
                KOKKOS_LAMBDA(int i, uint32_t& sum, bool fin) {
                    uint32_t v = (i < (int)numOfBlocks) ? flagArrSize[i] : 0u;
                    if (fin) flagArrOff[i] = sum;
                    sum += v;
                }
            );
            Kokkos::fence();

            Kokkos::parallel_scan("dataScan", (int)(numOfBlocks + 1),
                KOKKOS_LAMBDA(int i, uint32_t& sum, bool fin) {
                    uint32_t v = (i < (int)numOfBlocks) ? compDataSize[i] : 0u;
                    if (fin) compDataOff[i] = sum;
                    sum += v;
                }
            );
            Kokkos::fence();

            // --- Kernel III: pack into contiguous output arrays ---
            Kokkos::parallel_for("compressKernelIII",
                Kokkos::RangePolicy<ExecSpace>(0, (int)numOfBlocks),
                KOKKOS_LAMBDA(int blk) {
                    int fOff = (int)flagArrOff [blk];
                    int fSz  = (int)flagArrOff [blk + 1] - fOff;
                    int dOff = (int)compDataOff[blk];
                    int dSz  = (int)compDataOff[blk + 1] - dOff;

                    size_t fBase   = (size_t)blk * (BLOCK_ELEMS / 8);
                    size_t tmpBase = (size_t)blk * BLOCK_ELEMS * sizeof(INPUT_TYPE);

                    for (int i = 0; i < fSz; i++)
                        flagArr [fOff + i] = tmpFlagArr [fBase   + i];
                    for (int i = 0; i < dSz; i++)
                        compData[dOff + i] = tmpCompData[tmpBase + i];
                }
            );
            Kokkos::fence();
        }

        auto compStop = std::chrono::steady_clock::now();

        // ----------------------------------------------------------------
        // Decompression loop
        // ----------------------------------------------------------------
        auto decompStart = std::chrono::steady_clock::now();

        for (int iter = 0; iter < repeat; iter++) {
            Kokkos::deep_copy(deviceOutput, (INPUT_TYPE)0);

            Kokkos::parallel_for("decompressKernel",
                Kokkos::RangePolicy<ExecSpace>(0, numOfBlocks),
                KOKKOS_LAMBDA(int tid) {
                    int      fOff = (int)flagArrOff [tid];
                    int      fSz  = (int)flagArrOff [tid + 1] - fOff;
                    int      dOff = (int)compDataOff[tid];
                    uint32_t di   = 0, ci = 0;

                    for (int fi = 0; fi < fSz; fi++) {
                        uint8_t byteFlag = flagArr[fOff + fi];
                        for (int bit = 0; bit < 8; bit++) {
                            if ((byteFlag >> bit) & 1) {
                                // back-reference
                                int len = (int)compData[dOff + ci];
                                int off = (int)compData[dOff + ci + 1];
                                ci += 2;
                                uint32_t dstart = di;
                                for (int k = 0; k < len; k++) {
                                    deviceOutput[tid * BLOCK_ELEMS + di] =
                                        deviceOutput[tid * BLOCK_ELEMS + dstart - off + k];
                                    di++;
                                }
                            } else {
                                // literal
                                uint8_t *dst = (uint8_t *)&deviceOutput[tid * BLOCK_ELEMS + di];
                                for (int b = 0; b < (int)sizeof(INPUT_TYPE); b++)
                                    dst[b] = compData[dOff + ci + b];
                                ci += sizeof(INPUT_TYPE);
                                di++;
                            }
                            if (di >= (uint32_t)BLOCK_ELEMS) return;
                        }
                    }
                }
            );
            Kokkos::fence();
        }

        auto decompStop = std::chrono::steady_clock::now();

        // ----------------------------------------------------------------
        // Verify
        // ----------------------------------------------------------------
        auto h_out = Kokkos::create_mirror_view(deviceOutput);
        Kokkos::deep_copy(h_out, deviceOutput);
        bool ok = true;
        for (uint32_t i = 0; i < fileSize / sizeof(INPUT_TYPE); i++) {
            if (hostArray[i] != h_out[i]) {
                printf("verification failed at index %u\n", i);
                ok = false;
                break;
            }
        }
        if (ok) printf("verification passed\n");

        auto h_fOff = Kokkos::create_mirror_view(flagArrOff);
        auto h_dOff = Kokkos::create_mirror_view(compDataOff);
        Kokkos::deep_copy(h_fOff, flagArrOff);
        Kokkos::deep_copy(h_dOff, compDataOff);

        float origSz = (float)fileSize;
        float compSz = (float)(sizeof(uint32_t) * (numOfBlocks + 1) * 2
                               + h_fOff[numOfBlocks] + h_dOff[numOfBlocks]);
        std::cout << "compression ratio: " << (origSz / compSz) << "\n";

        float compMs  = std::chrono::duration<float, std::milli>(compStop  - compStart ).count();
        float decompMs = std::chrono::duration<float, std::milli>(decompStop - decompStart).count();
        std::cout << "compression e2e throughput: "
                  << origSz / 1024.f / 1024.f / (compMs  / repeat) << " GB/s\n";
        std::cout << "decompression e2e throughput: "
                  << origSz / 1024.f / 1024.f / (decompMs / repeat) << " GB/s\n";

        delete[] hostArray;
    }
    Kokkos::finalize();
    return 0;
}
