/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This is a simple test showing performance differences
 * between aligned and misaligned structures
 * (those having/missing __align__ keyword).
 * It measures per-element copy throughput for
 * aligned and misaligned structures on
 * big chunks of data.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

////////////////////////////////////////////////////////////////////////////////
// Misaligned types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned char uchar_misaligned;

typedef unsigned short int ushort_misaligned;

typedef struct
{
  unsigned char r, g, b, a;
} uchar4_misaligned;

typedef struct
{
  unsigned int l, a;
} uint2_misaligned;

typedef struct
{
  unsigned int r, g, b;
} uint3_misaligned;

typedef struct
{
  unsigned int r, g, b, a;
} uint4_misaligned;

typedef struct
{
  uint4_misaligned c1, c2;
} uint8_misaligned;


////////////////////////////////////////////////////////////////////////////////
// Aligned types
////////////////////////////////////////////////////////////////////////////////
typedef struct __attribute__((__aligned__(4)))
{
  unsigned char r, g, b, a;
}
uchar4_aligned;

typedef unsigned int uint_aligned;

typedef struct __attribute__((__aligned__(8)))
{
  unsigned int l, a;
}
uint2_aligned;

typedef struct __attribute__((__aligned__(16)))
{
  unsigned int r, g, b;
}
uint3_aligned;

typedef struct __attribute__((__aligned__(16)))
{
  unsigned int r, g, b, a;
}
uint4_aligned;

typedef struct __attribute__((__aligned__(16)))
{
  uint4_aligned c1, c2;
}
uint8_aligned;


////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
int iDivUp(int a, int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

int iDivDown(int a, int b)
{
  return a / b;
}

int iAlignUp(int a, int b)
{
  return (a % b != 0) ? (a - a % b + b) : a;
}

int iAlignDown(int a, int b)
{
  return a - a % b;
}


////////////////////////////////////////////////////////////////////////////////
// Validation routine for simple copy kernel.
////////////////////////////////////////////////////////////////////////////////
template<class TData> int testCPU(
    TData *h_odata,
    TData *h_idata,
    int numElements,
    int packedElementSize
    )
{
  for (int pos = 0; pos < numElements; pos++)
  {
    TData src = h_idata[pos];
    TData dst = h_odata[pos];

    for (int i = 0; i < packedElementSize; i++)
      if (((char *)&src)[i] != ((char *)&dst)[i])
      {
        return 0;
      }
  }
  return 1;
}


////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int       MEM_SIZE = 50000000;
const int NUM_ITERATIONS = 1000;

unsigned char *h_idataCPU;

template<class TData> int runTest(int packedElementSize, int memory_size)
{
  const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
  const int         numElements = iDivDown(memory_size, sizeof(TData));

  Kokkos::View<unsigned char*> d_idata("d_idata", memory_size);
  Kokkos::View<unsigned char*> d_odata("d_odata", memory_size);

  // Copy input data to device
  auto h_idata_mirror = Kokkos::create_mirror_view(d_idata);
  for (int i = 0; i < memory_size; i++)
    h_idata_mirror(i) = h_idataCPU[i];
  Kokkos::deep_copy(d_idata, h_idata_mirror);

  // Clean output buffer
  Kokkos::parallel_for("clear", memory_size, KOKKOS_LAMBDA(const int i) {
    d_odata(i) = 0;
  });

  // Run test
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    Kokkos::parallel_for("copy", numElements, KOKKOS_LAMBDA(const int pos) {
      reinterpret_cast<TData*>(d_odata.data())[pos] =
        reinterpret_cast<TData*>(d_idata.data())[pos];
    });
  }
  Kokkos::fence();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double gpuTime = (double)elapsed_seconds.count() / NUM_ITERATIONS;

  printf(
      "Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime * 1000,
      (double)totalMemSizeAligned / (gpuTime * 1073741824.0)
        );

  // Read back results and validate
  auto h_odata_mirror = Kokkos::create_mirror_view(d_odata);
  Kokkos::deep_copy(h_odata_mirror, d_odata);

  unsigned char *h_odata_cpu = (unsigned char *)malloc(memory_size);
  for (int i = 0; i < memory_size; i++)
    h_odata_cpu[i] = h_odata_mirror(i);

  int flag = testCPU(
      (TData *)h_odata_cpu,
      (TData *)h_idataCPU,
      numElements,
      packedElementSize
      );

  printf(flag ? "\tTEST OK\n" : "\tTEST FAILURE\n");

  free(h_odata_cpu);

  return !flag;
}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    int nTotalFailures = 0;

    printf("[%s] - Starting...\n", argv[0]);

    printf("Allocating memory...\n");
    int MemorySize = (int)(MEM_SIZE) & 0xffffff00; // force multiple of 256 bytes
    h_idataCPU = (unsigned char *)malloc(MemorySize);

    printf("Generating host input data array...\n");

    for (int i = 0; i < MemorySize; i++)
    {
      h_idataCPU[i] = (i & 0xFF) + 1;
    }

    printf("Testing misaligned types...\n");
    printf("uchar_misaligned...\n");
    nTotalFailures += runTest<uchar_misaligned>(1, MemorySize);

    printf("uchar4_misaligned...\n");
    nTotalFailures += runTest<uchar4_misaligned>(4, MemorySize);

    printf("uchar4_aligned...\n");
    nTotalFailures += runTest<uchar4_aligned>(4, MemorySize);

    printf("ushort_misaligned...\n");
    nTotalFailures += runTest<ushort_misaligned>(2, MemorySize);

    printf("uint_aligned...\n");
    nTotalFailures += runTest<uint_aligned>(4, MemorySize);

    printf("uint2_misaligned...\n");
    nTotalFailures += runTest<uint2_misaligned>(8, MemorySize);

    printf("uint2_aligned...\n");
    nTotalFailures += runTest<uint2_aligned>(8, MemorySize);

    printf("uint3_misaligned...\n");
    nTotalFailures += runTest<uint3_misaligned>(12, MemorySize);

    printf("uint3_aligned...\n");
    nTotalFailures += runTest<uint3_aligned>(12, MemorySize);

    printf("uint4_misaligned...\n");
    nTotalFailures += runTest<uint4_misaligned>(16, MemorySize);

    printf("uint4_aligned...\n");
    nTotalFailures += runTest<uint4_aligned>(16, MemorySize);

    printf("uint8_misaligned...\n");
    nTotalFailures += runTest<uint8_misaligned>(32, MemorySize);

    printf("uint8_aligned...\n");
    nTotalFailures += runTest<uint8_aligned>(32, MemorySize);

    printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);

    printf("Shutting down...\n");

    free(h_idataCPU);

    if (nTotalFailures != 0)
    {
      printf("Test failed!\n");
      Kokkos::finalize();
      exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
  }
  Kokkos::finalize();
  exit(EXIT_SUCCESS);
}
