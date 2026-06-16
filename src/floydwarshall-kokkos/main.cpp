/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define MAXDISTANCE (200)

void floydWarshallCPUReference(unsigned int *pathDistanceMatrix,
                               unsigned int *pathMatrix,
                               unsigned int numNodes)
{
  unsigned int distanceYtoX, distanceYtoK, distanceKtoX, indirectDistance;
  unsigned int width = numNodes;
  unsigned int yXwidth;

  for (unsigned int k = 0; k < numNodes; ++k) {
    for (unsigned int y = 0; y < numNodes; ++y) {
      yXwidth = y * numNodes;
      for (unsigned int x = 0; x < numNodes; ++x) {
        distanceYtoX = pathDistanceMatrix[yXwidth + x];
        distanceYtoK = pathDistanceMatrix[yXwidth + k];
        distanceKtoX = pathDistanceMatrix[k * width + x];

        indirectDistance = distanceYtoK + distanceKtoX;

        if (indirectDistance < distanceYtoX) {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x]         = k;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }

  unsigned int numNodes      = atoi(argv[1]);
  unsigned int numIterations = atoi(argv[2]);
  unsigned int blockSize     = atoi(argv[3]);

  // numNodes must be a multiple of blockSize
  if (numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  unsigned int matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);

  unsigned int *pathDistanceMatrix = (unsigned int *)malloc(matrixSizeBytes);
  assert(pathDistanceMatrix != NULL);
  unsigned int *pathMatrix = (unsigned int *)malloc(matrixSizeBytes);
  assert(pathMatrix != NULL);
  unsigned int *verificationPathDistanceMatrix =
      (unsigned int *)malloc(matrixSizeBytes);
  assert(verificationPathDistanceMatrix != NULL);
  unsigned int *verificationPathMatrix =
      (unsigned int *)malloc(matrixSizeBytes);
  assert(verificationPathMatrix != NULL);

  srand(2);
  for (unsigned int i = 0; i < numNodes; i++)
    for (unsigned int j = 0; j < numNodes; j++)
      pathDistanceMatrix[i * numNodes + j] = rand() % (MAXDISTANCE + 1);

  for (unsigned int i = 0; i < numNodes; ++i)
    pathDistanceMatrix[i * numNodes + i] = 0;

  for (unsigned int i = 0; i < numNodes; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix, matrixSizeBytes);
  memcpy(verificationPathMatrix, pathMatrix,
         numNodes * numNodes * sizeof(unsigned int));

  unsigned int numPasses = numNodes;

  Kokkos::initialize(argc, argv);
  {
    unsigned int N = numNodes;

    Kokkos::View<unsigned int *> d_pathDist("pathDistanceBuffer", N * N);
    Kokkos::View<unsigned int *> d_pathBuf("pathBuffer", N * N);

    // Host mirror for initial data copy
    auto h_pathDist = Kokkos::create_mirror_view(d_pathDist);

    for (unsigned int idx = 0; idx < N * N; idx++)
      h_pathDist(idx) = pathDistanceMatrix[idx];

    float total_time = 0.f;

    for (unsigned int n = 0; n < numIterations; n++) {
      // Reset device buffer from host initial state each iteration
      Kokkos::deep_copy(d_pathDist, h_pathDist);
      Kokkos::fence();

      auto start = std::chrono::steady_clock::now();

      for (unsigned int k = 0; k < numPasses; k++) {
        Kokkos::parallel_for(
            "floydWarshallPass",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {(int)N, (int)N}),
            KOKKOS_LAMBDA(int y, int x) {
              unsigned int oldWeight = d_pathDist(y * N + x);
              unsigned int tempWeight =
                  d_pathDist(y * N + k) + d_pathDist(k * N + x);
              if (tempWeight < oldWeight) {
                d_pathDist(y * N + x) = tempWeight;
                d_pathBuf(y * N + x)  = k;
              }
            });
        Kokkos::fence();
      }

      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end - start)
                      .count();
      total_time += (float)time;
    }

    printf("Average kernel execution time %f (s)\n",
           (total_time * 1e-9f) / numIterations);

    // Copy final result back to host
    auto h_result = Kokkos::create_mirror_view(d_pathDist);
    Kokkos::deep_copy(h_result, d_pathDist);

    for (unsigned int idx = 0; idx < N * N; idx++)
      pathDistanceMatrix[idx] = h_result(idx);
  }
  Kokkos::finalize();

  // Verify
  floydWarshallCPUReference(verificationPathDistanceMatrix,
                            verificationPathMatrix, numNodes);

  if (memcmp(pathDistanceMatrix, verificationPathDistanceMatrix,
             matrixSizeBytes) == 0) {
    printf("PASS\n");
  } else {
    printf("FAIL\n");
    if (numNodes <= 8) {
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("host: %u ", verificationPathDistanceMatrix[i * numNodes + j]);
        printf("\n");
      }
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("device: %u ", pathDistanceMatrix[i * numNodes + j]);
        printf("\n");
      }
    }
  }

  free(pathDistanceMatrix);
  free(pathMatrix);
  free(verificationPathDistanceMatrix);
  free(verificationPathMatrix);
  return 0;
}
