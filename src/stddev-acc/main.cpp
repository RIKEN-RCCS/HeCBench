/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "reference.h"

#include <openacc.h>

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param std the output stddev vector
 * @param data the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 */
template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

  static const int TeamX = (N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk;
  static const int TeamY = (D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk;
  static const int Teams = TeamX * TeamY;

  #pragma acc parallel loop present(std[0:D]) vector_length(TPB)
  for (int i = 0; i < D; i++) {
    std[i] = (Type)0;
  }

  #pragma acc parallel present(data[0:(size_t)D*(size_t)N], std[0:D]) num_gangs(Teams) vector_length(TPB)
  {
    #pragma acc loop gang
    for (int g = 0; g < Teams; ++g) {
      Type sstd[ColsPerBlk];

      const int bx = g % TeamX;
      const int by = g / TeamX;
      const int gridDim_x = TeamX;

      const int RowsPerBlkPerIter = TPB / ColsPerBlk;
      const IdxType stride = (IdxType)(RowsPerBlkPerIter * gridDim_x);

      #pragma acc loop vector
      for (int c = 0; c < ColsPerBlk; ++c) {
        sstd[c] = (Type)0;
      }
      
			#pragma acc loop vector
      for (int tx = 0; tx < TPB; ++tx) {
        const IdxType thisColId = (IdxType)(tx % ColsPerBlk);
        const IdxType thisRowId = (IdxType)(tx / ColsPerBlk);

        const IdxType colId = thisColId + ((IdxType)by * (IdxType)ColsPerBlk);
        const IdxType rowId = thisRowId + ((IdxType)bx * (IdxType)RowsPerBlkPerIter);

        Type thread_data = (Type)0;

        for (IdxType i = rowId; i < N; i += stride) {
          const Type val = (colId < D) ? data[(size_t)i * (size_t)D + (size_t)colId] : (Type)0;
          thread_data += val * val;
        }

        #pragma acc atomic update
        sstd[thisColId] += thread_data;
      }
      
			#pragma acc loop vector
      for (int c = 0; c < ColsPerBlk; ++c) {
        const IdxType colId = (IdxType)c + ((IdxType)by * (IdxType)ColsPerBlk);
        if (colId < D) {
          #pragma acc atomic update
          std[colId] += sstd[c];
        }
      }
    }
  }

  const IdxType sampleSize = sample ? (N - 1) : N;

  #pragma acc parallel loop present(std[0:D]) vector_length(TPB)
  for (int i = 0; i < D; i++) {
    std[i] = (Type)sqrtf((float)(std[i] / (Type)sampleSize));
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns of data (must be a multiple of 32)\n");
    printf("N: number of rows of data (at least one row)\n");
    return 1;
  }
  int D = atoi(argv[1]); // columns must be a multiple of 32
  int N = atoi(argv[2]); // at least one row
  int repeat = atoi(argv[3]);

  bool sample = true;
  long inputSize = (long)D * (long)N;
  long inputSizeByte = inputSize * (long)sizeof(float);
  float *data = (float*) malloc((size_t)inputSizeByte);

  // input data
  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
      data[i*D + j] = rand() / (float)RAND_MAX;

  // host and device results
  long outputSize = D;
  long outputSizeByte = outputSize * (long)sizeof(float);
  float *std  = (float*) malloc((size_t)outputSizeByte);
  float *std_ref  = (float*) malloc((size_t)outputSizeByte);

  #pragma acc data copyin(data[0:inputSize]) copyout(std[0:outputSize])
  {
    stddev(std, data, D, N, sample);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stddev(std, data, D, N, sample);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);
  }

  // verify (host reference)
  stddev_ref(std_ref, data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  free(std_ref);
  free(std);
  free(data);
  return 0;
}
