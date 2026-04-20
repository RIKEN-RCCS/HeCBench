/* Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

typedef struct {
  double i, c, h;
} checksum;

using DevView = Kokkos::View<float*>;

KOKKOS_INLINE_FUNCTION float sigmoidf(float in) {
  return 1.f / (1.f + expf(-in));
}

KOKKOS_INLINE_FUNCTION float LCG_random(unsigned int *seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return (float)(*seed) / (float)m;
}

void init(DevView data, int size) {
  Kokkos::parallel_for("init", Kokkos::RangePolicy<>(0, size),
    KOKKOS_LAMBDA(const int index) {
      unsigned int seed = (unsigned int)index ^ (unsigned int)size;
      data[index] = LCG_random(&seed);
    });
  Kokkos::fence();
}

void elementWise_fp(int hiddenSize, int miniBatch,
    DevView tmp_h,
    DevView tmp_i,
    DevView bias,
    DevView linearGates,
    DevView h_out,
    DevView i_out,
    DevView c_in,
    DevView c_out)
{
  int numElements = miniBatch * hiddenSize;

  Kokkos::parallel_for("elementWise", Kokkos::RangePolicy<>(0, numElements),
    KOKKOS_LAMBDA(const int index) {
      int batch = index / hiddenSize;
      int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

      float g[4];
      for (int i = 0; i < 4; i++) {
        g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
        g[i] += bias[i * hiddenSize + index % hiddenSize] +
                bias[(i + 4) * hiddenSize + index % hiddenSize];
        linearGates[gateIndex + i * hiddenSize] = g[i];
      }

      float in_gate     = sigmoidf(g[0]);
      float forget_gate = sigmoidf(g[1]);
      float in_gate2    = tanhf(g[2]);
      float out_gate    = sigmoidf(g[3]);

      float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
      c_out[index] = val;
      val = out_gate * tanhf(val);
      h_out[index] = val;
      i_out[index] = val;
    });
  Kokkos::fence();
}

void test(int hiddenSize, int miniBatch, int seqLength, int numLayers,
          checksum &cs, double &time)
{
  int numElements = hiddenSize * miniBatch;

  int hc_size         = (seqLength + 1) * numLayers * numElements;
  int i_size          = seqLength * (numLayers + 1) * numElements;
  int bias_size       = numLayers * hiddenSize * 8;
  int tmp_h_size      = 4 * numLayers * numElements;
  int tmp_i_size      = 4 * seqLength * numElements;
  int linearGates_size = 4 * seqLength * numLayers * numElements;

  DevView d_h_data("h_data", hc_size);
  DevView d_i_data("i_data", i_size);
  DevView d_c_data("c_data", hc_size);
  DevView d_bias("bias", bias_size);
  DevView d_tmp_h("tmp_h", tmp_h_size);
  DevView d_tmp_i("tmp_i", tmp_i_size);
  DevView d_linearGates("linearGates", linearGates_size);

  // Initialize device arrays with pseudo-random values
  init(d_tmp_h, tmp_h_size);
  init(d_tmp_i, tmp_i_size);
  init(d_c_data, hc_size);
  init(d_bias, bias_size);

  int lStart = 0, lEnd = 0, rStart = 0, rEnd = 0;
  int recurBatchSize = 2;
  double ktime = 0.0;

  while (true) {
    // Many layer "scheduling" (diagonal sweep over layers and sequence positions)
    if (lEnd == 0) {
      lStart = 0; lEnd = 1; rStart = 0;
    } else {
      lStart++; lEnd++;
      rStart -= recurBatchSize;

      if (lEnd > numLayers || rStart < 0) {
        rStart += (lStart + 1) * recurBatchSize;
        lStart = 0; lEnd = 1;
      }

      while (rStart >= seqLength && lEnd <= numLayers) {
        lStart++; lEnd++;
        rStart -= recurBatchSize;
      }

      if (lEnd > numLayers || rStart < 0) break;
    }

    rEnd = rStart + recurBatchSize;
    if (rEnd > seqLength) rEnd = seqLength;

    auto start = std::chrono::steady_clock::now();

    for (int layer = lStart; layer < lEnd; layer++) {
      for (int i = rStart; i < rEnd; i++) {
        // Build subviews at the appropriate offsets (mirroring pointer arithmetic
        // from the OMP version)
        int tmp_h_off = 4 * layer * numElements;
        int tmp_i_off = 4 * i * numElements;
        int bias_off  = 8 * layer * hiddenSize;
        int lg_off    = 4 * (i * numElements + layer * seqLength * numElements);
        int h_out_off = (i + 1) * numElements + layer * (seqLength + 1) * numElements;
        int i_out_off = i * numElements + (layer + 1) * seqLength * numElements;
        int c_in_off  = i * numElements + layer * (seqLength + 1) * numElements;
        int c_out_off = (i + 1) * numElements + layer * (seqLength + 1) * numElements;

        elementWise_fp(hiddenSize, miniBatch,
          Kokkos::subview(d_tmp_h,     Kokkos::make_pair(tmp_h_off, tmp_h_size)),
          Kokkos::subview(d_tmp_i,     Kokkos::make_pair(tmp_i_off, tmp_i_size)),
          Kokkos::subview(d_bias,      Kokkos::make_pair(bias_off,  bias_size)),
          Kokkos::subview(d_linearGates, Kokkos::make_pair(lg_off,  linearGates_size)),
          Kokkos::subview(d_h_data,    Kokkos::make_pair(h_out_off, hc_size)),
          Kokkos::subview(d_i_data,    Kokkos::make_pair(i_out_off, i_size)),
          Kokkos::subview(d_c_data,    Kokkos::make_pair(c_in_off,  hc_size)),
          Kokkos::subview(d_c_data,    Kokkos::make_pair(c_out_off, hc_size)));
      }
    }

    auto end = std::chrono::steady_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  time += ktime;

  // Copy results back to host
  auto h_i_data = Kokkos::create_mirror_view(d_i_data);
  auto h_h_data = Kokkos::create_mirror_view(d_h_data);
  auto h_c_data = Kokkos::create_mirror_view(d_c_data);
  Kokkos::deep_copy(h_i_data, d_i_data);
  Kokkos::deep_copy(h_h_data, d_h_data);
  Kokkos::deep_copy(h_c_data, d_c_data);

  float *testOutputi = (float*)malloc(numElements * seqLength * sizeof(float));
  float *testOutputh = (float*)malloc(numElements * numLayers * sizeof(float));
  float *testOutputc = (float*)malloc(numElements * numLayers * sizeof(float));

  memcpy(testOutputi,
    h_i_data.data() + numLayers * seqLength * numElements,
    seqLength * numElements * sizeof(float));

  for (int layer = 0; layer < numLayers; layer++) {
    memcpy(testOutputh + layer * numElements,
      h_h_data.data() + seqLength * numElements + layer * (seqLength + 1) * numElements,
      numElements * sizeof(float));
    memcpy(testOutputc + layer * numElements,
      h_c_data.data() + seqLength * numElements + layer * (seqLength + 1) * numElements,
      numElements * sizeof(float));
  }

  double checksumi = 0., checksumh = 0., checksumc = 0.;

  for (int m = 0; m < miniBatch; m++) {
    for (int j = 0; j < seqLength; j++)
      for (int i = 0; i < hiddenSize; i++)
        checksumi += testOutputi[j * numElements + m * hiddenSize + i];
    for (int j = 0; j < numLayers; j++)
      for (int i = 0; i < hiddenSize; i++) {
        checksumh += testOutputh[j * numElements + m * hiddenSize + i];
        checksumc += testOutputc[j * numElements + m * hiddenSize + i];
      }
  }

  free(testOutputi);
  free(testOutputc);
  free(testOutputh);

  cs.i = checksumi;
  cs.c = checksumc;
  cs.h = checksumh;
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int seqLength, numLayers, hiddenSize, miniBatch, numRuns;

    if (argc == 6) {
      seqLength  = atoi(argv[1]);
      numLayers  = atoi(argv[2]);
      hiddenSize = atoi(argv[3]);
      miniBatch  = atoi(argv[4]);
      numRuns    = atoi(argv[5]);
    } else if (argc == 1) {
      printf("Running with default settings\n");
      seqLength  = 100;
      numLayers  = 4;
      hiddenSize = 512;
      miniBatch  = 64;
      numRuns    = 1;
    } else {
      printf("Usage: %s <seqLength> <numLayers> <hiddenSize> <miniBatch> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }

    printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n",
           seqLength, numLayers, hiddenSize, miniBatch);

    checksum cs;
    double time = 0.0;

    for (int run = 0; run < numRuns; run++)
      test(hiddenSize, miniBatch, seqLength, numLayers, cs, time);

    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / numRuns);
    printf("i checksum %E     ", cs.i);
    printf("c checksum %E     ", cs.c);
    printf("h checksum %E\n", cs.h);
  }
  Kokkos::finalize();
  return 0;
}
