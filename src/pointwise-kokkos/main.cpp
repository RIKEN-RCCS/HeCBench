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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef struct {
  double i, c, h;
} checksum;

KOKKOS_INLINE_FUNCTION
float sigmoidf(float in) {
  return 1.f / (1.f + expf(-in));
}

KOKKOS_INLINE_FUNCTION
float LCG_random(unsigned int* seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return (float)(*seed) / (float)m;
}

// Fused LSTM elementwise kernel.
// All array parameters are full Views; integer offsets select the active region.
void elementWise_fp(
    int hiddenSize, int miniBatch,
    Kokkos::View<const float*> tmp_h,  int tmp_h_off,
    Kokkos::View<const float*> tmp_i,  int tmp_i_off,
    Kokkos::View<const float*> bias,   int bias_off,
    Kokkos::View<float*>       linearGates, int lg_off,
    Kokkos::View<float*>       h_out,  int h_off,
    Kokkos::View<float*>       i_out,  int i_off,
    Kokkos::View<const float*> c_in,   int cin_off,
    Kokkos::View<float*>       c_out,  int cout_off)
{
  const int numElements = miniBatch * hiddenSize;
  Kokkos::parallel_for("elementWise_fp", numElements,
    KOKKOS_LAMBDA(int index) {
      const int batch     = index / hiddenSize;
      const int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

      float g[4];
      for (int i = 0; i < 4; i++) {
        g[i] = tmp_i(tmp_i_off + i * hiddenSize + gateIndex)
             + tmp_h(tmp_h_off + i * hiddenSize + gateIndex);
        g[i] += bias(bias_off + i * hiddenSize + index % hiddenSize)
              + bias(bias_off + (i + 4) * hiddenSize + index % hiddenSize);
        linearGates(lg_off + gateIndex + i * hiddenSize) = g[i];
      }

      const float in_gate     = sigmoidf(g[0]);
      const float forget_gate = sigmoidf(g[1]);
      const float in_gate2    = tanhf(g[2]);
      const float out_gate    = sigmoidf(g[3]);

      float val = (forget_gate * c_in(cin_off + index)) + (in_gate * in_gate2);
      c_out(cout_off + index) = val;
      val = out_gate * tanhf(val);
      h_out(h_off + index) = val;
      i_out(i_off + index) = val;
    });
  Kokkos::fence();
}

void init(Kokkos::View<float*> data, int size) {
  Kokkos::parallel_for("init", size,
    KOKKOS_LAMBDA(int index) {
      unsigned int seed = (unsigned int)index ^ (unsigned int)size;
      data(index) = LCG_random(&seed);
    });
  Kokkos::fence();
}

void test(int hiddenSize, int miniBatch, int seqLength, int numLayers,
          checksum& cs, double& time)
{
  const int numElements = hiddenSize * miniBatch;
  const int hc_size     = (seqLength + 1) * numLayers * numElements;
  const int i_size      = seqLength * (numLayers + 1) * numElements;
  const int bias_size   = numLayers * hiddenSize * 8;
  const int tmp_h_size  = 4 * numLayers * numElements;
  const int tmp_i_size  = 4 * seqLength * numElements;
  const int lg_size     = 4 * seqLength * numLayers * numElements;

  // Allocate device Views (no host initialisation needed; init kernel runs on device)
  Kokkos::View<float*> d_h_data("h_data", hc_size);
  Kokkos::View<float*> d_i_data("i_data", i_size);
  Kokkos::View<float*> d_c_data("c_data", hc_size);
  Kokkos::View<float*> d_bias("bias", bias_size);
  Kokkos::View<float*> d_tmp_h("tmp_h", tmp_h_size);
  Kokkos::View<float*> d_tmp_i("tmp_i", tmp_i_size);
  Kokkos::View<float*> d_linearGates("linearGates", lg_size);

  // Initialise with random values on device
  init(d_tmp_h, tmp_h_size);
  init(d_tmp_i, tmp_i_size);
  init(d_c_data, hc_size);
  init(d_bias, bias_size);

  int lStart = 0, lEnd = 0, rStart = 0, rEnd = 0;
  const int recurBatchSize = 2;
  double ktime = 0.0;

  while (true) {
    if (lEnd == 0) {
      lStart = 0;
      lEnd   = 1;
      rStart = 0;
    } else {
      lStart++;
      lEnd++;
      rStart -= recurBatchSize;

      if (lEnd > numLayers || rStart < 0) {
        rStart += (lStart + 1) * recurBatchSize;
        lStart = 0;
        lEnd   = 1;
      }

      while (rStart >= seqLength && lEnd <= numLayers) {
        lStart++;
        lEnd++;
        rStart -= recurBatchSize;
      }

      if (lEnd > numLayers || rStart < 0) break;
    }

    rEnd = rStart + recurBatchSize;
    if (rEnd > seqLength) rEnd = seqLength;

    auto start = std::chrono::steady_clock::now();

    for (int layer = lStart; layer < lEnd; layer++) {
      for (int i = rStart; i < rEnd; i++) {
        elementWise_fp(
            hiddenSize, miniBatch,
            d_tmp_h, 4 * layer * numElements,
            d_tmp_i, 4 * i * numElements,
            d_bias,  8 * layer * hiddenSize,
            d_linearGates,
                4 * (i * numElements + layer * seqLength * numElements),
            d_h_data,
                (i + 1) * numElements + layer * (seqLength + 1) * numElements,
            d_i_data,
                i * numElements + (layer + 1) * seqLength * numElements,
            d_c_data,
                i * numElements + layer * (seqLength + 1) * numElements,
            d_c_data,
                (i + 1) * numElements + layer * (seqLength + 1) * numElements);
      }
    }

    auto end = std::chrono::steady_clock::now();
    ktime +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  time += ktime;

  // Copy results back to host
  auto h_i_data = Kokkos::create_mirror_view(d_i_data);
  auto h_h_data = Kokkos::create_mirror_view(d_h_data);
  auto h_c_data = Kokkos::create_mirror_view(d_c_data);
  Kokkos::deep_copy(h_i_data, d_i_data);
  Kokkos::deep_copy(h_h_data, d_h_data);
  Kokkos::deep_copy(h_c_data, d_c_data);

  // Compute checksums matching the OMP version's layout
  double checksumi = 0., checksumh = 0., checksumc = 0.;
  const int i_out_off = numLayers * seqLength * numElements;
  for (int m = 0; m < miniBatch; m++) {
    for (int j = 0; j < seqLength; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        checksumi += h_i_data(i_out_off + j * numElements + m * hiddenSize + i);
      }
    }
    for (int j = 0; j < numLayers; j++) {
      const int h_off = seqLength * numElements + j * (seqLength + 1) * numElements;
      for (int i = 0; i < hiddenSize; i++) {
        checksumh += h_h_data(h_off + m * hiddenSize + i);
        checksumc += h_c_data(h_off + m * hiddenSize + i);
      }
    }
  }

  cs.i = checksumi;
  cs.c = checksumc;
  cs.h = checksumh;
}

int main(int argc, char* argv[]) {
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
    printf("Usage: %s <seqLength> <numLayers> <hiddenSize> <miniBatch> <repeat>\n",
           argv[0]);
    return 1;
  }

  printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n",
         seqLength, numLayers, hiddenSize, miniBatch);

  Kokkos::initialize(argc, argv);
  {
    checksum cs;
    double time = 0.0;

    for (int run = 0; run < numRuns; run++) {
      test(hiddenSize, miniBatch, seqLength, numLayers, cs, time);
    }

    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / numRuns);
    printf("i checksum %E     ", cs.i);
    printf("c checksum %E     ", cs.c);
    printf("h checksum %E\n",    cs.h);
  }
  Kokkos::finalize();
  return 0;
}
