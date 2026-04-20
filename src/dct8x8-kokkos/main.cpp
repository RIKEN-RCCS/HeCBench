/*
 * DCT8x8 – Kokkos port
 * Applies forward and inverse 2D DCT on 8x8 blocks of an image.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "../dct8x8-omp/DCT8x8.h"

// DCT8x8_gold CPU reference (pulled from DCT8x8_gold.cpp inline here)
static void DCT8_cpu(float *dst, const float *src, unsigned int os, unsigned int is) {
  float X07P=src[0*is]+src[7*is], X16P=src[1*is]+src[6*is];
  float X25P=src[2*is]+src[5*is], X34P=src[3*is]+src[4*is];
  float X07M=src[0*is]-src[7*is], X61M=src[6*is]-src[1*is];
  float X25M=src[2*is]-src[5*is], X43M=src[4*is]-src[3*is];
  float X07P34PP=X07P+X34P, X07P34PM=X07P-X34P;
  float X16P25PP=X16P+X25P, X16P25PM=X16P-X25P;
  dst[0*os]=C_norm*(X07P34PP+X16P25PP); dst[4*os]=C_norm*(X07P34PP-X16P25PP);
  dst[2*os]=C_norm*(C_b*X07P34PM+C_e*X16P25PM); dst[6*os]=C_norm*(C_e*X07P34PM-C_b*X16P25PM);
  dst[1*os]=C_norm*(C_a*X07M-C_c*X61M+C_d*X25M-C_f*X43M);
  dst[3*os]=C_norm*(C_c*X07M+C_f*X61M-C_a*X25M+C_d*X43M);
  dst[5*os]=C_norm*(C_d*X07M+C_a*X61M+C_f*X25M-C_c*X43M);
  dst[7*os]=C_norm*(C_f*X07M+C_d*X61M+C_c*X25M+C_a*X43M);
}
static void IDCT8_cpu(float *dst, const float *src, unsigned int os, unsigned int is) {
  float Y04P=src[0*is]+src[4*is];
  float Y2b6eP=C_b*src[2*is]+C_e*src[6*is];
  float Y04P2b6ePP=Y04P+Y2b6eP, Y04P2b6ePM=Y04P-Y2b6eP;
  float Y7f1aP3c5dPP=C_f*src[7*is]+C_a*src[1*is]+C_c*src[3*is]+C_d*src[5*is];
  float Y7a1fM3d5cMP=C_a*src[7*is]-C_f*src[1*is]+C_d*src[3*is]-C_c*src[5*is];
  float Y04M=src[0*is]-src[4*is], Y2e6bM=C_e*src[2*is]-C_b*src[6*is];
  float Y04M2e6bMP=Y04M+Y2e6bM, Y04M2e6bMM=Y04M-Y2e6bM;
  float Y1c7dM3f5aPM=C_c*src[1*is]-C_d*src[7*is]-C_f*src[3*is]-C_a*src[5*is];
  float Y1d7cP3a5fMM=C_d*src[1*is]+C_c*src[7*is]-C_a*src[3*is]+C_f*src[5*is];
  dst[0*os]=C_norm*(Y04P2b6ePP+Y7f1aP3c5dPP); dst[7*os]=C_norm*(Y04P2b6ePP-Y7f1aP3c5dPP);
  dst[4*os]=C_norm*(Y04P2b6ePM+Y7a1fM3d5cMP); dst[3*os]=C_norm*(Y04P2b6ePM-Y7a1fM3d5cMP);
  dst[1*os]=C_norm*(Y04M2e6bMP+Y1c7dM3f5aPM); dst[5*os]=C_norm*(Y04M2e6bMM-Y1d7cP3a5fMM);
  dst[2*os]=C_norm*(Y04M2e6bMM+Y1d7cP3a5fMM); dst[6*os]=C_norm*(Y04M2e6bMP-Y1d7cP3a5fMM);
}

void DCT8x8CPU(float *dst, const float *src, unsigned int stride,
               unsigned int imageH, unsigned int imageW, int dir) {
  for (unsigned i = 0; i+7 < imageH; i += 8)
    for (unsigned j = 0; j+7 < imageW; j += 8) {
      for (unsigned k = 0; k < 8; k++)
        if (dir == DCT_FORWARD)
          DCT8_cpu(dst+(i+k)*stride+j, src+(i+k)*stride+j, 1, 1);
        else
          IDCT8_cpu(dst+(i+k)*stride+j, src+(i+k)*stride+j, 1, 1);
      for (unsigned k = 0; k < 8; k++)
        if (dir == DCT_FORWARD)
          DCT8_cpu(dst+i*stride+(j+k), dst+i*stride+(j+k), stride, stride);
        else
          IDCT8_cpu(dst+i*stride+(j+k), dst+i*stride+(j+k), stride, stride);
    }
}

// Device 8-point fast DCT/IDCT
KOKKOS_INLINE_FUNCTION void DCT8_dev(float D[8]) {
  float X07P=D[0]+D[7], X16P=D[1]+D[6], X25P=D[2]+D[5], X34P=D[3]+D[4];
  float X07M=D[0]-D[7], X61M=D[6]-D[1], X25M=D[2]-D[5], X43M=D[4]-D[3];
  float X07P34PP=X07P+X34P, X07P34PM=X07P-X34P;
  float X16P25PP=X16P+X25P, X16P25PM=X16P-X25P;
  D[0]=C_norm*(X07P34PP+X16P25PP); D[4]=C_norm*(X07P34PP-X16P25PP);
  D[2]=C_norm*(C_b*X07P34PM+C_e*X16P25PM); D[6]=C_norm*(C_e*X07P34PM-C_b*X16P25PM);
  D[1]=C_norm*(C_a*X07M-C_c*X61M+C_d*X25M-C_f*X43M);
  D[3]=C_norm*(C_c*X07M+C_f*X61M-C_a*X25M+C_d*X43M);
  D[5]=C_norm*(C_d*X07M+C_a*X61M+C_f*X25M-C_c*X43M);
  D[7]=C_norm*(C_f*X07M+C_d*X61M+C_c*X25M+C_a*X43M);
}

KOKKOS_INLINE_FUNCTION void IDCT8_dev(float D[8]) {
  float Y04P=D[0]+D[4], Y2b6eP=C_b*D[2]+C_e*D[6];
  float Y04P2b6ePP=Y04P+Y2b6eP, Y04P2b6ePM=Y04P-Y2b6eP;
  float Y7f1aP3c5dPP=C_f*D[7]+C_a*D[1]+C_c*D[3]+C_d*D[5];
  float Y7a1fM3d5cMP=C_a*D[7]-C_f*D[1]+C_d*D[3]-C_c*D[5];
  float Y04M=D[0]-D[4], Y2e6bM=C_e*D[2]-C_b*D[6];
  float Y04M2e6bMP=Y04M+Y2e6bM, Y04M2e6bMM=Y04M-Y2e6bM;
  float Y1c7dM3f5aPM=C_c*D[1]-C_d*D[7]-C_f*D[3]-C_a*D[5];
  float Y1d7cP3a5fMM=C_d*D[1]+C_c*D[7]-C_a*D[3]+C_f*D[5];
  D[0]=C_norm*(Y04P2b6ePP+Y7f1aP3c5dPP); D[7]=C_norm*(Y04P2b6ePP-Y7f1aP3c5dPP);
  D[4]=C_norm*(Y04P2b6ePM+Y7a1fM3d5cMP); D[3]=C_norm*(Y04P2b6ePM-Y7a1fM3d5cMP);
  D[1]=C_norm*(Y04M2e6bMP+Y1c7dM3f5aPM); D[5]=C_norm*(Y04M2e6bMM-Y1d7cP3a5fMM);
  D[2]=C_norm*(Y04M2e6bMM+Y1d7cP3a5fMM); D[6]=C_norm*(Y04M2e6bMP-Y1d7cP3a5fMM);
}

// Device DCT8x8: one workitem per 8x8 block
void DCT8x8_kokkos(Kokkos::View<float*> d_Dst, Kokkos::View<float*> d_Src,
                   unsigned int stride, unsigned int imageH, unsigned int imageW, int dir)
{
  const unsigned int blocksH = imageH / 8;
  const unsigned int blocksW = imageW / 8;
  const unsigned int nBlocks = blocksH * blocksW;

  Kokkos::parallel_for("DCT8x8", nBlocks,
    KOKKOS_LAMBDA(int blk) {
      unsigned int br = (blk / blocksW) * 8;   // block row start
      unsigned int bc = (blk % blocksW) * 8;   // block col start
      float D[8];

      if (dir == DCT_FORWARD) {
        // Row-wise DCT
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) D[c] = d_Src((br+r)*stride + bc+c);
          DCT8_dev(D);
          for (int c = 0; c < 8; c++) d_Dst((br+r)*stride + bc+c) = D[c];
        }
        // Column-wise DCT
        for (int c = 0; c < 8; c++) {
          for (int r = 0; r < 8; r++) D[r] = d_Dst((br+r)*stride + bc+c);
          DCT8_dev(D);
          for (int r = 0; r < 8; r++) d_Dst((br+r)*stride + bc+c) = D[r];
        }
      } else {
        // Row-wise IDCT
        for (int r = 0; r < 8; r++) {
          for (int c = 0; c < 8; c++) D[c] = d_Src((br+r)*stride + bc+c);
          IDCT8_dev(D);
          for (int c = 0; c < 8; c++) d_Dst((br+r)*stride + bc+c) = D[c];
        }
        // Column-wise IDCT
        for (int c = 0; c < 8; c++) {
          for (int r = 0; r < 8; r++) D[r] = d_Dst((br+r)*stride + bc+c);
          IDCT8_dev(D);
          for (int r = 0; r < 8; r++) d_Dst((br+r)*stride + bc+c) = D[r];
        }
      }
    });
}

void Verify(const float* h_GPU, float* h_CPU, const float* h_Input,
            unsigned int stride, unsigned int imageH, unsigned int imageW, int dir)
{
  printf("Comparing against Host/C++ computation...\n");
  DCT8x8CPU(h_CPU, h_Input, stride, imageH, imageW, dir);
  double sum = 0, delta = 0;
  for (unsigned i = 0; i < imageH; i++)
    for (unsigned j = 0; j < imageW; j++) {
      double v = h_CPU[i*stride+j];
      sum   += v*v;
      double d = h_GPU[i*stride+j] - v;
      delta += d*d;
    }
  double L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);
  printf(L2norm < 1e-6 ? "PASS\n" : "FAIL\n");
}

int main(int argc, char **argv) {
  if (argc != 4) { printf("Usage: %s <imageW> <imageH> <repeat>\n", argv[0]); return 1; }
  const unsigned int imageW  = atoi(argv[1]);
  const unsigned int imageH  = atoi(argv[2]);
  const int numIter           = atoi(argv[3]);
  const unsigned int stride   = imageW;

  float *h_Input     = (float*)malloc(imageH * stride * sizeof(float));
  float *h_OutputCPU = (float*)malloc(imageH * stride * sizeof(float));
  float *h_OutputGPU = (float*)malloc(imageH * stride * sizeof(float));

  srand(2009);
  for (unsigned i = 0; i < imageH; i++)
    for (unsigned j = 0; j < imageW; j++)
      h_Input[i*stride+j] = (float)rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*>       d_Output("output", imageH*stride);
    Kokkos::View<float*>       d_Input;
    {
      Kokkos::View<float*> d_in_tmp("in_tmp", imageH*stride);
      auto h_tmp = Kokkos::create_mirror_view(d_in_tmp);
      for (unsigned i = 0; i < imageH*stride; i++) h_tmp(i) = h_Input[i];
      Kokkos::deep_copy(d_in_tmp, h_tmp);
      d_Input = d_in_tmp;
    }

    // Forward DCT
    printf("Performing Forward DCT8x8 of %u x %u image on device\n\n", imageH, imageW);
    auto t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < numIter; i++)
      DCT8x8_kokkos(d_Output, d_Input, stride, imageH, imageW, DCT_FORWARD);
    Kokkos::fence();
    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average DCT8x8 kernel execution time %f (s)\n", (elapsed * 1e-9) / numIter);

    {
      auto h_out = Kokkos::create_mirror_view(d_Output);
      Kokkos::deep_copy(h_out, d_Output);
      for (unsigned i = 0; i < imageH*stride; i++) h_OutputGPU[i] = h_out(i);
    }
    Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, DCT_FORWARD);

    // Inverse DCT
    printf("Performing Inverse DCT8x8 of %u x %u image on device\n\n", imageH, imageW);
    t_start = std::chrono::steady_clock::now();
    for (int i = 0; i < numIter; i++)
      DCT8x8_kokkos(d_Output, d_Input, stride, imageH, imageW, DCT_INVERSE);
    Kokkos::fence();
    t_end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average IDCT8x8 kernel execution time %f (s)\n", (elapsed * 1e-9) / numIter);

    {
      auto h_out = Kokkos::create_mirror_view(d_Output);
      Kokkos::deep_copy(h_out, d_Output);
      for (unsigned i = 0; i < imageH*stride; i++) h_OutputGPU[i] = h_out(i);
    }
    Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, DCT_INVERSE);
  }
  Kokkos::finalize();

  free(h_Input); free(h_OutputCPU); free(h_OutputGPU);
  return 0;
}
