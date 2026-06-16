/*
 * Kokkos port of convolutionSeparable-omp/main.cpp.
 * The OMP target shared-memory tiled kernels are replaced with simple
 * Kokkos::parallel_for (no shared-memory tiling).
 * The CPU reference functions convolutionRowHost / convolutionColumnHost are
 * compiled from ../convolutionSeparable-omp/conv_gold.cpp.
 */

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "conv.h"   // for KERNEL_RADIUS, KERNEL_LENGTH and host prototypes

// ---- Device convolution kernels -------------------------------------------

void convolutionRows(
    Kokkos::View<float*>       dst,
    Kokkos::View<const float*> src,
    Kokkos::View<const float*> kernel,
    int imageW,
    int imageH,
    int pitch)
{
    Kokkos::parallel_for("convRows", imageW * imageH,
        KOKKOS_LAMBDA(int idx) {
            int y = idx / imageW;
            int x = idx % imageW;
            float sum = 0.0f;
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                int xj = x + j;
                if (xj >= 0 && xj < imageW)
                    sum += kernel[KERNEL_RADIUS - j] * src[y * pitch + xj];
            }
            dst[y * pitch + x] = sum;
        });
    Kokkos::fence();
}

void convolutionColumns(
    Kokkos::View<float*>       dst,
    Kokkos::View<const float*> src,
    Kokkos::View<const float*> kernel,
    int imageW,
    int imageH,
    int pitch)
{
    Kokkos::parallel_for("convCols", imageW * imageH,
        KOKKOS_LAMBDA(int idx) {
            int y = idx / imageW;
            int x = idx % imageW;
            float sum = 0.0f;
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                int yj = y + j;
                if (yj >= 0 && yj < imageH)
                    sum += kernel[KERNEL_RADIUS - j] * src[yj * pitch + x];
            }
            dst[y * pitch + x] = sum;
        });
    Kokkos::fence();
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <image_width> <image_height> <repeat>\n", argv[0]);
        return 1;
    }
    const int imageW        = atoi(argv[1]);
    const int imageH        = atoi(argv[2]);
    const int numIterations = atoi(argv[3]);

    float *h_Kernel    = (float*)malloc(KERNEL_LENGTH * sizeof(float));
    float *h_Input     = (float*)malloc(imageW * imageH * sizeof(float));
    float *h_Buffer    = (float*)malloc(imageW * imageH * sizeof(float));
    float *h_OutputCPU = (float*)malloc(imageW * imageH * sizeof(float));
    float *h_OutputGPU = (float*)malloc(imageW * imageH * sizeof(float));

    srand(2009);
    for (int i = 0; i < KERNEL_LENGTH;     i++) h_Kernel[i] = (float)(rand() % 16);
    for (int i = 0; i < imageW * imageH;   i++) h_Input[i]  = (float)(rand() % 16);

    Kokkos::initialize(argc, argv);
    {
        const int N = imageW * imageH;

        // ---- Device Views ------------------------------------------------
        Kokkos::View<float*> d_Kernel("d_kernel", KERNEL_LENGTH);
        Kokkos::View<float*> d_Input ("d_input",  N);
        Kokkos::View<float*> d_Buffer("d_buffer", N);
        Kokkos::View<float*> d_Output("d_output", N);

        // host mirrors
        auto hm_k = Kokkos::create_mirror_view(d_Kernel);
        auto hm_i = Kokkos::create_mirror_view(d_Input);
        for (int i = 0; i < KERNEL_LENGTH; i++) hm_k(i) = h_Kernel[i];
        for (int i = 0; i < N;             i++) hm_i(i) = h_Input[i];
        Kokkos::deep_copy(d_Kernel, hm_k);
        Kokkos::deep_copy(d_Input,  hm_i);

        Kokkos::View<const float*> d_Kernel_c = d_Kernel;
        Kokkos::View<const float*> d_Input_c  = d_Input;

        // Warmup pass
        convolutionRows   (d_Buffer, d_Input_c, d_Kernel_c, imageW, imageH, imageW);
        convolutionColumns(d_Output, Kokkos::View<const float*>(d_Buffer),
                           d_Kernel_c, imageW, imageH, imageW);

        auto t_start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numIterations; iter++) {
            convolutionRows   (d_Buffer, d_Input_c, d_Kernel_c, imageW, imageH, imageW);
            convolutionColumns(d_Output, Kokkos::View<const float*>(d_Buffer),
                               d_Kernel_c, imageW, imageH, imageW);
        }

        auto t_end = std::chrono::steady_clock::now();
        double ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         t_end - t_start).count();
        printf("Average kernel execution time %f (s)\n", (ns * 1e-9) / numIterations);

        // ---- Copy result back to host ------------------------------------
        auto hm_out = Kokkos::create_mirror_view(d_Output);
        Kokkos::deep_copy(hm_out, d_Output);
        for (int i = 0; i < N; i++) h_OutputGPU[i] = hm_out(i);
    }
    Kokkos::finalize();

    // ---- CPU reference ---------------------------------------------------
    printf("Comparing against Host/C++ computation...\n");
    convolutionRowHost   (h_Buffer, h_Input,  h_Kernel, imageW, imageH, KERNEL_RADIUS);
    convolutionColumnHost(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);

    double sum = 0, delta = 0;
    for (int i = 0; i < imageW * imageH; i++) {
        delta += (double)(h_OutputCPU[i] - h_OutputGPU[i]) *
                         (h_OutputCPU[i] - h_OutputGPU[i]);
        sum   += (double)h_OutputCPU[i] * h_OutputCPU[i];
    }
    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %.3e\n\n", L2norm);

    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    printf("%s\n", L2norm < 1e-6 ? "PASS" : "FAIL");
    return 0;
}
