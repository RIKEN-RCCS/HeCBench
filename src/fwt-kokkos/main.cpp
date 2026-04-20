/*
 * Fast Walsh-Hadamard Transform (FWT) - Kokkos port
 *
 * Ported from fwt-omp (NVIDIA CUDA sample, OpenMP target version).
 * Original copyright 1993-2015 NVIDIA Corporation.
 */

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

// ============================================================
// Types
// ============================================================
using View1Df = Kokkos::View<float*>;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using ScratchSpace = ExecSpace::scratch_memory_space;
using ScratchView1Df = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// ============================================================
// Elementary FWT block size (shared-memory threshold)
// ============================================================
static constexpr int ELEMENTARY_LOG2SIZE = 11;  // 2^11 = 2048

// ============================================================
// Reference CPU implementations (from reference.cpp)
// ============================================================
void fwtCPU(float *h_Output, float *h_Input, int log2N)
{
    const int N = 1 << log2N;
    for (int pos = 0; pos < N; pos++)
        h_Output[pos] = h_Input[pos];

    for (int stride = N / 2; stride >= 1; stride >>= 1)
        for (int base = 0; base < N; base += 2 * stride)
            for (int j = 0; j < stride; j++) {
                int i0 = base + j;
                int i1 = base + j + stride;
                float T1 = h_Output[i0];
                float T2 = h_Output[i1];
                h_Output[i0] = T1 + T2;
                h_Output[i1] = T1 - T2;
            }
}

void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN)
{
    const int   dataN = 1 << log2dataN;
    const int kernelN = 1 << log2kernelN;

    for (int i = 0; i < dataN; i++) {
        double sum = 0;
        for (int j = 0; j < kernelN; j++)
            sum += h_Data[i ^ j] * h_Kernel[j];
        h_Result[i] = (float)sum;
    }
}

// ============================================================
// GPU FWT kernels (Kokkos)
// ============================================================

// Outer radix-4 + elementary combined FWT on a batch of M vectors of length 2^log2N.
// d_Data layout: M vectors, each of length 2^log2N (original), contiguous.
void fwtBatchGPU(View1Df d_Data, int M, int log2N)
{
    int N = 1 << log2N;
    const int sN = N;   // original vector length (fixed throughout)
    const int sM = M;   // original batch size (fixed throughout)

    // Outer radix-4 stages: each stage reduces log2N by 2, doubles M.
    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2) {
        const int stride = N / 4;
        const int stride_cap = stride;
        const int sN_cap    = sN;

        Kokkos::parallel_for(
            "fwt_radix4",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {sM, sN_cap / 4}),
            KOKKOS_LAMBDA(int m, int pos) {
                const int base = m * sN_cap;
                const int lo   = pos & (stride_cap - 1);
                const int i0   = base + ((pos - lo) << 2) + lo;
                const int i1   = i0 + stride_cap;
                const int i2   = i1 + stride_cap;
                const int i3   = i2 + stride_cap;

                float D0 = d_Data(i0);
                float D1 = d_Data(i1);
                float D2 = d_Data(i2);
                float D3 = d_Data(i3);

                float T;
                T = D0; D0 = D0 + D2; D2 = T - D2;
                T = D1; D1 = D1 + D3; D3 = T - D3;
                T = D0;
                d_Data(i0) = D0 + D1;
                d_Data(i1) = T  - D1;
                T = D2;
                d_Data(i2) = D2 + D3;
                d_Data(i3) = T  - D3;
            });
        Kokkos::fence();
    }

    // Elementary in-register radix-4 + optional radix-2 stage.
    // After the outer loop: log2N <= ELEMENTARY_LOG2SIZE, N = 2^log2N, M has been updated.
    const int elN        = 1 << log2N;   // element count per sub-vector (≤ 2048)
    const int elTeamSize = elN / 4;       // threads per team
    const int log2N_fin  = log2N;
    const int N_fin      = elN;
    // scratch: one float array of length elN per team
    const int scratch_sz = static_cast<int>(sizeof(float)) * elN;

    auto policy = Kokkos::TeamPolicy<>(M, elTeamSize)
                      .set_scratch_size(0, Kokkos::PerTeam(scratch_sz));

    Kokkos::parallel_for(
        "fwt_elementary", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            ScratchView1Df s_data(team.team_scratch(0), N_fin);

            const int lid  = team.team_rank();
            const int gsz  = team.team_size();
            const int gid  = team.league_rank();
            const int base = gid * N_fin;

            // Load sub-vector into scratch
            for (int pos = lid; pos < N_fin; pos += gsz)
                s_data(pos) = d_Data(base + pos);
            team.team_barrier();

            // Main radix-4 stages — each thread owns position pos=lid
            const int pos = lid;
            for (int stride = N_fin >> 2; stride > 0; stride >>= 2) {
                const int lo = pos & (stride - 1);
                const int i0 = ((pos - lo) << 2) + lo;
                const int i1 = i0 + stride;
                const int i2 = i1 + stride;
                const int i3 = i2 + stride;

                team.team_barrier();
                float D0 = s_data(i0);
                float D1 = s_data(i1);
                float D2 = s_data(i2);
                float D3 = s_data(i3);

                float T;
                T = D0; D0 = D0 + D2; D2 = T - D2;
                T = D1; D1 = D1 + D3; D3 = T - D3;
                T = D0;
                s_data(i0) = D0 + D1;
                s_data(i1) = T  - D1;
                T = D2;
                s_data(i2) = D2 + D3;
                s_data(i3) = T  - D3;
            }

            // Single radix-2 stage for odd log2N
            if (log2N_fin & 1) {
                team.team_barrier();
                for (int p = lid; p < N_fin / 2; p += gsz) {
                    const int i0 = p << 1;
                    const int i1 = i0 + 1;
                    const float D0 = s_data(i0);
                    const float D1 = s_data(i1);
                    s_data(i0) = D0 + D1;
                    s_data(i1) = D0 - D1;
                }
            }

            team.team_barrier();
            // Write back
            for (int p = lid; p < N_fin; p += gsz)
                d_Data(base + p) = s_data(p);
        });
    Kokkos::fence();
}

// Element-wise modulation: d_A[i] *= d_B[i] / N
void modulateGPU(View1Df d_A, View1Df d_B, int N)
{
    const float rcpN = 1.0f / static_cast<float>(N);
    Kokkos::parallel_for(
        "modulate",
        Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(int pos) {
            d_A(pos) *= d_B(pos) * rcpN;
        });
    Kokkos::fence();
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    Kokkos::initialize(argc, argv);
    {
        const int log2Data   = 23;
        const int dataN      = 1 << log2Data;
        const int DATA_SIZE  = dataN * sizeof(float);

        const int log2Kernel = 7;
        const int kernelN    = 1 << log2Kernel;
        const int KERNEL_SIZE = kernelN * sizeof(float);

        printf("Data length: %d; kernel length: %d\n", dataN, kernelN);
        printf("Initializing data...\n");

        float *h_Kernel    = (float *)malloc(KERNEL_SIZE);
        float *h_Data      = (float *)malloc(DATA_SIZE);
        float *h_ResultCPU = (float *)malloc(DATA_SIZE);

        srand(123);
        for (int i = 0; i < kernelN; i++)
            h_Kernel[i] = (float)rand() / (float)RAND_MAX;
        for (int i = 0; i < dataN; i++)
            h_Data[i] = (float)rand() / (float)RAND_MAX;

        // Allocate device views
        View1Df d_Data  ("d_Data",   dataN);
        View1Df d_Kernel("d_Kernel", dataN);

        auto hm_Data   = Kokkos::create_mirror_view(d_Data);
        auto hm_Kernel = Kokkos::create_mirror_view(d_Kernel);

        printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");

        float total_time = 0.f;
        for (int i = 0; i < repeat; i++) {
            // Copy padded kernel (rest is zero) and full data to device
            memset(hm_Kernel.data(), 0, DATA_SIZE);
            memcpy(hm_Kernel.data(), h_Kernel, KERNEL_SIZE);
            memcpy(hm_Data.data(), h_Data, DATA_SIZE);
            Kokkos::deep_copy(d_Kernel, hm_Kernel);
            Kokkos::deep_copy(d_Data,   hm_Data);

            auto start = std::chrono::steady_clock::now();

            fwtBatchGPU(d_Data,   1, log2Data);
            fwtBatchGPU(d_Kernel, 1, log2Data);
            modulateGPU(d_Data, d_Kernel, dataN);
            fwtBatchGPU(d_Data,   1, log2Data);

            auto end = std::chrono::steady_clock::now();
            total_time += (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        printf("Average device execution time %f (s)\n", (total_time * 1e-9f) / repeat);

        // Retrieve result
        Kokkos::deep_copy(hm_Data, d_Data);

        // CPU reference
        printf("Running straightforward CPU dyadic convolution...\n");
        dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

        // Verify
        printf("Comparing the results...\n");
        double sum_delta2 = 0, sum_ref2 = 0;
        for (int i = 0; i < dataN; i++) {
            double delta = h_ResultCPU[i] - hm_Data(i);
            double ref   = h_ResultCPU[i];
            sum_delta2  += delta * delta;
            sum_ref2    += ref   * ref;
        }
        double L2norm = sqrt(sum_delta2 / sum_ref2);

        printf("Shutting down...\n");
        free(h_ResultCPU);
        free(h_Data);
        free(h_Kernel);

        printf("L2 norm: %E\n", L2norm);
        printf(L2norm < 1e-6 ? "PASS\n" : "FAIL\n");
    }
    Kokkos::finalize();
    return 0;
}
