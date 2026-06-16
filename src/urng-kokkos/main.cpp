/**********************************************************************
  Copyright 2013 Advanced Micro Devices, Inc. All rights reserved.
  Ported to Kokkos (self-contained, synthetic image).
 ********************************************************************/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <Kokkos_Core.hpp>

// ---------------------------------------------------------------------------
// URNG constants
// ---------------------------------------------------------------------------
#define GROUP_SIZE 256
#define FACTOR     25
#define IA         16807
#define IM         2147483647
#define AM         (1.f / (float)IM)
#define IQ         127773
#define IR         2836
#define NTAB       16
#define NDIV       (1 + (IM - 1) / NTAB)

// ---------------------------------------------------------------------------
// Simple pixel types (self-contained, no SDKBitMap dependency)
// ---------------------------------------------------------------------------
struct uchar4 { unsigned char x, y, z, w; };
struct float4 { float x, y, z, w; };

KOKKOS_INLINE_FUNCTION
float4 convert_float4(uchar4 v) {
    return {(float)v.x, (float)v.y, (float)v.z, (float)v.w};
}

KOKKOS_INLINE_FUNCTION
uchar4 convert_uchar4_sat(float4 v) {
    auto clamp = [](float c) -> unsigned char {
        return (unsigned char)(c > 255.f ? 255.f : (c < 0.f ? 0.f : c));
    };
    return {clamp(v.x), clamp(v.y), clamp(v.z), clamp(v.w)};
}

KOKKOS_INLINE_FUNCTION
float4 operator+(float4 a, float b) {
    return {a.x + b, a.y + b, a.z + b, a.w + b};
}

// ---------------------------------------------------------------------------
// Park-Miller PRNG with Bays-Durham shuffle
// tid  - thread index within the work-group (replaces omp_get_thread_num())
// iv   - pointer to the group's shared shuffle table
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
float ran1(int idum, int *iv, int tid) {
    int j, k, iy = 0;

    for (j = NTAB; j >= 0; j--) {
        k    = idum / IQ;
        idum = IA * (idum - k * IQ) - IR * k;
        if (idum < 0) idum += IM;
        if (j < NTAB)  iv[NTAB * tid + j] = idum;
    }
    iy = iv[NTAB * tid];

    k    = idum / IQ;
    idum = IA * (idum - k * IQ) - IR * k;
    if (idum < 0) idum += IM;

    j  = iy / NDIV;
    iy = iv[NTAB * tid + j];
    return AM * iy;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <block_size> <repeat>\n", argv[0]);
        return 1;
    }
    const int block_size = atoi(argv[1]);
    const int iterations = atoi(argv[2]);

    // Synthetic 256×256 RGBA image
    const int width    = 256;
    const int height   = 256;
    const int n_pixels = width * height;
    const int factor   = FACTOR;

    uchar4 *inputImageData  = (uchar4 *)malloc(n_pixels * sizeof(uchar4));
    uchar4 *outputImageData = (uchar4 *)malloc(n_pixels * sizeof(uchar4));
    memset(outputImageData, 0, n_pixels * sizeof(uchar4));

    // Fill with a deterministic pattern so the mean check is meaningful
    for (int i = 0; i < n_pixels; i++) {
        inputImageData[i] = {
            (unsigned char)((i * 3 + 50)  & 0xFF),
            (unsigned char)((i * 7 + 100) & 0xFF),
            (unsigned char)((i * 5 + 150) & 0xFF),
            (unsigned char)((i * 2 + 200) & 0xFF)
        };
    }

    Kokkos::initialize(argc, argv);
    {
        using exec_space   = Kokkos::DefaultExecutionSpace;
        using mem_space    = Kokkos::DefaultExecutionSpace::memory_space;
        using ScratchSpace = exec_space::scratch_memory_space;
        using IntScratch   = Kokkos::View<int*, ScratchSpace,
                                          Kokkos::MemoryUnmanaged>;

        Kokkos::View<uchar4*, mem_space> d_input("d_input",   n_pixels);
        Kokkos::View<uchar4*, mem_space> d_output("d_output", n_pixels);

        {
            auto h_in = Kokkos::View<uchar4*, Kokkos::HostSpace,
                                     Kokkos::MemoryUnmanaged>(
                            inputImageData, n_pixels);
            Kokkos::deep_copy(d_input, h_in);
        }

        const int block = block_size;          // threads per team
        const int teams = n_pixels / block;    // number of teams

        // Each team needs NTAB * block ints for the shuffle table
        int scratch_size = IntScratch::shmem_size(NTAB * block);

        auto policy =
            Kokkos::TeamPolicy<exec_space>(teams, block)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        std::cout << "Image height: " << height << " width: " << width
                  << std::endl;
        std::cout << "Executing kernel for " << iterations
                  << " iterations" << std::endl;
        std::cout << "-------------------------------------------"
                  << std::endl;

        auto start = std::chrono::steady_clock::now();

        for (int it = 0; it < iterations; it++) {
            Kokkos::parallel_for(
                "urng", policy,
                KOKKOS_LAMBDA(
                    const Kokkos::TeamPolicy<exec_space>::member_type
                        &team) {
                    // Per-team shuffle table in fast scratch memory
                    IntScratch iv(team.team_scratch(0),
                                  NTAB * team.team_size());

                    const int tid = team.team_rank();
                    const int pos = team.league_rank() * team.team_size()
                                    + tid;

                    float4 temp = convert_float4(d_input(pos));
                    float  avg  = (temp.x + temp.y + temp.z + temp.w)
                                  / 4.0f;

                    // Park-Miller requires a negative seed
                    int seed = -(int)avg;
                    if (seed >= 0) seed = -1;

                    float dev = ran1(seed, iv.data(), tid);
                    dev = (dev - 0.55f) * (float)factor;

                    d_output(pos) = convert_uchar4_sat(temp + dev);
                });
        }
        Kokkos::fence();

        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        std::cout << "Average kernel execution time: "
                  << (time * 1e-3f) / iterations << " (us)\n";

        // Copy result d2h
        {
            auto h_out = Kokkos::create_mirror_view(d_output);
            Kokkos::deep_copy(h_out, d_output);
            memcpy(outputImageData, h_out.data(),
                   n_pixels * sizeof(uchar4));
        }
    }
    Kokkos::finalize();

    // Verify: the mean deviation over all channels / factor should be < 1
    float mean = 0.f;
    for (int i = 0; i < n_pixels; i++) {
        mean += (float)outputImageData[i].x - (float)inputImageData[i].x;
        mean += (float)outputImageData[i].y - (float)inputImageData[i].y;
        mean += (float)outputImageData[i].z - (float)inputImageData[i].z;
        mean += (float)outputImageData[i].w - (float)inputImageData[i].w;
    }
    mean /= ((float)(n_pixels * 4) * factor);
    std::cout << "The averaged mean of the image: " << mean << std::endl;

    if (fabsf(mean) < 1.0f)
        std::cout << "PASS" << std::endl;
    else
        std::cout << "FAIL" << std::endl;

    free(inputImageData);
    free(outputImageData);
    return 0;
}
