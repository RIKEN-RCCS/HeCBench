#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define nx 680
#define ny 134
#define nz 450

KOKKOS_INLINE_FUNCTION
int indexTo1D(int x, int y, int z) {
    return x + y*nx + z*nx*ny;
}

// CPU reference implementation
void rtm8_cpu(float* vsq, float* current_s, float* current_r,
              float* next_s, float* next_r, float* image, float* a)
{
    for (int z = 4; z < nz - 4; z++) {
        for (int y = 4; y < ny - 4; y++) {
            for (int x = 4; x < nx - 4; x++) {
                float div =
                    a[0] * current_s[indexTo1D(x,y,z)] +
                    a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
                            current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
                            current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
                    a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
                            current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
                            current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
                    a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
                            current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
                            current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
                    a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
                            current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
                            current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

                next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)]
                    - next_s[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)]*div;

                div =
                    a[0] * current_r[indexTo1D(x,y,z)] +
                    a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
                            current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
                            current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
                    a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
                            current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
                            current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
                    a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
                            current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
                            current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
                    a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
                            current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
                            current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

                next_r[indexTo1D(x,y,z)] = 2*current_r[indexTo1D(x,y,z)]
                    - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)]*div;

                image[indexTo1D(x,y,z)] =
                    next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    const int ArraySize = nx * ny * nz;

    float* next_s    = (float*)malloc(ArraySize * sizeof(float));
    float* current_s = (float*)malloc(ArraySize * sizeof(float));
    float* next_r    = (float*)malloc(ArraySize * sizeof(float));
    float* current_r = (float*)malloc(ArraySize * sizeof(float));
    float* vsq       = (float*)malloc(ArraySize * sizeof(float));
    float* image_gpu = (float*)malloc(ArraySize * sizeof(float));
    float* image_cpu = (float*)malloc(ArraySize * sizeof(float));
    float  a[5];

    double memory = (double)ArraySize * sizeof(float) * 6;
    double pts    = (double)repeat * (nx-8) * (ny-8) * (nz-8);
    double flops  = 67.0 * pts;
    printf("memory (MB) = %lf\n", memory / 1e6);
    printf("pts (billions) = %lf\n", pts / 1e9);
    printf("Tflops = %lf\n", flops / 1e12);

    a[0] = -1.f/560.f;
    a[1] =  8.f/315.f;
    a[2] = -0.2f;
    a[3] =  1.6f;
    a[4] = -1435.f/504.f;

    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                vsq[indexTo1D(x,y,z)]       = 1.0f;
                next_s[indexTo1D(x,y,z)]    = 0.0f;
                current_s[indexTo1D(x,y,z)] = 1.0f;
                next_r[indexTo1D(x,y,z)]    = 0.0f;
                current_r[indexTo1D(x,y,z)] = 1.0f;
                image_gpu[indexTo1D(x,y,z)] = 0.5f;
                image_cpu[indexTo1D(x,y,z)] = 0.5f;
            }

    Kokkos::initialize(argc, argv);
    {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using MemSpace  = ExecSpace::memory_space;

        Kokkos::View<float*, MemSpace> d_vsq      ("vsq",       ArraySize);
        Kokkos::View<float*, MemSpace> d_current_s("current_s", ArraySize);
        Kokkos::View<float*, MemSpace> d_current_r("current_r", ArraySize);
        Kokkos::View<float*, MemSpace> d_next_s   ("next_s",    ArraySize);
        Kokkos::View<float*, MemSpace> d_next_r   ("next_r",    ArraySize);
        Kokkos::View<float*, MemSpace> d_image    ("image_gpu", ArraySize);
        Kokkos::View<float[5], MemSpace> d_a      ("a");

        // Copy inputs to device
        {
            auto h_vsq       = Kokkos::create_mirror_view(d_vsq);
            auto h_current_s = Kokkos::create_mirror_view(d_current_s);
            auto h_current_r = Kokkos::create_mirror_view(d_current_r);
            auto h_next_s    = Kokkos::create_mirror_view(d_next_s);
            auto h_next_r    = Kokkos::create_mirror_view(d_next_r);
            auto h_image     = Kokkos::create_mirror_view(d_image);
            auto h_a         = Kokkos::create_mirror_view(d_a);

            for (int i = 0; i < ArraySize; i++) {
                h_vsq(i)       = vsq[i];
                h_current_s(i) = current_s[i];
                h_current_r(i) = current_r[i];
                h_next_s(i)    = next_s[i];
                h_next_r(i)    = next_r[i];
                h_image(i)     = image_gpu[i];
            }
            for (int i = 0; i < 5; i++) h_a(i) = a[i];

            Kokkos::deep_copy(d_vsq,       h_vsq);
            Kokkos::deep_copy(d_current_s, h_current_s);
            Kokkos::deep_copy(d_current_r, h_current_r);
            Kokkos::deep_copy(d_next_s,    h_next_s);
            Kokkos::deep_copy(d_next_r,    h_next_r);
            Kokkos::deep_copy(d_image,     h_image);
            Kokkos::deep_copy(d_a,         h_a);
        }

        auto t0 = std::chrono::steady_clock::now();

        for (int t = 0; t < repeat; t++) {
            Kokkos::parallel_for("rtm8",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {4, 4, 4}, {nz-4, ny-4, nx-4}),
                KOKKOS_LAMBDA(int z, int y, int x) {
                    float div =
                        d_a[0] * d_current_s[indexTo1D(x,y,z)] +
                        d_a[1] * (d_current_s[indexTo1D(x+1,y,z)] + d_current_s[indexTo1D(x-1,y,z)] +
                                  d_current_s[indexTo1D(x,y+1,z)] + d_current_s[indexTo1D(x,y-1,z)] +
                                  d_current_s[indexTo1D(x,y,z+1)] + d_current_s[indexTo1D(x,y,z-1)]) +
                        d_a[2] * (d_current_s[indexTo1D(x+2,y,z)] + d_current_s[indexTo1D(x-2,y,z)] +
                                  d_current_s[indexTo1D(x,y+2,z)] + d_current_s[indexTo1D(x,y-2,z)] +
                                  d_current_s[indexTo1D(x,y,z+2)] + d_current_s[indexTo1D(x,y,z-2)]) +
                        d_a[3] * (d_current_s[indexTo1D(x+3,y,z)] + d_current_s[indexTo1D(x-3,y,z)] +
                                  d_current_s[indexTo1D(x,y+3,z)] + d_current_s[indexTo1D(x,y-3,z)] +
                                  d_current_s[indexTo1D(x,y,z+3)] + d_current_s[indexTo1D(x,y,z-3)]) +
                        d_a[4] * (d_current_s[indexTo1D(x+4,y,z)] + d_current_s[indexTo1D(x-4,y,z)] +
                                  d_current_s[indexTo1D(x,y+4,z)] + d_current_s[indexTo1D(x,y-4,z)] +
                                  d_current_s[indexTo1D(x,y,z+4)] + d_current_s[indexTo1D(x,y,z-4)]);

                    d_next_s[indexTo1D(x,y,z)] = 2*d_current_s[indexTo1D(x,y,z)]
                        - d_next_s[indexTo1D(x,y,z)] + d_vsq[indexTo1D(x,y,z)]*div;

                    div =
                        d_a[0] * d_current_r[indexTo1D(x,y,z)] +
                        d_a[1] * (d_current_r[indexTo1D(x+1,y,z)] + d_current_r[indexTo1D(x-1,y,z)] +
                                  d_current_r[indexTo1D(x,y+1,z)] + d_current_r[indexTo1D(x,y-1,z)] +
                                  d_current_r[indexTo1D(x,y,z+1)] + d_current_r[indexTo1D(x,y,z-1)]) +
                        d_a[2] * (d_current_r[indexTo1D(x+2,y,z)] + d_current_r[indexTo1D(x-2,y,z)] +
                                  d_current_r[indexTo1D(x,y+2,z)] + d_current_r[indexTo1D(x,y-2,z)] +
                                  d_current_r[indexTo1D(x,y,z+2)] + d_current_r[indexTo1D(x,y,z-2)]) +
                        d_a[3] * (d_current_r[indexTo1D(x+3,y,z)] + d_current_r[indexTo1D(x-3,y,z)] +
                                  d_current_r[indexTo1D(x,y+3,z)] + d_current_r[indexTo1D(x,y-3,z)] +
                                  d_current_r[indexTo1D(x,y,z+3)] + d_current_r[indexTo1D(x,y,z-3)]) +
                        d_a[4] * (d_current_r[indexTo1D(x+4,y,z)] + d_current_r[indexTo1D(x-4,y,z)] +
                                  d_current_r[indexTo1D(x,y+4,z)] + d_current_r[indexTo1D(x,y-4,z)] +
                                  d_current_r[indexTo1D(x,y,z+4)] + d_current_r[indexTo1D(x,y,z-4)]);

                    d_next_r[indexTo1D(x,y,z)] = 2*d_current_r[indexTo1D(x,y,z)]
                        - d_next_r[indexTo1D(x,y,z)] + d_vsq[indexTo1D(x,y,z)]*div;

                    d_image[indexTo1D(x,y,z)] =
                        d_next_s[indexTo1D(x,y,z)] * d_next_r[indexTo1D(x,y,z)];
                });
            Kokkos::fence();
        }

        auto t1 = std::chrono::steady_clock::now();
        double dt_gpu = std::chrono::duration<double>(t1 - t0).count();

        // Copy result back
        {
            auto h_image = Kokkos::create_mirror_view(d_image);
            Kokkos::deep_copy(h_image, d_image);
            for (int i = 0; i < ArraySize; i++) image_gpu[i] = h_image(i);
        }
        // Also need next_s/next_r for the CPU run (re-init from device)
        {
            auto h_next_s = Kokkos::create_mirror_view(d_next_s);
            auto h_next_r = Kokkos::create_mirror_view(d_next_r);
            Kokkos::deep_copy(h_next_s, d_next_s);
            Kokkos::deep_copy(h_next_r, d_next_r);
            for (int i = 0; i < ArraySize; i++) {
                next_s[i] = h_next_s(i);
                next_r[i] = h_next_r(i);
            }
        }

        // CPU reference run (re-initialise wavefield arrays first)
        float* cpu_next_s = (float*)malloc(ArraySize * sizeof(float));
        float* cpu_next_r = (float*)malloc(ArraySize * sizeof(float));
        for (int i = 0; i < ArraySize; i++) { cpu_next_s[i] = 0.0f; cpu_next_r[i] = 0.0f; }

        auto tc0 = std::chrono::steady_clock::now();
        for (int t = 0; t < repeat; t++)
            rtm8_cpu(vsq, current_s, current_r, cpu_next_s, cpu_next_r, image_cpu, a);
        auto tc1 = std::chrono::steady_clock::now();
        double dt_cpu = std::chrono::duration<double>(tc1 - tc0).count();

        // Verification
        bool ok = true;
        for (int i = 0; i < ArraySize; i++) {
            if (fabsf(image_cpu[i] - image_gpu[i]) > 0.1f) {
                printf("@index %d host: %f device %f\n", i, image_cpu[i], image_gpu[i]);
                ok = false;
                break;
            }
        }
        printf("%s\n", ok ? "PASS" : "FAIL");

        double pt_rate   = pts / dt_gpu;
        double flop_rate = flops / dt_gpu;
        double speedup   = dt_cpu / dt_gpu;
        printf("dt = %lf\n", dt_gpu);
        printf("pt_rate (millions/sec) = %lf\n", pt_rate / 1e6);
        printf("flop_rate (Gflops) = %lf\n", flop_rate / 1e9);
        printf("speedup over cpu = %lf\n", speedup);
        printf("average kernel execution time = %lf (s)\n", dt_gpu / repeat);

        free(cpu_next_s);
        free(cpu_next_r);
    }
    Kokkos::finalize();

    free(vsq); free(next_s); free(current_s);
    free(next_r); free(current_r);
    free(image_cpu); free(image_gpu);
    return 0;
}
