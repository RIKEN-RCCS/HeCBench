//Scott Grauer-Gray (sgrauerg@gmail.com) - original
// Ported to Kokkos (GPU-only, no CPU co-execution)

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>

// Pull in reference implementations and utilities from the OMP sibling.
// The Makefile adds -I../winograd-omp so this include resolves correctly.
#include "utils.h"
#include "utils.cpp"

int main(int argc, char *argv[]) {

    DATA_TYPE *A      = (DATA_TYPE *)malloc(MAP_SIZE * MAP_SIZE
                                            * sizeof(DATA_TYPE));
    DATA_TYPE *B      = (DATA_TYPE *)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2)
                                            * sizeof(DATA_TYPE));
    DATA_TYPE *B_ref  = (DATA_TYPE *)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2)
                                            * sizeof(DATA_TYPE));
    DATA_TYPE *C      = (DATA_TYPE *)malloc(4 * 4 * sizeof(DATA_TYPE));

    srand(42);
    for (int i = 0; i < MAP_SIZE; i++)
        for (int j = 0; j < MAP_SIZE; j++)
            A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

    WinogradConv2D_2x2_filter_transformation(C);

    const int tile_n = (MAP_SIZE - 2 + 1) / 2;

    Kokkos::initialize(argc, argv);
    {
        using exec_space = Kokkos::DefaultExecutionSpace;
        using mem_space  = Kokkos::DefaultExecutionSpace::memory_space;

        Kokkos::View<DATA_TYPE*, mem_space> d_A("d_A",
            MAP_SIZE * MAP_SIZE);
        Kokkos::View<DATA_TYPE*, mem_space> d_B("d_B",
            (MAP_SIZE - 2) * (MAP_SIZE - 2));
        Kokkos::View<DATA_TYPE*, mem_space> d_C("d_C", 16);

        {
            auto h_A = Kokkos::View<DATA_TYPE*, Kokkos::HostSpace,
                                    Kokkos::MemoryUnmanaged>(
                           A, MAP_SIZE * MAP_SIZE);
            Kokkos::deep_copy(d_A, h_A);
            auto h_C = Kokkos::View<DATA_TYPE*, Kokkos::HostSpace,
                                    Kokkos::MemoryUnmanaged>(C, 16);
            Kokkos::deep_copy(d_C, h_C);
        }

        const int map_sz   = MAP_SIZE;
        const int out_sz   = MAP_SIZE - 2;
        const int tile_n_  = tile_n;

        double co_time = 0.0;
        bool   pass    = true;

        // Mirror the OMP version's sweep over cpu_offset (0..100).
        // In this GPU-only port every iteration runs entirely on device.
        for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

            // Zero the output buffer before each run so d2h copy is fresh
            Kokkos::deep_copy(d_B, DATA_TYPE(0));

            double co_start = rtclock();

            Kokkos::parallel_for(
                "winograd",
                Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(
                    {0, 0}, {tile_n_, tile_n_}),
                KOKKOS_LAMBDA(int tile_i, int tile_j) {

                    // ---- input transformation ----
                    DATA_TYPE input_tile[4][4];
                    DATA_TYPE tmp_tile[4][4];
                    DATA_TYPE transformed_tile[4][4];

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            int x = 2 * tile_i + i;
                            int y = 2 * tile_j + j;
                            if (x >= map_sz || y >= map_sz)
                                input_tile[i][j] = 0;
                            else
                                input_tile[i][j] =
                                    d_A[x * map_sz + y];
                        }
                    }

                    // Bt * d
                    for (int j = 0; j < 4; j++) {
                        tmp_tile[0][j] =  input_tile[0][j]
                                        - input_tile[2][j];
                        tmp_tile[1][j] =  input_tile[1][j]
                                        + input_tile[2][j];
                        tmp_tile[2][j] = -input_tile[1][j]
                                        + input_tile[2][j];
                        tmp_tile[3][j] =  input_tile[1][j]
                                        - input_tile[3][j];
                    }
                    // d * B
                    for (int i = 0; i < 4; i++) {
                        transformed_tile[i][0] =  tmp_tile[i][0]
                                                 - tmp_tile[i][2];
                        transformed_tile[i][1] =  tmp_tile[i][1]
                                                 + tmp_tile[i][2];
                        transformed_tile[i][2] = -tmp_tile[i][1]
                                                 + tmp_tile[i][2];
                        transformed_tile[i][3] =  tmp_tile[i][1]
                                                 - tmp_tile[i][3];
                    }

                    // ---- element-wise multiply with transformed filter ----
                    DATA_TYPE multiplied_tile[4][4];
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                            multiplied_tile[i][j] =
                                transformed_tile[i][j]
                                * d_C[i * 4 + j];

                    // ---- output transformation ----
                    DATA_TYPE tmp_tile_1[2][4];
                    DATA_TYPE final_tile[2][2];

                    // At * I
                    for (int j = 0; j < 4; j++) {
                        tmp_tile_1[0][j] = multiplied_tile[0][j]
                                         + multiplied_tile[1][j]
                                         + multiplied_tile[2][j];
                        tmp_tile_1[1][j] = multiplied_tile[1][j]
                                         - multiplied_tile[2][j]
                                         - multiplied_tile[3][j];
                    }
                    // I * A
                    for (int i = 0; i < 2; i++) {
                        final_tile[i][0] = tmp_tile_1[i][0]
                                         + tmp_tile_1[i][1]
                                         + tmp_tile_1[i][2];
                        final_tile[i][1] = tmp_tile_1[i][1]
                                         - tmp_tile_1[i][2]
                                         - tmp_tile_1[i][3];
                    }

                    // Write output
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 2; j++) {
                            int x = 2 * tile_i + i;
                            int y = 2 * tile_j + j;
                            if (x < out_sz && y < out_sz)
                                d_B[x * out_sz + y] = final_tile[i][j];
                        }
                    }
                });
            Kokkos::fence();

            co_time += rtclock() - co_start;

            // Copy result d2h and verify against scalar reference
            {
                auto h_B = Kokkos::create_mirror_view(d_B);
                Kokkos::deep_copy(h_B, d_B);
                memcpy(B, h_B.data(),
                       (MAP_SIZE - 2) * (MAP_SIZE - 2)
                       * sizeof(DATA_TYPE));
            }

            WinogradConv2D_2x2(A, B_ref, C);
            pass &= compareResults(B_ref, B);
        }

        printf("%s\n", pass ? "PASS" : "FAIL");
        printf("Co-execution time (GPU only): %lf s\n", co_time);
    }
    Kokkos::finalize();

    free(A);
    free(B);
    free(B_ref);
    free(C);
    return 0;
}
