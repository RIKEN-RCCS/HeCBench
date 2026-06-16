#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ---- constants (from constants.h) ----
const float PI = 3.14159265f;
const unsigned WALLACE_POOL_SIZE        = 2048;
const unsigned WALLACE_POOL_SIZE_MASK   = WALLACE_POOL_SIZE - 1;
const unsigned WALLACE_RUNS_PER_THREAD  = 2;
const unsigned WALLACE_NUM_THREADS      = WALLACE_POOL_SIZE / (4 * WALLACE_RUNS_PER_THREAD); // 256
const unsigned WALLACE_MAX_OUTPUTS_PER_ITERATION = 4;
const unsigned WALLACE_OUTPUT_COMBINE_COUNT = 1;
const unsigned WALLACE_NUM_BLOCKS       = 16384;
const unsigned WALLACE_NUM_POOL_PASSES  = 1;
const unsigned WALLACE_NUM_OUTPUTS_PER_RUN = 1;
const unsigned WALLACE_NUM_RANDOM_NUMBERS_PER_THREAD =
    WALLACE_NUM_OUTPUTS_PER_RUN * WALLACE_RUNS_PER_THREAD * WALLACE_MAX_OUTPUTS_PER_ITERATION;
const unsigned WALLACE_TOTAL_NUM_THREADS   = WALLACE_NUM_BLOCKS * WALLACE_NUM_THREADS;
const unsigned WALLACE_TOTAL_POOL_SIZE     = WALLACE_POOL_SIZE * WALLACE_NUM_BLOCKS;
const unsigned WALLACE_NUM_RANDOM_NUMBERS_PER_BLOCK =
    WALLACE_NUM_RANDOM_NUMBERS_PER_THREAD * WALLACE_NUM_THREADS;
const unsigned WALLACE_OUTPUT_SIZE =
    WALLACE_NUM_RANDOM_NUMBERS_PER_BLOCK * WALLACE_NUM_BLOCKS;
const unsigned WALLACE_CHI2_VALUES_PER_BLOCK = WALLACE_NUM_OUTPUTS_PER_RUN;
const unsigned WALLACE_CHI2_COUNT = WALLACE_CHI2_VALUES_PER_BLOCK * WALLACE_NUM_BLOCKS;
const unsigned WALLACE_CHI2_OFFSET      = WALLACE_POOL_SIZE;
const unsigned WALLACE_CHI2_SHARED_SIZE = 1;

// ---- host-side RNG helpers (from rand_helpers.h) ----
static unsigned z_s=362436069, w_s=521288629, jsr_s=123456789, jcong_s=380116160;

static unsigned Kiss()
{
    z_s = 36969*(z_s&65535)+(z_s>>16);
    w_s = 18000*(w_s&65535)+(w_s>>16);
    unsigned mwc = (z_s<<16)+w_s;
    jsr_s ^= (jsr_s<<17); jsr_s ^= (jsr_s>>13); jsr_s ^= (jsr_s<<5);
    jcong_s = 69069*jcong_s+1234567;
    return (mwc^jcong_s)+jsr_s;
}

static double Rand()
{
    unsigned long long x = Kiss();
    x = (x<<32)|Kiss();
    return x * 5.4210108624275221703311375920553e-20;
}

static bool cached_rn = false;
static double cached_rn_val = 0.0;

static double RandN()
{
    if (cached_rn) { cached_rn = false; return cached_rn_val; }
    double a = sqrt(-2.0*log(Rand()));
    double b = 6.283185307179586476925286766559*Rand();
    cached_rn_val = sin(b)*a;
    cached_rn = true;
    return cos(b)*a;
}

static double MakeChi2Scale(unsigned N)
{
    const double chic1 = sqrt(sqrt(1.0 - 1.0/N));
    const double chic2 = sqrt(1.0 - chic1*chic1);
    return chic1 + chic2*RandN();
}

// ---- Kokkos scratch memory type ----
using ExecSpace   = Kokkos::DefaultExecutionSpace;
using ScratchSpace = ExecSpace::scratch_memory_space;
using ScratchView  = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged>;
using TeamPolicy   = Kokkos::TeamPolicy<ExecSpace>;
using MemberType   = TeamPolicy::member_type;

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    // Allocate and initialise host arrays
    float* h_globalPool       = (float*)malloc(4 * WALLACE_TOTAL_POOL_SIZE);
    float* h_chi2Corrections  = (float*)malloc(4 * WALLACE_CHI2_COUNT);
    float* h_randomNumbers    = (float*)malloc(4 * WALLACE_OUTPUT_SIZE);

    for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
        h_globalPool[i] = (float)RandN();

    for (unsigned i = 0; i < WALLACE_CHI2_COUNT; i++)
        h_chi2Corrections[i] = (float)MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);

    Kokkos::initialize(argc, argv);
    {
        // Device views
        Kokkos::View<float*, ExecSpace> d_globalPool("globalPool", WALLACE_TOTAL_POOL_SIZE);
        Kokkos::View<float*, ExecSpace> d_chi2("chi2Corrections", WALLACE_CHI2_COUNT);
        Kokkos::View<float*, ExecSpace> d_randomNumbers("randomNumbers", WALLACE_OUTPUT_SIZE);

        // Host mirrors
        auto h_gp   = Kokkos::create_mirror_view(d_globalPool);
        auto h_chi2 = Kokkos::create_mirror_view(d_chi2);

        for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++) h_gp(i)   = h_globalPool[i];
        for (unsigned i = 0; i < WALLACE_CHI2_COUNT; i++)      h_chi2(i) = h_chi2Corrections[i];

        Kokkos::deep_copy(d_globalPool, h_gp);
        Kokkos::deep_copy(d_chi2, h_chi2);

        // Scratch size: WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE floats per team
        const int scratch_size = ScratchView::shmem_size(WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE);

        auto policy = TeamPolicy(WALLACE_NUM_BLOCKS, WALLACE_NUM_THREADS)
                          .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        const unsigned m_seed = 1;

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < repeat; iter++) {
            Kokkos::parallel_for("rng_wallace", policy,
                KOKKOS_LAMBDA(const MemberType& team) {
                    ScratchView pool(team.team_scratch(0),
                                     WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE);

                    const unsigned lcg_a  = 241;
                    const unsigned lcg_c  = 59;
                    const unsigned lcg_m  = 256;
                    const unsigned mod_mask = lcg_m - 1;

                    const unsigned lid    = (unsigned)team.team_rank();
                    const unsigned gid    = (unsigned)team.league_rank();
                    const unsigned offset = WALLACE_POOL_SIZE * gid;

                    // Load pool from global memory (8 values per thread)
                    for (unsigned i = 0; i < 8; i++)
                        pool[lid + WALLACE_NUM_THREADS * i] =
                            d_globalPool[offset + lid + WALLACE_NUM_THREADS * i];

                    team.team_barrier();

                    unsigned t_seed = m_seed;

                    for (unsigned loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++) {
                        t_seed = (1664525U * t_seed + 1013904223U) & 0xFFFFFFFF;

                        unsigned intermediate_address =
                            loop * 8 * WALLACE_TOTAL_NUM_THREADS +
                            8 * WALLACE_NUM_THREADS * gid + lid;

                        if (lid == 0)
                            pool[WALLACE_CHI2_OFFSET] =
                                d_chi2[gid * WALLACE_NUM_OUTPUTS_PER_RUN + loop];

                        team.team_barrier();
                        float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];

                        for (unsigned i = 0; i < 8; i++) {
                            d_randomNumbers[intermediate_address + i * WALLACE_NUM_THREADS] =
                                pool[i * WALLACE_NUM_THREADS + lid] * chi2CorrAndScale;
                        }

                        float rin0_0, rin1_0, rin2_0, rin3_0;
                        float rin0_1, rin1_1, rin2_1, rin3_1;

                        for (unsigned i = 0; i < WALLACE_NUM_POOL_PASSES; i++) {
                            unsigned seed = (t_seed + lid) & mod_mask;
                            team.team_barrier();

                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin0_0 = pool[(seed << 3)];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin1_0 = pool[(seed << 3) + 1];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin2_0 = pool[(seed << 3) + 2];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin3_0 = pool[(seed << 3) + 3];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin0_1 = pool[(seed << 3) + 4];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin1_1 = pool[(seed << 3) + 5];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin2_1 = pool[(seed << 3) + 6];
                            seed = (seed * lcg_a + lcg_c) & mod_mask;
                            rin3_1 = pool[(seed << 3) + 7];

                            team.team_barrier();

                            // Hadamard4x4a
                            {
                                float t = (rin0_0 + rin1_0 + rin2_0 + rin3_0) / 2.f;
                                rin0_0 = rin0_0 - t;
                                rin1_0 = rin1_0 - t;
                                rin2_0 = t - rin2_0;
                                rin3_0 = t - rin3_0;
                            }
                            pool[0 * WALLACE_NUM_THREADS + lid] = rin0_0;
                            pool[1 * WALLACE_NUM_THREADS + lid] = rin1_0;
                            pool[2 * WALLACE_NUM_THREADS + lid] = rin2_0;
                            pool[3 * WALLACE_NUM_THREADS + lid] = rin3_0;

                            // Hadamard4x4b
                            {
                                float t = (rin0_1 + rin1_1 + rin2_1 + rin3_1) / 2.f;
                                rin0_1 = t - rin0_1;
                                rin1_1 = t - rin1_1;
                                rin2_1 = rin2_1 - t;
                                rin3_1 = rin3_1 - t;
                            }
                            pool[4 * WALLACE_NUM_THREADS + lid] = rin0_1;
                            pool[5 * WALLACE_NUM_THREADS + lid] = rin1_1;
                            pool[6 * WALLACE_NUM_THREADS + lid] = rin2_1;
                            pool[7 * WALLACE_NUM_THREADS + lid] = rin3_1;

                            team.team_barrier();
                        }
                    }

                    team.team_barrier();

                    // Write pool back to global memory
                    for (unsigned i = 0; i < 8; i++)
                        d_globalPool[offset + lid + WALLACE_NUM_THREADS * i] =
                            pool[lid + WALLACE_NUM_THREADS * i];
                });
            Kokkos::fence();
        }

        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

        // Copy results back for optional debug print
        auto h_rn = Kokkos::create_mirror_view(d_randomNumbers);
        Kokkos::deep_copy(h_rn, d_randomNumbers);

#ifdef DEBUG
        for (unsigned n = 0; n < WALLACE_OUTPUT_SIZE; n++)
            printf("%.3f\n", h_rn(n));
#endif
    }
    Kokkos::finalize();

    free(h_globalPool);
    free(h_chi2Corrections);
    free(h_randomNumbers);
    return 0;
}
