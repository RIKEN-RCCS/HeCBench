#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <Kokkos_Core.hpp>

// ---------------------------------------------------------------------------
// Type definitions (mirrors ../wlcpow-omp/utils.h but with Kokkos annotations)
// ---------------------------------------------------------------------------
typedef float  r32;
typedef double r64;
typedef int    i32;

// Define our own struct types; for CUDA/HIP these align with the built-ins
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// CUDA/HIP already provide float4, float3, int2 via their built-in headers.
// We only provide typedef aliases so the rest of the code compiles.
// (No struct redefinition needed.)
#else
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(16) float3 { float x, y, z; };
struct alignas(8)  int2   { int x, y; };
#endif

// ---------------------------------------------------------------------------
// Device-callable math helpers (replicate utils.h with KOKKOS annotations)
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
float minimum_image(float dr, float p) {
    float p_half = p * 0.5f;
    return dr + (dr > -p_half ? (dr < p_half ? 0.f : -p) : p);
}

KOKKOS_INLINE_FUNCTION
float bound(float x, float lower, float upper) {
    return fmaxf(lower, fminf(x, upper));
}

#define _LN_2           6.9314718055994528623E-1
#define _2_TO_MINUS_31  4.6566128730773925781E-10
#define _2_TO_MINUS_32  2.3283064365386962891E-10
#define _TEA_K0  0xA341316C
#define _TEA_K1  0xC8013EA4
#define _TEA_K2  0xAD90777D
#define _TEA_K3  0x7E95761E
#define _TEA_DT  0x9E3779B9

template<int N>
KOKKOS_INLINE_FUNCTION
void tea_core(unsigned int &v0, unsigned int &v1, unsigned int sum = 0) {
    sum += _TEA_DT;
    v0 += ((v1 << 4) + _TEA_K0) ^ (v1 + sum) ^ ((v1 >> 5) + _TEA_K1);
    v1 += ((v0 << 4) + _TEA_K2) ^ (v0 + sum) ^ ((v0 >> 5) + _TEA_K3);
    tea_core<N - 1>(v0, v1, sum);
}

template<>
KOKKOS_INLINE_FUNCTION
void tea_core<0>(unsigned int &, unsigned int &, unsigned int) {}

template<int N>
KOKKOS_INLINE_FUNCTION
float gaussian_TEA_fast(bool pred, int u, int v) {
    unsigned int v0 =  pred ? (unsigned)u : (unsigned)v;
    unsigned int v1 = !pred ? (unsigned)u : (unsigned)v;
    tea_core<N>(v0, v1);
    float f = sinf((float)M_PI * (int)v0 * (float)_2_TO_MINUS_31);
    float r = sqrtf(-2.0f * (float)_LN_2
                    * log2f(v1 * (float)_2_TO_MINUS_32));
    return bound(r * f, -4.0f, 4.0f);
}

// ---------------------------------------------------------------------------
// n_type is fixed at compile time (matches the OMP version's #define)
// ---------------------------------------------------------------------------
#define N_TYPE 32

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: ./%s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    float3 period  = {0.5f, 0.5f, 0.5f};
    int    padding = 1;
    int    n       = 1000000;   // problem size

    float4 *coord_merged = (float4 *)malloc((n + 1) * sizeof(float4));
    float4 *veloc        = (float4 *)malloc((n + 1) * sizeof(float4));
    int    *nbond        = (int *)   malloc((n + 1) * sizeof(int));
    int2   *bonds        = (int2 *)  malloc((n + n + 1) * sizeof(int2));
    r64    *bond_r0      = (r64 *)   malloc((n + n + 1) * sizeof(r64));
    r64    *force_x      = (r64 *)   malloc((n + 1) * sizeof(r64));
    r64    *force_y      = (r64 *)   malloc((n + 1) * sizeof(r64));
    r64    *force_z      = (r64 *)   malloc((n + 1) * sizeof(r64));
    r32    *bond_l0      = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *temp_h       = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *mu_targ      = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *qp           = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *gamc         = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *gamt         = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *sigc         = (r32 *)   malloc((n + 1) * sizeof(r32));
    r32    *sigt         = (r32 *)   malloc((n + 1) * sizeof(r32));

    std::mt19937 g(19937);
    std::uniform_real_distribution<r64> dist_r64(0.1, 0.9);
    std::uniform_real_distribution<r32> dist_r32(0.1, 0.9);
    std::uniform_int_distribution<i32>  dist_i32(0, N_TYPE);

    for (int i = 0; i < n + n + 1; i++) {
        bond_r0[i] = dist_r64(g) + 0.001;
        bonds[i]   = {(i + 1) % (n + 1), dist_i32(g)};
    }
    for (int i = 0; i < n + 1; i++) {
        force_x[i] = force_y[i] = force_z[i] = 0.0;
        nbond[i]  = dist_i32(g);
        r32 cx = dist_r32(g), cy = dist_r32(g), cz = dist_r32(g);
        coord_merged[i] = {cx, cy, cz, 0.f};
        r32 vx = dist_r32(g), vy = dist_r32(g), vz = dist_r32(g);
        veloc[i] = {vx, vy, vz, sqrtf(vx * vx + vy * vy + vz * vz)};
        bond_l0[i] = dist_r32(g);
        gamt[i]    = dist_r32(g);
        gamc[i]    = ((dist_i32(g) % 4) + 4) * gamt[i];
        temp_h[i]  = dist_r32(g);
        mu_targ[i] = dist_r32(g);
        qp[i]      = dist_r32(g);
        sigc[i]    = sqrtf(2.0f * temp_h[i] * (3.0f * gamc[i] - gamt[i]));
        sigt[i]    = 2.0f * sqrtf(gamt[i] * temp_h[i]);
    }

    Kokkos::initialize(argc, argv);
    {
        using exec_space = Kokkos::DefaultExecutionSpace;
        using mem_space  = Kokkos::DefaultExecutionSpace::memory_space;
        using ScratchSpace = exec_space::scratch_memory_space;
        using R32Scratch   = Kokkos::View<r32*, ScratchSpace,
                                          Kokkos::MemoryUnmanaged>;

        // Allocate device Views
        Kokkos::View<float4*, mem_space> d_coord("d_coord", n + 1);
        Kokkos::View<float4*, mem_space> d_veloc("d_veloc", n + 1);
        Kokkos::View<int*,    mem_space> d_nbond("d_nbond", n + 1);
        Kokkos::View<int2*,   mem_space> d_bonds("d_bonds", n + n + 1);
        Kokkos::View<r64*,    mem_space> d_bond_r0("d_bond_r0", n + n + 1);
        Kokkos::View<r64*,    mem_space> d_force_x("d_force_x", n + 1);
        Kokkos::View<r64*,    mem_space> d_force_y("d_force_y", n + 1);
        Kokkos::View<r64*,    mem_space> d_force_z("d_force_z", n + 1);
        Kokkos::View<r32*,    mem_space> d_bond_l0("d_bond_l0", n + 1);
        Kokkos::View<r32*,    mem_space> d_temp("d_temp",       n + 1);
        Kokkos::View<r32*,    mem_space> d_mu_targ("d_mu_targ", n + 1);
        Kokkos::View<r32*,    mem_space> d_qp("d_qp",           n + 1);
        Kokkos::View<r32*,    mem_space> d_gamc("d_gamc",       n + 1);
        Kokkos::View<r32*,    mem_space> d_gamt("d_gamt",       n + 1);
        Kokkos::View<r32*,    mem_space> d_sigc("d_sigc",       n + 1);
        Kokkos::View<r32*,    mem_space> d_sigt("d_sigt",       n + 1);

        // Helper to copy host array → device View
        auto h2d = [&](auto &d_view, auto *h_ptr, int count) {
            using T = typename std::remove_reference<decltype(d_view)>::type
                          ::value_type;
            auto h_view = Kokkos::View<T*, Kokkos::HostSpace,
                                       Kokkos::MemoryUnmanaged>(h_ptr, count);
            Kokkos::deep_copy(d_view, h_view);
        };

        h2d(d_coord,   coord_merged, n + 1);
        h2d(d_veloc,   veloc,        n + 1);
        h2d(d_nbond,   nbond,        n + 1);
        h2d(d_bonds,   bonds,        n + n + 1);
        h2d(d_bond_r0, bond_r0,      n + n + 1);
        h2d(d_force_x, force_x,      n + 1);
        h2d(d_force_y, force_y,      n + 1);
        h2d(d_force_z, force_z,      n + 1);
        h2d(d_bond_l0, bond_l0,      n + 1);
        h2d(d_temp,    temp_h,       n + 1);
        h2d(d_mu_targ, mu_targ,      n + 1);
        h2d(d_qp,      qp,           n + 1);
        h2d(d_gamc,    gamc,         n + 1);
        h2d(d_gamt,    gamt,         n + 1);
        h2d(d_sigc,    sigc,         n + 1);
        h2d(d_sigt,    sigt,         n + 1);

        const int teams_count = (n + 127) / 128;
        const int blocks_per_team = 128;

        // Shared scratch: (N_TYPE+1)*8 r32 values
        int scratch_size = R32Scratch::shmem_size((N_TYPE + 1) * 8);

        auto policy =
            Kokkos::TeamPolicy<exec_space>(teams_count, blocks_per_team)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        const float3 per = period;
        const int    pad = padding;
        const int    nl  = n;

        auto start = std::chrono::steady_clock::now();

        // Note: outputs are accumulated across all repetitions (matches OMP)
        for (int rep = 0; rep < repeat; rep++) {
            Kokkos::parallel_for(
                "wlcpow", policy,
                KOKKOS_LAMBDA(
                    const Kokkos::TeamPolicy<exec_space>::member_type
                        &team) {
                    R32Scratch shared_data(team.team_scratch(0),
                                          (N_TYPE + 1) * 8);

                    r32 *temp    = &shared_data(0);
                    r32 *r0      = &shared_data(1 * (N_TYPE + 1));
                    r32 *mu_targ = &shared_data(2 * (N_TYPE + 1));
                    r32 *qp      = &shared_data(3 * (N_TYPE + 1));
                    r32 *gamc    = &shared_data(4 * (N_TYPE + 1));
                    r32 *gamt    = &shared_data(5 * (N_TYPE + 1));
                    r32 *sigc    = &shared_data(6 * (N_TYPE + 1));
                    r32 *sigt    = &shared_data(7 * (N_TYPE + 1));

                    int tid    = team.team_rank();
                    int bid    = team.league_rank();
                    int bdim   = team.team_size();
                    int gdim   = team.league_size();

                    // Load bond-type parameters into scratch
                    for (int i = tid; i < N_TYPE + 1; i += bdim) {
                        temp[i]    = d_temp(i);
                        r0[i]      = d_bond_l0(i);
                        mu_targ[i] = d_mu_targ(i);
                        qp[i]      = d_qp(i);
                        gamc[i]    = d_gamc(i);
                        gamt[i]    = d_gamt(i);
                        sigc[i]    = d_sigc(i);
                        sigt[i]    = d_sigt(i);
                    }
                    team.team_barrier();

                    for (int i = bid * bdim + tid; i < nl;
                         i += gdim * bdim) {
                        int    nb     = d_nbond(i);
                        float4 coord1 = d_coord(i);
                        float4 veloc1 = d_veloc(i);
                        r32    fxi = 0.f, fyi = 0.f, fzi = 0.f;

                        for (int p = 0; p < nb; p++) {
                            int    j    = d_bonds(i + p * pad).x;
                            int    type = d_bonds(i + p * pad).y;
                            float4 coord2 = d_coord(j);

                            r32 delx = minimum_image(
                                coord1.x - coord2.x, per.x);
                            r32 dely = minimum_image(
                                coord1.y - coord2.y, per.y);
                            r32 delz = minimum_image(
                                coord1.z - coord2.z, per.z);

                            float4 veloc2 = d_veloc(j);
                            r32 dvx = veloc1.x - veloc2.x;
                            r32 dvy = veloc1.y - veloc2.y;
                            r32 dvz = veloc1.z - veloc2.z;

                            r32 l0     = (r32)d_bond_r0(i + p * pad);
                            r32 ra     = sqrtf(delx*delx + dely*dely
                                               + delz*delz);
                            r32 lmax   = l0 * r0[type];
                            r32 rr     = 1.0f / r0[type];
                            r32 sr     = (1.0f - rr) * (1.0f - rr);
                            r32 kph    = powf(l0, qp[type])
                                         * temp[type]
                                         * (0.25f / sr - 0.25f + rr);
                            r32 mu     = 0.433f * (
                                temp[type] * (-0.25f / sr + 0.25f
                                    + 0.5f * rr / (sr * (1.0f - rr)))
                                    / (lmax * rr)
                                + kph * (qp[type] + 1.0f)
                                    / powf(l0, qp[type] + 1.0f));
                            r32 lambda = mu / mu_targ[type];
                            kph = kph / lambda;
                            rr  = ra / lmax;

                            r32 rlogarg = powf(ra, qp[type] + 1.0f);
                            r32 vv      = (delx*dvx + dely*dvy
                                          + delz*dvz) / ra;

                            if (rr >= 0.99f)   rr      = 0.99f;
                            if (rlogarg < 0.01f) rlogarg = 0.01f;

                            float4 wrr;
                            r32    ww[3][3];

                            // Bit-cast float velocity tag → int seed
                            int iv1, iv2;
                            memcpy(&iv1, &veloc1.w, sizeof(int));
                            memcpy(&iv2, &veloc2.w, sizeof(int));

                            for (int tes = 0; tes < 3; tes++)
                                for (int see = 0; see < 3; see++)
                                    ww[tes][see] =
                                        gaussian_TEA_fast<4>(
                                            iv1 > iv2,
                                            iv1 + tes, iv2 + see);

                            wrr.w = (ww[0][0]+ww[1][1]+ww[2][2]) / 3.0f;
                            wrr.x = (ww[0][0]-wrr.w)*delx
                                  + 0.5f*(ww[0][1]+ww[1][0])*dely
                                  + 0.5f*(ww[0][2]+ww[2][0])*delz;
                            wrr.y = 0.5f*(ww[1][0]+ww[0][1])*delx
                                  + (ww[1][1]-wrr.w)*dely
                                  + 0.5f*(ww[1][2]+ww[2][1])*delz;
                            wrr.z = 0.5f*(ww[2][0]+ww[0][2])*delx
                                  + 0.5f*(ww[2][1]+ww[1][2])*dely
                                  + (ww[2][2]-wrr.w)*delz;

                            r32 fforce =
                                -temp[type]
                                * (0.25f / (1.0f-rr) / (1.0f-rr)
                                   - 0.25f + rr)
                                / lambda / ra
                                + kph / rlogarg
                                + (sigc[type]*wrr.w - gamc[type]*vv)
                                  / ra;
                            r32 fxij = delx*fforce - gamt[type]*dvx
                                     + sigt[type]*wrr.x / ra;
                            r32 fyij = dely*fforce - gamt[type]*dvy
                                     + sigt[type]*wrr.y / ra;
                            r32 fzij = delz*fforce - gamt[type]*dvz
                                     + sigt[type]*wrr.z / ra;

                            fxi += fxij;
                            fyi += fyij;
                            fzi += fzij;
                        }
                        d_force_x(i) += fxi;
                        d_force_y(i) += fyi;
                        d_force_z(i) += fzi;
                    }
                });
            Kokkos::fence();
        }

        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        printf("Average kernel execution time: %f (us)\n",
               time * 1e-3f / repeat);

        // Copy results d2h
        {
            auto h_fx = Kokkos::create_mirror_view(d_force_x);
            auto h_fy = Kokkos::create_mirror_view(d_force_y);
            auto h_fz = Kokkos::create_mirror_view(d_force_z);
            Kokkos::deep_copy(h_fx, d_force_x);
            Kokkos::deep_copy(h_fy, d_force_y);
            Kokkos::deep_copy(h_fz, d_force_z);
            memcpy(force_x, h_fx.data(), (n + 1) * sizeof(r64));
            memcpy(force_y, h_fy.data(), (n + 1) * sizeof(r64));
            memcpy(force_z, h_fz.data(), (n + 1) * sizeof(r64));
        }
    }
    Kokkos::finalize();

    // Check for NaN
    for (int i = 0; i < n + 1; i++) {
        if (isnan(force_x[i]) || isnan(force_y[i]) || isnan(force_z[i]))
            printf("There are NaN numbers at index %d\n", i);
    }

    double sum_x = 0, sum_y = 0, sum_z = 0;
    for (int i = 0; i < n + 1; i++) {
        sum_x += force_x[i];
        sum_y += force_y[i];
        sum_z += force_z[i];
    }
    printf("checksum: forceX=%lf forceY=%lf forceZ=%lf\n",
           sum_x / (n + 1), sum_y / (n + 1), sum_z / (n + 1));

    free(coord_merged); free(veloc);   free(force_x); free(force_y);
    free(force_z);      free(nbond);   free(bonds);   free(bond_r0);
    free(bond_l0);      free(temp_h);  free(mu_targ); free(qp);
    free(gamc);         free(gamt);    free(sigc);    free(sigt);
    return 0;
}
