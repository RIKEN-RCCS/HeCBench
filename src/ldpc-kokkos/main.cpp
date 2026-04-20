/*
 * LDPC decoder benchmark – Kokkos port of ldpc-omp
 *
 * Faithful translation of the WIMAX QC-LDPC belief-propagation decoder.
 * Sources ported:
 *   kernel.cpp  → Kokkos TeamPolicy parallel_for kernels
 *   cpu.cpp     → unchanged host functions
 *   main.cpp    → Kokkos-based host driver
 *   matrix.h    → h_base embedded here
 *   LDPC.h      → constants embedded here
 *
 * Key mapping from OpenMP → Kokkos:
 *   omp_get_team_num()    → team.league_rank()
 *   omp_get_thread_num()  → team.team_rank()
 *   team-shared arrays    → Kokkos scratch memory (set_scratch_size)
 *   #pragma omp barrier   → team.team_barrier()
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// LDPC.h constants (WIMAX mode)
// ============================================================================
#define YES 1
#define NO  0

#define WIMAX 0
#define WIFI  1
#define MODE  WIMAX
#define MIN_SUM YES

// Simulation parameters (reduced for quick benchmark run)
#define NUM_SNR      1
static float snr_array[NUM_SNR] = {3.0f};
#define MIN_FER         10
#define MIN_CODEWORD    80
#define MAX_ITERATION   10
#define MAX_SIM         2

#define CW  2   // code words per macro codeword
#define MCW 40  // number of macro codewords

// WIMAX fixed parameters
#define Z                   96
#define NON_EMPTY_ELMENT    7
#define NON_EMPTY_ELMENT_VNP 6

#define BLOCK_SIZE_X        ((Z + 32 - 1) / 32 * 32)   // = 96
#define THREADS_PER_BLOCK   (BLOCK_SIZE_X * CW)         // = 192

#define BLK_ROW  12
#define BLK_COL  24

#define ROW          (Z * BLK_ROW)        // 1152
#define COL          (Z * BLK_COL)        // 2304
#define INFO_LEN     (BLK_ROW * Z / 2)    // 576  (rate-1/2 code)
#define CODEWORD_LEN (BLK_COL * Z)        // 2304

// Actually for rate-1/2 WIMAX the information length is BLK_ROW/2 * Z = 576
// but the original uses INFO_LEN = BLK_INFO * Z where BLK_INFO = BLK_ROW = 12
// Wait – let me check: BLK_INFO = BLK_ROW = 12, INFO_LEN = 12*96 = 1152.
// Redefine to match the original exactly:
#undef INFO_LEN
#define BLK_INFO        BLK_ROW
#define INFO_LEN        (BLK_INFO * Z)     // 1152

#define H_COMPACT1_ROW  BLK_ROW
#define H_COMPACT1_COL  NON_EMPTY_ELMENT
#define H_COMPACT1      (BLK_ROW * NON_EMPTY_ELMENT)  // 84

#define H_COMPACT2_ROW  BLK_ROW
#define H_COMPACT2_COL  BLK_COL

// ============================================================================
// Types
// ============================================================================
struct error_result {
    int bit_error;
    int frame_error;
};

struct h_element {
    char x;      // block row
    char y;      // block col
    char value;  // cyclic shift
    char valid;
};

// ============================================================================
// matrix.h – WIMAX 802.16e base matrix
// ============================================================================
static int h_base[BLK_ROW][BLK_COL] = {
    {-1, 94, 73, -1, -1, -1, -1, -1, 55, 83, -1, -1,  7,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, 27, -1, -1, -1, 22, 79,  9, -1, -1, -1, 12, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, 24, 22, 81, -1, 33, -1, -1, -1,  0, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1},
    {61, -1, 47, -1, -1, -1, -1, -1, 65, 25, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 39, -1, -1, -1, 84, -1, -1, 41, 72, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, 46, 40, -1, 82, -1, -1, -1, 79,  0, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1},
    {-1, -1, 95, 53, -1, -1, -1, -1, -1, 14, 18, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1},
    {-1, 11, 73, -1, -1, -1,  2, -1, -1, 47, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1},
    {12, -1, -1, -1, 83, 24, -1, 43, -1, -1, -1, 51, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1},
    {-1, -1, -1, -1, -1, 94, -1, 59, -1, -1, 70, 72, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1},
    {-1, -1,  7, 65, -1, -1, -1, -1, 39, 49, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0},
    {43, -1, -1, -1, -1, 66, -1, 41, -1, -1, -1, 26,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0}
};

// ============================================================================
// Globals used by CPU functions
// ============================================================================
float sigma;
int  *info_bin;

// ============================================================================
// cpu.cpp – host utility functions (unchanged from original)
// ============================================================================
static void info_gen(int info[], long seed)
{
    std::srand((unsigned)seed);
    for (int i = 0; i < INFO_LEN; i++)
        info[i] = std::rand() % 2;
}

static void modulation(const int code[], float trans[])
{
    for (int i = 0; i < CODEWORD_LEN; i++)
        trans[i] = (code[i] == 0) ? 1.0f : -1.0f;
}

static void awgn(const float trans[], float recv[], long seed)
{
    std::srand((unsigned)seed);
    for (int i = 0; i < CODEWORD_LEN; i++) {
        float u1, u2, s;
        do {
            u1 = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;
            u2 = (float)std::rand() / RAND_MAX * 2.0f - 1.0f;
            s  = u1 * u1 + u2 * u2;
        } while (s >= 1.0f);
        float noise = u1 * std::sqrt(-2.0f * std::log(s) / s);
        recv[i] = trans[i] + noise * sigma;
    }
}

static void llr_init(float llr[], const float recv[])
{
    for (int i = 0; i < CODEWORD_LEN; i++)
        llr[i] = (recv[i] * 2.0f) / (sigma * sigma);
}

static error_result error_check(const int info_all[], const int hd_all[])
{
    error_result r = {0, 0};
    for (int i = 0; i < CW * MCW; i++) {
        int bit_err = 0;
        const int *hd   = hd_all   + i * CODEWORD_LEN;
        const int *info = info_all + i * INFO_LEN;
        for (int j = 0; j < INFO_LEN; j++)
            if (info[j] != hd[j]) bit_err++;
        if (bit_err) r.frame_error++;
        r.bit_error += bit_err;
    }
    return r;
}

static void structure_encode(const int s[], int code[], int h[][BLK_COL])
{
    int x[BLK_INFO][Z] = {};
    int sum_x[Z] = {};

    for (int i = 0; i < BLK_INFO; i++)
        for (int j = 0; j < BLK_INFO; j++) {
            int shift = h[i][j];
            if (shift >= 0)
                for (int k = 0; k < Z; k++) {
                    int sk = (k + shift) % Z;
                    int jj = j * Z + sk;
                    x[i][k] = (x[i][k] + s[jj]) % 2;
                }
        }

    for (int i = 0; i < Z; i++)
        for (int j = 0; j < BLK_INFO; j++)
            sum_x[i] = (x[j][i] + sum_x[i]) % 2;

    int id = INFO_LEN;

    // p0
    int p0[Z], p1[Z], p2[Z], p3[Z], p4[Z], p5[Z];
    int p6[Z], p7[Z], p8[Z], p9[Z], p10[Z], pp[Z];

    for (int i = 0; i < Z; i++) code[id++] = p0[i] = sum_x[i];

    int shift0 = h[0][BLK_INFO];
    for (int i = 0; i < Z; i++) {
        int j = (shift0 >= 0) ? (i + shift0) % Z : i;
        pp[i] = p0[j];
    }

    for (int i = 0; i < Z; i++) code[id++] = p1[i]  = (x[0][i] + pp[i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p2[i]  = (p1[i] + x[1][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p3[i]  = (p2[i] + x[2][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p4[i]  = (p3[i] + x[3][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p5[i]  = (p4[i] + x[4][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p6[i]  = (p5[i] + x[5][i] + p0[i]) % 2;
    for (int i = 0; i < Z; i++) code[id++] = p7[i]  = (p6[i] + x[6][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p8[i]  = (p7[i] + x[7][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p9[i]  = (p8[i] + x[8][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++] = p10[i] = (p9[i] + x[9][i])  % 2;
    for (int i = 0; i < Z; i++) code[id++]           = (p10[i]+ x[10][i]) % 2;

    for (int i = 0; i < INFO_LEN; i++) code[i] = s[i];
}

// ============================================================================
// Kokkos kernels
// ============================================================================

// Scratch-memory type alias used across all kernels
using ScratchF =
    Kokkos::View<float *,
                 Kokkos::DefaultExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryUnmanaged>;

// ----------------------------------------------------------------------------
// Kernel 1a – check-node processing, 1st iteration (no RCache needed)
// ----------------------------------------------------------------------------
static void ldpc_cnp_kernel_1st_iter(
    Kokkos::View<const float *> dev_llr,
    Kokkos::View<float *>       dev_dt,
    Kokkos::View<float *>       dev_R,
    Kokkos::View<const char *>  dev_h_element_count1,
    Kokkos::View<const h_element *> dev_h_compact1)
{
    const int size_llr_CW = COL;
    const int size_R_CW   = ROW * BLK_COL;

    auto policy = Kokkos::TeamPolicy<>(BLK_ROW * MCW, THREADS_PER_BLOCK);

    Kokkos::parallel_for(
        "ldpc_cnp_1st", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            const int iSubRow = team.team_rank() % BLOCK_SIZE_X;
            const int iCW     = team.team_rank() / BLOCK_SIZE_X;
            const int iMCW    = team.league_rank() / BLK_ROW;
            const int iBlkRow = team.league_rank() % BLK_ROW;
            const int iCurrentCW = iMCW * CW + iCW;

            int s = dev_h_element_count1(iBlkRow);
            int offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

            char  Q_sign  = 0;
            char  sq;
            float sign    = 1.0f;
            float rmin1   = 1000.0f;
            float rmin2   = 1000.0f;
            char  idx_min = 0;

            // 1st recursion: find two minima
            for (int i = 0; i < s; i++) {
                h_element he = dev_h_compact1(i * H_COMPACT1_ROW + iBlkRow);
                int shift_t = (int)(unsigned char)he.value;
                shift_t = (iSubRow + shift_t);
                if (shift_t >= Z) shift_t -= Z;
                int iCol = (int)(unsigned char)he.y * Z + shift_t;

                float Q     = dev_llr(size_llr_CW * iCurrentCW + iCol);
                float Q_abs = fabsf(Q);
                sq = Q < 0;
                sign    *= (1 - sq * 2);
                Q_sign  |= sq << i;

                if (Q_abs < rmin1) {
                    rmin2 = rmin1; rmin1 = Q_abs; idx_min = i;
                } else if (Q_abs < rmin2) {
                    rmin2 = Q_abs;
                }
            }

            // 2nd recursion: write messages
            for (int i = 0; i < s; i++) {
                sq = 1 - 2 * ((Q_sign >> i) & 0x01);
                float R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

                h_element he   = dev_h_compact1(i * H_COMPACT1_ROW + iBlkRow);
                int addr_temp  = offsetR + (int)(unsigned char)he.y * ROW;
                dev_dt(addr_temp) = R_temp;
                dev_R(addr_temp)  = R_temp;
            }
        });
}

// ----------------------------------------------------------------------------
// Kernel 1b – check-node processing, subsequent iterations (uses RCache)
// ----------------------------------------------------------------------------
static void ldpc_cnp_kernel(
    Kokkos::View<const float *> dev_llr,
    Kokkos::View<float *>       dev_dt,
    Kokkos::View<float *>       dev_R,
    Kokkos::View<const char *>  dev_h_element_count1,
    Kokkos::View<const h_element *> dev_h_compact1)
{
    const int size_llr_CW = COL;
    const int size_R_CW   = ROW * BLK_COL;
    const int scratch_floats = THREADS_PER_BLOCK * NON_EMPTY_ELMENT;

    auto policy = Kokkos::TeamPolicy<>(BLK_ROW * MCW, THREADS_PER_BLOCK)
                      .set_scratch_size(
                          0, Kokkos::PerTeam(scratch_floats * sizeof(float)));

    Kokkos::parallel_for(
        "ldpc_cnp", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            ScratchF RCache(team.team_scratch(0), scratch_floats);

            const int iSubRow    = team.team_rank() % BLOCK_SIZE_X;
            const int iCW        = team.team_rank() / BLOCK_SIZE_X;
            const int iMCW       = team.league_rank() / BLK_ROW;
            const int iBlkRow    = team.league_rank() % BLK_ROW;
            const int iCurrentCW = iMCW * CW + iCW;
            const int iRCacheLine = iCW * BLOCK_SIZE_X + iSubRow;

            int s       = dev_h_element_count1(iBlkRow);
            int offsetR = size_R_CW * iCurrentCW + iBlkRow * Z + iSubRow;

            char  Q_sign  = 0;
            char  sq;
            float sign    = 1.0f;
            float rmin1   = 1000.0f;
            float rmin2   = 1000.0f;
            char  idx_min = 0;

            // 1st recursion: load RCache, find two minima
            for (int i = 0; i < s; i++) {
                h_element he = dev_h_compact1(i * H_COMPACT1_ROW + iBlkRow);
                int iBlkCol = (int)(unsigned char)he.y;
                int shift_t = (int)(unsigned char)he.value;
                shift_t = (iSubRow + shift_t);
                if (shift_t >= Z) shift_t -= Z;
                int iCol = iBlkCol * Z + shift_t;

                float R_temp = dev_R(offsetR + iBlkCol * ROW);
                RCache(i * THREADS_PER_BLOCK + iRCacheLine) = R_temp;

                float Q     = dev_llr(size_llr_CW * iCurrentCW + iCol) - R_temp;
                float Q_abs = fabsf(Q);
                sq = Q < 0;
                sign   *= (1 - sq * 2);
                Q_sign |= sq << i;

                if (Q_abs < rmin1) {
                    rmin2 = rmin1; rmin1 = Q_abs; idx_min = i;
                } else if (Q_abs < rmin2) {
                    rmin2 = Q_abs;
                }
            }

            team.team_barrier();

            // 2nd recursion: write messages
            for (int i = 0; i < s; i++) {
                sq = 1 - 2 * ((Q_sign >> i) & 0x01);
                float R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

                h_element he  = dev_h_compact1(i * H_COMPACT1_ROW + iBlkRow);
                int addr_temp = (int)(unsigned char)he.y * ROW + offsetR;
                dev_dt(addr_temp) = R_temp -
                                    RCache(i * THREADS_PER_BLOCK + iRCacheLine);
                dev_R(addr_temp)  = R_temp;
            }
        });
}

// ----------------------------------------------------------------------------
// Kernel 2a – variable-node processing (non-last iteration)
// ----------------------------------------------------------------------------
static void ldpc_vnp_kernel_normal(
    Kokkos::View<float *>       dev_llr,
    Kokkos::View<const float *> dev_dt,
    Kokkos::View<const char *>  dev_h_element_count2,
    Kokkos::View<const h_element *> dev_h_compact2)
{
    const int size_llr_CW = COL;
    const int size_R_CW   = ROW * BLK_COL;

    auto policy = Kokkos::TeamPolicy<>(BLK_COL * MCW, THREADS_PER_BLOCK);

    Kokkos::parallel_for(
        "ldpc_vnp", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            const int iSubCol    = team.team_rank() % BLOCK_SIZE_X;
            const int iCW        = team.team_rank() / BLOCK_SIZE_X;
            const int iMCW       = team.league_rank() / BLK_COL;
            const int iBlkCol    = team.league_rank() % BLK_COL;
            const int iCurrentCW = iMCW * CW + iCW;

            int iCol      = iBlkCol * Z + iSubCol;
            int llr_index = size_llr_CW * iCurrentCW + iCol;
            int offsetDt  = size_R_CW   * iCurrentCW + iBlkCol * ROW;

            float APP = dev_llr(llr_index);
            int   ns  = dev_h_element_count2(iBlkCol);

            for (int i = 0; i < ns; i++) {
                h_element he = dev_h_compact2(i * H_COMPACT2_COL + iBlkCol);
                int shift_t  = (int)(unsigned char)he.value;
                int iBlkRow  = (int)(unsigned char)he.x;
                int sf = iSubCol - shift_t;
                if (sf < 0) sf += Z;
                int iRow = iBlkRow * Z + sf;
                APP += dev_dt(offsetDt + iRow);
            }
            dev_llr(llr_index) = APP;
        });
}

// ----------------------------------------------------------------------------
// Kernel 2b – variable-node processing (last iteration, makes hard decisions)
// ----------------------------------------------------------------------------
static void ldpc_vnp_kernel_last_iter(
    Kokkos::View<const float *> dev_llr,
    Kokkos::View<const float *> dev_dt,
    Kokkos::View<int *>         dev_hd,
    Kokkos::View<const char *>  dev_h_element_count2,
    Kokkos::View<const h_element *> dev_h_compact2)
{
    const int size_llr_CW = COL;
    const int size_R_CW   = ROW * BLK_COL;

    auto policy = Kokkos::TeamPolicy<>(BLK_COL * MCW, THREADS_PER_BLOCK);

    Kokkos::parallel_for(
        "ldpc_vnp_last", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            const int iSubCol    = team.team_rank() % BLOCK_SIZE_X;
            const int iCW        = team.team_rank() / BLOCK_SIZE_X;
            const int iMCW       = team.league_rank() / BLK_COL;
            const int iBlkCol    = team.league_rank() % BLK_COL;
            const int iCurrentCW = iMCW * CW + iCW;

            int iCol      = iBlkCol * Z + iSubCol;
            int llr_index = size_llr_CW * iCurrentCW + iCol;
            int offsetDt  = size_R_CW   * iCurrentCW + iBlkCol * ROW;

            float APP = dev_llr(llr_index);
            int   ns  = dev_h_element_count2(iBlkCol);

            for (int i = 0; i < ns; i++) {
                h_element he = dev_h_compact2(i * H_COMPACT2_COL + iBlkCol);
                int shift_t  = (int)(unsigned char)he.value;
                int iBlkRow  = (int)(unsigned char)he.x;
                int sf = iSubCol - shift_t;
                if (sf < 0) sf += Z;
                int iRow = iBlkRow * Z + sf;
                APP += dev_dt(offsetDt + iRow);
            }
            dev_hd(llr_index) = (APP > 0) ? 0 : 1;
        });
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    printf("GPU LDPC Decoder (Kokkos)\nComputing...\n");

    Kokkos::initialize();
    {
        // ---- Build compact H matrices (identical to original main.cpp) ----
        const char h_element_count1[BLK_ROW] = {6,7,7,6,6,7,6,6,7,6,6,6};
        const char h_element_count2[BLK_COL] = {3,3,6,3,3,6,3,6,3,6,3,6,
                                                  3,2,2,2,2,2,2,2,2,2,2,2};

        h_element h_compact1[H_COMPACT1_COL * H_COMPACT1_ROW];
        h_element h_compact2[H_COMPACT2_ROW * H_COMPACT2_COL];

        h_element init = {0, 0, static_cast<char>(-1), 0};
        for (int i = 0; i < H_COMPACT1_COL * H_COMPACT1_ROW; i++) h_compact1[i] = init;
        for (int i = 0; i < H_COMPACT2_ROW * H_COMPACT2_COL; i++) h_compact2[i] = init;

        // h_compact1: column-major compact (for CNP)
        for (int i = 0; i < BLK_ROW; i++) {
            int k = 0;
            for (int j = 0; j < BLK_COL; j++) {
                if (h_base[i][j] != -1) {
                    h_compact1[k * H_COMPACT1_ROW + i] = {(char)i, (char)j,
                                                           (char)h_base[i][j], 1};
                    k++;
                }
            }
        }

        // h_compact2: column-major compact (for VNP)
        for (int j = 0; j < BLK_COL; j++) {
            int k = 0;
            for (int i = 0; i < BLK_ROW; i++) {
                if (h_base[i][j] != -1) {
                    h_compact2[k * H_COMPACT2_COL + j] = {(char)i, (char)j,
                                                           (char)h_base[i][j], 1};
                    k++;
                }
            }
        }

        // ---- Device views for constant data --------------------------------
        const int wh1 = H_COMPACT1_ROW * H_COMPACT1_COL;
        const int wh2 = H_COMPACT2_ROW * H_COMPACT2_COL;

        Kokkos::View<h_element *>    d_hc1("hc1",  wh1);
        Kokkos::View<h_element *>    d_hc2("hc2",  wh2);
        Kokkos::View<char *>         d_hec1("hec1", BLK_ROW);
        Kokkos::View<char *>         d_hec2("hec2", BLK_COL);

        {
            auto h = Kokkos::create_mirror_view(d_hc1);
            for (int i = 0; i < wh1; i++) h(i) = h_compact1[i];
            Kokkos::deep_copy(d_hc1, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_hc2);
            for (int i = 0; i < wh2; i++) h(i) = h_compact2[i];
            Kokkos::deep_copy(d_hc2, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_hec1);
            for (int i = 0; i < BLK_ROW; i++) h(i) = h_element_count1[i];
            Kokkos::deep_copy(d_hec1, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_hec2);
            for (int i = 0; i < BLK_COL; i++) h(i) = h_element_count2[i];
            Kokkos::deep_copy(d_hec2, h);
        }

        // ---- Simulation arrays ---------------------------------------------
        const int wordSize_llr  = MCW * CW * CODEWORD_LEN;
        const int wordSize_dt   = MCW * CW * ROW * BLK_COL;
        const int wordSize_R    = MCW * CW * ROW * BLK_COL;
        const int wordSize_hd   = MCW * CW * CODEWORD_LEN;

        Kokkos::View<float *>  d_llr("llr", wordSize_llr);
        Kokkos::View<float *>  d_dt("dt",   wordSize_dt);
        Kokkos::View<float *>  d_R("R",     wordSize_R);
        Kokkos::View<int *>    d_hd("hd",   wordSize_hd);

        // Const-qualified views for read-only kernel arguments
        Kokkos::View<const float *>     d_llr_c  = d_llr;
        Kokkos::View<const float *>     d_dt_c   = d_dt;
        Kokkos::View<const h_element *> d_hc1_c  = d_hc1;
        Kokkos::View<const h_element *> d_hc2_c  = d_hc2;
        Kokkos::View<const char *>      d_hec1_c = d_hec1;
        Kokkos::View<const char *>      d_hec2_c = d_hec2;

        auto h_llr = Kokkos::create_mirror_view(d_llr);
        auto h_hd  = Kokkos::create_mirror_view(d_hd);

        // CPU-side buffers
        info_bin = (int *)std::malloc(MCW * CW * INFO_LEN * sizeof(int));
        int   *codeword   = (int *)  std::malloc(CODEWORD_LEN * sizeof(int));
        float *trans      = (float *)std::malloc(CODEWORD_LEN * sizeof(float));
        float *recv       = (float *)std::malloc(CODEWORD_LEN * sizeof(float));
        float *llr_cpu    = (float *)std::malloc(CODEWORD_LEN * sizeof(float));

        const float rate = 0.5f;
        std::srand(69012);

        int total_frame_error = 0, total_bit_error = 0, total_codeword = 0;

        for (int snri = 0; snri < NUM_SNR; snri++) {
            float snr = snr_array[snri];
            sigma = 1.0f / std::sqrt(2.0f * rate *
                                     std::pow(10.0f, snr / 10.0f));
            total_codeword = total_frame_error = total_bit_error = 0;

            float total_time = 0.0f;

            while (total_frame_error <= MIN_FER &&
                   total_codeword    <= MIN_CODEWORD)
            {
                total_codeword += CW * MCW;

                // Generate CW*MCW codewords and LLRs
                for (int i = 0; i < CW * MCW; i++) {
                    info_gen(info_bin + i * INFO_LEN, std::rand());
                    structure_encode(info_bin + i * INFO_LEN, codeword, h_base);
                    modulation(codeword, trans);
                    awgn(trans, recv, std::rand());
                    llr_init(llr_cpu, recv);
                    std::memcpy(h_llr.data() + i * CODEWORD_LEN, llr_cpu,
                                CODEWORD_LEN * sizeof(float));
                }

                for (int j = 0; j < MAX_SIM; j++) {
                    Kokkos::deep_copy(d_llr, h_llr);

                    auto t0 = std::chrono::steady_clock::now();

                    for (int ii = 0; ii < MAX_ITERATION; ii++) {
                        if (ii == 0)
                            ldpc_cnp_kernel_1st_iter(d_llr_c, d_dt, d_R,
                                                     d_hec1_c, d_hc1_c);
                        else
                            ldpc_cnp_kernel(d_llr_c, d_dt, d_R,
                                            d_hec1_c, d_hc1_c);

                        if (ii < MAX_ITERATION - 1)
                            ldpc_vnp_kernel_normal(d_llr, d_dt_c,
                                                   d_hec2_c, d_hc2_c);
                        else
                            ldpc_vnp_kernel_last_iter(d_llr_c, d_dt_c, d_hd,
                                                      d_hec2_c, d_hc2_c);
                    }
                    Kokkos::fence();
                    total_time += std::chrono::duration<float>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count();

                    Kokkos::deep_copy(h_hd, d_hd);
                    error_result e = error_check(info_bin, h_hd.data());
                    total_bit_error   += e.bit_error;
                    total_frame_error += e.frame_error;
                }
            }

            printf("\nTotal kernel time: %.4f s\n", total_time);
            printf("# codewords = %d  (CW=%d, MCW=%d)\n",
                   total_codeword, CW, MCW);
            printf("total bit errors   = %d\n", total_bit_error);
            printf("total frame errors = %d\n", total_frame_error);
            printf("BER = %.2e,  FER = %.2e\n",
                   (float)total_bit_error  / total_codeword / INFO_LEN,
                   (float)total_frame_error / total_codeword);
        }

        std::free(info_bin);
        std::free(codeword);
        std::free(trans);
        std::free(recv);
        std::free(llr_cpu);
    }
    Kokkos::finalize();
    return 0;
}
