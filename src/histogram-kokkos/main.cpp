/*
 * Histogram benchmark – Kokkos port of histogram-omp
 *
 * Tests two GPU histogram implementations on a 1920×1080 RGBA image:
 *   1. Global-memory atomics  (one parallel_for over all pixels)
 *   2. Shared-memory atomics  (TeamPolicy: per-team scratch histogram,
 *                              then a global reduction pass)
 *
 * Both results are verified against a CPU reference.
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
struct uchar4 {
    unsigned char x, y, z, w;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int NUM_CHANNELS = 4;
static constexpr int NUM_BINS     = 256;
static constexpr int WIDTH        = 1920;
static constexpr int HEIGHT       = 1080;
static constexpr int N_PIXELS     = WIDTH * HEIGHT;

// Match the OMP source: 256 teams × 128 threads = 32 768 virtual threads
// mapped on a 512-column × 64-row stride grid.
static constexpr int TOTAL_TEAMS = 256;
static constexpr int TEAM_SIZE   = 128;

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
static void cpu_histogram(const uchar4 *img, unsigned int *hist, int n)
{
    std::memset(hist, 0, NUM_CHANNELS * NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < n; i++) {
        hist[0 * NUM_BINS + img[i].x]++;
        hist[1 * NUM_BINS + img[i].y]++;
        hist[2 * NUM_BINS + img[i].z]++;
        hist[3 * NUM_BINS + img[i].w]++;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // ---- Build synthetic image ----------------------------------------
        Kokkos::View<uchar4 *> d_image("image", N_PIXELS);
        auto h_image = Kokkos::create_mirror_view(d_image);

        for (int i = 0; i < N_PIXELS; i++) {
            h_image(i).x = static_cast<unsigned char>((i * 37 + 13) & 0xFF);
            h_image(i).y = static_cast<unsigned char>((i * 53 + 29) & 0xFF);
            h_image(i).z = static_cast<unsigned char>((i * 71 + 41) & 0xFF);
            h_image(i).w = static_cast<unsigned char>((i * 89 + 57) & 0xFF);
        }
        Kokkos::deep_copy(d_image, h_image);

        // ---- CPU reference -----------------------------------------------
        std::vector<unsigned int> ref(NUM_CHANNELS * NUM_BINS, 0u);
        cpu_histogram(h_image.data(), ref.data(), N_PIXELS);

        // ==================================================================
        // 1. Global-memory atomics
        // ==================================================================
        Kokkos::View<unsigned int *> d_hist_g("hist_g", NUM_CHANNELS * NUM_BINS);
        Kokkos::deep_copy(d_hist_g, 0u);

        auto t0 = std::chrono::steady_clock::now();

        Kokkos::parallel_for("hist_gmem", N_PIXELS, KOKKOS_LAMBDA(int idx) {
            uchar4 p = d_image(idx);
            Kokkos::atomic_fetch_add(&d_hist_g(0 * 256 + p.x), 1u);
            Kokkos::atomic_fetch_add(&d_hist_g(1 * 256 + p.y), 1u);
            Kokkos::atomic_fetch_add(&d_hist_g(2 * 256 + p.z), 1u);
            Kokkos::atomic_fetch_add(&d_hist_g(3 * 256 + p.w), 1u);
        });
        Kokkos::fence();

        double ms_g = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count() * 1e3;

        auto h_hist_g = Kokkos::create_mirror_view(d_hist_g);
        Kokkos::deep_copy(h_hist_g, d_hist_g);

        bool pass_g = true;
        for (int i = 0; i < NUM_CHANNELS * NUM_BINS; i++)
            if (h_hist_g(i) != ref[i]) { pass_g = false; break; }

        printf("Gmem atomics:  %.3f ms  %s\n", ms_g, pass_g ? "PASSED" : "FAILED");

        // ==================================================================
        // 2. Shared-memory (scratch) atomics  –  TeamPolicy version
        //    Each team accumulates into its own scratch histogram, then
        //    the partial histograms are reduced into the final result.
        // ==================================================================
        using ScratchUI =
            Kokkos::View<unsigned int *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>;

        const int smem_bytes = NUM_CHANNELS * NUM_BINS * sizeof(unsigned int);

        // Partial histograms stored in global memory (TOTAL_TEAMS rows)
        Kokkos::View<unsigned int **> d_part("part_hist",
                                             TOTAL_TEAMS,
                                             NUM_CHANNELS * NUM_BINS);
        Kokkos::deep_copy(d_part, 0u);

        auto policy = Kokkos::TeamPolicy<>(TOTAL_TEAMS, TEAM_SIZE)
                          .set_scratch_size(0, Kokkos::PerTeam(smem_bytes));

        auto t2 = std::chrono::steady_clock::now();

        Kokkos::parallel_for(
            "hist_smem", policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
                const int g  = team.league_rank();
                const int t  = team.team_rank();
                const int nt = team.team_size();

                ScratchUI smem(team.team_scratch(0), NUM_CHANNELS * NUM_BINS);

                // Initialise scratch histogram
                for (int i = t; i < NUM_CHANNELS * NUM_BINS; i += nt)
                    smem(i) = 0u;
                team.team_barrier();

                // Stride pixel traversal (mirrors the OMP 512×64 grid)
                const int gid  = g * nt + t;
                const int col0 = gid % 512;
                const int row0 = gid / 512;

                for (int col = col0; col < WIDTH; col += 512) {
                    for (int row = row0; row < HEIGHT; row += 64) {
                        uchar4 p = d_image(row * WIDTH + col);
                        Kokkos::atomic_fetch_add(&smem(0 * 256 + p.x), 1u);
                        Kokkos::atomic_fetch_add(&smem(1 * 256 + p.y), 1u);
                        Kokkos::atomic_fetch_add(&smem(2 * 256 + p.z), 1u);
                        Kokkos::atomic_fetch_add(&smem(3 * 256 + p.w), 1u);
                    }
                }
                team.team_barrier();

                // Flush scratch to global partial histogram
                for (int i = t; i < NUM_CHANNELS * NUM_BINS; i += nt)
                    d_part(g, i) = smem(i);
            });

        // Reduce partial histograms → final result
        Kokkos::View<unsigned int *> d_hist_s("hist_s", NUM_CHANNELS * NUM_BINS);
        Kokkos::parallel_for(
            "reduce_parts", NUM_CHANNELS * NUM_BINS,
            KOKKOS_LAMBDA(int i) {
                unsigned int total = 0u;
                for (int j = 0; j < TOTAL_TEAMS; j++)
                    total += d_part(j, i);
                d_hist_s(i) = total;
            });
        Kokkos::fence();

        double ms_s = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t2).count() * 1e3;

        auto h_hist_s = Kokkos::create_mirror_view(d_hist_s);
        Kokkos::deep_copy(h_hist_s, d_hist_s);

        bool pass_s = true;
        for (int i = 0; i < NUM_CHANNELS * NUM_BINS; i++)
            if (h_hist_s(i) != ref[i]) { pass_s = false; break; }

        printf("Smem atomics:  %.3f ms  %s\n", ms_s, pass_s ? "PASSED" : "FAILED");
    }
    Kokkos::finalize();
    return 0;
}
