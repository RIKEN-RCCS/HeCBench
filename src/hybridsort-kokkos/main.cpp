/*
 * Hybridsort benchmark – Kokkos port of hybridsort-omp
 *
 * Algorithm (mirrors the original):
 *   1. Build a 1024-bin histogram of the input floats (GPU parallel).
 *   2. Compute 1024 pivot points from the histogram (CPU, exact copy of
 *      the original calcPivotPoints).
 *   3. Count elements per bucket and compute exclusive prefix offsets (GPU+CPU).
 *   4. Scatter elements to contiguous per-bucket regions (GPU parallel).
 *   5. Sort each bucket with an in-place bitonic sort (GPU, TeamPolicy).
 *   6. Verify the result against std::sort.
 *
 * Test size: N = 1 000 000 random floats.
 */

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Constants (mirror bucketsort.h)
// ---------------------------------------------------------------------------
static constexpr int DIVISIONS    = 1024;
static constexpr int HIST_SIZE    = 1024;
static constexpr int MAX_BUCKET   = 2048;   // max elements per bucket (power-of-2)
static constexpr int SORT_THREADS = 256;    // threads per team for bitonic sort
static constexpr int N            = 1000000;

// ---------------------------------------------------------------------------
// Device utility: find bucket index via binary search through pivot array
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int find_bucket(float val, const float *pivots, int n_pivots)
{
    int lo = 0, hi = n_pivots - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (val <= pivots[mid]) hi = mid;
        else                    lo = mid + 1;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// CPU: compute pivot points from histogram (exact copy of original)
// ---------------------------------------------------------------------------
static void calcPivotPoints(const float *histogram, int histosize, int listsize,
                             int divisions, float minv, float maxv,
                             float *pivotPoints, float histo_width)
{
    float elemsPerSlice = listsize / (float)divisions;
    float startsAt      = minv;
    float endsAt        = minv + histo_width;
    float we_need       = elemsPerSlice;
    int   p_idx         = 0;

    // Work on a mutable copy
    std::vector<float> hist(histogram, histogram + histosize);

    for (int i = 0; i < histosize; i++) {
        if (i == histosize - 1) {
            if (p_idx < divisions)
                pivotPoints[p_idx++] =
                    startsAt + (we_need / hist[i]) * histo_width;
            break;
        }
        while (hist[i] > we_need) {
            if (p_idx >= divisions) break;
            pivotPoints[p_idx++] =
                startsAt + (we_need / hist[i]) * histo_width;
            startsAt += (we_need / hist[i]) * histo_width;
            hist[i]  -= we_need;
            we_need   = elemsPerSlice;
        }
        we_need  -= hist[i];
        startsAt  = endsAt;
        endsAt   += histo_width;
    }
    while (p_idx < divisions) {
        pivotPoints[p_idx] = pivotPoints[p_idx - 1];
        p_idx++;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // ---- Generate input -----------------------------------------------
        std::srand(42);
        std::vector<float> h_input(N);
        float datamin =  FLT_MAX;
        float datamax = -FLT_MAX;
        for (int i = 0; i < N; i++) {
            h_input[i] = (float)std::rand() / RAND_MAX;
            datamin = std::fmin(h_input[i], datamin);
            datamax = std::fmax(h_input[i], datamax);
        }

        // Reference: sorted copy for verification
        std::vector<float> ref(h_input);
        std::sort(ref.begin(), ref.end());

        // ---- Device views ------------------------------------------------
        Kokkos::View<float *>        d_input("input",  N);
        Kokkos::View<unsigned int *> d_hist("hist",    HIST_SIZE);
        Kokkos::View<float *>        d_pivots("pivots", DIVISIONS);
        Kokkos::View<int *>          d_counts("counts", DIVISIONS);
        Kokkos::View<int *>          d_write_offsets("write_off", DIVISIONS);
        // Padded sorted storage: [bucket][slot] – init to FLT_MAX
        Kokkos::View<float *>        d_padded("padded",
                                              (long long)DIVISIONS * MAX_BUCKET);

        {
            auto h = Kokkos::create_mirror_view(d_input);
            for (int i = 0; i < N; i++) h(i) = h_input[i];
            Kokkos::deep_copy(d_input, h);
        }

        auto t_start = std::chrono::steady_clock::now();

        // ==================================================================
        // Step 1 – Histogram
        // ==================================================================
        Kokkos::deep_copy(d_hist, 0u);

        const float histo_width = (datamax - datamin) / HIST_SIZE;
        const float inv_hwidth  = 1.0f / histo_width;
        const float dmin        = datamin;

        Kokkos::parallel_for("histogram", N, KOKKOS_LAMBDA(int i) {
            int bin = (int)((d_input(i) - dmin) * inv_hwidth);
            if (bin < 0)         bin = 0;
            if (bin >= HIST_SIZE) bin = HIST_SIZE - 1;
            Kokkos::atomic_fetch_add(&d_hist(bin), 1u);
        });
        Kokkos::fence();

        // ==================================================================
        // Step 2 – CPU pivot computation
        // ==================================================================
        auto h_hist = Kokkos::create_mirror_view(d_hist);
        Kokkos::deep_copy(h_hist, d_hist);

        std::vector<float> h_hist_f(HIST_SIZE);
        for (int i = 0; i < HIST_SIZE; i++) h_hist_f[i] = (float)h_hist(i);

        std::vector<float> h_pivots(DIVISIONS);
        calcPivotPoints(h_hist_f.data(), HIST_SIZE, N, DIVISIONS,
                        datamin, datamax, h_pivots.data(), histo_width);

        {
            auto hp = Kokkos::create_mirror_view(d_pivots);
            for (int i = 0; i < DIVISIONS; i++) hp(i) = h_pivots[i];
            Kokkos::deep_copy(d_pivots, hp);
        }

        // ==================================================================
        // Step 3 – Count elements per bucket
        // ==================================================================
        Kokkos::deep_copy(d_counts, 0);

        Kokkos::parallel_for("count_buckets", N, KOKKOS_LAMBDA(int i) {
            int b = find_bucket(d_input(i), d_pivots.data(), DIVISIONS);
            Kokkos::atomic_fetch_add(&d_counts(b), 1);
        });
        Kokkos::fence();

        // ==================================================================
        // Step 4 – CPU prefix scan for scatter offsets
        // ==================================================================
        auto h_counts = Kokkos::create_mirror_view(d_counts);
        Kokkos::deep_copy(h_counts, d_counts);

        auto h_woff = Kokkos::create_mirror_view(d_write_offsets);
        std::vector<int> h_offsets(DIVISIONS + 1, 0);
        for (int i = 0; i < DIVISIONS; i++)
            h_offsets[i + 1] = h_offsets[i] + h_counts(i);

        for (int i = 0; i < DIVISIONS; i++) h_woff(i) = h_offsets[i];
        Kokkos::deep_copy(d_write_offsets, h_woff);

        // Init padded storage with FLT_MAX
        Kokkos::deep_copy(d_padded, FLT_MAX);

        // Reuse d_write_offsets as atomic scatter pointers (reset to offsets)
        Kokkos::View<int *> d_atomic_off("atomic_off", DIVISIONS);
        Kokkos::deep_copy(d_atomic_off, d_write_offsets);

        // ==================================================================
        // Step 5 – Scatter elements into padded bucket storage
        // ==================================================================
        Kokkos::parallel_for("scatter", N, KOKKOS_LAMBDA(int i) {
            int b   = find_bucket(d_input(i), d_pivots.data(), DIVISIONS);
            int pos = Kokkos::atomic_fetch_add(&d_atomic_off(b), 1);
            // pos is the absolute position in the padded array
            int slot = pos - d_write_offsets(b);   // position within bucket
            if (slot < MAX_BUCKET)
                d_padded((long long)b * MAX_BUCKET + slot) = d_input(i);
        });
        Kokkos::fence();

        // ==================================================================
        // Step 6 – Per-bucket bitonic sort (TeamPolicy)
        // ==================================================================
        auto sort_policy = Kokkos::TeamPolicy<>(DIVISIONS, SORT_THREADS);

        Kokkos::parallel_for(
            "bitonic_sort", sort_policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
                const int bid  = team.league_rank();
                const long long base = (long long)bid * MAX_BUCKET;

                // Bitonic sort over MAX_BUCKET elements (FLT_MAX pads the tail)
                for (int k = 2; k <= MAX_BUCKET; k <<= 1) {
                    for (int j = k >> 1; j >= 1; j >>= 1) {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, MAX_BUCKET),
                            [&](int idx) {
                                int l = idx ^ j;
                                if (l > idx) {
                                    bool ascending = ((idx & k) == 0);
                                    float ai = d_padded(base + idx);
                                    float al = d_padded(base + l);
                                    if ((ai > al) == ascending) {
                                        d_padded(base + idx) = al;
                                        d_padded(base + l)   = ai;
                                    }
                                }
                            });
                        team.team_barrier();
                    }
                }
            });
        Kokkos::fence();

        double ms = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start).count() * 1e3;

        // ==================================================================
        // Gather sorted result
        // ==================================================================
        auto h_padded = Kokkos::create_mirror_view(d_padded);
        Kokkos::deep_copy(h_padded, d_padded);

        std::vector<float> result;
        result.reserve(N);
        for (int b = 0; b < DIVISIONS; b++) {
            int cnt = h_counts(b);
            for (int s = 0; s < cnt && s < MAX_BUCKET; s++) {
                float v = h_padded((long long)b * MAX_BUCKET + s);
                if (v < FLT_MAX) result.push_back(v);
            }
        }

        // ==================================================================
        // Verification
        // ==================================================================
        bool pass = ((int)result.size() == N);
        if (pass) {
            for (int i = 0; i < N; i++) {
                if (result[i] != ref[i]) { pass = false; break; }
            }
        }

        printf("Sorted %d floats in %.3f ms  %s\n",
               N, ms, pass ? "PASSED" : "FAILED");
        if (!pass) {
            // Show first mismatch
            for (int i = 0; i < (int)result.size() && i < N; i++) {
                if (result[i] != ref[i]) {
                    printf("  First mismatch at index %d: got %f, expected %f\n",
                           i, result[i], ref[i]);
                    break;
                }
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
