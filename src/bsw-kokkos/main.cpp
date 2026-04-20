// Smith-Waterman sequence alignment benchmark – Kokkos port
// Implements affine-gap SW alignment for batches of amino-acid sequence pairs.
// Uses the BLOSUM62 scoring matrix and amino-acid encoding from bsw-cuda.
//
// The CUDA source uses warp-shuffle anti-diagonal wavefront parallelism
// (one CUDA block per pair).  Here each Kokkos thread handles one pair
// entirely using a two-row DP formulation, achieving the same O(L_A * L_B)
// work while avoiding warp-shuffle intrinsics.
//
// Usage: ./main <numPairs> <avgLength> <repeat>
// Example: ./main 100 100 100

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <string>

// ---------------------------------------------------------------------------
// BLOSUM62 scoring matrix (24×24) – taken verbatim from bsw-cuda/main.cu
// Amino-acid order: A R N D C Q E G H I L K M F P S T W Y V B Z X *
//                   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
// ---------------------------------------------------------------------------
static const short BLOSUM62[24 * 24] = {
    4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4,
   -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4,
   -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4,
   -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4,
    0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
   -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4,
   -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
    0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4,
   -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4,
   -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4,
   -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4,
   -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4,
   -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4,
   -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4,
   -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
    1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4,
    0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4,
   -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4,
   -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4,
    0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4,
   -2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4,
   -1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
    0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4,
   -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1
};

// ---------------------------------------------------------------------------
// Encoding matrix: ASCII char → BLOSUM62 row/col index (0-based, 0..23).
// Taken from the h_encoding_matrix in bsw-cuda/driver.cu.
// Only indices 42–90 are non-zero (covers '*' and 'A'-'Z').
// ---------------------------------------------------------------------------
static const short AA_ENCODE[128] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  // 0-15
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  // 16-31
    0,0,0,0,0,0,0,0,0,0,               // 32-41
    23,                                 // 42 '*'
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 43-64
    0,                                  // 65 'A' → 0
    20,                                 // 66 'B' → 20
    4,                                  // 67 'C' → 4
    3,                                  // 68 'D' → 3
    6,                                  // 69 'E' → 6
    13,                                 // 70 'F' → 13
    7,                                  // 71 'G' → 7
    8,                                  // 72 'H' → 8
    9,                                  // 73 'I' → 9
    0,                                  // 74 'J'
    11,                                 // 75 'K' → 11
    10,                                 // 76 'L' → 10
    12,                                 // 77 'M' → 12
    2,                                  // 78 'N' → 2
    0,                                  // 79 'O'
    14,                                 // 80 'P' → 14
    5,                                  // 81 'Q' → 5
    1,                                  // 82 'R' → 1
    15,                                 // 83 'S' → 15
    16,                                 // 84 'T' → 16
    0,                                  // 85 'U'
    19,                                 // 86 'V' → 19
    17,                                 // 87 'W' → 17
    22,                                 // 88 'X' → 22
    18,                                 // 89 'Y' → 18
    21,                                 // 90 'Z' → 21
    0                                   // 91+
};

// The 20 standard amino-acid characters used when generating synthetic sequences
static const char AA_CHARS[] = "ACDEFGHIKLMNPQRSTVWY";
static constexpr int NUM_AA   = 20;

// Gap penalties (same as bsw-cuda default: openGap=-6, extendGap=-1 in driver.cu)
static constexpr short GAP_OPEN   = -6;
static constexpr short GAP_EXTEND = -1;

// Maximum sequence length supported without heap allocation
static constexpr int MAX_SEQ_LEN = 512;

// ---------------------------------------------------------------------------
// Smith-Waterman with affine gaps, two-row formulation.
// Returns the alignment score and end position (seqA_end, seqB_end).
// Scratch arrays H_prev/H_curr/F must each be >= lenB+1 shorts.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
short sw_align(const char* seqA, int lenA,
               const char* seqB, int lenB,
               const short* blosum,
               const short* enc,
               short* H_prev,  // [lenB+1]
               short* H_curr,  // [lenB+1]
               short* F_arr,   // [lenB+1]  F gap (gap in B / vertical)
               short& best_i, short& best_j)
{
    short best = 0;
    best_i = 0; best_j = 0;

    for (int j = 0; j <= lenB; ++j) {
        H_prev[j] = 0;
        F_arr[j]  = 0;
    }

    for (int i = 1; i <= lenA; ++i) {
        H_curr[0] = 0;
        short E   = 0;  // E[i][j-1]: gap in A (horizontal)
        const short enc_ai = enc[(int)(unsigned char)seqA[i-1]];

        for (int j = 1; j <= lenB; ++j) {
            // F[i][j] = max(H[i-1][j] + gapOpen, F[i-1][j] + gapExtend)
            const short F = (H_prev[j] + GAP_OPEN > F_arr[j] + GAP_EXTEND)
                            ? H_prev[j] + GAP_OPEN
                            : F_arr[j]  + GAP_EXTEND;
            F_arr[j] = F;

            // E[i][j] = max(H[i][j-1] + gapOpen, E[i][j-1] + gapExtend)
            const short E_new = (H_curr[j-1] + GAP_OPEN > E + GAP_EXTEND)
                                ? H_curr[j-1] + GAP_OPEN
                                : E           + GAP_EXTEND;
            E = E_new;

            // Substitution
            const short enc_bj  = enc[(int)(unsigned char)seqB[j-1]];
            const short sub     = blosum[enc_ai * 24 + enc_bj];
            const short diag    = H_prev[j-1] + sub;

            // H[i][j] = max(0, diag, F, E)
            short h = 0;
            if (diag > h) h = diag;
            if (F    > h) h = F;
            if (E    > h) h = E;
            H_curr[j] = h;

            if (h > best) {
                best   = h;
                best_i = (short)i;
                best_j = (short)j;
            }
        }

        // Swap rows
        for (int j = 0; j <= lenB; ++j) H_prev[j] = H_curr[j];
    }
    return best;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int numPairs  = 100;
    int avgLength = 100;
    int repeat    = 100;

    if (argc >= 2) numPairs  = std::atoi(argv[1]);
    if (argc >= 3) avgLength = std::atoi(argv[2]);
    if (argc >= 4) repeat    = std::atoi(argv[3]);

    // Length range: [avgLength/2, 3*avgLength/2]
    const int minLen = Kokkos::max(1, avgLength / 2);
    const int maxLen = Kokkos::min(MAX_SEQ_LEN, 3 * avgLength / 2);

    Kokkos::initialize(argc, argv);
    {
        // ---------------------------------------------------------------
        // Generate synthetic amino-acid sequences
        // ---------------------------------------------------------------
        Kokkos::View<char**>  seqA_d("seqA", numPairs, maxLen);
        Kokkos::View<char**>  seqB_d("seqB", numPairs, maxLen);
        Kokkos::View<int*>    lenA_d("lenA", numPairs);
        Kokkos::View<int*>    lenB_d("lenB", numPairs);
        Kokkos::View<short*>  scores_d("scores", numPairs);
        Kokkos::View<short*>  endA_d("endA", numPairs);
        Kokkos::View<short*>  endB_d("endB", numPairs);

        {
            auto seqAh = Kokkos::create_mirror_view(seqA_d);
            auto seqBh = Kokkos::create_mirror_view(seqB_d);
            auto lenAh = Kokkos::create_mirror_view(lenA_d);
            auto lenBh = Kokkos::create_mirror_view(lenB_d);

            std::mt19937 rng(42);
            std::uniform_int_distribution<int>  len_dist(minLen, maxLen);
            std::uniform_int_distribution<int>  aa_dist (0, NUM_AA - 1);

            for (int p = 0; p < numPairs; ++p) {
                const int la = len_dist(rng);
                const int lb = len_dist(rng);
                lenAh(p) = la;
                lenBh(p) = lb;
                for (int i = 0; i < la; ++i) seqAh(p, i) = AA_CHARS[aa_dist(rng)];
                for (int j = 0; j < lb; ++j) seqBh(p, j) = AA_CHARS[aa_dist(rng)];
            }

            Kokkos::deep_copy(seqA_d, seqAh);
            Kokkos::deep_copy(seqB_d, seqBh);
            Kokkos::deep_copy(lenA_d, lenAh);
            Kokkos::deep_copy(lenB_d, lenBh);
        }

        // Copy scoring and encoding matrices to device
        Kokkos::View<short*> blosum_d("blosum", 24 * 24);
        Kokkos::View<short*> enc_d   ("enc",    128);
        {
            auto bh = Kokkos::create_mirror_view(blosum_d);
            auto eh = Kokkos::create_mirror_view(enc_d);
            for (int i = 0; i < 24*24; ++i) bh(i) = BLOSUM62[i];
            for (int i = 0; i < 128;   ++i) eh(i) = AA_ENCODE[i];
            Kokkos::deep_copy(blosum_d, bh);
            Kokkos::deep_copy(enc_d,    eh);
        }

        // Scratch storage for DP rows – one set per pair (avoids register spill)
        Kokkos::View<short**> H_prev_d("H_prev", numPairs, MAX_SEQ_LEN + 1);
        Kokkos::View<short**> H_curr_d("H_curr", numPairs, MAX_SEQ_LEN + 1);
        Kokkos::View<short**> F_arr_d ("F_arr",  numPairs, MAX_SEQ_LEN + 1);

        Kokkos::fence();
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int rep = 0; rep < repeat; ++rep) {
            Kokkos::parallel_for("sw_kernel",
                Kokkos::RangePolicy<>(0, numPairs),
                KOKKOS_LAMBDA(int p) {
                    const int la = lenA_d(p);
                    const int lb = lenB_d(p);

                    short* Hp  = &H_prev_d(p, 0);
                    short* Hc  = &H_curr_d(p, 0);
                    short* Fp  = &F_arr_d (p, 0);

                    short ei = 0, ej = 0;
                    const short score = sw_align(
                        &seqA_d(p, 0), la,
                        &seqB_d(p, 0), lb,
                        blosum_d.data(),
                        enc_d.data(),
                        Hp, Hc, Fp,
                        ei, ej);

                    scores_d(p) = score;
                    endA_d(p)   = ei;
                    endB_d(p)   = ej;
                });
        }

        Kokkos::fence();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Check results on host
        auto scores_h = Kokkos::create_mirror_view(scores_d);
        auto endA_h   = Kokkos::create_mirror_view(endA_d);
        auto endB_h   = Kokkos::create_mirror_view(endB_d);
        Kokkos::deep_copy(scores_h, scores_d);
        Kokkos::deep_copy(endA_h,   endA_d);
        Kokkos::deep_copy(endB_h,   endB_d);

        // Print a sample of results
        const int show = (numPairs < 5) ? numPairs : 5;
        printf("BSW Kokkos benchmark\n");
        printf("Pairs: %d, avgLen: %d, repeat: %d\n", numPairs, avgLength, repeat);
        printf("Sample results (pair, score, endA, endB):\n");
        for (int p = 0; p < show; ++p)
            printf("  pair %3d  score=%5d  endA=%4d  endB=%4d\n",
                   p, scores_h(p), endA_h(p), endB_h(p));

        printf("Total time       : %.3f s\n", elapsed);
        printf("Avg per repeat   : %.3f ms\n", elapsed * 1e3 / repeat);
        printf("Throughput       : %.1f GCUPS\n",
               (double)numPairs * avgLength * avgLength * repeat / elapsed / 1e9);
    }
    Kokkos::finalize();
    return 0;
}
