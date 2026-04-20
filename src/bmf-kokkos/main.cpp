// Binary Matrix Factorization (BMF) benchmark – Kokkos port
// Implements simulated-annealing BMF: find low-rank boolean matrices A (m×d)
// and B (d×n) such that A*B (Boolean product) approximates C.
//
// Rows of A and columns of B are stored as d-bit integers (uint32_t).
// C is stored column-major in 32-row chunks: C[(i/32)*padded_n + j] bit (i%32).
// Matches the bit_vector_kernels.cuh approach from the CUDA source.

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
static constexpr int   M          = 1000;  // matrix rows
static constexpr int   N          = 1000;  // matrix columns
static constexpr int   D          = 20;    // factorisation rank (bits per vector)
static constexpr int   MAX_ITER   = 100;   // SA iterations
static constexpr float TEMP_START = 1.0f;
static constexpr float TEMP_END   = 0.01f;
static constexpr float TEMP_DECAY = 0.9f;

// Pad columns to next multiple of 32 for the bit-packed C layout
static constexpr int PADDED_N = ((N + 31) / 32) * 32;
// Number of 32-row groups
static constexpr int ROW_GROUPS = (M + 31) / 32;
// Total uint32_t entries in packed C: ROW_GROUPS * N
static constexpr int C_FLAT = ROW_GROUPS * N;

// ---------------------------------------------------------------------------
// Minimal xorshift32 PRNG usable on device
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
uint32_t xorshift32(uint32_t s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

// ---------------------------------------------------------------------------
// Helpers to access the packed C matrix
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
int get_C(const Kokkos::View<uint32_t*>& C_d, int i, int j)
{
    return (int)((C_d((i / 32) * N + j) >> (i % 32)) & 1u);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // ---------------------------------------------------------------
        // Allocate device views
        // ---------------------------------------------------------------
        Kokkos::View<uint32_t*> A_d("A", M);      // m rows as D-bit vectors
        Kokkos::View<uint32_t*> B_d("B", N);      // n cols as D-bit vectors
        Kokkos::View<uint32_t*> C_d("C", C_FLAT); // packed boolean matrix

        const uint32_t dmask = (D < 32) ? ((1u << D) - 1u) : ~0u;

        // ---------------------------------------------------------------
        // Initialise A, B, C on host then copy
        // ---------------------------------------------------------------
        {
            auto Ah = Kokkos::create_mirror_view(A_d);
            auto Bh = Kokkos::create_mirror_view(B_d);
            auto Ch = Kokkos::create_mirror_view(C_d);

            std::mt19937 rng(12345);
            std::uniform_int_distribution<uint32_t> udist(0, ~0u);

            // Random D-bit vectors for A and B
            for (int i = 0; i < M; ++i) Ah(i) = udist(rng) & dmask;
            for (int j = 0; j < N; ++j) Bh(j) = udist(rng) & dmask;

            // Ground-truth C: product of random A0 and B0 (noiseless BMF target)
            std::fill(Ch.data(), Ch.data() + C_FLAT, 0u);
            for (int i = 0; i < M; ++i) {
                uint32_t Ai = Ah(i);
                for (int j = 0; j < N; ++j) {
                    if (Ai & Bh(j)) // Boolean inner product != 0
                        Ch((i / 32) * N + j) |= (1u << (i % 32));
                }
            }
            // Re-randomise A and B as starting point for SA
            for (int i = 0; i < M; ++i) Ah(i) = udist(rng) & dmask;
            for (int j = 0; j < N; ++j) Bh(j) = udist(rng) & dmask;

            Kokkos::deep_copy(A_d, Ah);
            Kokkos::deep_copy(B_d, Bh);
            Kokkos::deep_copy(C_d, Ch);
        }

        // ---------------------------------------------------------------
        // Compute initial error
        // ---------------------------------------------------------------
        auto compute_error = [&]() -> int {
            int err = 0;
            Kokkos::parallel_reduce("error",
                Kokkos::RangePolicy<>(0, M),
                KOKKOS_LAMBDA(int i, int& local) {
                    const uint32_t Ai = A_d(i);
                    for (int j = 0; j < N; ++j) {
                        const int prod  = (Ai & B_d(j)) ? 1 : 0;
                        const int C_ij  = get_C(C_d, i, j);
                        local += (prod != C_ij) ? 1 : 0;
                    }
                }, err);
            return err;
        };

        const int init_error = compute_error();
        printf("Initial error: %d / %d  (%.2f%%)\n",
               init_error, M * N, 100.0 * init_error / (M * N));

        // ---------------------------------------------------------------
        // Simulated annealing loop
        // ---------------------------------------------------------------
        Kokkos::fence();
        auto t0 = std::chrono::high_resolution_clock::now();

        float temp = TEMP_START;
        for (int iter = 0; iter < MAX_ITER && temp > TEMP_END; ++iter) {
            // Seeds change each iteration so threads get different random choices
            const uint32_t seed_A = xorshift32((uint32_t)(iter * 6364136223846793005u + 1));
            const uint32_t seed_B = xorshift32(seed_A + 0xDEADBEEFu);

            // ----------------------------------------------------------
            // Update rows of A in parallel
            // Each thread i independently tries flipping one bit of A[i].
            // B is read-only; different threads write different A[i], no race.
            // ----------------------------------------------------------
            Kokkos::parallel_for("updateA",
                Kokkos::RangePolicy<>(0, M),
                KOKKOS_LAMBDA(int i) {
                    uint32_t rng = xorshift32(seed_A ^ (uint32_t)(i + 1));
                    rng = xorshift32(rng);
                    const int bit = (int)(rng % (uint32_t)D);

                    const uint32_t Ai     = A_d(i);
                    const uint32_t Ai_new = Ai ^ (1u << bit);

                    int delta = 0;
                    for (int j = 0; j < N; ++j) {
                        const int old_prod = (Ai     & B_d(j)) ? 1 : 0;
                        const int new_prod = (Ai_new & B_d(j)) ? 1 : 0;
                        if (old_prod == new_prod) continue;
                        const int C_ij = get_C(C_d, i, j);
                        delta += (new_prod != C_ij) - (old_prod != C_ij);
                    }

                    bool accept = (delta <= 0);
                    if (!accept) {
                        rng = xorshift32(rng ^ (uint32_t)delta);
                        const float p = Kokkos::exp(-(float)delta / temp);
                        const float r = (float)(rng >> 8) / (float)(1u << 24);
                        accept = (r < p);
                    }
                    if (accept) A_d(i) = Ai_new;
                });

            // ----------------------------------------------------------
            // Update columns of B in parallel
            // A is read-only in this step; different threads write different B[j].
            // ----------------------------------------------------------
            Kokkos::parallel_for("updateB",
                Kokkos::RangePolicy<>(0, N),
                KOKKOS_LAMBDA(int j) {
                    uint32_t rng = xorshift32(seed_B ^ (uint32_t)(j + 1));
                    rng = xorshift32(rng);
                    const int bit = (int)(rng % (uint32_t)D);

                    const uint32_t Bj     = B_d(j);
                    const uint32_t Bj_new = Bj ^ (1u << bit);

                    int delta = 0;
                    for (int i = 0; i < M; ++i) {
                        const int old_prod = (A_d(i) & Bj    ) ? 1 : 0;
                        const int new_prod = (A_d(i) & Bj_new) ? 1 : 0;
                        if (old_prod == new_prod) continue;
                        const int C_ij = get_C(C_d, i, j);
                        delta += (new_prod != C_ij) - (old_prod != C_ij);
                    }

                    bool accept = (delta <= 0);
                    if (!accept) {
                        rng = xorshift32(rng ^ (uint32_t)delta);
                        const float p = Kokkos::exp(-(float)delta / temp);
                        const float r = (float)(rng >> 8) / (float)(1u << 24);
                        accept = (r < p);
                    }
                    if (accept) B_d(j) = Bj_new;
                });

            temp *= TEMP_DECAY;
        }

        Kokkos::fence();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        const int final_error = compute_error();
        printf("BMF Kokkos benchmark\n");
        printf("Matrix %dx%d, rank %d, SA iterations %d\n", M, N, D, MAX_ITER);
        printf("Final error  : %d / %d  (%.2f%%)\n",
               final_error, M * N, 100.0 * final_error / (M * N));
        printf("Total time   : %.3f s\n", elapsed);
        printf("Avg per iter : %.3f ms\n", elapsed * 1e3 / MAX_ITER);
    }
    Kokkos::finalize();
    return 0;
}
