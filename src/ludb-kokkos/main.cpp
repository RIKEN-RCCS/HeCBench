// LU Decomposition Batched - Kokkos port
// Computes LU factorization with partial pivoting for BATCH_SIZE matrices of size NxN
// Column-major storage: mat[j*N + i] = element at row i, column j

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define N          48
#define BATCH_SIZE 10000

// ---- CPU verification helpers -----------------------------------------------

static void getLUdecoded(const float* mat, float* L, float* U) {
    memset(L, 0, N * N * sizeof(float));
    memset(U, 0, N * N * sizeof(float));
    // Unit diagonal for L
    for (int i = 0; i < N; i++) L[i * N + i] = 1.0f;
    // Below-diagonal elements come from factored matrix
    for (int i = 0; i < N; i++)
        for (int j = 0; j < i; j++) L[j * N + i] = mat[j * N + i];
    // Upper-triangular elements
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++) U[j * N + i] = mat[j * N + i];
}

static void matrixMultiply(float* res, const float* mat1, const float* mat2) {
    memset(res, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                res[j * N + i] += mat1[k * N + i] * mat2[j * N + k];
}

// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int repeat = (argc > 1) ? atoi(argv[1]) : 1;

        // Allocate host matrices and fill with random values.
        // Add diagonal dominance so LU is numerically well-conditioned.
        float* orig_mats = new float[(size_t)BATCH_SIZE * N * N];
        srand(42);
        for (int b = 0; b < BATCH_SIZE; b++) {
            float* mat = orig_mats + (size_t)b * N * N;
            for (int j = 0; j < N; j++)
                for (int i = 0; i < N; i++)
                    mat[j * N + i] = (float)rand() / RAND_MAX;
            // Make diagonally dominant (column-major diagonal: mat[i*N+i])
            for (int i = 0; i < N; i++) mat[i * N + i] += N;
        }

        Kokkos::View<float*> d_mats("d_mats",   (size_t)BATCH_SIZE * N * N);
        Kokkos::View<int*>   d_pivots("d_pivots", (size_t)BATCH_SIZE * N);

        // Use an explicit HostSpace view for the original data so that
        // deep_copy always transfers host→device regardless of backend.
        Kokkos::View<float*, Kokkos::HostSpace> h_orig("h_orig",
                                                        (size_t)BATCH_SIZE * N * N);
        memcpy(h_orig.data(), orig_mats, (size_t)BATCH_SIZE * N * N * sizeof(float));

        double total_time = 0.0;

        for (int r = 0; r < repeat; r++) {
            // Reset working matrices to original values for each repeat
            Kokkos::deep_copy(d_mats, h_orig);
            Kokkos::fence();

            Kokkos::Timer timer;

            Kokkos::parallel_for(
                "lu_batch", BATCH_SIZE,
                KOKKOS_LAMBDA(int b) {
                    const int base = b * N * N;
                    const int pb   = b * N;

                    for (int k = 0; k < N; k++) {
                        // Find pivot row (maximum absolute value in column k below row k)
                        int   piv    = k;
                        float maxval = fabsf(d_mats(base + k * N + k));
                        for (int i = k + 1; i < N; i++) {
                            float v = fabsf(d_mats(base + k * N + i));
                            if (v > maxval) { maxval = v; piv = i; }
                        }
                        d_pivots(pb + k) = piv + 1; // store 1-based pivot index

                        // Swap rows k and piv across all columns
                        if (piv != k) {
                            for (int j = 0; j < N; j++) {
                                float tmp             = d_mats(base + j * N + k);
                                d_mats(base + j * N + k)   = d_mats(base + j * N + piv);
                                d_mats(base + j * N + piv) = tmp;
                            }
                        }

                        // Gaussian elimination: scale subcolumn, then update submatrix
                        const float diag = d_mats(base + k * N + k);
                        if (diag != 0.0f) {
                            const float inv = 1.0f / diag;
                            for (int i = k + 1; i < N; i++)
                                d_mats(base + k * N + i) *= inv;
                            for (int j = k + 1; j < N; j++)
                                for (int i = k + 1; i < N; i++)
                                    d_mats(base + j * N + i) -=
                                        d_mats(base + j * N + k) * d_mats(base + k * N + i);
                        }
                    }
                });
            Kokkos::fence();
            total_time += timer.seconds();
        }

        const double avg_ms = total_time * 1000.0 / repeat;
        // ~(2/3)*N^3 flops per matrix for LU
        const double gflops = (double)BATCH_SIZE * (2.0 / 3.0 * N * N * N) * repeat
                              / total_time * 1e-9;
        printf("LU batch (%d x %dx%d), repeat=%d\n", BATCH_SIZE, N, N, repeat);
        printf("Average kernel time : %.3f ms\n", avg_ms);
        printf("Throughput          : %.2f GFlops\n", gflops);

        // ---- Verification: check the first matrix in the batch ---------------
        Kokkos::View<float*, Kokkos::HostSpace> h_mats(
            "h_mats_out", (size_t)BATCH_SIZE * N * N);
        Kokkos::View<int*, Kokkos::HostSpace> h_pivots(
            "h_pivots_out", (size_t)BATCH_SIZE * N);
        Kokkos::deep_copy(h_mats,   d_mats);
        Kokkos::deep_copy(h_pivots, d_pivots);
        {
            const float* lu_mat = h_mats.data();          // factored matrix
            const int*   pivot  = h_pivots.data();         // pivot indices (1-based)
            const float* A      = orig_mats;               // original matrix

            float L[N * N], U[N * N], LU[N * N], PA[N * N];
            getLUdecoded(lu_mat, L, U);
            matrixMultiply(LU, L, U);

            // Build PA = P*A by applying the same row swaps to the original matrix
            memcpy(PA, A, N * N * sizeof(float));
            for (int k = 0; k < N; k++) {
                const int pk = pivot[k] - 1; // 0-based
                if (pk != k) {
                    for (int j = 0; j < N; j++) {
                        float tmp      = PA[j * N + k];
                        PA[j * N + k]  = PA[j * N + pk];
                        PA[j * N + pk] = tmp;
                    }
                }
            }

            float maxerr = 0.0f;
            for (int idx = 0; idx < N * N; idx++)
                maxerr = fmaxf(maxerr, fabsf(LU[idx] - PA[idx]));
            printf("Verification (matrix 0): max|LU - PA| = %e  %s\n",
                   maxerr, maxerr < 1e-3f ? "PASS" : "FAIL");
        }

        delete[] orig_mats;
    }
    Kokkos::finalize();
    return 0;
}
