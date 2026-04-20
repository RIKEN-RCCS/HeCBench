/*
HPL (High Performance Linpack) simplified single-node version.
Kokkos port (OpenMP backend).

Solves Ax = b using LU factorization with partial pivoting (DGETRF).

For each pivot column k:
  1. Find pivot row (parallel_reduce for max)
  2. Swap rows k and pivot (parallel_for)
  3. Scale column k below pivot (parallel_for)
  4. Update trailing submatrix (parallel_for over rows)

Usage: ./main [n]   (default n=1024)
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int n = 1024;
    if (argc > 1) n = atoi(argv[1]);

    printf("Matrix size: %d\n", n);

    // Allocate and initialize on host
    std::vector<double> A_init(n * n);
    std::vector<double> b_init(n);
    std::vector<double> x_ref (n, 1.0); // true solution x = 1

    srand(12345);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            A_init[i*n + j] = (double)rand() / RAND_MAX + (i == j ? n : 0.0);
        // b = A * x_ref  (x_ref = all ones)
    }
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) s += A_init[i*n + j]; // * 1
        b_init[i] = s;
    }

    Kokkos::initialize(argc, argv);
    {
        using ViewD2 = Kokkos::View<double**>; // row-major n×n
        using ViewD1 = Kokkos::View<double*>;
        using ViewI1 = Kokkos::View<int*>;

        ViewD2 d_A  ("A",    n, n);
        ViewD1 d_b  ("b",    n);
        ViewI1 d_piv("piv",  n); // pivot row indices

        // Copy data to device
        {
            auto hA = Kokkos::create_mirror_view(d_A);
            auto hb = Kokkos::create_mirror_view(d_b);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) hA(i, j) = A_init[i*n + j];
                hb(i) = b_init[i];
            }
            Kokkos::deep_copy(d_A, hA);
            Kokkos::deep_copy(d_b, hb);
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // LU factorization with partial pivoting
        for (int k = 0; k < n; k++) {

            // Step 1: Find pivot row (max |A[i][k]| for i >= k)
            // With HostSpace (OpenMP) backend, d_A is directly accessible.
            // We do a parallel_reduce to find the maximum absolute value,
            // then a second pass to locate its row index.
            double max_val = 0.0;
            Kokkos::parallel_reduce("findPivotVal", Kokkos::RangePolicy<>(k, n),
                KOKKOS_LAMBDA(int i, double& lmax) {
                    double av = d_A(i, k);
                    if (av < 0.0) av = -av;
                    if (av > lmax) lmax = av;
                },
                Kokkos::Max<double>(max_val));

            // Find the first row with that max value
            int pivot_row = k;
            const double pv = max_val;
            Kokkos::parallel_reduce("findPivotIdx", Kokkos::RangePolicy<>(k, n),
                KOKKOS_LAMBDA(int i, int& best) {
                    double av = d_A(i, k);
                    if (av < 0.0) av = -av;
                    // Track the smallest index among maximizers
                    if (av == pv && i < best) best = i;
                },
                Kokkos::Min<int>(pivot_row));

            // Store pivot index
            const int kk_store = k;
            const int pr_store = pivot_row;
            Kokkos::parallel_for("setPiv", 1, KOKKOS_LAMBDA(int) {
                d_piv(kk_store) = pr_store;
            });

            // Step 2: Swap rows k and pivot_row
            if (pivot_row != k) {
                const int pr = pivot_row;
                Kokkos::parallel_for("swapRows", n, KOKKOS_LAMBDA(int j) {
                    double tmp    = d_A(k,  j);
                    d_A(k,  j)   = d_A(pr, j);
                    d_A(pr, j)   = tmp;
                });
                // Swap b entries
                Kokkos::parallel_for("swapB", 1, KOKKOS_LAMBDA(int) {
                    double tmp = d_b(k); d_b(k) = d_b(pr); d_b(pr) = tmp;
                });
            }

            // Step 3: Scale column k below pivot
            {
                const int kk = k;
                Kokkos::parallel_for("scaleCol", Kokkos::RangePolicy<>(k+1, n),
                    KOKKOS_LAMBDA(int i) {
                        d_A(i, kk) /= d_A(kk, kk);
                    });
            }

            // Step 4: Update trailing submatrix A[i][j] -= A[i][k]*A[k][j]
            //         for i in [k+1,n), j in [k+1,n)
            {
                const int kk = k;
                const int nn = n;
                Kokkos::parallel_for("updateTrailing",
                    Kokkos::RangePolicy<>(k+1, n),
                    KOKKOS_LAMBDA(int i) {
                        double aik = d_A(i, kk);
                        for (int j = kk + 1; j < nn; j++)
                            d_A(i, j) -= aik * d_A(kk, j);
                    });
            }
        } // end LU

        Kokkos::fence();
        auto t_end = std::chrono::high_resolution_clock::now();
        double lu_time = std::chrono::duration<double>(t_end - t_start).count();
        printf("LU factorization time: %.6f (s)\n", lu_time);

        // ---------------------------------------------------------------
        // Forward substitution: Ly = b  (L has implicit unit diagonal)
        // ---------------------------------------------------------------
        ViewD1 d_y("y", n);
        Kokkos::deep_copy(d_y, d_b);

        for (int i = 0; i < n; i++) {
            const int ii = i;
            Kokkos::parallel_for("fwdSub", Kokkos::RangePolicy<>(i+1, n),
                KOKKOS_LAMBDA(int j) {
                    d_y(j) -= d_A(j, ii) * d_y(ii);
                });
        }

        // ---------------------------------------------------------------
        // Backward substitution: Ux = y
        // ---------------------------------------------------------------
        ViewD1 d_x("x", n);
        Kokkos::deep_copy(d_x, d_y);

        for (int i = n-1; i >= 0; i--) {
            // x[i] /= U[i][i]
            const int ii = i;
            Kokkos::parallel_for("backDiag", 1, KOKKOS_LAMBDA(int) {
                d_x(ii) /= d_A(ii, ii);
            });
            // Update x[j] for j < i
            Kokkos::parallel_for("backSub", Kokkos::RangePolicy<>(0, i),
                KOKKOS_LAMBDA(int j) {
                    d_x(j) -= d_A(j, ii) * d_x(ii);
                });
        }

        // ---------------------------------------------------------------
        // Compute residual ||Ax - b|| / (||A|| * ||x|| * n)
        // ---------------------------------------------------------------
        // Recompute Ax using original A (A_init)
        // Copy x back to host
        auto h_x = Kokkos::create_mirror_view(d_x);
        Kokkos::deep_copy(h_x, d_x);

        double norm_r = 0.0, norm_A = 0.0, norm_x = 0.0;
        for (int i = 0; i < n; i++) {
            double ax_i = 0.0;
            for (int j = 0; j < n; j++) ax_i += A_init[i*n + j] * h_x(j);
            double ri = ax_i - b_init[i];
            norm_r += ri * ri;
        }
        norm_r = sqrt(norm_r);

        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) row_sum += fabs(A_init[i*n + j]);
            norm_A = std::max(norm_A, row_sum);
            norm_x += h_x(i) * h_x(i);
        }
        norm_x = sqrt(norm_x);

        double residual = norm_r / (norm_A * norm_x * n);
        printf("Residual ||Ax-b||/||A||/||x||/N: %e\n", residual);
        printf("%s\n", residual < 1e-8 ? "PASSED" : "FAILED");
    }
    Kokkos::finalize();
    return 0;
}
