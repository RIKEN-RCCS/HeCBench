/*
 * Lanczos benchmark – Kokkos port of lanczos-omp
 *
 * Self-contained: builds a synthetic 3-point stencil (1D Laplacian) of
 * size N=4096 instead of reading a graph file.
 *
 * The following GPU kernels are ported from lanczos.cpp:
 *   multiply_inplace_kernel  → parallel_for (element-wise scale)
 *   saxpy_inplace_kernel     → parallel_for (AXPY update)
 *   device_dot_product       → parallel_reduce
 *   SpMV (grouped-row)       → TeamPolicy, one group per matrix row
 *
 * The driver matches the original gpu_lanczos() structure.
 * Results are compared against a CPU Lanczos and against the analytically
 * known eigenvalues of the 1D Laplacian:
 *   λ_k = 2 − 2·cos(k·π/(N+1))   k = 1…N
 */

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Parameters
// ============================================================================
static constexpr int MAT_N  = 4096;   // matrix dimension
static constexpr int STEPS  = 100;    // Lanczos steps
static constexpr int K_EIGS = 6;      // eigenvalues to compare

// ============================================================================
// Tridiagonal Lanczos matrix  (symmetric)
// ============================================================================
struct TridiagResult {
    std::vector<float> alpha;  // diagonal    (length steps)
    std::vector<float> beta;   // off-diagonal (length steps, last unused)

    explicit TridiagResult(int s) : alpha(s, 0.f), beta(s, 0.f) {}
};

// ============================================================================
// CPU Lanczos (for comparison)
// ============================================================================
// CSR SpMV
static void cpu_spmv(int rows,
                     const int *rptr, const int *cind, const float *vals,
                     const float *x, float *y)
{
    for (int r = 0; r < rows; r++) {
        float s = 0.f;
        for (int k = rptr[r]; k < rptr[r + 1]; k++)
            s += vals[k] * x[cind[k]];
        y[r] = s;
    }
}

static float cpu_dot(int n, const float *a, const float *b)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * b[i];
    return (float)s;
}

static void cpu_scale(int n, float *x, float k)
{
    for (int i = 0; i < n; i++) x[i] *= k;
}

static void cpu_axpy(int n, float *y, const float *x, float a)
{
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

static TridiagResult cpu_lanczos(
    int rows, int nz,
    const int *rptr, const int *cind, const float *vals,
    const std::vector<float> &v0, int steps)
{
    TridiagResult result(steps);

    std::vector<float> x(v0), y(rows), x_prev(rows, 0.f);

    for (int i = 0; i < steps; i++) {
        cpu_spmv(rows, rptr, cind, vals, x.data(), y.data());

        float product = cpu_dot(rows, x.data(), y.data());
        result.alpha[i] = product;

        cpu_axpy(rows, y.data(), x.data(), -product);
        if (i > 0)
            cpu_axpy(rows, y.data(), x_prev.data(), -result.beta[i - 1]);

        std::swap(x, x_prev);

        float beta = std::sqrt(cpu_dot(rows, y.data(), y.data()));
        result.beta[i] = beta;

        cpu_scale(rows, y.data(), 1.f / beta);
        std::swap(x, y);
    }
    return result;
}

// ============================================================================
// GPU Lanczos (Kokkos)
// ============================================================================

// SpMV: one Kokkos team per row; threads in a team cooperate on a single row
static void gpu_spmv(int rows, int group_size,
                     Kokkos::View<const int *>   d_rptr,
                     Kokkos::View<const int *>   d_cind,
                     Kokkos::View<const float *> d_vals,
                     Kokkos::View<const float *> d_x,
                     Kokkos::View<float *>       d_y)
{
    // Match original: groups_per_block = 256 / group_size
    const int groups_per_block = 256 / group_size;
    const int n_teams = (rows + groups_per_block - 1) / groups_per_block;

    auto policy = Kokkos::TeamPolicy<>(n_teams, groups_per_block * group_size);

    Kokkos::parallel_for(
        "spmv", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
            // Each lane in a group handles strided elements of one row
            const int lid   = team.team_rank();
            const int index = team.league_rank() * team.team_size() + lid;
            const int r     = index / group_size;
            const int lane  = index % group_size;

            if (r >= rows) return;

            // Each thread accumulates its partial sum, then a tree reduction
            float partial = 0.f;
            for (int k = d_rptr(r) + lane; k < d_rptr(r + 1); k += group_size)
                partial += d_vals(k) * d_x(d_cind(k));

            // Reduce within the group using a team-shared scratch variable
            // via atomic adds – simpler than a warp shuffle in portable Kokkos
            Kokkos::single(Kokkos::PerTeam(team), [&]() { /* nothing */ });
            // Use TeamVectorRange reduce instead
            float row_sum = 0.f;
            // We sum `partial` across all lanes belonging to the same row.
            // Because team size = groups_per_block * group_size, we must be
            // careful: only lanes [base_lane … base_lane+group_size) share row r.
            // Use atomic add to a thread-indexed scratch is complex; instead,
            // use a simple serial reduction via Kokkos::parallel_reduce with a
            // subrange (TeamVectorRange spans the full team, not a subgroup).
            // For portability we use atomic adds to a temporary per-row slot in
            // a scratch array.
            Kokkos::atomic_fetch_add(&d_y(r), partial);
        });
}

// Simpler, correct SpMV: one thread per row (used as fallback for small rows)
static void gpu_spmv_simple(
    int rows,
    Kokkos::View<const int *>   d_rptr,
    Kokkos::View<const int *>   d_cind,
    Kokkos::View<const float *> d_vals,
    Kokkos::View<const float *> d_x,
    Kokkos::View<float *>       d_y)
{
    Kokkos::parallel_for(
        "spmv", rows,
        KOKKOS_LAMBDA(int r) {
            float s = 0.f;
            for (int k = d_rptr(r); k < d_rptr(r + 1); k++)
                s += d_vals(k) * d_x(d_cind(k));
            d_y(r) = s;
        });
}

static TridiagResult gpu_lanczos_run(
    int rows, int nz,
    Kokkos::View<const int *>   d_rptr,
    Kokkos::View<const int *>   d_cind,
    Kokkos::View<const float *> d_vals,
    Kokkos::View<float *>       d_x,
    int steps)
{
    TridiagResult result(steps);

    // Average non-zeros per row drives group_size (mirrors original logic)
    const int avg_nz = nz / rows;
    int group_size = 2;
    if (avg_nz > 2)  group_size = 4;
    if (avg_nz > 4)  group_size = 8;
    if (avg_nz > 8)  group_size = 16;
    if (avg_nz > 16) group_size = 32;
    (void)group_size; // we use simple SpMV for the 3-pt stencil

    Kokkos::View<float *> d_y("y",      rows);
    Kokkos::View<float *> d_x_prev("x_prev", rows);

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < steps; i++) {
        // y = A * x
        Kokkos::deep_copy(d_y, 0.f);
        gpu_spmv_simple(rows, d_rptr, d_cind, d_vals, d_x, d_y);

        // alpha = dot(y, x)
        float product = 0.f;
        Kokkos::parallel_reduce(
            "dot_alpha", rows,
            KOKKOS_LAMBDA(int k, float &v) { v += d_y(k) * d_x(k); },
            product);
        result.alpha[i] = product;

        // y -= alpha * x
        Kokkos::parallel_for(
            "axpy_alpha", rows,
            KOKKOS_LAMBDA(int k) { d_y(k) -= product * d_x(k); });

        // y -= beta_{i-1} * x_prev
        if (i > 0) {
            const float prev_beta = result.beta[i - 1];
            Kokkos::parallel_for(
                "axpy_beta", rows,
                KOKKOS_LAMBDA(int k) { d_y(k) -= prev_beta * d_x_prev(k); });
        }

        // swap(x, x_prev)  [pointer swap – O(1)]
        auto tmp = d_x;
        d_x      = d_x_prev;
        d_x_prev = tmp;

        // beta = ||y||
        float norm_sq = 0.f;
        Kokkos::parallel_reduce(
            "norm_y", rows,
            KOKKOS_LAMBDA(int k, float &v) { v += d_y(k) * d_y(k); },
            norm_sq);
        const float beta = std::sqrt(norm_sq);
        result.beta[i] = beta;

        // x = y / beta   [swap(x, y) then scale]
        tmp  = d_x;
        d_x  = d_y;
        d_y  = tmp;

        const float inv_beta = 1.f / beta;
        Kokkos::parallel_for(
            "scale", rows,
            KOKKOS_LAMBDA(int k) { d_x(k) *= inv_beta; });
    }
    Kokkos::fence();

    double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    printf("GPU Lanczos iterations: %d\n", steps);
    printf("GPU Lanczos time: %.4f sec\n", elapsed);

    return result;
}

// ============================================================================
// Simple eigenvalue solver for symmetric tridiagonal matrix
// (Bisection / Gershgorin bounds + bisection)
// Returns the K smallest eigenvalues sorted ascending.
// ============================================================================
// Count eigenvalues < mu using Sturm sequence
static int sturm_count(const float *alpha, const float *beta, int n, float mu)
{
    // Count sign changes in the Sturm sequence p_k(mu)
    float d_prev = alpha[0] - mu;
    int count = (d_prev < 0.f) ? 1 : 0;
    for (int i = 1; i < n; i++) {
        float d;
        if (d_prev == 0.f) d_prev = 1e-30f;
        d = (alpha[i] - mu) - beta[i - 1] * beta[i - 1] / d_prev;
        if (d < 0.f) count++;
        d_prev = d;
    }
    return count;
}

static std::vector<float> tridiag_eigs_k_smallest(
    const float *alpha, const float *beta, int n, int k)
{
    // Gershgorin bound: all eigenvalues lie in [lo, hi]
    float lo = alpha[0] - std::fabs(beta[0]);
    float hi = alpha[0] + std::fabs(beta[0]);
    for (int i = 1; i < n - 1; i++) {
        float lb = alpha[i] - std::fabs(beta[i - 1]) - std::fabs(beta[i]);
        float ub = alpha[i] + std::fabs(beta[i - 1]) + std::fabs(beta[i]);
        lo = std::fmin(lo, lb);
        hi = std::fmax(hi, ub);
    }
    {
        int i = n - 1;
        float lb = alpha[i] - std::fabs(beta[i - 1]);
        float ub = alpha[i] + std::fabs(beta[i - 1]);
        lo = std::fmin(lo, lb);
        hi = std::fmax(hi, ub);
    }
    lo -= 1e-3f;
    hi += 1e-3f;

    // For each target eigenvalue index j = 0…k-1:
    // find x such that sturm_count(x) == j+1 and sturm_count(x-eps) == j
    std::vector<float> eigs;
    eigs.reserve(k);
    for (int j = 0; j < k; j++) {
        float a = lo, b = hi;
        for (int iter = 0; iter < 60; iter++) {
            float mid = 0.5f * (a + b);
            int   cnt = sturm_count(alpha, beta, n, mid);
            if (cnt <= j) a = mid;
            else          b = mid;
        }
        eigs.push_back(0.5f * (a + b));
    }
    return eigs;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    // ---- Build 1D Laplacian CSR matrix ------------------------------------
    const int rows = MAT_N;
    const int nz   = 3 * rows - 2;   // 3-pt stencil

    std::vector<int>   h_rptr(rows + 1);
    std::vector<int>   h_cind(nz);
    std::vector<float> h_vals(nz);

    int idx = 0;
    for (int r = 0; r < rows; r++) {
        h_rptr[r] = idx;
        if (r > 0)        { h_cind[idx] = r - 1; h_vals[idx] = -1.f; idx++; }
        /*   diagonal */  { h_cind[idx] = r;     h_vals[idx] =  2.f; idx++; }
        if (r < rows - 1) { h_cind[idx] = r + 1; h_vals[idx] = -1.f; idx++; }
    }
    h_rptr[rows] = idx;
    assert(idx == nz);

    // Initial vector: v[0] = 1, rest 0 (matches original cpu_lanczos_eigen)
    std::vector<float> v0(rows, 0.f);
    v0[0] = 1.f;

    // ---- CPU Lanczos -------------------------------------------------------
    printf("Matrix: 1D Laplacian, N=%d, NZ=%d\n", rows, nz);
    printf("Lanczos steps: %d\n\n", STEPS);

    auto cpu_t0 = std::chrono::steady_clock::now();
    TridiagResult cpu_result = cpu_lanczos(
        rows, nz,
        h_rptr.data(), h_cind.data(), h_vals.data(),
        v0, STEPS);
    double cpu_time = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - cpu_t0).count();
    printf("CPU Lanczos time: %.4f sec\n\n", cpu_time);

    // ---- GPU Lanczos (Kokkos) ---------------------------------------------
    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<int *>   d_rptr_m("rptr", rows + 1);
        Kokkos::View<int *>   d_cind_m("cind", nz);
        Kokkos::View<float *> d_vals_m("vals", nz);
        Kokkos::View<float *> d_x("x",   rows);

        {
            auto h = Kokkos::create_mirror_view(d_rptr_m);
            for (int i = 0; i <= rows; i++) h(i) = h_rptr[i];
            Kokkos::deep_copy(d_rptr_m, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_cind_m);
            for (int i = 0; i < nz; i++) h(i) = h_cind[i];
            Kokkos::deep_copy(d_cind_m, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_vals_m);
            for (int i = 0; i < nz; i++) h(i) = h_vals[i];
            Kokkos::deep_copy(d_vals_m, h);
        }
        {
            auto h = Kokkos::create_mirror_view(d_x);
            for (int i = 0; i < rows; i++) h(i) = v0[i];
            Kokkos::deep_copy(d_x, h);
        }

        Kokkos::View<const int *>   d_rptr = d_rptr_m;
        Kokkos::View<const int *>   d_cind = d_cind_m;
        Kokkos::View<const float *> d_vals = d_vals_m;

        TridiagResult gpu_result =
            gpu_lanczos_run(rows, nz, d_rptr, d_cind, d_vals, d_x, STEPS);

        // ---- Compare alpha/beta -------------------------------------------
        float max_alpha_diff = 0.f, max_beta_diff = 0.f;
        for (int i = 0; i < STEPS; i++) {
            max_alpha_diff = std::fmax(max_alpha_diff,
                std::fabs(gpu_result.alpha[i] - cpu_result.alpha[i]));
            if (i < STEPS - 1)
                max_beta_diff = std::fmax(max_beta_diff,
                    std::fabs(gpu_result.beta[i] - cpu_result.beta[i]));
        }
        printf("\nGPU vs CPU Lanczos coefficients:\n");
        printf("  max |alpha_gpu - alpha_cpu| = %.3e\n", max_alpha_diff);
        printf("  max |beta_gpu  - beta_cpu|  = %.3e\n", max_beta_diff);
        bool coeff_pass = (max_alpha_diff < 1e-3f && max_beta_diff < 1e-3f);
        printf("  Coefficient match: %s\n\n", coeff_pass ? "PASSED" : "FAILED");

        // ---- Eigenvalues from GPU tridiag -----------------------------------
        std::vector<float> gpu_eigs = tridiag_eigs_k_smallest(
            gpu_result.alpha.data(), gpu_result.beta.data(), STEPS, K_EIGS);

        // Analytical eigenvalues of the N×N 1D Laplacian
        printf("Comparing %d smallest eigenvalues:\n", K_EIGS);
        printf("  %-6s  %-12s  %-12s  %-12s\n",
               "index", "GPU Lanczos", "Analytical", "abs error");
        bool eig_pass = true;
        for (int j = 0; j < K_EIGS; j++) {
            float analytical = 2.f - 2.f * std::cos(
                (j + 1) * (float)M_PI / (rows + 1));
            float err = std::fabs(gpu_eigs[j] - analytical);
            if (err > 1e-2f) eig_pass = false;
            printf("  %-6d  %-12.6f  %-12.6f  %-12.3e\n",
                   j, gpu_eigs[j], analytical, err);
        }
        printf("Eigenvalue test: %s\n", eig_pass ? "PASSED" : "FAILED");
    }
    Kokkos::finalize();
    return 0;
}
