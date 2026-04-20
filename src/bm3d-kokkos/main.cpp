// BM3D image denoising benchmark – Kokkos port
// Implements the core compute kernels: 2D DCT, Fast Walsh-Hadamard Transform,
// hard-thresholding, 2D IDCT, and aggregation.
// No image I/O dependency; uses a synthetic 256×256 noisy image.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <random>

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
static constexpr int REPEAT       = 100;
static constexpr int IMG_W        = 256;
static constexpr int IMG_H        = 256;
static constexpr int K            = 8;    // patch side length
static constexpr int N_STACK      = 16;   // stack depth (must be power-of-2)
static constexpr float SIGMA      = 25.0f;
static constexpr float L3D        = 2.7f; // threshold factor (same as CUDA params)
static constexpr int PATCH_STRIDE = 8;    // spacing between reference patches

// ---------------------------------------------------------------------------
// DCT-8 butterfly constants  (sqrt(2)*cos(k*pi/16), from dct8x8.cu)
// ---------------------------------------------------------------------------
static constexpr float C_a    = 1.387039845322148f;
static constexpr float C_b    = 1.306562964876377f;
static constexpr float C_c    = 1.175875602419359f;
static constexpr float C_d    = 0.785694958387102f;
static constexpr float C_e    = 0.541196100146197f;
static constexpr float C_f    = 0.275899379282943f;
static constexpr float C_norm = 0.3535533905932737f; // 1/sqrt(8)

// ---------------------------------------------------------------------------
// In-place forward DCT of 8 elements accessed with stride `step`.
// Matches InplaceDCTvector() from dct8x8.cu exactly.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void dct8(float* v, int step)
{
    float v0=v[0],       v1=v[step],   v2=v[2*step], v3=v[3*step];
    float v4=v[4*step],  v5=v[5*step], v6=v[6*step], v7=v[7*step];

    float X07P = v0+v7,  X16P = v1+v6,  X25P = v2+v5,  X34P = v3+v4;
    float X07M = v0-v7,  X61M = v6-v1,  X25M = v2-v5,  X43M = v4-v3;

    float X07P34PP = X07P+X34P,  X07P34PM = X07P-X34P;
    float X16P25PP = X16P+X25P,  X16P25PM = X16P-X25P;

    v[0]      = C_norm*(X07P34PP + X16P25PP);
    v[2*step] = C_norm*(C_b*X07P34PM + C_e*X16P25PM);
    v[4*step] = C_norm*(X07P34PP - X16P25PP);
    v[6*step] = C_norm*(C_e*X07P34PM - C_b*X16P25PM);
    v[step]   = C_norm*(C_a*X07M - C_c*X61M + C_d*X25M - C_f*X43M);
    v[3*step] = C_norm*(C_c*X07M + C_f*X61M - C_a*X25M + C_d*X43M);
    v[5*step] = C_norm*(C_d*X07M + C_a*X61M + C_f*X25M - C_c*X43M);
    v[7*step] = C_norm*(C_f*X07M + C_d*X61M + C_c*X25M + C_a*X43M);
}

// ---------------------------------------------------------------------------
// In-place inverse DCT of 8 elements. Matches InplaceIDCTvector() from dct8x8.cu.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void idct8(float* v, int step)
{
    float v0=v[0],       v1=v[step],   v2=v[2*step], v3=v[3*step];
    float v4=v[4*step],  v5=v[5*step], v6=v[6*step], v7=v[7*step];

    float Y04P       = v0+v4;
    float Y2b6eP     = C_b*v2 + C_e*v6;
    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;

    float Y7f1aP3c5dPP = C_f*v7 + C_a*v1 + C_c*v3 + C_d*v5;
    float Y7a1fM3d5cMP = C_a*v7 - C_f*v1 + C_d*v3 - C_c*v5;

    float Y04M       = v0-v4;
    float Y2e6bM     = C_e*v2 - C_b*v6;
    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;

    float Y1c7dM3f5aPM = C_c*v1 - C_d*v7 - C_f*v3 - C_a*v5;
    float Y1d7cP3a5fMM = C_d*v1 + C_c*v7 - C_a*v3 + C_f*v5;

    v[0]      = C_norm*(Y04P2b6ePP + Y7f1aP3c5dPP);
    v[7*step] = C_norm*(Y04P2b6ePP - Y7f1aP3c5dPP);
    v[4*step] = C_norm*(Y04P2b6ePM + Y7a1fM3d5cMP);
    v[3*step] = C_norm*(Y04P2b6ePM - Y7a1fM3d5cMP);
    v[step]   = C_norm*(Y04M2e6bMP + Y1c7dM3f5aPM);
    v[5*step] = C_norm*(Y04M2e6bMM - Y1d7cP3a5fMM);
    v[2*step] = C_norm*(Y04M2e6bMM + Y1d7cP3a5fMM);
    v[6*step] = C_norm*(Y04M2e6bMP - Y1c7dM3f5aPM);
}

// ---------------------------------------------------------------------------
// In-place Fast Walsh-Hadamard Transform (power-of-2 length).
// The inverse is the same operation divided by n.
// Matches fwht() from filtering.cu.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void fwht(float* data, int n)
{
    for (int step = 1; step < n; step <<= 1) {
        for (int i = 0; i < n; i += step * 2) {
            for (int j = i; j < i + step; ++j) {
                float a = data[j], b = data[j + step];
                data[j]        = a + b;
                data[j + step] = a - b;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // Reference patch grid: every PATCH_STRIDE pixels
        const int num_ref_x = IMG_W / PATCH_STRIDE; // 32
        const int num_ref_y = IMG_H / PATCH_STRIDE; // 32
        const int num_ref   = num_ref_x * num_ref_y; // 1024

        // Block-matching offsets: 4×4 neighbourhood of patch positions.
        // N_STACK = 16 = 4×4, so each reference patch collects exactly N_STACK
        // neighbours (offset by ±1 or +2 strides).
        static_assert(N_STACK == 16, "Adjust offset generation for different N_STACK");
        Kokkos::View<int*> d_dx("dx", N_STACK), d_dy("dy", N_STACK);
        {
            auto hx = Kokkos::create_mirror_view(d_dx);
            auto hy = Kokkos::create_mirror_view(d_dy);
            int ii = 0;
            for (int dy = 0; dy < 4; ++dy)
                for (int dx = 0; dx < 4; ++dx, ++ii) {
                    hx(ii) = (dx - 2) * PATCH_STRIDE;
                    hy(ii) = (dy - 2) * PATCH_STRIDE;
                }
            Kokkos::deep_copy(d_dx, hx);
            Kokkos::deep_copy(d_dy, hy);
        }

        // Flat storage: stacks[ref * N_STACK * K*K + s * K*K + ki*K + kj]
        const int total_patches = num_ref * N_STACK;
        Kokkos::View<float*> image_d    ("image",       IMG_H * IMG_W);
        Kokkos::View<float*> stacks_d   ("stacks",      total_patches * K * K);
        Kokkos::View<int*>   nonzero_d  ("nonzero",     num_ref);
        Kokkos::View<float*> weights_d  ("weights",     num_ref);
        Kokkos::View<float*> numerator_d("numerator",   IMG_H * IMG_W);
        Kokkos::View<float*> denominator_d("denominator", IMG_H * IMG_W);
        Kokkos::View<float*> result_d   ("result",      IMG_H * IMG_W);

        // Synthetic noisy image
        {
            auto img_h = Kokkos::create_mirror_view(image_d);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(0.0f, 255.0f);
            for (int i = 0; i < IMG_H * IMG_W; ++i)
                img_h(i) = dist(rng);
            Kokkos::deep_copy(image_d, img_h);
        }

        Kokkos::fence();
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int rep = 0; rep < REPEAT; ++rep) {

            // Reset aggregation buffers
            Kokkos::deep_copy(numerator_d,   0.0f);
            Kokkos::deep_copy(denominator_d, 0.0f);
            Kokkos::deep_copy(nonzero_d,     0);

            // ------------------------------------------------------------------
            // Step 1: Gather patches (simplified block matching using fixed offsets)
            // ------------------------------------------------------------------
            Kokkos::parallel_for("gather",
                Kokkos::RangePolicy<>(0, total_patches),
                KOKKOS_LAMBDA(int idx) {
                    const int ref = idx / N_STACK;
                    const int s   = idx % N_STACK;
                    const int rx  = (ref % num_ref_x) * PATCH_STRIDE;
                    const int ry  = (ref / num_ref_x) * PATCH_STRIDE;
                    int px = rx + d_dx(s);
                    int py = ry + d_dy(s);
                    px = Kokkos::max(0, Kokkos::min(IMG_W - K, px));
                    py = Kokkos::max(0, Kokkos::min(IMG_H - K, py));
                    const int base = idx * K * K;
                    for (int ki = 0; ki < K; ++ki)
                        for (int kj = 0; kj < K; ++kj)
                            stacks_d(base + ki*K + kj) =
                                image_d((py + ki)*IMG_W + (px + kj));
                });

            // ------------------------------------------------------------------
            // Step 2: 2D DCT on every K×K patch (row DCT then column DCT)
            // ------------------------------------------------------------------
            Kokkos::parallel_for("dct2d",
                Kokkos::RangePolicy<>(0, total_patches),
                KOKKOS_LAMBDA(int idx) {
                    const int base = idx * K * K;
                    float blk[K * K];
                    for (int i = 0; i < K*K; ++i) blk[i] = stacks_d(base + i);

                    // Row-wise DCT
                    for (int r = 0; r < K; ++r) dct8(blk + r*K, 1);
                    // Column-wise DCT
                    for (int c = 0; c < K; ++c) dct8(blk + c,   K);

                    for (int i = 0; i < K*K; ++i) stacks_d(base + i) = blk[i];
                });

            // ------------------------------------------------------------------
            // Step 3: FWHT along z-axis + hard threshold + IFWHT + weight
            // Each thread processes one (ki,kj) position across all N_STACK slices
            // of one reference-patch group.
            // ------------------------------------------------------------------
            Kokkos::parallel_for("fwht_thr",
                Kokkos::RangePolicy<>(0, num_ref * K * K),
                KOKKOS_LAMBDA(int idx) {
                    const int ref = idx / (K * K);
                    const int kij = idx % (K * K);

                    float z[N_STACK];
                    for (int s = 0; s < N_STACK; ++s)
                        z[s] = stacks_d((ref * N_STACK + s) * K*K + kij);

                    fwht(z, N_STACK);

                    const float thr = L3D * Kokkos::sqrt((float)N_STACK) * SIGMA;
                    int nz = 0;
                    for (int s = 0; s < N_STACK; ++s) {
                        if (Kokkos::fabs(z[s]) < thr) z[s] = 0.0f;
                        else ++nz;
                    }

                    fwht(z, N_STACK); // inverse WHT
                    const float inv_n = 1.0f / N_STACK;
                    for (int s = 0; s < N_STACK; ++s)
                        stacks_d((ref * N_STACK + s) * K*K + kij) = z[s] * inv_n;

                    Kokkos::atomic_add(&nonzero_d(ref), nz);
                });

            // Step 3b: Derive per-group weight from non-zero count
            Kokkos::parallel_for("weights",
                Kokkos::RangePolicy<>(0, num_ref),
                KOKKOS_LAMBDA(int ref) {
                    const int nz = nonzero_d(ref);
                    weights_d(ref) = (nz > 0) ? 1.0f / (float)nz : 1.0f;
                });

            // ------------------------------------------------------------------
            // Step 4: 2D IDCT on every patch (separable, same order as DCT)
            // ------------------------------------------------------------------
            Kokkos::parallel_for("idct2d",
                Kokkos::RangePolicy<>(0, total_patches),
                KOKKOS_LAMBDA(int idx) {
                    const int base = idx * K * K;
                    float blk[K * K];
                    for (int i = 0; i < K*K; ++i) blk[i] = stacks_d(base + i);

                    for (int r = 0; r < K; ++r) idct8(blk + r*K, 1);
                    for (int c = 0; c < K; ++c) idct8(blk + c,   K);

                    for (int i = 0; i < K*K; ++i) stacks_d(base + i) = blk[i];
                });

            // ------------------------------------------------------------------
            // Step 5: Aggregate – weighted scatter of filtered patches back to image
            // ------------------------------------------------------------------
            Kokkos::parallel_for("aggregate",
                Kokkos::RangePolicy<>(0, total_patches * K * K),
                KOKKOS_LAMBDA(int idx) {
                    const int ref = idx / (N_STACK * K * K);
                    const int rem = idx % (N_STACK * K * K);
                    const int s   = rem / (K * K);
                    const int kij = rem % (K * K);
                    const int ki  = kij / K;
                    const int kj  = kij % K;

                    const int rx = (ref % num_ref_x) * PATCH_STRIDE;
                    const int ry = (ref / num_ref_x) * PATCH_STRIDE;
                    int px = rx + d_dx(s);
                    int py = ry + d_dy(s);
                    px = Kokkos::max(0, Kokkos::min(IMG_W - K, px));
                    py = Kokkos::max(0, Kokkos::min(IMG_H - K, py));

                    const float wp  = weights_d(ref);
                    const float val = stacks_d((ref * N_STACK + s) * K*K + kij);
                    const int   img_idx = (py + ki)*IMG_W + (px + kj);
                    Kokkos::atomic_add(&numerator_d(img_idx),   val * wp);
                    Kokkos::atomic_add(&denominator_d(img_idx), wp);
                });

            // Step 6: Final normalisation
            Kokkos::parallel_for("normalize",
                Kokkos::RangePolicy<>(0, IMG_H * IMG_W),
                KOKKOS_LAMBDA(int i) {
                    const float d = denominator_d(i);
                    result_d(i) = (d > 0.0f) ? numerator_d(i) / d : image_d(i);
                });
        }

        Kokkos::fence();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // Quick sanity check: mean output pixel value
        double sum = 0.0;
        Kokkos::parallel_reduce("verify",
            Kokkos::RangePolicy<>(0, IMG_H * IMG_W),
            KOKKOS_LAMBDA(int i, double& s) { s += result_d(i); },
            sum);

        printf("BM3D Kokkos benchmark\n");
        printf("Image %dx%d, patch %dx%d, stack %d, iterations %d\n",
               IMG_W, IMG_H, K, K, N_STACK, REPEAT);
        printf("Total time   : %.3f s\n", elapsed);
        printf("Avg per iter : %.3f ms\n", elapsed * 1e3 / REPEAT);
        printf("Result mean  : %.3f\n", sum / (IMG_H * IMG_W));
    }
    Kokkos::finalize();
    return 0;
}
