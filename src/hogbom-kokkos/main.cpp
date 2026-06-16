/*
 * Port of hogbom-omp to Kokkos.
 * Self-contained: generates synthetic dirty image and PSF.
 *
 * findPeak   -> Kokkos::parallel_reduce with custom functor reducer
 * subtractPSF -> Kokkos::parallel_for with MDRangePolicy (2D)
 *
 * Usage: ./main [image_size] [niters]
 *   image_size : side length of the square dirty/PSF image (default 512)
 *   niters     : deconvolution iterations (default 1000)
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cassert>
#include <vector>
#include <algorithm>

static const float GAIN      = 0.1f;
static const float THRESHOLD = 0.00001f;

// ── Peak type ────────────────────────────────────────────────────────────────
struct Peak {
    float  val = 0.f;
    size_t pos = 0;
};

// ── Custom functor-based reducer for findPeak ─────────────────────────────────
struct FindPeakFunctor {
    using value_type = Peak;

    Kokkos::View<const float*> image_;

    KOKKOS_INLINE_FUNCTION
    void operator()(size_t i, Peak& lmax) const {
        float v = image_(i);
        if (fabsf(v) > fabsf(lmax.val)) { lmax.val = v; lmax.pos = i; }
    }

    KOKKOS_INLINE_FUNCTION
    void join(Peak& dst, const Peak& src) const {
        if (fabsf(src.val) > fabsf(dst.val)) dst = src;
    }

    KOKKOS_INLINE_FUNCTION
    static void init(Peak& v) { v.val = 0.f; v.pos = 0; }
};

static Peak findPeak(const Kokkos::View<const float*>& img)
{
    Peak result;
    Kokkos::parallel_reduce("findPeak",
        Kokkos::RangePolicy<Kokkos::IndexType<size_t>>(0, img.extent(0)),
        FindPeakFunctor{img},
        result);
    return result;
}

// ── subtractPSF ───────────────────────────────────────────────────────────────
static void subtractPSF(const Kokkos::View<const float*>& d_psf, int psfWidth,
                        Kokkos::View<float*>& d_residual,          int residualWidth,
                        size_t peakPos, size_t psfPeakPos,
                        float absPeakVal, float gain)
{
    const int rx = (int)(peakPos  % residualWidth);
    const int ry = (int)(peakPos  / residualWidth);
    const int px = (int)(psfPeakPos % psfWidth);
    const int py = (int)(psfPeakPos / psfWidth);

    const int diffx = rx - px;
    const int diffy = ry - py;

    const int startx = std::max(0, rx - px);
    const int starty = std::max(0, ry - py);
    const int stopx  = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy  = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

    if (startx > stopx || starty > stopy) return;

    const float scale = gain * absPeakVal;
    const int rw = residualWidth;
    const int pw = psfWidth;
    auto psf = d_psf;
    auto res = d_residual;

    Kokkos::parallel_for("subtractPSF",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {(int64_t)starty, (int64_t)startx},
            {(int64_t)stopy + 1, (int64_t)stopx + 1}),
        KOKKOS_LAMBDA(int64_t y, int64_t x) {
            res[(int)(y * rw + x)] -= scale * psf[(int)((y - diffy)*pw + (x - diffx))];
        });
}

// ── Synthetic data generation ─────────────────────────────────────────────────
static void makeSyntheticImages(std::vector<float>& dirty,  size_t dim,
                                std::vector<float>& psf,    size_t psfDim)
{
    // PSF: normalised Gaussian centred at (psfDim/2, psfDim/2)
    float sigma2 = (float)(psfDim * psfDim) / 64.0f;
    float psfSum = 0.f;
    for (size_t y = 0; y < psfDim; y++) {
        for (size_t x = 0; x < psfDim; x++) {
            float dx = (float)x - (float)(psfDim/2);
            float dy = (float)y - (float)(psfDim/2);
            float v = expf(-(dx*dx + dy*dy) / (2.f * sigma2));
            psf[y*psfDim + x] = v;
            psfSum += v;
        }
    }
    for (auto& v : psf) v /= psfSum;

    // Dirty: background noise + one bright point source
    srand(42);
    for (size_t i = 0; i < dim*dim; i++)
        dirty[i] = 5e-4f * ((float)rand() / RAND_MAX - 0.5f);
    dirty[(dim/2)*dim + dim/2] = 1.0f;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        size_t dim     = 512;
        int    niters  = 1000;
        if (argc > 1) dim    = (size_t)atoi(argv[1]);
        if (argc > 2) niters = atoi(argv[2]);

        const size_t psfDim = dim;
        const size_t imageSize = dim * dim;
        const size_t psfSize   = psfDim * psfDim;

        printf("Image %zux%zu, PSF %zux%zu, iterations %d\n",
               dim, dim, psfDim, psfDim, niters);

        std::vector<float> h_dirty(imageSize), h_psf(psfSize);
        makeSyntheticImages(h_dirty, dim, h_psf, psfDim);

        std::vector<float> h_model(imageSize, 0.f);
        std::vector<float> h_residual = h_dirty;

        // Upload to device
        Kokkos::View<float*> d_residual("d_residual", imageSize);
        Kokkos::View<float*> d_psf     ("d_psf",      psfSize);
        {
            auto hm = Kokkos::create_mirror_view(d_residual);
            for (size_t i = 0; i < imageSize; i++) hm(i) = h_residual[i];
            Kokkos::deep_copy(d_residual, hm);
        }
        {
            auto hm = Kokkos::create_mirror_view(d_psf);
            for (size_t i = 0; i < psfSize; i++) hm(i) = h_psf[i];
            Kokkos::deep_copy(d_psf, hm);
        }

        Kokkos::View<const float*> d_psf_c = d_psf;

        // Find peak of PSF
        Peak psfPeak = findPeak(d_psf_c);
        printf("PSF peak: val=%.6f  pos=%zu  (%d,%d)\n",
               psfPeak.val, psfPeak.pos,
               (int)(psfPeak.pos % psfDim), (int)(psfPeak.pos / psfDim));
        assert(psfPeak.pos < psfSize);

        auto t0 = std::chrono::steady_clock::now();

        for (int iter = 0; iter < niters; iter++) {
            Kokkos::View<const float*> d_res_c = d_residual;
            Peak peak = findPeak(d_res_c);

            assert(peak.pos < imageSize);

            if (fabsf(peak.val) < THRESHOLD) {
                printf("Reached stopping threshold at iteration %d\n", iter);
                break;
            }

            subtractPSF(d_psf_c, (int)psfDim,
                        d_residual, (int)dim,
                        peak.pos, psfPeak.pos, peak.val, GAIN);

            h_model[peak.pos] += peak.val * GAIN;
        }
        Kokkos::fence();

        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        printf("Time: %.3f s  (%.3f ms/iter)\n", elapsed, elapsed/niters*1e3);
        printf("Done\n");
    }
    Kokkos::finalize();
    return 0;
}
