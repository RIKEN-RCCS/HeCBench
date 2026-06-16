// Kokkos port of Horn-Schunck optical flow benchmark.
// Synthetic source/target images replace PPM file loading.
// Implements coarse-to-fine pyramid with warping iterations.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================
// Parameters (matching CUDA benchmark defaults)
// ============================================================
static constexpr int   IMG_W        = 768;
static constexpr int   IMG_H        = 576;
static constexpr float ALPHA        = 0.2f;
static constexpr int   N_LEVELS     = 5;
static constexpr int   N_WARP_ITERS = 3;
static constexpr int   N_SOLVER_ITERS = 500;

using View1D = Kokkos::View<float*>;

// ============================================================
// Bilinear interpolation (clamp-to-border) for image lookup
// ============================================================
KOKKOS_INLINE_FUNCTION
float bilinear(const float* img, int w, int h, int stride, float x, float y) {
  // x, y in pixel coordinates
  int x0 = (int)Kokkos::floor(x);
  int y0 = (int)Kokkos::floor(y);
  int x1 = x0 + 1, y1 = y0 + 1;
  float fx = x - x0, fy = y - y0;

  // clamp
  x0 = Kokkos::max(0, Kokkos::min(w - 1, x0));
  x1 = Kokkos::max(0, Kokkos::min(w - 1, x1));
  y0 = Kokkos::max(0, Kokkos::min(h - 1, y0));
  y1 = Kokkos::max(0, Kokkos::min(h - 1, y1));

  float v00 = img[y0 * stride + x0];
  float v10 = img[y0 * stride + x1];
  float v01 = img[y1 * stride + x0];
  float v11 = img[y1 * stride + x1];

  return (1.f - fy) * ((1.f - fx) * v00 + fx * v10)
       +        fy  * ((1.f - fx) * v01 + fx * v11);
}

// ============================================================
// Image derivative computation (5-point stencil, matching CUDA derivativesKernel)
// ============================================================
static void computeDerivatives(const View1D& I0, const View1D& I1,
                                int w, int h, int stride,
                                const View1D& Ix, const View1D& Iy, const View1D& Iz) {
  Kokkos::parallel_for("derivatives",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, w}),
    KOKKOS_LAMBDA(int iy, int ix) {
      const int pos = iy * stride + ix;

      // Helper lambdas for clamped access
      auto I0at = [&](int r, int c) {
        r = Kokkos::max(0, Kokkos::min(h-1, r));
        c = Kokkos::max(0, Kokkos::min(w-1, c));
        return I0(r * stride + c);
      };
      auto I1at = [&](int r, int c) {
        r = Kokkos::max(0, Kokkos::min(h-1, r));
        c = Kokkos::max(0, Kokkos::min(w-1, c));
        return I1(r * stride + c);
      };

      // x-derivative (5-point, average of I0 and I1)
      float t0 = (-I0at(iy, ix-2) + 8.f*I0at(iy, ix-1) - 8.f*I0at(iy, ix+1) + I0at(iy, ix+2)) / 12.f;
      float t1 = (-I1at(iy, ix-2) + 8.f*I1at(iy, ix-1) - 8.f*I1at(iy, ix+1) + I1at(iy, ix+2)) / 12.f;
      Ix(pos) = (t0 + t1) * 0.5f;

      // y-derivative
      t0 = (-I0at(iy-2, ix) + 8.f*I0at(iy-1, ix) - 8.f*I0at(iy+1, ix) + I0at(iy+2, ix)) / 12.f;
      t1 = (-I1at(iy-2, ix) + 8.f*I1at(iy-1, ix) - 8.f*I1at(iy+1, ix) + I1at(iy+2, ix)) / 12.f;
      Iy(pos) = (t0 + t1) * 0.5f;

      // temporal derivative
      Iz(pos) = I1(pos) - I0(pos);
    });
}

// ============================================================
// Jacobi iteration for Horn-Schunck optical flow
// One step: du1, dv1 = Jacobi(du0, dv0, Ix, Iy, Iz, alpha)
// ============================================================
static void jacobiIteration(const View1D& du0, const View1D& dv0,
                             const View1D& Ix, const View1D& Iy, const View1D& Iz,
                             int w, int h, int stride, float alpha,
                             const View1D& du1, const View1D& dv1) {
  Kokkos::parallel_for("jacobi",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, w}),
    KOKKOS_LAMBDA(int iy, int ix) {
      const int pos = iy * stride + ix;

      // Neighbours (clamp)
      int left  = iy * stride + Kokkos::max(0, ix - 1);
      int right = iy * stride + Kokkos::min(w-1, ix + 1);
      int up    = Kokkos::max(0,   iy - 1) * stride + ix;
      int down  = Kokkos::min(h-1, iy + 1) * stride + ix;

      // Average of du0 and dv0 over 4-neighbourhood (Laplacian approximation)
      float uAvg = 0.25f * (du0(left) + du0(right) + du0(up) + du0(down));
      float vAvg = 0.25f * (dv0(left) + dv0(right) + dv0(up) + dv0(down));

      float ix_ = Ix(pos), iy_ = Iy(pos), iz_ = Iz(pos);
      float denom = alpha * alpha + ix_*ix_ + iy_*iy_;
      float num   = ix_ * uAvg + iy_ * vAvg + iz_;

      du1(pos) = uAvg - ix_ * num / denom;
      dv1(pos) = vAvg - iy_ * num / denom;
    });
}

// ============================================================
// Image warping: out[pos] = bilinear(src, pos + u, pos + v)
// ============================================================
static void warpImage(const View1D& src, int w, int h, int stride,
                      const View1D& u, const View1D& v, const View1D& out) {
  Kokkos::parallel_for("warp",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, w}),
    KOKKOS_LAMBDA(int iy, int ix) {
      const int pos = iy * stride + ix;
      float sx = ix + u(pos);
      float sy = iy + v(pos);

      // Clamp to image bounds
      sx = Kokkos::max(0.f, Kokkos::min((float)(w-1), sx));
      sy = Kokkos::max(0.f, Kokkos::min((float)(h-1), sy));

      int x0 = (int)sx, y0 = (int)sy;
      int x1 = Kokkos::min(w-1, x0+1), y1 = Kokkos::min(h-1, y0+1);
      float fx = sx - x0, fy = sy - y0;

      out(pos) = (1-fy)*((1-fx)*src(y0*stride+x0) + fx*src(y0*stride+x1))
                +   fy *((1-fx)*src(y1*stride+x0) + fx*src(y1*stride+x1));
    });
}

// ============================================================
// Image downscale by 0.5 (average-pool 2x2)
// ============================================================
static void downscaleImage(const View1D& fine, int wFine, int hFine, int sFine,
                            const View1D& coarse, int wCoarse, int hCoarse, int sCoarse) {
  Kokkos::parallel_for("downscale",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {hCoarse, wCoarse}),
    KOKKOS_LAMBDA(int iy, int ix) {
      // sample four points around the 2x downsampled position
      float x = (ix + 0.5f) * 2.f - 0.5f;
      float y = (iy + 0.5f) * 2.f - 0.5f;

      auto at = [&](float xx, float yy) {
        int cx = Kokkos::max(0, Kokkos::min(wFine-1, (int)xx));
        int cy = Kokkos::max(0, Kokkos::min(hFine-1, (int)yy));
        return fine(cy * sFine + cx);
      };
      coarse(iy * sCoarse + ix) = 0.25f * (at(x-0.25f, y) + at(x+0.25f, y)
                                          + at(x, y-0.25f) + at(x, y+0.25f));
    });
}

// ============================================================
// Flow upscale by 2x (bilinear + scale by 2)
// ============================================================
static void upscaleFlow(const View1D& coarse, int wC, int hC, int sC,
                         const View1D& fine,   int wF, int hF, int sF,
                         float scale) {
  Kokkos::parallel_for("upscale_flow",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {hF, wF}),
    KOKKOS_LAMBDA(int iy, int ix) {
      // map fine pixel to coarse coordinate
      float cx = (ix + 0.5f) / scale - 0.5f;
      float cy = (iy + 0.5f) / scale - 0.5f;

      int x0 = (int)cx, y0 = (int)cy;
      float fx = cx - x0, fy = cy - y0;

      int x1 = Kokkos::min(wC-1, x0+1), y1 = Kokkos::min(hC-1, y0+1);
      x0 = Kokkos::max(0, Kokkos::min(wC-1, x0));
      y0 = Kokkos::max(0, Kokkos::min(hC-1, y0));

      float v = (1-fy)*((1-fx)*coarse(y0*sC+x0) + fx*coarse(y0*sC+x1))
              +    fy *((1-fx)*coarse(y1*sC+x0) + fx*coarse(y1*sC+x1));
      fine(iy * sF + ix) = v * scale;
    });
}

// ============================================================
// Compute coarse-to-fine optical flow (Horn-Schunck)
// ============================================================
static void computeFlow(const View1D& I0_full, const View1D& I1_full,
                         int width, int height,
                         float alpha, int nLevels, int nWarpIters, int nSolverIters,
                         const View1D& u_out, const View1D& v_out) {
  // Build pyramids
  struct Level { View1D I0, I1; int w, h, stride; };
  std::vector<Level> pyramid(nLevels);

  // Level 0 = finest = original
  pyramid[0].w = width; pyramid[0].h = height; pyramid[0].stride = width;
  pyramid[0].I0 = View1D("I0_L0", width * height);
  pyramid[0].I1 = View1D("I1_L0", width * height);
  Kokkos::deep_copy(pyramid[0].I0, I0_full);
  Kokkos::deep_copy(pyramid[0].I1, I1_full);

  for (int l = 1; l < nLevels; ++l) {
    int pw = pyramid[l-1].w, ph = pyramid[l-1].h, ps = pyramid[l-1].stride;
    int cw = (pw + 1) / 2, ch = (ph + 1) / 2;
    pyramid[l].w = cw; pyramid[l].h = ch; pyramid[l].stride = cw;
    pyramid[l].I0 = View1D(std::string("I0_L") + std::to_string(l), cw * ch);
    pyramid[l].I1 = View1D(std::string("I1_L") + std::to_string(l), cw * ch);
    downscaleImage(pyramid[l-1].I0, pw, ph, ps, pyramid[l].I0, cw, ch, cw);
    downscaleImage(pyramid[l-1].I1, pw, ph, ps, pyramid[l].I1, cw, ch, cw);
  }

  // Coarsest level: initialize flow to zero
  {
    int cw = pyramid[nLevels-1].w, ch = pyramid[nLevels-1].h;
    int cs = pyramid[nLevels-1].stride;
    int sz = cw * ch;
    View1D u("u_coarse", sz), v("v_coarse", sz);
    View1D u_tmp("u_tmp", sz), v_tmp("v_tmp", sz);
    View1D Ix("Ix_c", sz), Iy("Iy_c", sz), Iz("Iz_c", sz);
    View1D I1w("I1w_c", sz);
    Kokkos::deep_copy(u, 0.f); Kokkos::deep_copy(v, 0.f);

    for (int l = nLevels - 1; l >= 0; --l) {
      int lw = pyramid[l].w, lh = pyramid[l].h, ls = pyramid[l].stride;
      int lsz = lw * lh;

      // Upscale flow from coarser level if needed
      View1D ul("u_l", lsz), vl("v_l", lsz);
      if (l == nLevels - 1) {
        Kokkos::deep_copy(ul, 0.f); Kokkos::deep_copy(vl, 0.f);
      } else {
        int nextW = pyramid[l+1].w, nextH = pyramid[l+1].h, nextS = pyramid[l+1].stride;
        float scale = 2.f;
        upscaleFlow(u, nextW, nextH, nextS, ul, lw, lh, ls, scale);
        upscaleFlow(v, nextW, nextH, nextS, vl, lw, lh, ls, scale);
      }

      // Re-allocate per-level temporaries
      u = View1D("u_l2", lsz);   Kokkos::deep_copy(u, ul);
      v = View1D("v_l2", lsz);   Kokkos::deep_copy(v, vl);

      View1D Ixl("Ix_l", lsz), Iyl("Iy_l", lsz), Izl("Iz_l", lsz);
      View1D I1wl("I1w_l", lsz);
      View1D du0("du0", lsz), dv0("dv0", lsz);
      View1D du1("du1", lsz), dv1("dv1", lsz);

      for (int warp = 0; warp < nWarpIters; ++warp) {
        // Warp I1 with current flow
        warpImage(pyramid[l].I1, lw, lh, ls, u, v, I1wl);

        // Compute derivatives between I0 and warped I1
        computeDerivatives(pyramid[l].I0, I1wl, lw, lh, ls, Ixl, Iyl, Izl);

        // Solve linear system (incremental flow du, dv)
        Kokkos::deep_copy(du0, 0.f);
        Kokkos::deep_copy(dv0, 0.f);

        for (int iter = 0; iter < nSolverIters; ++iter) {
          jacobiIteration(du0, dv0, Ixl, Iyl, Izl, lw, lh, ls, alpha, du1, dv1);
          Kokkos::deep_copy(du0, du1);
          Kokkos::deep_copy(dv0, dv1);
        }

        // Update total flow
        Kokkos::parallel_for("update_flow",
          Kokkos::RangePolicy<>(0, lsz),
          KOKKOS_LAMBDA(int i) { u(i) += du0(i); v(i) += dv0(i); });
      }
    }

    // Copy final flow to output
    Kokkos::deep_copy(u_out, u);
    Kokkos::deep_copy(v_out, v);
  }
}

// ============================================================
// Generate synthetic image: checkerboard + sine gradient
// ============================================================
static void generateImage(const View1D& img, int w, int h, float phase) {
  auto h_img = Kokkos::create_mirror_view(img);
  for (int iy = 0; iy < h; ++iy)
    for (int ix = 0; ix < w; ++ix) {
      float xn = ix / (float)w;
      float yn = iy / (float)h;
      h_img(iy * w + ix) = 0.5f + 0.3f * std::sin(2.f * 3.14159f * (xn + phase))
                                 + 0.2f * std::cos(4.f * 3.14159f * (yn + phase * 0.5f));
      // clamp
      h_img(iy * w + ix) = std::min(1.f, std::max(0.f, h_img(iy * w + ix)));
    }
  Kokkos::deep_copy(img, h_img);
}

// ============================================================
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int width   = IMG_W;
  const int height  = IMG_H;
  const float alpha = ALPHA;

  printf("Optical flow: %dx%d, alpha=%.2f, levels=%d, warpIters=%d, solverIters=%d\n",
         width, height, alpha, N_LEVELS, N_WARP_ITERS, N_SOLVER_ITERS);

  Kokkos::initialize(argc, argv);
  {
    const int stride = width;
    const int imgSz  = width * height;

    View1D I0("I0", imgSz), I1("I1", imgSz);
    View1D u ("u",  imgSz), v ("v",  imgSz);

    // Synthetic images: slight phase shift simulates small displacement
    generateImage(I0, width, height, 0.f);
    generateImage(I1, width, height, 0.01f);  // ~7px horizontal shift at 768px width

    // Warm-up
    computeFlow(I0, I1, width, height, alpha,
                N_LEVELS, N_WARP_ITERS, N_SOLVER_ITERS, u, v);
    Kokkos::fence();

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
      computeFlow(I0, I1, width, height, alpha,
                  N_LEVELS, N_WARP_ITERS, N_SOLVER_ITERS, u, v);
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();

    double elapsed_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
    printf("Average optical flow computation time: %f (us)\n", elapsed_us / repeat);

    // Print a simple checksum
    float sumU = 0.f, sumV = 0.f;
    Kokkos::parallel_reduce("sum_u", Kokkos::RangePolicy<>(0, imgSz),
      KOKKOS_LAMBDA(int i, float& s) { s += u(i); }, sumU);
    Kokkos::parallel_reduce("sum_v", Kokkos::RangePolicy<>(0, imgSz),
      KOKKOS_LAMBDA(int i, float& s) { s += v(i); }, sumV);
    printf("Flow checksum: u_sum=%f  v_sum=%f\n", sumU, sumV);
  }
  Kokkos::finalize();
  return 0;
}
