// Kokkos port of resnet-kernels benchmark.
// Implements Winograd convolution (6x6 BtdB transform + GEMM + AtIA)
// and 1x1 convolution for 128 and 256 channel variants.
// Uses synthetic random data (no binary data files required).

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>

using View1D = Kokkos::View<float*>;

// ============================================================
// RNG helper: fill a View with uniform random data
// ============================================================
static void fillRandom(const View1D& v, float lo = -0.1f, float hi = 0.1f) {
  auto h = Kokkos::create_mirror_view(v);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (int i = 0; i < (int)v.extent(0); ++i) h(i) = dist(rng);
  Kokkos::deep_copy(v, h);
}

// ============================================================
// Winograd BtdB transform (6x6 kernel, 4x4 input tiles -> 4x4 output tiles)
//
// B^T = [[4, 0,-5, 0, 1, 0],
//         [0,-4,-4, 1, 1, 0],
//         [0, 4,-4,-1, 1, 0],
//         [0,-2,-1, 2, 1, 0],
//         [0, 2,-1,-2, 1, 0],
//         [0, 4, 0,-5, 0, 1]]
//
// For input tile d[6][6][C], output is V[6][6][C] = B^T d B
//
// Parameters for mode 0 (128 channels):
//   Input:  14x14 spatial, 128 channels  → 7x7 = 49 tiles of 6x6
//   Winograd domain: 36 x (7*7) x 128
// ============================================================

// Apply B^T on one row of a 6-element vector (one output row index 'r')
KOKKOS_INLINE_FUNCTION float btRow(int r, float d0, float d1, float d2,
                                   float d3, float d4, float d5) {
  switch (r) {
    case 0: return  4*d0          - 5*d2          + d4;
    case 1: return        -4*d1   - 4*d2 +  d3   + d4;
    case 2: return         4*d1   - 4*d2 -  d3   + d4;
    case 3: return        -2*d1   -   d2 + 2*d3  + d4;
    case 4: return         2*d1   -   d2 - 2*d3  + d4;
    default:return         4*d1          - 5*d3        + d5;
  }
}

// BtdB transform: Input [nTiles * 36 * C], Output [36 * nTiles * C]
// nTiles = numTilesH * numTilesW, each tile is 6x6 in Winograd domain
static void winograd_BtdB(const View1D& input, const View1D& output,
                           int nTilesH, int nTilesW, int C) {
  // input layout: tile_h, tile_w, 6, 6, C  -> flat as  [nTilesH*nTilesW * 36 * C]
  // For each tile and channel, apply B^T on rows then columns
  // output layout: [36 * nTilesH * nTilesW * C]  (winograd_index * tile * channel)

  const int nTiles = nTilesH * nTilesW;

  Kokkos::parallel_for("BtdB",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nTiles, C, 1}),
    KOKKOS_LAMBDA(int tile, int c, int /*dummy*/) {
      // Load 6x6 input patch for this tile+channel
      float d[6][6];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          d[i][j] = input(tile * 36 * C + (i*6 + j) * C + c);

      // Apply B^T on rows: tmp = B^T * d
      float tmp[6][6];
      for (int r = 0; r < 6; ++r)
        for (int j = 0; j < 6; ++j)
          tmp[r][j] = btRow(r, d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j]);

      // Apply B on columns: out = tmp * B  (= tmp * B, using B^T symmetry)
      float out[6][6];
      for (int i = 0; i < 6; ++i)
        for (int r = 0; r < 6; ++r)
          out[i][r] = btRow(r, tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], tmp[i][5]);

      // Write output: layout [36 * nTiles * C]
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          output((i*6 + j) * nTiles * C + tile * C + c) = out[i][j];
    });
}

// ============================================================
// Winograd GEMM: for each of the 36 Winograd elements,
// compute M_w = V_w * U_w  where
//   V_w: [nTiles x Cin]  (transformed input)
//   U_w: [Cin x Cout]    (transformed weight)
//   M_w: [nTiles x Cout] (transformed output)
// ============================================================
static void winograd_gemm(const View1D& V, const View1D& U, const View1D& M,
                           int nTiles, int Cin, int Cout) {
  // V layout: [36 * nTiles * Cin]  -> V_w at offset w*nTiles*Cin
  // U layout: [36 * Cin * Cout]
  // M layout: [36 * nTiles * Cout]
  Kokkos::parallel_for("winograd_gemm",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {36, nTiles, Cout}),
    KOKKOS_LAMBDA(int w, int tile, int cout) {
      float sum = 0.f;
      for (int cin = 0; cin < Cin; ++cin)
        sum += V(w * nTiles * Cin + tile * Cin + cin)
             * U(w * Cin * Cout  + cin  * Cout + cout);
      M(w * nTiles * Cout + tile * Cout + cout) = sum;
    });
}

// ============================================================
// Winograd output transform A^T * M * A
// A^T = [[1, 1, 1, 1, 1, 0],
//         [0, 1,-1, 2,-2, 0],
//         [0, 1, 1, 4, 4, 0],
//         [0, 1,-1, 8,-8, 1]]
// Input: [36 * nTiles * Cout], Output: [nTiles * 4 * 4 * Cout]
// ============================================================
KOKKOS_INLINE_FUNCTION float atRow(int r, float m0, float m1, float m2,
                                   float m3, float m4, float m5) {
  switch (r) {
    case 0: return m0 + m1 + m2 + m3 + m4;
    case 1: return      m1 - m2 + 2*m3 - 2*m4;
    case 2: return      m1 + m2 + 4*m3 + 4*m4;
    default: return     m1 - m2 + 8*m3 - 8*m4 + m5;
  }
}

static void winograd_AtIA(const View1D& M, const View1D& bias,
                           const View1D& bnScale, const View1D& output,
                           int nTiles, int Cout) {
  Kokkos::parallel_for("AtIA",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nTiles, Cout}),
    KOKKOS_LAMBDA(int tile, int cout) {
      // Load 6x6 patch from M
      float m[6][6];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          m[i][j] = M((i*6 + j) * nTiles * Cout + tile * Cout + cout);

      // A^T * m
      float tmp[4][6];
      for (int j = 0; j < 6; ++j) {
        float col[4];
        for (int r = 0; r < 4; ++r)
          col[r] = atRow(r, m[0][j], m[1][j], m[2][j], m[3][j], m[4][j], m[5][j]);
        for (int r = 0; r < 4; ++r) tmp[r][j] = col[r];
      }

      // tmp * A  (same transform on columns)
      float out[4][4];
      for (int i = 0; i < 4; ++i)
        for (int r = 0; r < 4; ++r)
          out[i][r] = atRow(r, tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4], tmp[i][5]);

      // Apply bias+scale and ReLU, write output
      float b = bias(cout), s = bnScale(cout);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          float v = s * out[i][j] + b;
          output(tile * 16 * Cout + (i*4 + j) * Cout + cout) = v > 0.f ? v : 0.f;
        }
    });
}

// ============================================================
// 1x1 convolution: C = ReLU(bnScale * (A * W) + bnBias)
//   A: [nTiles x Cin], W: [Cin x Cout], C: [nTiles x Cout]
// ============================================================
static void conv1x1(const View1D& A, const View1D& W,
                    const View1D& bnBias, const View1D& bnScale,
                    const View1D& C,
                    int nTiles, int Cin, int Cout) {
  Kokkos::parallel_for("conv1x1",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nTiles, Cout}),
    KOKKOS_LAMBDA(int tile, int cout) {
      float sum = 0.f;
      for (int cin = 0; cin < Cin; ++cin)
        sum += A(tile * Cin + cin) * W(cin * Cout + cout);
      float v = bnScale(cout) * sum + bnBias(cout);
      C(tile * Cout + cout) = v > 0.f ? v : 0.f;
    });
}

// ============================================================
// Kernel functions (matching CUDA benchmark interface)
// ============================================================

struct KernelResult { double time_us; double ktime_us; };

// kernel_128: 14x14x128 input, 128 output channels, Winograd 6x6
KernelResult kernel_128(int repeat) {
  const int H = 14, W = 14, Cin = 128, Cout = 128;
  const int tileSz = 6;
  const int nTilesH = H - tileSz + 1; // 9 (step by 2 for 4x4 output tiles... but use step 2: ceil(14/2))
  // Winograd F(4x4, 3x3): input tile 6x6, output tile 4x4, stride 4 -> floor((14-6)/4)+1 = 3
  // Standard: floor((14-2)/2) = 6 tiles per dim
  const int nTH = (H - 2) / 2;  // 6
  const int nTW = (W - 2) / 2;  // 6
  const int nTiles = nTH * nTW;   // 36

  View1D d_input  ("input",   nTiles * 36 * Cin);
  View1D d_V      ("V",       36 * nTiles * Cin);
  View1D d_U      ("U",       36 * Cin * Cout);
  View1D d_M      ("M",       36 * nTiles * Cout);
  View1D d_bias   ("bias",    Cout);
  View1D d_scale  ("scale",   Cout);
  View1D d_output ("output",  nTiles * 16 * Cout);

  fillRandom(d_input);
  fillRandom(d_V);
  fillRandom(d_U);
  fillRandom(d_bias,  0.f, 0.01f);
  fillRandom(d_scale, 0.9f, 1.1f);

  // warm-up
  winograd_BtdB(d_input, d_V, nTH, nTW, Cin);
  winograd_gemm(d_V, d_U, d_M, nTiles, Cin, Cout);
  winograd_AtIA(d_M, d_bias, d_scale, d_output, nTiles, Cout);
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    winograd_BtdB(d_input, d_V, nTH, nTW, Cin);
    winograd_gemm(d_V, d_U, d_M, nTiles, Cin, Cout);
    winograd_AtIA(d_M, d_bias, d_scale, d_output, nTiles, Cout);
  }
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  return {total / repeat, total / repeat};
}

// kernel_256: same as 128 but with 256 channels
KernelResult kernel_256(int repeat) {
  const int H = 14, W = 14, Cin = 256, Cout = 256;
  const int nTH = (H - 2) / 2;
  const int nTW = (W - 2) / 2;
  const int nTiles = nTH * nTW;

  View1D d_input  ("input256",   nTiles * 36 * Cin);
  View1D d_V      ("V256",       36 * nTiles * Cin);
  View1D d_U      ("U256",       36 * Cin * Cout);
  View1D d_M      ("M256",       36 * nTiles * Cout);
  View1D d_bias   ("bias256",    Cout);
  View1D d_scale  ("scale256",   Cout);
  View1D d_output ("output256",  nTiles * 16 * Cout);

  fillRandom(d_input);
  fillRandom(d_V);
  fillRandom(d_U);
  fillRandom(d_bias, 0.f, 0.01f);
  fillRandom(d_scale, 0.9f, 1.1f);

  // warm-up
  winograd_BtdB(d_input, d_V, nTH, nTW, Cin);
  winograd_gemm(d_V, d_U, d_M, nTiles, Cin, Cout);
  winograd_AtIA(d_M, d_bias, d_scale, d_output, nTiles, Cout);
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    winograd_BtdB(d_input, d_V, nTH, nTW, Cin);
    winograd_gemm(d_V, d_U, d_M, nTiles, Cin, Cout);
    winograd_AtIA(d_M, d_bias, d_scale, d_output, nTiles, Cout);
  }
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  return {total / repeat, total / repeat};
}

// kernel_128_1_in: 512->128 1x1 conv (14x14 spatial)
KernelResult kernel_128_1_in(int repeat) {
  const int spatial = 14 * 14; // 196 tiles
  const int Cin = 512, Cout = 128;

  View1D d_A    ("A_128_1in",  spatial * Cin);
  View1D d_W    ("W_128_1in",  Cin * Cout);
  View1D d_bias ("bias_128_1in", Cout);
  View1D d_scale("scale_128_1in",Cout);
  View1D d_C    ("C_128_1in",  spatial * Cout);

  fillRandom(d_A);
  fillRandom(d_W);
  fillRandom(d_bias, 0.f, 0.01f);
  fillRandom(d_scale, 0.9f, 1.1f);

  conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r)
    conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  return {total / repeat, total / repeat};
}

// kernel_128_1_out: 128->512 1x1 conv
KernelResult kernel_128_1_out(int repeat) {
  const int spatial = 14 * 14;
  const int Cin = 128, Cout = 512;

  View1D d_A    ("A_128_1out",   spatial * Cin);
  View1D d_W    ("W_128_1out",   Cin * Cout);
  View1D d_bias ("bias_128_1out",Cout);
  View1D d_scale("scale_128_1out",Cout);
  View1D d_C    ("C_128_1out",   spatial * Cout);

  fillRandom(d_A);
  fillRandom(d_W);
  fillRandom(d_bias, 0.f, 0.01f);
  fillRandom(d_scale, 0.9f, 1.1f);

  conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r)
    conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  return {total / repeat, total / repeat};
}

// kernel_256_1_in: 1024->256 1x1 conv (7x7 spatial)
KernelResult kernel_256_1_in(int repeat) {
  const int spatial = 7 * 7;
  const int Cin = 1024, Cout = 256;

  View1D d_A    ("A_256_1in",   spatial * Cin);
  View1D d_W    ("W_256_1in",   Cin * Cout);
  View1D d_bias ("bias_256_1in",Cout);
  View1D d_scale("scale_256_1in",Cout);
  View1D d_C    ("C_256_1in",  spatial * Cout);

  fillRandom(d_A);
  fillRandom(d_W);
  fillRandom(d_bias, 0.f, 0.01f);
  fillRandom(d_scale, 0.9f, 1.1f);

  conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r)
    conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  return {total / repeat, total / repeat};
}

// kernel_256_1_out: 256->1024 1x1 conv
KernelResult kernel_256_1_out(int repeat) {
  const int spatial = 7 * 7;
  const int Cin = 256, Cout = 1024;

  View1D d_A    ("A_256_1out",   spatial * Cin);
  View1D d_W    ("W_256_1out",   Cin * Cout);
  View1D d_bias ("bias_256_1out",Cout);
  View1D d_scale("scale_256_1out",Cout);
  View1D d_C    ("C_256_1out",  spatial * Cout);

  fillRandom(d_A);
  fillRandom(d_W);
  fillRandom(d_bias, 0.f, 0.01f);
  fillRandom(d_scale, 0.9f, 1.1f);

  conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r)
    conv1x1(d_A, d_W, d_bias, d_scale, d_C, spatial, Cin, Cout);
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double total = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
  return {total / repeat, total / repeat};
}

// ============================================================
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <mode> <repeat (>2)>\n", argv[0]);
    return 1;
  }
  const int mode   = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    double time_total = 0, ktime_total = 0;
    const int effective = (repeat > 2) ? repeat - 2 : 1;

    for (int i = 0; i < repeat; ++i) {
      KernelResult r;
      switch (mode) {
        case 0: r = kernel_128(1);       break;
        case 1: r = kernel_256(1);       break;
        case 2: r = kernel_128_1_in(1);  break;
        case 3: r = kernel_128_1_out(1); break;
        case 4: r = kernel_256_1_in(1);  break;
        case 5: r = kernel_256_1_out(1); break;
        default:
          printf("Unknown mode %d\n", mode);
          Kokkos::finalize();
          return 1;
      }
      if (i > 1) {
        time_total  += r.time_us;
        ktime_total += r.ktime_us;
      }
    }

    printf("Case %d: Average device offload time: [%lf us]\n", mode, time_total / effective);
    printf("        Average kernel time: [%lf us]\n", ktime_total / effective);
  }
  Kokkos::finalize();
  return 0;
}
