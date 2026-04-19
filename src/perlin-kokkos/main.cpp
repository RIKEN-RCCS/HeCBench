#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>

static constexpr int WIN_WIDTH  = 800;
static constexpr int WIN_HEIGHT = 600;

struct NoiseParams {
  float ppu;        // pixels per unit
  int   seed;
  int   octaves;
  float lacunarity; // frequency modulation rate per octave
  float persistence;// amplitude modulation rate per octave
};

// --------------------------------------------------------------------------
// Perlin noise primitives – all callable on device
// --------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION float lerp(float a, float b, float t) {
  return a + t * (b - a);
}

// Smooth: 6t^5 - 15t^4 + 10t^3
KOKKOS_INLINE_FUNCTION float smooth(float t) {
  return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

// Ken Perlin's reference permutation (256 entries).
// Access pattern is _hash[A + B] where A,B ∈ [0,255], so we double the table
// to 512 entries to avoid out-of-bounds reads.
KOKKOS_INLINE_FUNCTION int perm(int i) {
  static constexpr int H[512] = {
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
    // repeat
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
    184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
    222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
  };
  return H[i & 511];
}

// 8 base gradients (x and y components stored separately)
KOKKOS_INLINE_FUNCTION float gradX(int i) {
  static constexpr float gx[8] = {
    1.f, -1.f, 0.f, 0.f,
     0.70710678118654752440f, -0.70710678118654752440f,
     0.70710678118654752440f, -0.70710678118654752440f
  };
  return gx[i & 7];
}
KOKKOS_INLINE_FUNCTION float gradY(int i) {
  static constexpr float gy[8] = {
    0.f, 0.f, 1.f, -1.f,
     0.70710678118654752440f,  0.70710678118654752440f,
    -0.70710678118654752440f, -0.70710678118654752440f
  };
  return gy[i & 7];
}

// Compute Perlin noise at (x, y) with given seed.
// Returns value in [0, 1].
KOKKOS_INLINE_FUNCTION float noiseAt(float x, float y, int seed) {
  const int ix = static_cast<int>(Kokkos::floor(x));
  const int iy = static_cast<int>(Kokkos::floor(y));

  const float wx = x - (float)ix;
  const float wy = y - (float)iy;

  const int ix0 = ix & 255, iy0 = iy & 255;
  const int ix1 = (ix0 + 1) & 255, iy1 = (iy0 + 1) & 255;

  const int h0 = perm(ix0), h1 = perm(ix1);
  const int iTL = (perm(h0 + iy0) + seed) & 7;
  const int iTR = (perm(h1 + iy0) + seed) & 7;
  const int iBL = (perm(h0 + iy1) + seed) & 7;
  const int iBR = (perm(h1 + iy1) + seed) & 7;

  const float dTL = gradX(iTL) * wx       + gradY(iTL) * wy;
  const float dTR = gradX(iTR) * (wx - 1) + gradY(iTR) * wy;
  const float dBL = gradX(iBL) * wx       + gradY(iBL) * (wy - 1);
  const float dBR = gradX(iBR) * (wx - 1) + gradY(iBR) * (wy - 1);

  const float tx = smooth(wx), ty = smooth(wy);
  const float left  = lerp(dTL, dBL, ty);
  const float right = lerp(dTR, dBR, ty);

  return (lerp(left, right, tx) + 1.f) * 0.5f;
}

// Fractional Brownian motion: sum several octaves of Perlin noise.
KOKKOS_INLINE_FUNCTION float sumOctaves(float x, float y, NoiseParams p) {
  float frequency = 1.f;
  float sum       = noiseAt(x, y, p.seed);
  float amplitude = 1.f;
  float range     = 1.f;
  for (int i = 1; i < p.octaves; i++) {
    frequency *= p.lacunarity;
    amplitude *= p.persistence;
    range     += amplitude;
    sum       += amplitude * noiseAt(x * frequency, y * frequency, p.seed);
  }
  return sum / range;
}

// --------------------------------------------------------------------------
// Benchmark driver
// --------------------------------------------------------------------------

int main(int argc, char** argv) {
  NoiseParams params;
  params.ppu         = 250.f;
  params.seed        = 0;
  params.octaves     = 3;
  params.lacunarity  = 2.f;
  params.persistence = 0.5f;

  Kokkos::initialize(argc, argv);
  {
    const int npix = WIN_WIDTH * WIN_HEIGHT;

    // Host pixel buffer (RGBA)
    std::vector<uint8_t> h_pixels(4 * npix);

    // Device pixel buffer
    Kokkos::View<uint8_t*> d_pixels("d_pixels", 4 * npix);

    for (int nStreams = 1; nStreams <= 32; nStreams *= 2) {
      std::cout << "\nUsing " << nStreams << " streams." << std::endl;

      const int partialHeight = WIN_HEIGHT / nStreams;
      std::cout << "Each stream will calculate "
                << partialHeight * WIN_WIDTH << " pixels." << std::endl;

      Kokkos::fence();
      auto t0 = std::chrono::steady_clock::now();

      // Launch one parallel_for per "stream" (sequential in Kokkos, but
      // demonstrates the same decomposition as the CUDA version).
      for (int s = 0; s < nStreams; s++) {
        const int yStart = s * partialHeight;
        const int nPix   = partialHeight * WIN_WIDTH;
        NoiseParams lp   = params; // copy for lambda capture

        Kokkos::parallel_for(
            "perlin", nPix, KOKKOS_LAMBDA(int idx) {
              const int px = idx % WIN_WIDTH;
              const int py = yStart + idx / WIN_WIDTH;

              const float noise =
                  sumOctaves(px / lp.ppu, py / lp.ppu, lp);

              const int base = 4 * (py * WIN_WIDTH + px);
              const uint8_t val = (uint8_t)(noise * 255.f);
              d_pixels[base + 0] = val;
              d_pixels[base + 1] = val;
              d_pixels[base + 2] = val;
              d_pixels[base + 3] = 255;
            });
      }

      Kokkos::fence();
      auto t1   = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      std::cout << "Total kernel execution time " << time * 1e-6 << " (ms)"
                << std::endl;

      // Copy back and compute checksum
      {
        auto hv = Kokkos::View<uint8_t*, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            h_pixels.data(), 4 * npix);
        Kokkos::deep_copy(hv, d_pixels);
      }

      uint64_t checksum = 0;
      for (auto v : h_pixels) checksum += v;
      std::cout << "checksum = " << checksum / (4 * npix) << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
