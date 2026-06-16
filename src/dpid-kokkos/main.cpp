// Kokkos port of dpid-cuda (Detail-Preserving Image Downscaling)
// Warp-shuffle reductions replaced by serial loops within each output pixel's
// parallel_for work-item.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <Kokkos_Core.hpp>

struct Params {
  uint32_t oWidth, oHeight;
  uint32_t iWidth, iHeight;
  float    pWidth, pHeight;
  float    lambda;
  uint32_t repeat;
};

struct uchar3 { unsigned char x, y, z; };
struct float4 { float x, y, z, w; };

double LCG_random_double(uint64_t* seed)
{
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    printf("Usage: %s <output width> <output height> <lambda> <repeat>\n", argv[0]);
    return 1;
  }

  Params p;
  p.oWidth  = (uint32_t)atoi(argv[1]);
  p.oHeight = (uint32_t)atoi(argv[2]);
  p.lambda  = (float)atof(argv[3]);
  p.repeat  = (uint32_t)atoi(argv[4]);

  if (p.oWidth == 0 && p.oHeight == 0) {
    printf("only one dimension can be 0!\n");
    return 1;
  }

  p.iWidth  = 8192;
  p.iHeight = 8192;

  if (p.oWidth  == 0) p.oWidth  = (uint32_t)std::round((p.oHeight / (double)p.iHeight) * p.iWidth);
  if (p.oHeight == 0) p.oHeight = (uint32_t)std::round((p.oWidth  / (double)p.iWidth)  * p.iHeight);

  p.pWidth  = p.iWidth  / (float)p.oWidth;
  p.pHeight = p.iHeight / (float)p.oHeight;

  const size_t inSz  = (size_t)p.iWidth * p.iHeight;
  const size_t outSz = (size_t)p.oWidth * p.oHeight;

  uchar3* hInput  = (uchar3*)malloc(sizeof(uchar3) * inSz);
  uchar3* hOutput = (uchar3*)malloc(sizeof(uchar3) * outSz);

  uint64_t seed = 123;
  for (uint32_t i = 0; i < inSz; i++) {
    hInput[i].x = (unsigned char)(256 * LCG_random_double(&seed));
    hInput[i].y = (unsigned char)(256 * LCG_random_double(&seed));
    hInput[i].z = (unsigned char)(256 * LCG_random_double(&seed));
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<uchar3*>  dInput  ("input",    inSz);
    Kokkos::View<uchar3*>  dOutput ("output",   outSz);
    Kokkos::View<uchar3*>  dGuidance("guidance", outSz);

    {
      auto hv = Kokkos::create_mirror_view(dInput);
      for (size_t i = 0; i < inSz; i++) hv(i) = hInput[i];
      Kokkos::deep_copy(dInput, hv);
    }

    Kokkos::fence();
    auto tStart = std::chrono::steady_clock::now();

    for (uint32_t iter = 0; iter < p.repeat; iter++) {

      // --- kernelGuidance ---
      Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, (size_t)p.oWidth * p.oHeight),
        KOKKOS_LAMBDA(size_t idx) {
          uint32_t PX = (uint32_t)(idx % p.oWidth);
          uint32_t PY = (uint32_t)(idx / p.oWidth);

          float sx = Kokkos::fmax(PX * p.pWidth,  0.f);
          float ex = Kokkos::fmin((PX+1) * p.pWidth,  (float)p.iWidth);
          float sy = Kokkos::fmax(PY * p.pHeight, 0.f);
          float ey = Kokkos::fmin((PY+1) * p.pHeight, (float)p.iHeight);

          uint32_t sxr = (uint32_t)floorf(sx);
          uint32_t syr = (uint32_t)floorf(sy);
          uint32_t exr = (uint32_t)ceilf(ex);
          uint32_t eyr = (uint32_t)ceilf(ey);
          uint32_t xCount = exr - sxr;
          uint32_t yCount = eyr - syr;
          uint32_t pixelCount = xCount * yCount;

          float cx = 0, cy = 0, cz = 0, cw = 0;
          for (uint32_t i = 0; i < pixelCount; i++) {
            uint32_t x = sxr + (i % xCount);
            uint32_t y = syr + (i / xCount);
            float f = 1.f;
            if (x < sx)        f *= 1.f - (sx - x);
            if ((x+1.f) > ex)  f *= 1.f - ((x+1.f) - ex);
            if (y < sy)        f *= 1.f - (sy - y);
            if ((y+1.f) > ey)  f *= 1.f - ((y+1.f) - ey);
            const uchar3& pix = dInput(x + y * p.iWidth);
            cx += pix.x * f;
            cy += pix.y * f;
            cz += pix.z * f;
            cw += f;
          }
          if (cw > 0) { cx /= cw; cy /= cw; cz /= cw; }
          uchar3 g; g.x = (unsigned char)cx; g.y = (unsigned char)cy; g.z = (unsigned char)cz;
          dGuidance(PX + PY * p.oWidth) = g;
        });

      // --- kernelDownsampling ---
      Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, (size_t)p.oWidth * p.oHeight),
        KOKKOS_LAMBDA(size_t idx) {
          uint32_t PX = (uint32_t)(idx % p.oWidth);
          uint32_t PY = (uint32_t)(idx / p.oWidth);

          // Compute average color from 3x3 guidance neighbourhood
          const float corner = 1.f, edge = 2.f, center = 4.f;
          float ax = 0, ay = 0, az = 0, aw = 0;
          auto addG = [&](uint32_t gx, uint32_t gy, float w) {
            const uchar3& g = dGuidance(gx + gy * p.oWidth);
            ax += g.x * w; ay += g.y * w; az += g.z * w; aw += w;
          };
          if (PY > 0) {
            if (PX > 0)             addG(PX-1, PY-1, corner);
                                    addG(PX,   PY-1, edge);
            if (PX+1 < p.oWidth)    addG(PX+1, PY-1, corner);
          }
          if (PX > 0)               addG(PX-1, PY,   edge);
                                    addG(PX,   PY,   center);
          if (PX+1 < p.oWidth)      addG(PX+1, PY,   edge);
          if (PY+1 < p.oHeight) {
            if (PX > 0)             addG(PX-1, PY+1, corner);
                                    addG(PX,   PY+1, edge);
            if (PX+1 < p.oWidth)    addG(PX+1, PY+1, corner);
          }
          if (aw > 0) { ax /= aw; ay /= aw; az /= aw; }

          float sx = Kokkos::fmax(PX * p.pWidth,  0.f);
          float ex = Kokkos::fmin((PX+1) * p.pWidth,  (float)p.iWidth);
          float sy = Kokkos::fmax(PY * p.pHeight, 0.f);
          float ey = Kokkos::fmin((PY+1) * p.pHeight, (float)p.iHeight);
          uint32_t sxr = (uint32_t)floorf(sx);
          uint32_t syr = (uint32_t)floorf(sy);
          uint32_t exr = (uint32_t)ceilf(ex);
          uint32_t eyr = (uint32_t)ceilf(ey);
          uint32_t xCount = exr - sxr;
          uint32_t yCount = eyr - syr;
          uint32_t pixelCount = xCount * yCount;

          float cx = 0, cy = 0, cz = 0, cw = 0;
          for (uint32_t i = 0; i < pixelCount; i++) {
            uint32_t x = sxr + (i % xCount);
            uint32_t y = syr + (i / xCount);
            const uchar3& pix = dInput(x + y * p.iWidth);
            float dx = ax - pix.x, dy = ay - pix.y, dz = az - pix.z;
            float dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz) / 441.6729559f;
            float lam = (p.lambda == 0.f) ? 1.f :
                        (p.lambda == 1.f) ? dist : Kokkos::pow(dist, p.lambda);
            float f = lam;
            if (x < sx)        f *= 1.f - (sx - x);
            if ((x+1.f) > ex)  f *= 1.f - ((x+1.f) - ex);
            if (y < sy)        f *= 1.f - (sy - y);
            if ((y+1.f) > ey)  f *= 1.f - ((y+1.f) - ey);
            cx += pix.x * f; cy += pix.y * f; cz += pix.z * f; cw += f;
          }

          uchar3 out;
          if (cw == 0.f) {
            out.x = (unsigned char)ax;
            out.y = (unsigned char)ay;
            out.z = (unsigned char)az;
          } else {
            out.x = (unsigned char)(cx / cw);
            out.y = (unsigned char)(cy / cw);
            out.z = (unsigned char)(cz / cw);
          }
          dOutput(idx) = out;
        });
    }

    Kokkos::fence();
    auto tEnd = std::chrono::steady_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tEnd - tStart).count();
    printf("Average kernel execution time %f (s)\n", (ns * 1e-9) / p.repeat);

    {
      auto hv = Kokkos::create_mirror_view(dOutput);
      Kokkos::deep_copy(hv, dOutput);
      for (size_t i = 0; i < outSz; i++) hOutput[i] = hv(i);
    }
  }
  Kokkos::finalize();

  int sx = 0, sy = 0, sz = 0;
  for (uint32_t i = 0; i < p.oWidth * p.oHeight; i++) {
    sx += hOutput[i].x;
    sy += hOutput[i].y;
    sz += hOutput[i].z;
  }
  printf("Checksums %d %d %d\n", sx, sy, sz);

  free(hInput);
  free(hOutput);
  return 0;
}
