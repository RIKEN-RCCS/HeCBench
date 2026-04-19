/*
 * medianfilter – Image Median Filter using binary-search algorithm
 * Kokkos port from the OMP offload version.
 *
 * Original Copyright 1993-2010 NVIDIA Corporation (see EULA)
 *
 * Usage: ./main <image.ppm> <num_cycles>
 *
 * Requires a binary PPM (P6) image.  A 1920x1080 image is recommended.
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

// ---------------------------------------------------------------------------
// uchar4 (RGBA pixel)
// ---------------------------------------------------------------------------
struct alignas(4) uchar4 {
  unsigned char x, y, z, w;
};

// ---------------------------------------------------------------------------
// Simple inline PPM loader (replaces shrLoadPPM4ub)
// Loads a P6 binary PPM and pads to 4-channel (RGBA, w=255).
// Returns true on success; allocates *OutData (caller must free).
// ---------------------------------------------------------------------------
static bool loadPPM4ub(const char *file,
                        unsigned char **OutData,
                        unsigned int *w,
                        unsigned int *h)
{
  FILE *fp = fopen(file, "rb");
  if (!fp) { fprintf(stderr, "Cannot open %s\n", file); return false; }

  char hdr[256];
  // read magic
  if (!fgets(hdr, sizeof(hdr), fp)) { fclose(fp); return false; }
  if (strncmp(hdr, "P6", 2) != 0) {
    fprintf(stderr, "Not a P6 PPM: %s\n", file); fclose(fp); return false;
  }
  // skip comments
  do { if (!fgets(hdr, sizeof(hdr), fp)) { fclose(fp); return false; } }
  while (hdr[0] == '#');
  unsigned int width, height, maxval;
  sscanf(hdr, "%u %u", &width, &height);
  if (!fgets(hdr, sizeof(hdr), fp)) { fclose(fp); return false; }
  sscanf(hdr, "%u", &maxval);

  *w = width; *h = height;
  int size = width * height;
  unsigned char *rgb = (unsigned char*)malloc(size * 3);
  if ((int)fread(rgb, 3, size, fp) != size) {
    fprintf(stderr, "Short read\n"); free(rgb); fclose(fp); return false;
  }
  fclose(fp);

  if (*OutData == nullptr)
    *OutData = (unsigned char*)malloc(size * 4);
  unsigned char *src = rgb, *dst = *OutData;
  for (int i = 0; i < size; i++, src += 3, dst += 4) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = 255;
  }
  free(rgb);
  return true;
}

// ---------------------------------------------------------------------------
// Host median filter reference implementation (replaces MedianFilterHost)
// ---------------------------------------------------------------------------
static void MedianFilterHost(unsigned int *uiInputImage,
                              unsigned int *uiOutputImage,
                              unsigned int uiWidth,
                              unsigned int uiHeight)
{
  for (unsigned int y = 0; y < uiHeight; y++) {
    for (unsigned int x = 0; x < uiWidth; x++) {
      const unsigned int uiZero = 0U;
      float fMedianEstimate[3] = {128.f, 128.f, 128.f};
      float fMinBound[3] = {0.f, 0.f, 0.f};
      float fMaxBound[3]  = {255.f, 255.f, 255.f};

      for (int iSearch = 0; iSearch < 8; iSearch++) {
        unsigned int uiHighCount[3] = {0, 0, 0};
        for (int iRow = -1; iRow <= 1; iRow++) {
          int iLocalOffset = (int)((iRow + y) * uiWidth) + x - 1;
          for (int iCol = -1; iCol <= 1; iCol++, iLocalOffset++) {
            unsigned char *ucRGBA;
            bool valid = ((x + iCol) >= 0) && ((x + iCol) < (int)uiWidth) &&
                         ((y + iRow) >= 0) && ((y + iRow) < (int)uiHeight);
            ucRGBA = valid ? (unsigned char*)&uiInputImage[iLocalOffset]
                           : (unsigned char*)&uiZero;
            uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);
            uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);
            uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);
          }
        }
        for (int c = 0; c < 3; c++) {
          if (uiHighCount[c] > 4) fMinBound[c] = fMedianEstimate[c];
          else                    fMaxBound[c] = fMedianEstimate[c];
          fMedianEstimate[c] = 0.5f * (fMaxBound[c] + fMinBound[c]);
        }
      }
      unsigned int uiPackedPix =
          (0x000000FFu & (unsigned int)(fMedianEstimate[0] + 0.5f))       |
          (0x0000FF00u & (((unsigned int)(fMedianEstimate[1] + 0.5f)) << 8))  |
          (0x00FF0000u & (((unsigned int)(fMedianEstimate[2] + 0.5f)) << 16));
      uiOutputImage[y * uiWidth + x] = uiPackedPix;
    }
  }
}

// ---------------------------------------------------------------------------
// Compare two unsigned int arrays with float threshold
// (replaces shrCompareuit)
// ---------------------------------------------------------------------------
static bool shrCompareuit(const unsigned int *ref, const unsigned int *data,
                           unsigned int len, float epsilon, float threshold)
{
  unsigned int error_count = 0;
  for (unsigned int i = 0; i < len; i++) {
    float diff = fabsf((float)ref[i] - (float)data[i]);
    if (diff > epsilon) error_count++;
  }
  float ratio = (float)error_count / (float)len;
  return ratio <= threshold;
}

// ---------------------------------------------------------------------------
// Round up global_size to the next multiple of group_size
// ---------------------------------------------------------------------------
static size_t shrRoundUp(int group_size, int global_size) {
  int r = global_size % group_size;
  return (r == 0) ? (size_t)global_size
                  : (size_t)(global_size + group_size - r);
}

// ---------------------------------------------------------------------------
// Kokkos GPU median filter kernel
// ---------------------------------------------------------------------------
static double MedianFilterGPU(Kokkos::View<uchar4*>       d_source,
                               Kokkos::View<unsigned int*> d_dest,
                               int iImageWidth, int iImageHeight)
{
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<uchar4*, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const int iBlockDimX      = 16;
  const int iBlockDimY      = 4;
  const int iLocalPixPitch  = iBlockDimX + 2;   // 18

  int szGlobalX = (int)shrRoundUp(iBlockDimX, iImageWidth);
  int szGlobalY = (int)shrRoundUp(iBlockDimY, iImageHeight);

  int iTeamX    = szGlobalX / iBlockDimX;
  int iTeamY    = szGlobalY / iBlockDimY;
  int iNumTeams = iTeamX * iTeamY;
  int iNumThreads = iBlockDimX * iBlockDimY;   // 64

  // Scratch: uchar4 uc4LocalData[iLocalPixPitch * (iBlockDimY + 2)]
  // = 18 * 6 = 108 uchar4 elements
  const int LMEM_H      = iBlockDimY + 2;        // 6
  const int scratch_elems = iLocalPixPitch * LMEM_H;  // 108
  const int scratch_bytes = scratch_elems * (int)sizeof(uchar4);

  Kokkos::TeamPolicy<> policy(iNumTeams, iNumThreads);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  auto t0 = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
    "median_filter",
    policy,
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      ScratchView uc4LocalData(team.team_scratch(0), scratch_elems);

      const int iLocalIdX  = team.team_rank() % iBlockDimX;
      const int iLocalIdY  = team.team_rank() / iBlockDimX;
      const int iGroupIdX  = team.league_rank() % iTeamX;
      const int iGroupIdY  = team.league_rank() / iTeamX;

      const int iBlockX = iBlockDimX;
      const int iBlockY = iBlockDimY;
      const int iImageX = iTeamX * iBlockDimX;   // total image stride

      const int iImagePosX  = iGroupIdX * iBlockX + iLocalIdX;
      const int iDevYPrime  = iGroupIdY * iBlockY + iLocalIdY - 1;

      const int iDevGMEMOffset = iDevYPrime * iImageX + iImagePosX;

      // initial LMEM offset for main read
      int iLocalPixOffset = iLocalIdY * iLocalPixPitch + iLocalIdX + 1;

      // ---- Main read into LMEM ----
      {
        uchar4 zero = {0, 0, 0, 0};
        uchar4 val = ((iDevYPrime > -1) && (iDevYPrime < iImageHeight) &&
                      (iImagePosX < iImageWidth))
                     ? d_source(iDevGMEMOffset) : zero;
        uc4LocalData(iLocalPixOffset) = val;
      }

      // ---- Bottom 2 rows ----
      if (iLocalIdY < 2) {
        iLocalPixOffset += iBlockY * iLocalPixPitch;
        uchar4 zero = {0, 0, 0, 0};
        uchar4 val = (((iDevYPrime + iBlockY) < iImageHeight) &&
                      (iImagePosX < iImageWidth))
                     ? d_source(iDevGMEMOffset + iBlockY * iImageX) : zero;
        uc4LocalData(iLocalPixOffset) = val;
        iLocalPixOffset -= iBlockY * iLocalPixPitch;  // restore
      }

      // ---- Left apron (rightmost thread in X reads left edge) ----
      if (iLocalIdX == iBlockX - 1) {
        int off = iLocalIdY * iLocalPixPitch;
        uchar4 zero = {0, 0, 0, 0};
        uchar4 val = ((iDevYPrime > -1) && (iDevYPrime < iImageHeight) &&
                      (iGroupIdX > 0))
                     ? d_source(iDevYPrime * iImageX + iGroupIdX * iBlockX - 1)
                     : zero;
        uc4LocalData(off) = val;
        if (iLocalIdY < 2) {
          int off2 = off + iBlockY * iLocalPixPitch;
          uchar4 v2 = (((iDevYPrime + iBlockY) < iImageHeight) && (iGroupIdX > 0))
                      ? d_source((iDevYPrime + iBlockY)*iImageX + iGroupIdX*iBlockX - 1)
                      : zero;
          uc4LocalData(off2) = v2;
        }
      }
      // ---- Right apron (leftmost thread in X reads right edge) ----
      else if (iLocalIdX == 0) {
        int off = (iLocalIdY + 1) * iLocalPixPitch - 1;
        uchar4 zero = {0, 0, 0, 0};
        uchar4 val = ((iDevYPrime > -1) && (iDevYPrime < iImageHeight) &&
                      ((iGroupIdX + 1) * iBlockX < iImageWidth))
                     ? d_source(iDevYPrime * iImageX + (iGroupIdX + 1) * iBlockX)
                     : zero;
        uc4LocalData(off) = val;
        if (iLocalIdY < 2) {
          int off2 = off + iBlockY * iLocalPixPitch;
          uchar4 v2 = (((iDevYPrime + iBlockY) < iImageHeight) &&
                       ((iGroupIdX + 1) * iBlockX < iImageWidth))
                      ? d_source((iDevYPrime+iBlockY)*iImageX + (iGroupIdX+1)*iBlockX)
                      : zero;
          uc4LocalData(off2) = v2;
        }
      }

      team.team_barrier();

      // ---- Binary-search median computation ----
      float fMedianEstimate[3] = {128.f, 128.f, 128.f};
      float fMinBound[3]       = {0.f,   0.f,   0.f};
      float fMaxBound[3]       = {255.f, 255.f, 255.f};

      for (int iSearch = 0; iSearch < 8; iSearch++) {
        unsigned int uiHighCount[3] = {0, 0, 0};

        // row 0 of 3x3 neighborhood
        int lpo = iLocalIdY * iLocalPixPitch + iLocalIdX;
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo++).z);
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo++).z);
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo).z);

        // row 1
        lpo += iLocalPixPitch - 2;
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo++).z);
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo++).z);
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo).z);

        // row 2
        lpo += iLocalPixPitch - 2;
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo++).z);
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo++).z);
        uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData(lpo).x);
        uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData(lpo).y);
        uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData(lpo).z);

        for (int c = 0; c < 3; c++) {
          if (uiHighCount[c] > 4) fMinBound[c] = fMedianEstimate[c];
          else                    fMaxBound[c] = fMedianEstimate[c];
          fMedianEstimate[c] = 0.5f * (fMaxBound[c] + fMinBound[c]);
        }
      }

      // Pack and write output (offset by +1 row since iDevYPrime is shifted -1)
      unsigned int uiPackedPix =
          (0x000000FFu & (unsigned int)(fMedianEstimate[0] + 0.5f))          |
          (0x0000FF00u & (((unsigned int)(fMedianEstimate[1] + 0.5f)) << 8)) |
          (0x00FF0000u & (((unsigned int)(fMedianEstimate[2] + 0.5f)) << 16));

      if ((iDevYPrime < iImageHeight) && (iImagePosX < iImageWidth))
        d_dest(iDevGMEMOffset + iImageX) = uiPackedPix;
    });

  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <image.ppm> <repeat>\n", argv[0]);
    return 1;
  }
  const char *cPathAndName = argv[1];
  const int   iCycles      = atoi(argv[2]);

  unsigned int uiImageWidth  = 1920;
  unsigned int uiImageHeight = 1080;

  unsigned int *uiInput  = nullptr;
  unsigned int *uiOutput = nullptr;

  if (!loadPPM4ub(cPathAndName,
                   (unsigned char**)&uiInput,
                   &uiImageWidth, &uiImageHeight)) {
    fprintf(stderr, "Failed to load image: %s\n", cPathAndName);
    return 1;
  }

  printf("Image File\t = %s\nImage Dimensions = %u w x %u h x %zu bpp\n\n",
         cPathAndName, uiImageWidth, uiImageHeight, sizeof(unsigned int) * 8);

  size_t szBuffWords = (size_t)uiImageHeight * uiImageWidth;
  uiOutput = (unsigned int*)malloc(szBuffWords * sizeof(unsigned int));

  uchar4 *uc4Source = (uchar4*)uiInput;

  Kokkos::initialize(argc, argv);
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = ExecSpace::memory_space;

    Kokkos::View<uchar4*,       MemSpace> d_source("source", szBuffWords);
    Kokkos::View<unsigned int*, MemSpace> d_dest  ("dest",   szBuffWords);

    // Upload source
    auto h_source = Kokkos::create_mirror_view(d_source);
    for (size_t i = 0; i < szBuffWords; i++) h_source(i) = uc4Source[i];
    Kokkos::deep_copy(d_source, h_source);

    // Warmup
    MedianFilterGPU(d_source, d_dest, (int)uiImageWidth, (int)uiImageHeight);

    // Timed runs
    printf("Running MedianFilterGPU for %d cycles...\n\n", iCycles);
    double time = 0.0;
    for (int i = 0; i < iCycles; i++)
      time += MedianFilterGPU(d_source, d_dest,
                              (int)uiImageWidth, (int)uiImageHeight);
    printf("Average kernel execution time: %f (s)\n\n", (time * 1e-9f) / iCycles);

    // Download result
    auto h_dest = Kokkos::create_mirror_view(d_dest);
    Kokkos::deep_copy(h_dest, d_dest);
    for (size_t i = 0; i < szBuffWords; i++) uiOutput[i] = h_dest(i);
  }
  Kokkos::finalize();

  // Host reference
  unsigned int *uiGolden = (unsigned int*)malloc(szBuffWords * sizeof(unsigned int));
  MedianFilterHost(uiInput, uiGolden, uiImageWidth, uiImageHeight);

  // Compare
  printf("Comparing GPU Result to CPU Result...\n");
  bool bMatch = shrCompareuit(uiGolden, uiOutput,
                               (unsigned int)(uiImageWidth * uiImageHeight),
                               1.0f, 0.0001f);
  printf("\nGPU Result %s CPU Result within tolerance...\n",
         bMatch ? "matches" : "DOESN'T match");

  free(uiGolden);
  free(uiInput);
  free(uiOutput);

  printf(bMatch ? "PASS\n" : "FAIL\n");
  return bMatch ? EXIT_SUCCESS : EXIT_FAILURE;
}
