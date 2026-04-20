#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// -----------------------------------------------------------------------
// Command-line defaults
// -----------------------------------------------------------------------
static int  g_width      = 512;
static int  g_height     = 512;
static int  g_components = 3;
static int  g_bitDepth   = 8;
static int  g_dwtLvls    = 3;
static bool g_forward    = true;
static char g_srcFile[256] = "";
static char g_outFile[256] = "out.dwt";

// -----------------------------------------------------------------------
// Parse "WxH" dimension string
// -----------------------------------------------------------------------
static void parseDim(const char *s, int &w, int &h) {
  sscanf(s, "%dx%d", &w, &h);
}

static void usage() {
  printf("dwt2d [options] src_img.rgb <out.dwt>\n"
         "  -d WxH  dimensions (e.g. 1920x1080)\n"
         "  -c N    components (default 3)\n"
         "  -b N    bit depth (default 8)\n"
         "  -l N    DWT levels (default 3)\n"
         "  -f      forward transform\n"
         "  -r      reverse transform\n"
         "  -5      5/3 transform\n");
}

// -----------------------------------------------------------------------
// 1-D in-place forward 5/3 DWT (floating point lifting)
//   data[0..n-1]  — processed in-place
// -----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
static void dwt1d_forward(float *x, int n) {
  if (n < 2) return;
  // Predict step: update odd samples
  for (int i = 1; i < n - 1; i += 2)
    x[i] -= 0.5f * (x[i - 1] + x[i + 1]);
  if ((n & 1) == 0)  // even length: last sample is odd
    x[n - 1] -= x[n - 2];

  // Update step: update even samples
  x[0] += 0.5f * x[1];
  for (int i = 2; i < n - 1; i += 2)
    x[i] += 0.25f * (x[i - 1] + x[i + 1]);
  if ((n & 1) == 1 && n > 1)
    x[n - 1] += 0.5f * x[n - 2];
}

// -----------------------------------------------------------------------
// Apply one stage of 2-D DWT to the sub-image [0..h-1][0..w-1]
// stored row-major in `data` (stride = fullWidth).
// Kokkos parallel over rows, then over columns.
// -----------------------------------------------------------------------
static void applyDWT2D(Kokkos::View<float *> data,
                       int w, int h, int fullWidth) {
  // --- Row transforms ---
  Kokkos::parallel_for(
    "dwt_rows", h,
    KOKKOS_LAMBDA(int row) {
      // scratch buffer for the row
      float tmp[4096];  // supports up to 4096-wide images per stage
      for (int c = 0; c < w; c++)
        tmp[c] = data(row * fullWidth + c);
      // in-place DWT
      // Predict
      for (int i = 1; i < w - 1; i += 2)
        tmp[i] -= 0.5f * (tmp[i - 1] + tmp[i + 1]);
      if ((w & 1) == 0)
        tmp[w - 1] -= tmp[w - 2];
      // Update
      tmp[0] += 0.5f * tmp[1];
      for (int i = 2; i < w - 1; i += 2)
        tmp[i] += 0.25f * (tmp[i - 1] + tmp[i + 1]);
      if ((w & 1) == 1 && w > 1)
        tmp[w - 1] += 0.5f * tmp[w - 2];

      for (int c = 0; c < w; c++)
        data(row * fullWidth + c) = tmp[c];
    });
  Kokkos::fence();

  // --- Column transforms ---
  Kokkos::parallel_for(
    "dwt_cols", w,
    KOKKOS_LAMBDA(int col) {
      float tmp[4096];
      for (int r = 0; r < h; r++)
        tmp[r] = data(r * fullWidth + col);
      // Predict
      for (int i = 1; i < h - 1; i += 2)
        tmp[i] -= 0.5f * (tmp[i - 1] + tmp[i + 1]);
      if ((h & 1) == 0)
        tmp[h - 1] -= tmp[h - 2];
      // Update
      tmp[0] += 0.5f * tmp[1];
      for (int i = 2; i < h - 1; i += 2)
        tmp[i] += 0.25f * (tmp[i - 1] + tmp[i + 1]);
      if ((h & 1) == 1 && h > 1)
        tmp[h - 1] += 0.5f * tmp[h - 2];

      for (int r = 0; r < h; r++)
        data(r * fullWidth + col) = tmp[r];
    });
  Kokkos::fence();
}

// -----------------------------------------------------------------------
// Multi-stage 2-D DWT
// -----------------------------------------------------------------------
static void nStage2dDWT(Kokkos::View<float *> data,
                        int pixWidth, int pixHeight, int levels) {
  int w = pixWidth, h = pixHeight;
  for (int lvl = 0; lvl < levels; lvl++) {
    applyDWT2D(data, w, h, pixWidth);
    w = (w + 1) / 2;
    h = (h + 1) / 2;
  }
}

// -----------------------------------------------------------------------
// Write component to file (convert float→uchar)
// -----------------------------------------------------------------------
static void writeComponent(const float *src, int pixels, const char *filename,
                            const char *suffix) {
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s%s", filename, suffix);
  auto *result = new unsigned char[pixels];
  for (int i = 0; i < pixels; i++) {
    int v = (int)(src[i] + 128.0f);
    if (v > 255) v = 255;
    if (v < 0)   v = 0;
    result[i] = (unsigned char)v;
  }
  int fd = open(outfile, O_CREAT | O_WRONLY, 0644);
  if (fd != -1) {
    printf("\nWriting to %s (%d x %d)\n", outfile, g_width, g_height);
    write(fd, result, pixels);
    close(fd);
  }
  delete[] result;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
  // Parse args
  int argi = 1;
  while (argi < argc) {
    if (!strcmp(argv[argi], "-d") && argi + 1 < argc) {
      parseDim(argv[++argi], g_width, g_height);
    } else if (!strcmp(argv[argi], "-c") && argi + 1 < argc) {
      g_components = atoi(argv[++argi]);
    } else if (!strcmp(argv[argi], "-b") && argi + 1 < argc) {
      g_bitDepth = atoi(argv[++argi]);
    } else if (!strcmp(argv[argi], "-l") && argi + 1 < argc) {
      g_dwtLvls = atoi(argv[++argi]);
    } else if (!strcmp(argv[argi], "-f")) {
      g_forward = true;
    } else if (!strcmp(argv[argi], "-r")) {
      g_forward = false;
    } else if (!strcmp(argv[argi], "-5")) {
      // 5/3 is the only implemented filter — ignore
    } else if (!strcmp(argv[argi], "--help") || !strcmp(argv[argi], "-h")) {
      usage(); return 0;
    } else if (argv[argi][0] != '-') {
      if (g_srcFile[0] == '\0')
        strncpy(g_srcFile, argv[argi], sizeof(g_srcFile) - 1);
      else
        strncpy(g_outFile, argv[argi], sizeof(g_outFile) - 1);
    }
    argi++;
  }

  Kokkos::initialize(argc, argv);
  {
    const int pixels    = g_width * g_height;
    const int imgBytes  = pixels * g_components;

    // Load or generate source image
    auto *srcImg = new unsigned char[imgBytes];
    bool loaded = false;
    if (g_srcFile[0] != '\0') {
      char path[512];
      snprintf(path, sizeof(path), "../data/dwt2d/%s", g_srcFile);
      int fd = open(path, O_RDONLY, 0644);
      if (fd == -1) fd = open(g_srcFile, O_RDONLY, 0644);
      if (fd != -1) {
        if (read(fd, srcImg, imgBytes) == imgBytes) loaded = true;
        close(fd);
      }
    }
    if (!loaded) {
      printf("Input file not found — generating synthetic image "
             "(%dx%d, %d components)\n", g_width, g_height, g_components);
      for (int i = 0; i < imgBytes; i++)
        srcImg[i] = (unsigned char)(i * 7 + 13 * (i / g_width));
    }

    // Allocate per-component Kokkos Views
    // We work in float for the DWT
    const int nComp = g_components;
    std::vector<Kokkos::View<float *>> compView(nComp);
    for (int c = 0; c < nComp; c++)
      compView[c] = Kokkos::View<float *>("comp", pixels);

    // Separate RGB components (parallel)
    for (int c = 0; c < nComp; c++) {
      auto view = compView[c];
      // Mirror src to device via host mirror
      auto h_view = Kokkos::create_mirror_view(view);
      for (int i = 0; i < pixels; i++)
        h_view(i) = (float)srcImg[i * nComp + c] - 128.0f;
      Kokkos::deep_copy(view, h_view);
    }

    // --- Timed DWT ---
    Kokkos::fence();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < nComp; c++)
      nStage2dDWT(compView[c], g_width, g_height, g_dwtLvls);

    Kokkos::fence();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("DWT processing time: %.3f (ms)\n", ms);

#ifdef OUTPUT
    // Write results
    for (int c = 0; c < nComp; c++) {
      auto h_view = Kokkos::create_mirror_view(compView[c]);
      Kokkos::deep_copy(h_view, compView[c]);
      const char *sfx = (c == 0) ? ".r" : (c == 1) ? ".g" : ".b";
      writeComponent(h_view.data(), pixels, g_outFile, sfx);
    }
#endif

    delete[] srcImg;
  }
  Kokkos::finalize();
  return 0;
}
