// Kokkos port of morphology-omp
// Binary morphology (erode/dilate) using the van Herk/Gil-Werman algorithm.
// Each parallel work-item processes one complete sel-width pixel group serially,
// which avoids team-size restrictions on CPU while preserving the same algorithm.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define BLACK 0
#define WHITE 255

// Maximum sel size supported (must be >= any hsize/vsize used at runtime).
// Stack buffers inside the lambda are sized to this constant.
#define MAX_SEL 128

static inline int roundUp(int x, int y) { return (x + y - 1) / y; }

enum class MorphOpType { ERODE, DILATE };

template<MorphOpType opType>
KOKKOS_INLINE_FUNCTION unsigned char elementOp(unsigned char lhs, unsigned char rhs) {
  if constexpr (opType == MorphOpType::ERODE) return lhs < rhs ? lhs : rhs;
  else                                         return lhs > rhs ? lhs : rhs;
}

template<MorphOpType opType>
KOKKOS_INLINE_FUNCTION unsigned char borderValue() {
  if constexpr (opType == MorphOpType::ERODE) return (unsigned char)BLACK;
  else                                         return (unsigned char)WHITE;
}

// Serial two-way scan: same recurrence as the parallel SIMT version but
// processed by a single thread.  Processing order is chosen so reads always
// see the pre-step values, matching the parallel barrier-separated semantics.
//   buffer [0..selSize-1]     : first half of pixel window
//   buffer [selSize..2*selSize-1]: second half of pixel window
//   opArray[0..selSize-1]     : forward suffix-scan result after the call
//   opArray[selSize..2*selSize-1]: backward prefix-scan result after the call
template<MorphOpType opType>
KOKKOS_INLINE_FUNCTION
void twoWayScanSerial(const unsigned char* buffer, unsigned char* opArray, int selSize)
{
  for (int tx = 0; tx < selSize; tx++) {
    opArray[tx]           = buffer[tx];
    opArray[tx + selSize] = buffer[tx + selSize];
  }
  for (int offset = 1; offset < selSize; offset *= 2) {
    // Forward suffix scan (indices 0..selSize-1) – process in INCREASING order
    // so each tx reads opArray[tx+offset] before that slot is written.
    for (int tx = 0; tx < selSize; tx++) {
      if (tx <= selSize - 1 - offset)
        opArray[tx] = elementOp<opType>(opArray[tx], opArray[tx + offset]);
    }
    // Backward prefix scan (indices selSize..2*selSize-2) – process in DECREASING
    // order so each tx reads opArray[tx+selSize-1-offset] before that slot is written.
    for (int tx = selSize - 1; tx >= 0; tx--) {
      if (tx >= offset)
        opArray[tx + selSize - 1] =
            elementOp<opType>(opArray[tx + selSize - 1],
                              opArray[tx + selSize - 1 - offset]);
    }
  }
}

template<MorphOpType opType>
double morphology(Kokkos::View<unsigned char*> img_d,
                  Kokkos::View<unsigned char*> tmp_d,
                  int width, int height, int hsize, int vsize)
{
  Kokkos::deep_copy(tmp_d, (unsigned char)0);

  const int gx_h = roundUp(width,  hsize);  // groups in x for horizontal pass
  const int gy_h = height;                   // one group per row

  const int gx_v = width;                    // one group per column
  const int gy_v = roundUp(height, vsize);   // groups in y for vertical pass

  auto t_start = std::chrono::steady_clock::now();

  // ---- Horizontal pass ----
  // Each work-item covers one row-segment of hsize pixels (bx, by).
  Kokkos::parallel_for("morph_h", gx_h * gy_h, KOKKOS_LAMBDA(int group_id) {
    const int bx = group_id % gx_h;
    const int by = group_id / gx_h;

    if (by >= height) return;

    unsigned char buffer [2 * MAX_SEL];
    unsigned char opArray[2 * MAX_SEL];

    for (int tx = 0; tx < hsize; tx++) {
      const int tidx = tx + bx * hsize;
      buffer[tx] = (tidx < width)
                   ? img_d[by * width + tidx]
                   : borderValue<opType>();
      buffer[tx + hsize] = (tidx + hsize < width)
                           ? img_d[by * width + tidx + hsize]
                           : borderValue<opType>();
    }

    twoWayScanSerial<opType>(buffer, opArray, hsize);

    for (int tx = 0; tx < hsize; tx++) {
      const int tidx = tx + bx * hsize;
      if (tidx < width && tidx + hsize / 2 < width - hsize / 2)
        tmp_d[by * width + tidx + hsize / 2] =
            elementOp<opType>(opArray[tx], opArray[tx + hsize - 1]);
    }
  });
  Kokkos::fence();

  // ---- Vertical pass ----
  // Each work-item covers one column-segment of vsize pixels (bx, by).
  Kokkos::parallel_for("morph_v", gx_v * gy_v, KOKKOS_LAMBDA(int group_id) {
    const int bx = group_id % gx_v;
    const int by = group_id / gx_v;

    if (bx >= width) return;

    unsigned char buffer [2 * MAX_SEL];
    unsigned char opArray[2 * MAX_SEL];

    for (int ty = 0; ty < vsize; ty++) {
      const int tidy = ty + by * vsize;
      buffer[ty] = (tidy < height)
                   ? tmp_d[tidy * width + bx]
                   : borderValue<opType>();
      buffer[ty + vsize] = (tidy + vsize < height)
                           ? tmp_d[(tidy + vsize) * width + bx]
                           : borderValue<opType>();
    }

    twoWayScanSerial<opType>(buffer, opArray, vsize);

    for (int ty = 0; ty < vsize; ty++) {
      const int tidy = ty + by * vsize;
      if (tidy < height) {
        if (tidy + vsize / 2 < height - vsize / 2)
          img_d[(tidy + vsize / 2) * width + bx] =
              elementOp<opType>(opArray[ty], opArray[ty + vsize - 1]);
        if (tidy < vsize / 2 || tidy >= height - vsize / 2)
          img_d[tidy * width + bx] = borderValue<opType>();
      }
    }
  });
  Kokkos::fence();

  auto t_end = std::chrono::steady_clock::now();
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <kernel_width> <kernel_height> <image_width> <image_height> <repeat>\n",
           argv[0]);
    return 1;
  }

  const int hsize  = atoi(argv[1]);
  const int vsize  = atoi(argv[2]);
  const int width  = atoi(argv[3]);
  const int height = atoi(argv[4]);
  const int repeat = atoi(argv[5]);

  if (hsize > MAX_SEL || vsize > MAX_SEL) {
    fprintf(stderr, "Error: sel size exceeds MAX_SEL=%d\n", MAX_SEL);
    return 1;
  }

  const unsigned int memSize = width * height * sizeof(unsigned char);

  Kokkos::initialize(argc, argv);
  {
    std::vector<unsigned char> srcImg(width * height, BLACK);
    srcImg[(height / 2 - 1) * width + (width / 2 - 1)] = WHITE;

    Kokkos::View<unsigned char*> img_d("img_d", width * height);
    Kokkos::View<unsigned char*> tmp_d("tmp_d", width * height);

    auto img_h = Kokkos::create_mirror_view(img_d);
    std::memcpy(img_h.data(), srcImg.data(), memSize);
    Kokkos::deep_copy(img_d, img_h);

    double dilate_time = 0.0, erode_time = 0.0;
    for (int n = 0; n < repeat; n++) {
      dilate_time += morphology<MorphOpType::DILATE>(img_d, tmp_d, width, height, hsize, vsize);
      erode_time  += morphology<MorphOpType::ERODE> (img_d, tmp_d, width, height, hsize, vsize);
    }

    printf("Average kernel execution time (dilate): %f (s)\n",
           (dilate_time * 1e-9) / repeat);
    printf("Average kernel execution time (erode):  %f (s)\n",
           (erode_time  * 1e-9) / repeat);

    Kokkos::deep_copy(img_h, img_d);
    int s = 0;
    for (unsigned int i = 0; i < (unsigned int)(width * height); i++)
      s += img_h[i];
    printf("%s\n", s == WHITE ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
