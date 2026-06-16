#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>

// Color encoding of flow vectors
// adapted from the color circle idea described at
//   http://members.shaw.ca/quadibloc/other/colint.htm

#define RY  15
#define YG  6
#define GC  4
#define CB  11
#define BM  13
#define MR  6
#define MAXCOLS  (RY + YG + GC + CB + BM + MR)
typedef unsigned char uchar;

KOKKOS_INLINE_FUNCTION
void setcols(int cw[MAXCOLS][3], int r, int g, int b, int k)
{
  cw[k][0] = r;
  cw[k][1] = g;
  cw[k][2] = b;
}

KOKKOS_INLINE_FUNCTION
void computeColor(float fx, float fy, uchar *pix)
{
  int cw[MAXCOLS][3];  // color wheel

  int i;
  int k = 0;
  for (i = 0; i < RY; i++) setcols(cw, 255,        255*i/RY,   0,           k++);
  for (i = 0; i < YG; i++) setcols(cw, 255-255*i/YG, 255,     0,           k++);
  for (i = 0; i < GC; i++) setcols(cw, 0,           255,     255*i/GC,     k++);
  for (i = 0; i < CB; i++) setcols(cw, 0,           255-255*i/CB, 255,     k++);
  for (i = 0; i < BM; i++) setcols(cw, 255*i/BM,   0,           255,       k++);
  for (i = 0; i < MR; i++) setcols(cw, 255,         0,         255-255*i/MR, k++);

  float rad = Kokkos::sqrt(fx * fx + fy * fy);
  float a = Kokkos::atan2(-fy, -fx) / (float)M_PI;
  float fk = (a + 1.f) / 2.f * (MAXCOLS - 1);
  int k0 = (int)fk;
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  for (int b = 0; b < 3; b++) {
    float col0 = cw[k0][b] / 255.f;
    float col1 = cw[k1][b] / 255.f;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col);
    else
      col *= .75f;
    pix[2 - b] = (int)(255.f * col);
  }
}

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <range> <size> <repeat>\n", argv[0]);
    return 1;
  }
  const float truerange = atof(argv[1]);
  const int size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  float range = 1.04f * truerange;
  const int half_size = size / 2;

  size_t imgSize = (size_t)size * size * 3;
  uchar* pix   = (uchar*) malloc(imgSize);
  uchar* d_pix = (uchar*) malloc(imgSize);

  memset(pix,   0, imgSize);
  memset(d_pix, 0, imgSize);

  // Reference serial computation
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      float fx = (float)x / (float)half_size * range - range;
      float fy = (float)y / (float)half_size * range - range;
      if (x == half_size || y == half_size) continue;
      size_t idx = ((size_t)y * size + x) * 3;
      computeColor(fx / truerange, fy / truerange, pix + idx);
    }
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<uchar*> d_img("d_img", imgSize);
    auto h_img = Kokkos::create_mirror_view(d_img);
    Kokkos::deep_copy(d_img, 0);

    printf("Start execution on a device\n");

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("colorwheel", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {size, size}),
          KOKKOS_LAMBDA(int y, int x) {
        float fx = (float)x / (float)half_size * range - range;
        float fy = (float)y / (float)half_size * range - range;
        if (x != half_size && y != half_size) {
          size_t idx = ((size_t)y * size + x) * 3;
          computeColor(fx / truerange, fy / truerange, d_img.data() + idx);
        }
      });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time : %f (ms)\n", (time * 1e-6f) / repeat);

    Kokkos::deep_copy(h_img, d_img);
    for (size_t i = 0; i < imgSize; i++) d_pix[i] = h_img(i);
  }
  Kokkos::finalize();

  // verify
  int fail = memcmp(pix, d_pix, imgSize);
  if (fail) {
    int max_error = 0;
    for (size_t i = 0; i < imgSize; i++) {
      int e = abs(d_pix[i] - pix[i]);
      if (e > max_error) max_error = e;
    }
    printf("Maximum error between host and device results: %d\n", max_error);
  } else {
    printf("%s\n", "PASS");
  }

  free(d_pix);
  free(pix);
  return 0;
}
