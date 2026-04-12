#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define TOL        (0.001)
#define STR_SIZE   (256)
#define MAX_PD     (3.0e6)
#define PRECISION  0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI       100
#define FACTOR_CHIP 0.5

static float t_chip      = 0.0005f;
static float chip_height = 0.016f;
static float chip_width  = 0.016f;
static float amb_temp_g  = 80.0f;

static long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000LL) + tv.tv_usec;
}

static void fatal(const char *s) {
  fprintf(stderr, "Error: %s\n", s);
  exit(1);
}

static void readinput(float *vect, int grid_rows, int grid_cols, int layers, const char *file) {
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if ((fp = fopen(file, "r")) == NULL)
    fatal("The file was not opened");

  for (int i = 0; i <= grid_rows - 1; i++)
    for (int j = 0; j <= grid_cols - 1; j++)
      for (int k = 0; k <= layers - 1; k++) {
        if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
        if (feof(fp)) fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1)) fatal("invalid file format");
        vect[i * grid_cols + j + k * grid_rows * grid_cols] = val;
      }
  fclose(fp);
}

static void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, const char *file) {
  FILE *fp;
  char str[STR_SIZE];

  if ((fp = fopen(file, "w")) == NULL) {
    printf("The file was not opened\n");
    return;
  }

  int index = 0;
  for (int i = 0; i < grid_rows; i++)
    for (int j = 0; j < grid_cols; j++)
      for (int k = 0; k < layers; k++) {
        sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j + k * grid_rows * grid_cols]);
        fputs(str, fp);
        index++;
      }
  fclose(fp);
}

static void computeTempCPU(float *pIn, float *tIn, float *tOut,
    int nx, int ny, int nz, float Cap,
    float Rx, float Ry, float Rz,
    float dt, float amb_temp, int numiter) {
  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;
  cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);

  int i = 0;
  do {
    for (int z = 0; z < nz; z++)
      for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
          int c = x + y * nx + z * nx * ny;
          int w = (x == 0)      ? c : c - 1;
          int e = (x == nx - 1) ? c : c + 1;
          int n = (y == 0)      ? c : c - nx;
          int s = (y == ny - 1) ? c : c + nx;
          int b = (z == 0)      ? c : c - nx * ny;
          int t = (z == nz - 1) ? c : c + nx * ny;
          tOut[c] = tIn[c] * cc + tIn[n] * cn + tIn[s] * cs + tIn[e] * ce + tIn[w] * cw +
                    tIn[t] * ct + tIn[b] * cb + (dt / Cap) * pIn[c] + ct * amb_temp;
        }
    float *temp = tIn; tIn = tOut; tOut = temp;
    i++;
  } while (i < numiter);
}

static float accuracy(float *arr1, float *arr2, int len) {
  float err = 0.0f;
  for (int i = 0; i < len; i++)
    err += (arr1[i] - arr2[i]) * (arr1[i] - arr2[i]);
  return sqrtf(err / len);
}

static void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 7) usage(argc, argv);

  int numCols    = atoi(argv[1]);
  int numRows    = atoi(argv[1]);
  int layers     = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  const char *pfile = argv[4];
  const char *tfile = argv[5];
  const char *ofile = argv[6];

  float dx = chip_height / numRows;
  float dy = chip_width  / numCols;
  float dz = t_chip      / layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0f * K_SI * t_chip * dx);
  float Ry         = dx / (2.0f * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);
  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float stepDivCap = dt / Cap;
  float ce, cw, cn, cs, ct, cb, cc;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;
  cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);

  int size = numCols * numRows * layers;
  float *tIn   = (float *) calloc(size, sizeof(float));
  float *pIn   = (float *) calloc(size, sizeof(float));
  float *tCopy = (float *) malloc(size * sizeof(float));
  float *tOut  = (float *) calloc(size, sizeof(float));

  readinput(tIn, numRows, numCols, layers, tfile);
  readinput(pIn, numRows, numCols, layers, pfile);
  memcpy(tCopy, tIn, size * sizeof(float));

  long long start = get_time();

  Kokkos::initialize(argc, argv);
  {
    using ViewF = Kokkos::View<float *>;
    ViewF d_tIn("d_tIn", size);
    ViewF d_pIn("d_pIn", size);
    ViewF d_tOut("d_tOut", size);

    auto h_tIn = Kokkos::create_mirror_view(d_tIn);
    auto h_pIn = Kokkos::create_mirror_view(d_pIn);

    for (int i = 0; i < size; i++) { h_tIn(i) = tIn[i]; h_pIn(i) = pIn[i]; }
    Kokkos::deep_copy(d_tIn, h_tIn);
    Kokkos::deep_copy(d_pIn, h_pIn);

    int xy = numCols * numRows;
    const float amb = amb_temp_g;

    Kokkos::fence();
    auto kstart = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
      Kokkos::parallel_for("hotspot3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {layers, numRows, numCols}),
        KOKKOS_LAMBDA(int k, int j, int i) {
          int c = i + j * numCols + k * xy;
          int W = (i == 0)          ? c : c - 1;
          int E = (i == numCols - 1)? c : c + 1;
          int N = (j == 0)          ? c : c - numCols;
          int S = (j == numRows - 1)? c : c + numCols;
          int B = (k == 0)          ? c : c - xy;
          int T = (k == layers - 1) ? c : c + xy;
          d_tOut(c) = cc * d_tIn(c) + cw * d_tIn(W) + ce * d_tIn(E)
                    + cs * d_tIn(S) + cn * d_tIn(N) + cb * d_tIn(B)
                    + ct * d_tIn(T) + stepDivCap * d_pIn(c) + ct * amb;
        });
      Kokkos::fence();

      // swap input/output
      ViewF tmp = d_tIn;
      d_tIn  = d_tOut;
      d_tOut = tmp;
    }

    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    printf("Average kernel execution time %f (us)\n", (ktime * 1e-3f) / iterations);

    // After iterations, d_tIn holds the final result (due to swap)
    auto h_tOut = Kokkos::create_mirror_view(d_tIn);
    Kokkos::deep_copy(h_tOut, d_tIn);
    for (int i = 0; i < size; i++) tOut[i] = h_tOut(i);
  }
  Kokkos::finalize();

  long long stop = get_time();

  float *answer = (float *) calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp_g, iterations);

  float acc  = accuracy(tOut, answer, numRows * numCols * layers);
  float time = (float)((stop - start) / (1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n", time);
  printf("Root-mean-square error: %e\n", acc);

  writeoutput(tOut, numRows, numCols, layers, ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}
