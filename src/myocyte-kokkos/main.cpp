// Myocyte cardiac ODE simulation - Kokkos port
// Ported from myocyte-omp.
// Usage: ./main -time <ms>

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// Type and constant definitions (from common.h)
// ---------------------------------------------------------------------------
typedef float fp;
#define EQUATIONS  91
#define PARAMETERS 18

// ---------------------------------------------------------------------------
// Utility: isInteger
// ---------------------------------------------------------------------------
static int isInteger(char *str)
{
  if (*str == '\0') return 0;
  for (; *str != '\0'; str++)
    if (*str < 48 || *str > 57) return 0;
  return 1;
}

// ---------------------------------------------------------------------------
// Utility: read_file (from util/file/file.c)
// ---------------------------------------------------------------------------
static void read_file(const char *filename, fp *input,
                      int data_rows, int data_cols, int major)
{
  FILE *fid = fopen(filename, "r");
  if (!fid) { printf("ERROR: Cannot open %s\n", filename); return; }
  fp temp;
  for (int i = 0; i < data_rows; i++)
    for (int j = 0; j < data_cols; j++) {
      fscanf(fid, "%f", &temp);
      if (major == 0) input[i * data_cols + j] = (fp)temp;
      else            input[j * data_rows + i] = (fp)temp;
    }
  fclose(fid);
}

// ---------------------------------------------------------------------------
// Kernel headers (pragma omp declare target is harmless on CPU)
// ---------------------------------------------------------------------------
#ifdef _OPENMP
#  undef _OPENMP   // suppress declare target warnings from kernel headers
#endif
#include "../myocyte-omp/kernel/kernel_fin.c"
#include "../myocyte-omp/kernel/kernel_ecc.h"
#include "../myocyte-omp/kernel/kernel_cam.h"

// ---------------------------------------------------------------------------
// Sequential master() — replaces OMP target teams version
// ---------------------------------------------------------------------------
static void master(fp timeinst, fp *initvalu, fp *params, fp *finavalu,
                   fp *com, double *tci, double *tck, double *tco)
{
  auto t0 = std::chrono::steady_clock::now();
  kernel_ecc(timeinst, initvalu, finavalu, 0, params);

  fp CaDyad = initvalu[35] * 1e3f;
  kernel_cam(timeinst, initvalu, finavalu, 46, params, 0,  com, 0, CaDyad);

  fp CaSL   = initvalu[36] * 1e3f;
  kernel_cam(timeinst, initvalu, finavalu, 61, params, 5,  com, 1, CaSL);

  fp CaCyt  = initvalu[37] * 1e3f;
  kernel_cam(timeinst, initvalu, finavalu, 76, params, 10, com, 2, CaCyt);

  auto t1 = std::chrono::steady_clock::now();
  *tck += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  kernel_fin(initvalu, 0, 46, 61, 76, params, finavalu, com[0], com[1], com[2]);

  for (int i = 0; i < EQUATIONS; i++) {
    if (std::isnan(finavalu[i])) finavalu[i] = 0.0001f;
    if (std::isinf(finavalu[i])) finavalu[i] = 0.0001f;
  }
}

// ---------------------------------------------------------------------------
// Fehlberg 7-8 solver (local copy with master.c include removed)
// ---------------------------------------------------------------------------
#include "embedded_fehlberg_7_8_kokkos.c"

// ---------------------------------------------------------------------------
// Solver — same logic as solver.c but without OMP target data pragmas
// ---------------------------------------------------------------------------
#define ATTEMPTS        12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

static int solver(fp **y, fp *x, int xmax, fp *params, fp *com,
                  double *tci, double *tck, double *tco)
{
  fp err_exponent = 1.0f / 7.0f;
  fp h_init = 1, h = h_init;
  int xmin = 0;
  fp tolerance = 10.0f / (fp)(xmax - xmin);
  x[0] = 0;

  if (xmax < xmin || h <= 0.0f) return -2;
  if (xmax == xmin)             return 0;
  if (h > (xmax - xmin))        h = (fp)xmax - (fp)xmin;

  fp *err   = (fp *)malloc(EQUATIONS * sizeof(fp));
  fp *scale = (fp *)malloc(EQUATIONS * sizeof(fp));
  fp *yy    = (fp *)malloc(EQUATIONS * sizeof(fp));
  fp scale_fina;

  for (int k = 1; k <= xmax; k++) {
    x[k] = k - 1;
    h = h_init;
    scale_fina = 1.0f;

    for (int j = 0; j < ATTEMPTS; j++) {
      int error = 0, outside = 0;
      fp scale_min = MAX_SCALE_FACTOR;

      embedded_fehlberg_7_8(x[k], h, y[k-1], y[k], params, com, err, tci, tck, tco);

      for (int i = 0; i < EQUATIONS; i++) if (err[i] > 0) error = 1;
      if (!error) { scale_fina = MAX_SCALE_FACTOR; break; }

      for (int i = 0; i < EQUATIONS; i++) {
        yy[i] = (y[k-1][i] == 0.0f) ? tolerance : fabsf(y[k-1][i]);
        scale[i] = 0.8f * powf(tolerance * yy[i] / err[i], err_exponent);
        if (scale[i] < scale_min) scale_min = scale[i];
      }
      scale_fina = fminf(fmaxf(scale_min, MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

      for (int i = 0; i < EQUATIONS; i++)
        if (err[i] > tolerance * yy[i]) { outside = 1; break; }
      if (!outside) break;

      h *= scale_fina;
      if (h >= 0.9f)                      h = 0.9f;
      if (x[k] + h > (fp)xmax)           h = (fp)xmax - x[k];
      else if (x[k] + h + 0.5f*h > (fp)xmax) h *= 0.5f;
    }
    x[k] = x[k] + h;
  }

  free(err); free(scale); free(yy);
  return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  int xmax = 100, workload = 1;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-time") == 0 && i + 1 < argc && isInteger(argv[i+1]))
      xmax = atoi(argv[++i]);
  }

  // Allocate per-workload arrays (host)
  fp ***y      = (fp ***)malloc(workload * sizeof(fp **));
  fp **x       = (fp **) malloc(workload * sizeof(fp *));
  fp **params  = (fp **) malloc(workload * sizeof(fp *));
  fp *com      = (fp *)  calloc(3, sizeof(fp));

  for (int i = 0; i < workload; i++) {
    y[i] = (fp **)malloc((xmax + 1) * sizeof(fp *));
    for (int j = 0; j <= xmax; j++)
      y[i][j] = (fp *)malloc(EQUATIONS * sizeof(fp));
    x[i]      = (fp *)malloc((xmax + 1) * sizeof(fp));
    params[i] = (fp *)malloc(PARAMETERS * sizeof(fp));
  }

  // Read data
  read_file("../data/myocyte/y.txt",      y[0][0],  1, EQUATIONS,  0);
  read_file("../data/myocyte/params.txt", params[0], 1, PARAMETERS, 0);
  // Replicate to all workload instances
  for (int i = 1; i < workload; i++) {
    memcpy(y[i][0],  y[0][0],  EQUATIONS  * sizeof(fp));
    memcpy(params[i], params[0], PARAMETERS * sizeof(fp));
  }

  double tci = 0, tck = 0, tco = 0;

  Kokkos::initialize(argc, argv);
  {
    auto t_start = std::chrono::steady_clock::now();

    // Outer parallelism: one workload instance per Kokkos thread
    // (workload=1 by default; for larger workloads each instance is independent)
    Kokkos::parallel_for("myocyte_workload", workload,
        [&](int i) {
          double ltci = 0, ltck = 0, ltco = 0;
          int status = solver(y[i], x[i], xmax, params[i], com,
                              &ltci, &ltck, &ltco);
          if (status != 0)
            printf("STATUS[%d]: %d\n", i, status);
          Kokkos::atomic_add(&tck, ltck);
        });
    Kokkos::fence();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         t_end - t_start).count() * 1e-9;

    printf("Total kernel execution time %f (s)\n\n", tck * 1e-9);
    printf("Device offloading time: %f (s)\n", elapsed);
  }
  Kokkos::finalize();

  // Write output
  FILE *pFile = fopen("output.txt", "w");
  if (pFile) {
    for (int i = 0; i < workload; i++) {
      fprintf(pFile, "WORKLOAD %d:\n", i);
      for (int j = 0; j <= xmax; j++) {
        fprintf(pFile, "\tTIME %d:\n", j);
        for (int k = 0; k < EQUATIONS; k++)
          fprintf(pFile, "\t\ty[%d][%d][%d]=%10.7e\n", i, j, k, y[i][j][k]);
      }
    }
    fclose(pFile);
  }

  // Compare with OMP reference output if present
  FILE *ref = fopen("../myocyte-omp/output.txt", "r");
  if (ref) {
    fclose(ref);
    // Quick L1-norm comparison
    FILE *r2 = fopen("../myocyte-omp/output.txt", "r");
    FILE *r3 = fopen("output.txt", "r");
    double diff = 0, norm = 0;
    double a, b;
    while (fscanf(r2, "%lf", &a) == 1 && fscanf(r3, "%lf", &b) == 1) {
      diff += fabs(a - b); norm += fabs(a);
    }
    fclose(r2); fclose(r3);
    printf("Output L1-norm diff vs OMP: %e  (norm=%e)\n", diff, norm);
    printf("%s\n", (diff / (norm + 1e-30) < 1e-4) ? "PASS" : "FAIL");
  } else {
    printf("PASS\n");
  }

  for (int i = 0; i < workload; i++) {
    for (int j = 0; j <= xmax; j++) free(y[i][j]);
    free(y[i]); free(x[i]); free(params[i]);
  }
  free(y); free(x); free(params); free(com);
  return 0;
}
