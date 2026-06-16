/*
 * Tissue simulation (biophysical tissue model).
 * Each tissue point accumulates contributions from all other tissue points.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

void tissue(
    Kokkos::View<const int*>   d_tisspoints,
    Kokkos::View<const float*> d_gtt,
    Kokkos::View<const float*> d_gbartt,
    Kokkos::View<float*>       d_ct,
    Kokkos::View<const float*> d_ctprev,
    Kokkos::View<const float*> d_qt,
    int nnt, int nntDev, int isp)
{
  // For each target tissue point, sum contributions from all other tissue points.
  // This is an N^2 computation without the step-based partitioning.
  Kokkos::parallel_for("tissue", nnt, KOKKOS_LAMBDA(int itp) {
    int nnt2 = 2 * nnt;
    int ix = d_tisspoints(itp);
    int iy = d_tisspoints(itp + nnt);
    int iz = d_tisspoints(itp + nnt2);
    float p = 0.f;
    for (int jtp = 0; jtp < nnt; jtp++) {
      int jx = d_tisspoints(jtp);
      int jy = d_tisspoints(jtp + nnt);
      int jz = d_tisspoints(jtp + nnt2);
      int ixyz = abs(jx - ix) + abs(jy - iy) + abs(jz - iz) + (isp - 1) * nntDev;
      p += d_gtt(ixyz) * d_ctprev(jtp) + d_gbartt(ixyz) * d_qt(jtp);
    }
    d_ct(itp) = p;
  });
  Kokkos::fence();
}

void reference(
    const int*   d_tisspoints,
    const float* d_gtt,
    const float* d_gbartt,
          float* d_ct,
    const float* d_ctprev,
    const float* d_qt,
    int nnt, int nntDev, int step, int isp)
{
  for (int i = 0; i < step * nnt; i++) {
    int jtp, ixyz, ix, iy, iz, jx, jy, jz, istep;
    int nnt2 = 2 * nnt;
    float p = 0.f;
    int itp = i / step;
    int itp1 = i % step;
    if (itp < nnt) {
      ix = d_tisspoints[itp];
      iy = d_tisspoints[itp + nnt];
      iz = d_tisspoints[itp + nnt2];
      for (jtp = itp1; jtp < nnt; jtp += step) {
        jx = d_tisspoints[jtp];
        jy = d_tisspoints[jtp + nnt];
        jz = d_tisspoints[jtp + nnt2];
        ixyz = abs(jx - ix) + abs(jy - iy) + abs(jz - iz) + (isp - 1) * nntDev;
        p += d_gtt[ixyz] * d_ctprev[jtp] + d_gbartt[ixyz] * d_qt[jtp];
      }
      if (itp1 == 0) d_ct[itp] = p;
    }
    for (istep = 1; istep < step; istep++)
      if (itp1 == istep && itp < nnt) d_ct[itp] += p;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <dimension of a 3D grid> <repeat>\n", argv[0]);
    return 1;
  }
  const int dim = atoi(argv[1]);
  if (dim > 32) { printf("Maximum dimension is 32\n"); return 1; }
  const int repeat = atoi(argv[2]);

  const int nnt    = dim * dim * dim;
  const int nntDev = 32 * 32 * 32;
  const int nsp    = 2;

  int   *h_tisspoints = (int*)   malloc(3 * nntDev * sizeof(int));
  float *h_gtt        = (float*) malloc(nsp * nntDev * sizeof(float));
  float *h_gbartt     = (float*) malloc(nsp * nntDev * sizeof(float));
  float *h_ct         = (float*) malloc(nntDev * sizeof(float));
  float *h_ctprev     = (float*) malloc(nntDev * sizeof(float));
  float *h_qt         = (float*) malloc(nntDev * sizeof(float));
  float *h_ct_gold    = (float*) malloc(nntDev * sizeof(float));

  srand(123);
  for (int i = 0; i < 3 * nntDev; i++) h_tisspoints[i] = rand() % (nntDev / 3);
  for (int i = 0; i < nsp * nntDev; i++) { h_gtt[i] = rand() / (float)RAND_MAX; h_gbartt[i] = rand() / (float)RAND_MAX; }
  for (int i = 0; i < nntDev; i++) {
    h_ct[i] = h_ct_gold[i] = 0;
    h_ctprev[i] = rand() / (float)RAND_MAX;
    h_qt[i] = rand() / (float)RAND_MAX;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*>   d_tp("d_tp",   3 * nntDev);
    Kokkos::View<float*> d_gtt("d_gtt", nsp * nntDev);
    Kokkos::View<float*> d_gbartt("d_gbartt", nsp * nntDev);
    Kokkos::View<float*> d_ct("d_ct",   nntDev);
    Kokkos::View<float*> d_ctprev("d_ctprev", nntDev);
    Kokkos::View<float*> d_qt("d_qt",   nntDev);

    auto h_tp_v      = Kokkos::create_mirror_view(d_tp);
    auto h_gtt_v     = Kokkos::create_mirror_view(d_gtt);
    auto h_gbartt_v  = Kokkos::create_mirror_view(d_gbartt);
    auto h_ct_v      = Kokkos::create_mirror_view(d_ct);
    auto h_ctprev_v  = Kokkos::create_mirror_view(d_ctprev);
    auto h_qt_v      = Kokkos::create_mirror_view(d_qt);

    for (int i = 0; i < 3 * nntDev; i++) h_tp_v(i) = h_tisspoints[i];
    for (int i = 0; i < nsp * nntDev; i++) { h_gtt_v(i) = h_gtt[i]; h_gbartt_v(i) = h_gbartt[i]; }
    for (int i = 0; i < nntDev; i++) { h_ct_v(i) = 0; h_ctprev_v(i) = h_ctprev[i]; h_qt_v(i) = h_qt[i]; }

    Kokkos::deep_copy(d_tp, h_tp_v); Kokkos::deep_copy(d_gtt, h_gtt_v);
    Kokkos::deep_copy(d_gbartt, h_gbartt_v); Kokkos::deep_copy(d_ct, h_ct_v);
    Kokkos::deep_copy(d_ctprev, h_ctprev_v); Kokkos::deep_copy(d_qt, h_qt_v);

    // Warmup and verify
    tissue(d_tp, d_gtt, d_gbartt, d_ct, d_ctprev, d_qt, nnt, nntDev, 1);
    tissue(d_tp, d_gtt, d_gbartt, d_ct, d_ctprev, d_qt, nnt, nntDev, 2);

    int step = 4;
    reference(h_tisspoints, h_gtt, h_gbartt, h_ct_gold, h_ctprev, h_qt, nnt, nntDev, step, 1);
    reference(h_tisspoints, h_gtt, h_gbartt, h_ct_gold, h_ctprev, h_qt, nnt, nntDev, step, 2);

    Kokkos::deep_copy(h_ct_v, d_ct);
    bool ok = true;
    for (int i = 0; i < nntDev; i++) {
      if (fabsf(h_ct_v(i) - h_ct_gold[i]) > 1e-2f) { ok = false; break; }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    // Timing
    Kokkos::deep_copy(d_ct, h_ct_v);  // reset
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      tissue(d_tp, d_gtt, d_gbartt, d_ct, d_ctprev, d_qt, nnt, nntDev, 1);
      tissue(d_tp, d_gtt, d_gbartt, d_ct, d_ctprev, d_qt, nnt, nntDev, 2);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }
  Kokkos::finalize();

  free(h_tisspoints); free(h_gtt); free(h_gbartt);
  free(h_ct); free(h_ct_gold); free(h_ctprev); free(h_qt);
  return 0;
}
