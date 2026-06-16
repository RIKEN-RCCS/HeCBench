#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "SimpleMOC-kernel_header.h"

// ---- init helpers (from init.cpp) ----------------------------------------

static Input* set_default_input(void)
{
  Input* I = (Input*)malloc(sizeof(Input));
  I->source_2D_regions    = 5000;
  I->coarse_axial_intervals = 27;
  I->fine_axial_intervals = 5;
  I->decomp_assemblies_ax = 20;
  I->segments             = 50000000;
  I->egroups              = 128;
  I->repeat               = 1;
  return I;
}

static Source* initialize_sources(Input* I)
{
  I->nbytes = 0;
  Source* sources = (Source*)malloc(I->source_3D_regions * sizeof(Source));

  float* data = (float*)malloc(I->source_3D_regions * I->fine_axial_intervals * I->egroups * sizeof(float));
  for (int i = 0; i < I->source_3D_regions; i++)
    sources[i].fine_source = &data[i * I->fine_axial_intervals * I->egroups];

  data = (float*)malloc(I->source_3D_regions * I->fine_axial_intervals * I->egroups * sizeof(float));
  for (int i = 0; i < I->source_3D_regions; i++)
    sources[i].fine_flux = &data[i * I->fine_axial_intervals * I->egroups];

  data = (float*)malloc(I->source_3D_regions * I->egroups * sizeof(float));
  for (int i = 0; i < I->source_3D_regions; i++)
    sources[i].sigT = &data[i * I->egroups];

  for (int i = 0; i < I->source_3D_regions; i++)
    for (int j = 0; j < I->fine_axial_intervals; j++)
      for (int k = 0; k < I->egroups; k++) {
        sources[i].fine_source[j * I->egroups + k] = rand() / (float)RAND_MAX;
        sources[i].fine_flux[j * I->egroups + k]   = rand() / (float)RAND_MAX;
      }
  for (int i = 0; i < I->source_3D_regions; i++)
    for (int j = 0; j < I->egroups; j++)
      sources[i].sigT[j] = rand() / (float)RAND_MAX;

  return sources;
}

// Copy sources to flat arrays
static void copy_sources_flat(Input* I, Source* S,
                              float* fine_source_flat,
                              float* fine_flux_flat,
                              float* sigT_flat)
{
  for (int i = 0; i < I->source_3D_regions; i++)
    for (int j = 0; j < I->fine_axial_intervals; j++)
      for (int k = 0; k < I->egroups; k++) {
        fine_source_flat[i * I->fine_axial_intervals * I->egroups + j * I->egroups + k]
          = S[i].fine_source[j * I->egroups + k];
        fine_flux_flat[i * I->fine_axial_intervals * I->egroups + j * I->egroups + k]
          = S[i].fine_flux[j * I->egroups + k];
      }
  for (int i = 0; i < I->source_3D_regions; i++)
    for (int j = 0; j < I->egroups; j++)
      sigT_flat[i * I->egroups + j] = S[i].sigT[j];
}

// ---- io helpers (minimal) ------------------------------------------------

static void border_print(void) {
  printf("======================================================================" "=========\n");
}
static void center_print(const char* s, int width) {
  int length = (int)strlen(s);
  for (int i = 0; i <= (width - length) / 2; i++) fputs(" ", stdout);
  fputs(s, stdout); fputs("\n", stdout);
}

// ---- main ----------------------------------------------------------------

int main(int argc, char* argv[])
{
  unsigned int seed = 2;
  srand(seed);

  Input* I = set_default_input();

  if (argc >= 2) I->repeat = atoi(argv[1]);

  I->source_3D_regions = (int)ceil((double)I->source_2D_regions *
      I->coarse_axial_intervals / I->decomp_assemblies_ax);

  border_print();
  center_print("SimpleMOC-kernel (Kokkos)", 79);
  border_print();

  Source* S = initialize_sources(I);

  int fine_axial_intervals = I->fine_axial_intervals;
  int egroups              = I->egroups;
  int segments             = I->segments;
  int source_3D_regions    = I->source_3D_regions;

  // Flatten sources for device
  size_t src_sz = (size_t)source_3D_regions * fine_axial_intervals * egroups;
  float* fine_source_flat = (float*)malloc(src_sz * sizeof(float));
  float* fine_flux_flat   = (float*)malloc(src_sz * sizeof(float));
  float* sigT_flat        = (float*)malloc((size_t)source_3D_regions * egroups * sizeof(float));
  copy_sources_flat(I, S, fine_source_flat, fine_flux_flat, sigT_flat);

  float* state_flux_host = (float*)malloc(egroups * sizeof(float));
  for (int i = 0; i < egroups; i++)
    state_flux_host[i] = rand_r(&seed) / (float)RAND_MAX;

  // Build segment arrays
  int* QSR_id_arr = (int*)malloc(sizeof(int) * segments);
  int* FAI_id_arr = (int*)malloc(sizeof(int) * segments);
  for (long i = 0; i < segments; i++) {
    QSR_id_arr[i] = rand_r(&seed) % source_3D_regions;
    FAI_id_arr[i] = rand_r(&seed) % fine_axial_intervals;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*>   d_QSR("QSR",       segments);
    Kokkos::View<int*>   d_FAI("FAI",       segments);
    Kokkos::View<float*> d_sigT("sigT",     (size_t)source_3D_regions * egroups);
    Kokkos::View<float*> d_fine_source("fine_source", src_sz);
    Kokkos::View<float*> d_fine_flux("fine_flux",     src_sz);
    Kokkos::View<float*> d_state_flux("state_flux",   egroups);

    {
      auto hQSR  = Kokkos::View<int*,   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(QSR_id_arr, segments);
      auto hFAI  = Kokkos::View<int*,   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(FAI_id_arr, segments);
      auto hsigT = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(sigT_flat, (size_t)source_3D_regions * egroups);
      auto hFS   = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(fine_source_flat, src_sz);
      auto hFF   = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(fine_flux_flat,   src_sz);
      auto hSF   = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(state_flux_host, egroups);
      Kokkos::deep_copy(d_QSR, hQSR); Kokkos::deep_copy(d_FAI, hFAI);
      Kokkos::deep_copy(d_sigT, hsigT); Kokkos::deep_copy(d_fine_source, hFS);
      Kokkos::deep_copy(d_fine_flux, hFF); Kokkos::deep_copy(d_state_flux, hSF);
    }

    auto kstart = std::chrono::steady_clock::now();

    for (int rep = 0; rep < I->repeat; rep++) {
      Kokkos::parallel_for("simplemoc",
        Kokkos::RangePolicy<>(0, segments),
        KOKKOS_LAMBDA(int gid) {
          const float dz = 0.1f, zin = 0.3f, weight = 0.5f;
          const float mu = 0.9f, mu2 = 0.3f, ds = 0.7f;

          int QSR_id = d_QSR(gid);
          int FAI_id = d_FAI(gid);
          int offset = QSR_id * fine_axial_intervals * egroups;

          for (int g = 0; g < egroups; g++) {
            float q0, q1, q2;
            if (FAI_id == 0) {
              float y2 = d_fine_source(offset + FAI_id * egroups + g);
              float y3 = d_fine_source(offset + (FAI_id + 1) * egroups + g);
              float c0 = y2, c1 = (y3 - y2) / dz;
              q0 = c0 + c1 * zin; q1 = c1; q2 = 0;
            } else if (FAI_id == fine_axial_intervals - 1) {
              float y1 = d_fine_source(offset + (FAI_id - 1) * egroups + g);
              float y2 = d_fine_source(offset + FAI_id * egroups + g);
              float c0 = y2, c1 = (y2 - y1) / dz;
              q0 = c0 + c1 * zin; q1 = c1; q2 = 0;
            } else {
              float y1 = d_fine_source(offset + (FAI_id - 1) * egroups + g);
              float y2 = d_fine_source(offset + FAI_id * egroups + g);
              float y3 = d_fine_source(offset + (FAI_id + 1) * egroups + g);
              float c0 = y2, c1 = (y1 - y3) / (2.f * dz),
                    c2 = (y1 - 2.f * y2 + y3) / (2.f * dz * dz);
              q0 = c0 + c1 * zin + c2 * zin * zin;
              q1 = c1 + 2.f * c2 * zin; q2 = c2;
            }

            float sigT_g  = d_sigT(QSR_id * egroups + g);
            float tau     = sigT_g * ds;
            float sigT2   = sigT_g * sigT_g;
            float expVal  = 1.f - expf(-tau);
            float reuse   = tau * (tau - 2.f) + 2.f * expVal / (sigT_g * sigT2);

            float sf = d_state_flux(g);
            float fi = (q0 * tau + (sigT_g * sf - q0) * expVal) / sigT2
                     + q1 * mu * reuse
                     + q2 * mu2 * (tau * (tau * (tau - 3.f) + 6.f) - 6.f * expVal)
                       / (3.f * sigT2 * sigT2);

            Kokkos::atomic_add(&d_fine_flux(offset + FAI_id * egroups + g), weight * fi);

            float t1 = q0 * expVal / sigT_g;
            float t2 = q1 * mu * (tau - expVal) / sigT2;
            float t3 = q2 * mu2 * reuse;
            float t4 = sf * (1.f - expVal);
            // state_flux is shared - just write, no sync needed for benchmark
            d_state_flux(g) = t1 + t2 + t3 + t4;
          }
        });
    }

    Kokkos::fence();
    auto kstop = std::chrono::steady_clock::now();
    double ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kstop - kstart).count() * 1e-9;

    border_print();
    center_print("RESULTS SUMMARY", 79);
    border_print();
    printf("%-25s%.3lf seconds\n", "Total kernel time:", ktime);
    double tpi = (ktime / I->repeat / (double)segments / (double)egroups) * 1.0e9;
    printf("%-25s%.3lf ns\n", "Time per Intersection:", tpi);
    border_print();
  }
  Kokkos::finalize();

  free(S[0].fine_source); free(S[0].fine_flux); free(S[0].sigT);
  free(S); free(fine_source_flat); free(fine_flux_flat); free(sigT_flat);
  free(state_flux_host); free(QSR_id_arr); free(FAI_id_arr); free(I);
  return 0;
}
