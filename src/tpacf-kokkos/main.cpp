// Port of tpacf CUDA benchmark to Kokkos
// Two-Point Angular Correlation Function
// Generates synthetic astronomical catalog data and computes DD, DR, RR histograms

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "tpacf.h"

int main(int argc, char **argv) {
  if (argc < 5) {
    printf("Usage: %s <data_size> <random_size> <random_count> <repeat>\n", argv[0]);
    printf("  (Synthetic data generated internally)\n");
    return 1;
  }
  int nd = atoi(argv[1]);
  int nr = atoi(argv[2]);
  int nrandom = atoi(argv[3]);
  int repeat  = atoi(argv[4]);

  // Round to multiple of 4 as original
  nd -= nd % 4;
  nr -= nr % 4;

  const int bins_per_dec = 5;
  const float min_angle = 0.01f;
  const float max_angle = 10.0f;
  const int angle_units = 0;

  int nbins = 0;
  double *binb = init_bins(bins_per_dec, min_angle, max_angle, angle_units, &nbins);

  printf("\ndata size: %d\n", nd);
  printf("random size: %d x %d\n", nr, nrandom);
  printf("nbins: %d\n\n", nbins);

  CartesianData data   = generate_data(nd, 1);
  CartesianData random = generate_data(nr, 2);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_dx("dx", nd), d_dy("dy", nd), d_dz("dz", nd);
    Kokkos::View<double*> d_rx("rx", nr), d_ry("ry", nr), d_rz("rz", nr);
    Kokkos::View<double*> d_binb("binb", NUMBINS);
    Kokkos::View<long long*> d_DD("DD", NUMBINS);
    Kokkos::View<long long*> d_DR("DR", NUMBINS);
    Kokkos::View<long long*> d_RR("RR", NUMBINS);

    {
      auto hx = Kokkos::create_mirror_view(d_dx), hy = Kokkos::create_mirror_view(d_dy), hz = Kokkos::create_mirror_view(d_dz);
      auto rx = Kokkos::create_mirror_view(d_rx), ry = Kokkos::create_mirror_view(d_ry), rz = Kokkos::create_mirror_view(d_rz);
      auto hb = Kokkos::create_mirror_view(d_binb);
      for (int i = 0; i < nd; i++) { hx(i) = data.x[i]; hy(i) = data.y[i]; hz(i) = data.z[i]; }
      for (int i = 0; i < nr; i++) { rx(i) = random.x[i]; ry(i) = random.y[i]; rz(i) = random.z[i]; }
      for (int b = 0; b < NUMBINS; b++) hb(b) = binb[b];
      Kokkos::deep_copy(d_dx, hx); Kokkos::deep_copy(d_dy, hy); Kokkos::deep_copy(d_dz, hz);
      Kokkos::deep_copy(d_rx, rx); Kokkos::deep_copy(d_ry, ry); Kokkos::deep_copy(d_rz, rz);
      Kokkos::deep_copy(d_binb, hb);
    }

    Kokkos::fence();
    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::deep_copy(d_DD, 0LL);
      Kokkos::deep_copy(d_DR, 0LL);
      Kokkos::deep_copy(d_RR, 0LL);
      compute_histogram(d_dx, d_dy, d_dz, d_dx, d_dy, d_dz, nd, nd, d_DD, d_binb, NUMBINS);
      compute_histogram(d_dx, d_dy, d_dz, d_rx, d_ry, d_rz, nd, nr, d_DR, d_binb, NUMBINS);
      compute_histogram(d_rx, d_ry, d_rz, d_rx, d_ry, d_rz, nr, nr, d_RR, d_binb, NUMBINS);
    }
    Kokkos::fence();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
    printf("DONE! after %f\n", elapsed);

    // Print first few bins
    auto hDD = Kokkos::create_mirror_view(d_DD);
    auto hDR = Kokkos::create_mirror_view(d_DR);
    auto hRR = Kokkos::create_mirror_view(d_RR);
    Kokkos::deep_copy(hDD, d_DD); Kokkos::deep_copy(hDR, d_DR); Kokkos::deep_copy(hRR, d_RR);
    printf("bin  DD         DR         RR\n");
    for (int b = 0; b < 10 && b < NUMBINS; b++)
      printf("%3d  %lld  %lld  %lld\n", b, hDD(b), hDR(b), hRR(b));
  }
  Kokkos::finalize();
  return 0;
}
