// Port of tpacf CUDA benchmark to Kokkos
// Two-Point Angular Correlation Function
// Generates synthetic astronomical catalog data and computes DD, DR, RR histograms

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#define NUMBINS 64
#define D2R (M_PI / 180.0)

// Global bin boundaries in constant-like view
static double g_binbounds[NUMBINS];

static double* init_bins(int bins_per_dec, float min_angle, float max_angle,
                         int angle_units, int *nbins) {
  *nbins = (int)floor(bins_per_dec * (log10(max_angle) - log10(min_angle)));
  static double binb[31];
  int binoffset = 30 - (*nbins);
  for (int k = 0; k < (*nbins)+1; k++) {
    double bb = pow(10.0, log10(min_angle) + k * 1.0 / bins_per_dec);
    binb[k + binoffset] = cos(bb / (angle_units ? 60.0 : 1.0) * D2R);
  }
  for (int k = 0; k < binoffset; k++) binb[k] = -5.0;
  binb[30] = -5.0;
  return binb;
}

struct CartesianData {
  std::vector<double> x, y, z;
  int n;
  CartesianData(int n_) : n(n_), x(n_), y(n_), z(n_) {}
};

// Generate synthetic spherical point cloud
static CartesianData generate_data(int n, unsigned seed) {
  CartesianData cd(n);
  srand(seed);
  for (int i = 0; i < n; i++) {
    double ra  = (rand() / (double)RAND_MAX) * 360.0;  // degrees
    double dec = (rand() / (double)RAND_MAX) * 180.0 - 90.0;
    double ra_r  = ra  * D2R;
    double dec_r = dec * D2R;
    cd.x[i] = cos(dec_r) * cos(ra_r);
    cd.y[i] = cos(dec_r) * sin(ra_r);
    cd.z[i] = sin(dec_r);
  }
  return cd;
}

// Find bin for a dot product value using waterfall search
KOKKOS_INLINE_FUNCTION
int find_bin(double dot, const double *binb, int nbins_minus1) {
  for (int b = 0; b < nbins_minus1; b++) {
    if (dot > binb[b]) return b;
  }
  return nbins_minus1;
}

// Compute histogram for all pairs between set1 and set2
static void compute_histogram(
    const Kokkos::View<double*> &x1, const Kokkos::View<double*> &y1, const Kokkos::View<double*> &z1,
    const Kokkos::View<double*> &x2, const Kokkos::View<double*> &y2, const Kokkos::View<double*> &z2,
    int n1, int n2,
    Kokkos::View<long long*> &histo,
    const Kokkos::View<double*> &d_binb, int nbins)
{
  Kokkos::parallel_for("acf", n1, KOKKOS_LAMBDA(const int i) {
    double xi = x1(i), yi = y1(i), zi = z1(i);
    for (int j = 0; j < n2; j++) {
      double dot = xi * x2(j) + yi * y2(j) + zi * z2(j);
      // clamp
      if (dot >  1.0) dot =  1.0;
      if (dot < -1.0) dot = -1.0;
      // find bin
      int bin = nbins - 1;
      for (int b = 0; b < nbins - 1; b++) {
        if (dot > d_binb(b)) { bin = b; break; }
      }
      Kokkos::atomic_add(&histo(bin), 1LL);
    }
  });
  Kokkos::fence();
}

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
