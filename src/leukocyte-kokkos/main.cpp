#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define NCIRCLES 7
#define NPOINTS  150
#define RADIUS   10
#define MIN_RAD  (RADIUS - 2)
#define MAX_RAD  (RADIUS * 2)
#define MaxR     (MAX_RAD + 2)  // = 22

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <grad_m> <repeat>\n", argv[0]);
    return 1;
  }

  const int grad_m = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int N = grad_m * grad_m;

  // Generate random gradient data on the host
  float* h_grad_x_raw = new float[N];
  float* h_grad_y_raw = new float[N];

  srand(42);
  for (int i = 0; i < N; i++) {
    h_grad_x_raw[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    h_grad_y_raw[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
  }

  // Precompute sin/cos lookup tables
  float h_sin_angle[NPOINTS], h_cos_angle[NPOINTS];
  for (int p = 0; p < NPOINTS; p++) {
    double angle = p * 2.0 * M_PI / NPOINTS;
    h_sin_angle[p] = (float)sin(angle);
    h_cos_angle[p] = (float)cos(angle);
  }

  // Precompute circle sample-point offsets
  int h_tX[NCIRCLES * NPOINTS];
  int h_tY[NCIRCLES * NPOINTS];
  for (int k = 0; k < NCIRCLES; k++) {
    float radius = MIN_RAD + k * (float)(MAX_RAD - MIN_RAD) / (float)(NCIRCLES - 1);
    for (int p = 0; p < NPOINTS; p++) {
      h_tX[k * NPOINTS + p] = (int)(radius * cos(p * 2.0 * M_PI / NPOINTS));
      h_tY[k * NPOINTS + p] = (int)(radius * sin(p * 2.0 * M_PI / NPOINTS));
    }
  }

  Kokkos::initialize(argc, argv);
  {
    // Allocate device views
    Kokkos::View<float*> d_grad_x("grad_x", N);
    Kokkos::View<float*> d_grad_y("grad_y", N);
    Kokkos::View<float*> d_gicov("gicov", N);
    Kokkos::View<float*> d_dilated("dilated", N);
    Kokkos::View<float*> d_sin_angle("sin_angle", NPOINTS);
    Kokkos::View<float*> d_cos_angle("cos_angle", NPOINTS);
    Kokkos::View<int*>   d_tX("tX", NCIRCLES * NPOINTS);
    Kokkos::View<int*>   d_tY("tY", NCIRCLES * NPOINTS);

    // Copy gradient data to device
    {
      auto h = Kokkos::create_mirror_view(d_grad_x);
      for (int i = 0; i < N; i++) h(i) = h_grad_x_raw[i];
      Kokkos::deep_copy(d_grad_x, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_grad_y);
      for (int i = 0; i < N; i++) h(i) = h_grad_y_raw[i];
      Kokkos::deep_copy(d_grad_y, h);
    }

    // Copy precomputed lookup tables to device
    {
      auto h = Kokkos::create_mirror_view(d_sin_angle);
      for (int p = 0; p < NPOINTS; p++) h(p) = h_sin_angle[p];
      Kokkos::deep_copy(d_sin_angle, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_cos_angle);
      for (int p = 0; p < NPOINTS; p++) h(p) = h_cos_angle[p];
      Kokkos::deep_copy(d_cos_angle, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_tX);
      for (int i = 0; i < NCIRCLES * NPOINTS; i++) h(i) = h_tX[i];
      Kokkos::deep_copy(d_tX, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_tY);
      for (int i = 0; i < NCIRCLES * NPOINTS; i++) h(i) = h_tY[i];
      Kokkos::deep_copy(d_tY, h);
    }

    // Border pixels are never written by the GICOV kernel; zero them once.
    Kokkos::deep_copy(d_gicov, 0.0f);

    // The GICOV kernel iterates over this inner region
    const int gicov_size = grad_m - 2 * MaxR;

    // Dilation structuring element: 25x25 disk (radius = 12)
    constexpr int strel_r = 12;

    using MDPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    // Warm-up fence so the clock starts clean
    Kokkos::fence();

    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {

      // ----------------------------------------------------------------
      // GICOV kernel
      // For each interior pixel (m, n) compute the maximal GICOV score
      // across all NCIRCLES sample circles.
      // ----------------------------------------------------------------
      Kokkos::parallel_for(
        "gicov",
        MDPolicy({0, 0}, {gicov_size, gicov_size}),
        KOKKOS_LAMBDA(int m, int n) {
          // Local array for per-circle per-point gradient projections.
          // NCIRCLES * NPOINTS = 1050 entries; kept in thread-local storage.
          float grad_mag[NCIRCLES * NPOINTS];

          for (int k = 0; k < NCIRCLES; k++) {
            for (int p = 0; p < NPOINTS; p++) {
              int x = m + MaxR + d_tX(k * NPOINTS + p);
              int y = n + MaxR + d_tY(k * NPOINTS + p);
              float gx = d_grad_x(y * grad_m + x);
              float gy = d_grad_y(y * grad_m + x);
              grad_mag[k * NPOINTS + p] = gx * d_sin_angle(p) + gy * d_cos_angle(p);
            }
          }

          float max_gicov = 0.0f;
          for (int k = 0; k < NCIRCLES; k++) {
            float sum_k = 0.0f;
            for (int p = 0; p < NPOINTS; p++) {
              float val = grad_mag[k * NPOINTS + p];
              if (val > 0.0f) sum_k += val;
            }
            float g = 2.0f * sum_k / NPOINTS - 1.0f;
            g = g * g;
            if (g > max_gicov) max_gicov = g;
          }

          d_gicov((n + MaxR) * grad_m + (m + MaxR)) = max_gicov;
        });

      // ----------------------------------------------------------------
      // Dilation kernel
      // Morphological dilation with a 25x25 (radius 12) structuring element.
      // ----------------------------------------------------------------
      Kokkos::parallel_for(
        "dilation",
        MDPolicy({0, 0}, {grad_m, grad_m}),
        KOKKOS_LAMBDA(int row, int col) {
          float max_val = 0.0f;
          for (int i = -strel_r; i <= strel_r; i++) {
            int ri = row + i;
            if (ri < 0 || ri >= grad_m) continue;
            for (int j = -strel_r; j <= strel_r; j++) {
              int cj = col + j;
              if (cj < 0 || cj >= grad_m) continue;
              float val = d_gicov(ri * grad_m + cj);
              if (val > max_val) max_val = val;
            }
          }
          d_dilated(row * grad_m + col) = max_val;
        });
    }

    Kokkos::fence();

    auto t_end = std::chrono::steady_clock::now();
    double total_time =
      std::chrono::duration<double>(t_end - t_start).count();

    printf("Total kernel time: %f seconds\n", total_time);
  }
  Kokkos::finalize();

  delete[] h_grad_x_raw;
  delete[] h_grad_y_raw;

  return 0;
}
