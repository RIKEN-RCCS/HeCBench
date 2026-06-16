#include <cstdio>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <algorithm>
#include <vector>
#include <Kokkos_Core.hpp>

// Pixel stored as packed RGBA (only RGB used; A ignored).
// We keep each pixel as an int with R in bits [0..7], G in [8..15], B in [16..23].
static inline int make_pixel(int r, int g, int b)
{
  return (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16);
}
static inline int pixel_r(int p) { return  p        & 0xFF; }
static inline int pixel_g(int p) { return (p >>  8) & 0xFF; }
static inline int pixel_b(int p) { return (p >> 16) & 0xFF; }

int main(int argc, char *argv[])
{
  if (argc != 5) {
    printf("Usage: %s <width> <height> <seams_to_remove> <repeat>\n", argv[0]);
    return 1;
  }

  int width          = atoi(argv[1]);
  int height         = atoi(argv[2]);
  int seams_to_remove = atoi(argv[3]);
  int repeat         = atoi(argv[4]);

  if (seams_to_remove < 0 || seams_to_remove >= width) {
    printf("ERROR: seams_to_remove must be in [0, width).\n");
    return 1;
  }

  printf("Image size: %dx%d\n", width, height);
  printf("Removing %d seams...\n", seams_to_remove);

  // Generate synthetic pixel data on host
  int total_pixels = width * height;
  std::vector<int> h_pixels(total_pixels);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      int r = (x * 37 + y * 53) % 256;
      int g = (x * 61 + y * 29) % 256;
      int b = (x * 13 + y * 97) % 256;
      h_pixels[y * width + x] = make_pixel(r, g, b);
    }

  Kokkos::initialize(argc, argv);
  {
    // We run the whole seam-carving repeat times to average timing.
    // Re-initialise pixel data before each run.

    int cur_w = width;

    // Allocate max-width buffers
    Kokkos::View<int*>    d_pixels ("d_pixels",  (long long)width * height);
    Kokkos::View<int*>    d_energy ("d_energy",  (long long)width * height);
    Kokkos::View<int*>    d_dp     ("d_dp",      (long long)width * height);
    Kokkos::View<int*>    d_seam   ("d_seam",    height);
    // Output pixel buffer (after seam removal)
    Kokkos::View<int*>    d_out    ("d_out",     (long long)width * height);

    double total_time_us = 0.0;

    for (int run = 0; run < repeat; run++) {
      // Reset to original image
      cur_w = width;
      {
        auto hm = Kokkos::create_mirror_view(d_pixels);
        for (int i = 0; i < total_pixels; i++) hm(i) = h_pixels[i];
        Kokkos::deep_copy(d_pixels, hm);
      }

      auto t0 = std::chrono::steady_clock::now();

      for (int s = 0; s < seams_to_remove; s++) {
        const int W = cur_w;
        const int H = height;

        // --- Energy computation (parallel over all pixels) ---
        Kokkos::parallel_for("energy", W * H, KOKKOS_LAMBDA(const int idx) {
          int x = idx % W;
          int y = idx / W;

          int xl = (x > 0)     ? x - 1 : x;
          int xr = (x < W - 1) ? x + 1 : x;
          int yu = (y > 0)     ? y - 1 : y;
          int yd = (y < H - 1) ? y + 1 : y;

          int pl = d_pixels[y  * W + xl];
          int pr = d_pixels[y  * W + xr];
          int pu = d_pixels[yu * W + x];
          int pd = d_pixels[yd * W + x];

          int e = 0;
          e += abs(pixel_r(pr) - pixel_r(pl)) + abs(pixel_g(pr) - pixel_g(pl)) + abs(pixel_b(pr) - pixel_b(pl));
          e += abs(pixel_r(pd) - pixel_r(pu)) + abs(pixel_g(pd) - pixel_g(pu)) + abs(pixel_b(pd) - pixel_b(pu));
          d_energy[idx] = e;
        });
        Kokkos::fence();

        // --- DP: copy first row ---
        Kokkos::parallel_for("dp_row0", W, KOKKOS_LAMBDA(const int x) {
          d_dp[x] = d_energy[x];
        });
        Kokkos::fence();

        // --- DP: row by row (sequential rows, parallel within row) ---
        for (int y = 1; y < H; y++) {
          const int row = y;
          Kokkos::parallel_for("dp_row", W, KOKKOS_LAMBDA(const int x) {
            int best = d_dp[(row - 1) * W + x];
            if (x > 0)     best = Kokkos::min(best, d_dp[(row - 1) * W + x - 1]);
            if (x < W - 1) best = Kokkos::min(best, d_dp[(row - 1) * W + x + 1]);
            d_dp[row * W + x] = d_energy[row * W + x] + best;
          });
          Kokkos::fence();
        }

        // --- Seam finding (sequential on host via mirror) ---
        auto h_dp   = Kokkos::create_mirror_view(d_dp);
        auto h_seam = Kokkos::create_mirror_view(d_seam);
        Kokkos::deep_copy(h_dp, d_dp);

        // Find minimum in last row
        int min_idx = 0;
        for (int x = 1; x < W; x++)
          if (h_dp[(H - 1) * W + x] < h_dp[(H - 1) * W + min_idx])
            min_idx = x;
        h_seam(H - 1) = min_idx;

        // Backtrack
        for (int y = H - 2; y >= 0; y--) {
          int cx  = h_seam(y + 1);
          int best = cx;
          if (cx > 0     && h_dp[y * W + cx - 1] < h_dp[y * W + best]) best = cx - 1;
          if (cx < W - 1 && h_dp[y * W + cx + 1] < h_dp[y * W + best]) best = cx + 1;
          h_seam(y) = best;
        }
        Kokkos::deep_copy(d_seam, h_seam);

        // --- Seam removal (parallel over rows) ---
        const int newW = W - 1;
        Kokkos::parallel_for("seam_remove", H, KOKKOS_LAMBDA(const int y) {
          int sc = d_seam[y];
          for (int x = 0; x < sc; x++)
            d_out[y * newW + x] = d_pixels[y * W + x];
          for (int x = sc; x < newW; x++)
            d_out[y * newW + x] = d_pixels[y * W + x + 1];
        });
        Kokkos::fence();

        // Swap pixel buffers via deep_copy
        Kokkos::parallel_for("swap", newW * H, KOKKOS_LAMBDA(const int i) {
          d_pixels[i] = d_out[i];
        });
        Kokkos::fence();

        cur_w = newW;
      } // end seam loop

      auto t1 = std::chrono::steady_clock::now();
      total_time_us += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;
    } // end repeat

    double avg_us = total_time_us / repeat / seams_to_remove;
    printf("Average time per seam: %f (us)\n", avg_us);
    printf("Done. Final width: %d\n", width - seams_to_remove);
  }
  Kokkos::finalize();
  return 0;
}
