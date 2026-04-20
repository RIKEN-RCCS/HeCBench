// Canny Edge Detection benchmark – Kokkos port.
// Stages: Gaussian blur → Sobel + angle → NMS → double threshold
// Usage: ./main <rows> <cols> <repeat>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>

// ─── Constants ────────────────────────────────────────────────────────────────

static const float c_gaus[9] = {
  0.0625f, 0.125f, 0.0625f,
  0.1250f, 0.250f, 0.1250f,
  0.0625f, 0.125f, 0.0625f
};
static const int c_sobx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
static const int c_soby[9] = { -1, -2, -1,  0, 0, 0,  1, 2, 1 };

// ─── Gaussian blur ────────────────────────────────────────────────────────────

void run_gaussian(
    const Kokkos::View<uint8_t*>& in,
    Kokkos::View<uint8_t*>& out,
    const Kokkos::View<float*>& gaus,
    int rows, int cols)
{
  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("gaussian",
    Policy({1, 1}, {rows - 1, cols - 1}),
    KOKKOS_LAMBDA(int row, int col) {
      int sum = 0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          sum += (int)(gaus(i * 3 + j) * (float)in((row - 1 + i) * cols + (col - 1 + j)));
      out(row * cols + col) = (uint8_t)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
    });
}

// ─── Sobel filter + angle quantisation ────────────────────────────────────────

void run_sobel(
    const Kokkos::View<uint8_t*>& in,
    Kokkos::View<uint8_t*>& out,
    Kokkos::View<uint8_t*>& theta,
    const Kokkos::View<int*>& sobx,
    const Kokkos::View<int*>& soby,
    int rows, int cols)
{
  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("sobel",
    Policy({1, 1}, {rows - 1, cols - 1}),
    KOKKOS_LAMBDA(int row, int col) {
      const float PI = 3.14159265f;
      float sumx = 0.f, sumy = 0.f;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          float px = (float)in((row - 1 + i) * cols + (col - 1 + j));
          sumx += (float)sobx(i * 3 + j) * px;
          sumy += (float)soby(i * 3 + j) * px;
        }
      float mag = Kokkos::sqrt(sumx * sumx + sumy * sumy);
      mag = mag < 0.f ? 0.f : (mag > 255.f ? 255.f : mag);
      out(row * cols + col) = (uint8_t)mag;

      float angle = Kokkos::atan2(sumy, sumx);
      if (angle < 0.f)
        angle = Kokkos::fmod(angle + 2.f * PI, 2.f * PI);

      uint8_t dir;
      if      (angle <= PI / 8.f)          dir = 0;
      else if (angle <= 3.f * PI / 8.f)    dir = 45;
      else if (angle <= 5.f * PI / 8.f)    dir = 90;
      else if (angle <= 7.f * PI / 8.f)    dir = 135;
      else if (angle <= 9.f * PI / 8.f)    dir = 0;
      else if (angle <= 11.f * PI / 8.f)   dir = 45;
      else if (angle <= 13.f * PI / 8.f)   dir = 90;
      else if (angle <= 15.f * PI / 8.f)   dir = 135;
      else                                  dir = 0;
      theta(row * cols + col) = dir;
    });
}

// ─── Non-maximum suppression ──────────────────────────────────────────────────

void run_nms(
    const Kokkos::View<uint8_t*>& in,
    Kokkos::View<uint8_t*>& out,
    const Kokkos::View<uint8_t*>& theta,
    int rows, int cols)
{
  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("nms",
    Policy({1, 1}, {rows - 1, cols - 1}),
    KOKKOS_LAMBDA(int row, int col) {
      const int pos = row * cols + col;
      uint8_t mag = in(pos);
      uint8_t dir = theta(pos);
      bool suppress;
      switch (dir) {
        case 0:   // East/West neighbors
          suppress = (mag <= in(row * cols + col + 1)) ||
                     (mag <= in(row * cols + col - 1));
          break;
        case 45:  // NE/SW neighbors
          suppress = (mag <= in((row - 1) * cols + col + 1)) ||
                     (mag <= in((row + 1) * cols + col - 1));
          break;
        case 90:  // North/South neighbors
          suppress = (mag <= in((row - 1) * cols + col)) ||
                     (mag <= in((row + 1) * cols + col));
          break;
        default:  // 135: NW/SE neighbors
          suppress = (mag <= in((row - 1) * cols + col - 1)) ||
                     (mag <= in((row + 1) * cols + col + 1));
          break;
      }
      out(pos) = suppress ? 0 : mag;
    });
}

// ─── Double threshold (hysteresis) ────────────────────────────────────────────

void run_threshold(
    const Kokkos::View<uint8_t*>& in,
    Kokkos::View<uint8_t*>& out,
    int rows, int cols)
{
  const float lowThresh  = 10.f;
  const float highThresh = 70.f;
  const float med        = (highThresh + lowThresh) / 2.f;
  const uint8_t EDGE     = 255;

  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  Kokkos::parallel_for("threshold",
    Policy({1, 1}, {rows - 1, cols - 1}),
    KOKKOS_LAMBDA(int row, int col) {
      const int pos = row * cols + col;
      float mag = (float)in(pos);
      if      (mag >= highThresh)  out(pos) = EDGE;
      else if (mag <= lowThresh)   out(pos) = 0;
      else if (mag >= med)         out(pos) = EDGE;
      else                         out(pos) = 0;
    });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  int rows   = 2048;
  int cols   = 2048;
  int repeat = 100;
  if (argc > 1) rows   = std::atoi(argv[1]);
  if (argc > 2) cols   = std::atoi(argv[2]);
  if (argc > 3) repeat = std::atoi(argv[3]);

  printf("Image: %d x %d, repeat: %d\n", rows, cols, repeat);

  const int N = rows * cols;

  // Generate random input on host
  std::vector<uint8_t> h_input(N);
  {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : h_input) v = (uint8_t)dist(rng);
  }

  Kokkos::initialize(argc, argv);
  {
    // Upload constant arrays
    Kokkos::View<float*> d_gaus("gaus", 9);
    Kokkos::View<int*>   d_sobx("sobx", 9);
    Kokkos::View<int*>   d_soby("soby", 9);
    {
      auto hg = Kokkos::create_mirror_view(d_gaus);
      auto hx = Kokkos::create_mirror_view(d_sobx);
      auto hy = Kokkos::create_mirror_view(d_soby);
      for (int i = 0; i < 9; ++i) { hg(i) = c_gaus[i]; hx(i) = c_sobx[i]; hy(i) = c_soby[i]; }
      Kokkos::deep_copy(d_gaus, hg);
      Kokkos::deep_copy(d_sobx, hx);
      Kokkos::deep_copy(d_soby, hy);
    }

    // Device buffers
    Kokkos::View<uint8_t*> d_in("d_in",   N);
    Kokkos::View<uint8_t*> d_blur("blur",  N);
    Kokkos::View<uint8_t*> d_grad("grad",  N);
    Kokkos::View<uint8_t*> d_nms("nms",   N);
    Kokkos::View<uint8_t*> d_out("d_out", N);
    Kokkos::View<uint8_t*> d_theta("theta", N);

    // Upload input
    {
      auto h_in = Kokkos::create_mirror_view(d_in);
      std::memcpy(h_in.data(), h_input.data(), N * sizeof(uint8_t));
      Kokkos::deep_copy(d_in, h_in);
    }

    Kokkos::fence();
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < repeat; ++r) {
      run_gaussian(d_in,   d_blur, d_gaus, rows, cols);
      run_sobel   (d_blur, d_grad, d_theta, d_sobx, d_soby, rows, cols);
      run_nms     (d_grad, d_nms,  d_theta, rows, cols);
      run_threshold(d_nms, d_out, rows, cols);
    }
    Kokkos::fence();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                       t_end - t_start).count() / 1e6;
    printf("Total time: %.4f s  (%.4f ms/iter)\n",
           elapsed, elapsed / repeat * 1e3);

    // Verify: check there are non-zero pixels in output
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_out);
    int nonzero = 0;
    for (int i = 0; i < N; ++i) if (h_out(i) != 0) ++nonzero;
    if (nonzero == 0)
      printf("VERIFICATION FAILED: output is all zeros\n");
    else
      printf("PASS: %d non-zero edge pixels\n", nonzero);
  }
  Kokkos::finalize();
  return 0;
}
