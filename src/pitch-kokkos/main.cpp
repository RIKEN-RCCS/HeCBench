/*
   Pitched vs simple 2D/3D memory access benchmark using Kokkos.
   Ported from pitch-sycl.
   Simulates pitched memory by padding row width to 64-byte alignment.
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// Compute padded width (in float elements) for 64-byte row alignment
static int padded_width(int w) {
  return ((w * (int)sizeof(float) + 63) & ~63) / (int)sizeof(float);
}

void malloc2D(int repeat, int width, int height) {
  printf("Dimension: (%d %d)\n", width, height);

  const int wp = padded_width(width);

  Kokkos::View<float*> d_pitched("pitched2d", (size_t)wp * height);
  Kokkos::View<float*> d_simple("simple2d",   (size_t)width * height);

  auto h_pitched = Kokkos::create_mirror_view(d_pitched);
  auto h_simple  = Kokkos::create_mirror_view(d_simple);

  srand(42);
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      float val = (float)rand() / (float)RAND_MAX;
      h_pitched(r * wp + c) = val;
      h_simple(r * width + c) = val;
    }
  }
  Kokkos::deep_copy(d_pitched, h_pitched);
  Kokkos::deep_copy(d_simple, h_simple);

  // warm-up
  Kokkos::parallel_for("pitched2d_warmup",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
    KOKKOS_LAMBDA(int r, int c) {
      int idx = r * wp + c;
      d_pitched(idx) = 1.0f / (1.0f + Kokkos::exp(-d_pitched(idx)));
    });
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("pitched2d",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
      KOKKOS_LAMBDA(int r, int c) {
        int idx = r * wp + c;
        d_pitched(idx) = 1.0f / (1.0f + Kokkos::exp(-d_pitched(idx)));
      });
  }
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double time_pitched = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;

  // warm-up
  Kokkos::parallel_for("simple2d_warmup",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
    KOKKOS_LAMBDA(int r, int c) {
      int idx = r * width + c;
      d_simple(idx) = 1.0f / (1.0f + Kokkos::exp(-d_simple(idx)));
    });
  Kokkos::fence();

  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("simple2d",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {height, width}),
      KOKKOS_LAMBDA(int r, int c) {
        int idx = r * width + c;
        d_simple(idx) = 1.0f / (1.0f + Kokkos::exp(-d_simple(idx)));
      });
  }
  Kokkos::fence();
  auto t3 = std::chrono::steady_clock::now();
  double time_simple = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * 1e-3;

  printf("Average execution time (pitched vs simple): %f %f (us)\n",
         (float)(time_pitched / repeat), (float)(time_simple / repeat));
}

void malloc3D(int repeat, int width, int height, int depth) {
  printf("Dimension: (%d %d %d)\n", width, height, depth);

  const int wp = padded_width(width);

  Kokkos::View<float*> d_pitched("pitched3d", (size_t)wp * height * depth);
  Kokkos::View<float*> d_simple("simple3d",   (size_t)width * height * depth);

  auto h_pitched = Kokkos::create_mirror_view(d_pitched);
  auto h_simple  = Kokkos::create_mirror_view(d_simple);

  srand(42);
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float val = (float)rand() / (float)RAND_MAX;
        h_pitched(z * height * wp + y * wp + x) = val;
        h_simple(z * height * width + y * width + x) = val;
      }
    }
  }
  Kokkos::deep_copy(d_pitched, h_pitched);
  Kokkos::deep_copy(d_simple, h_simple);

  // warm-up
  Kokkos::parallel_for("pitched3d_warmup",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {depth, height, width}),
    KOKKOS_LAMBDA(int z, int y, int x) {
      int idx = z * height * wp + y * wp + x;
      d_pitched(idx) = 1.0f / (1.0f + Kokkos::exp(-d_pitched(idx)));
    });
  Kokkos::fence();

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("pitched3d",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {depth, height, width}),
      KOKKOS_LAMBDA(int z, int y, int x) {
        int idx = z * height * wp + y * wp + x;
        d_pitched(idx) = 1.0f / (1.0f + Kokkos::exp(-d_pitched(idx)));
      });
  }
  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  double time_pitched = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3;

  // warm-up
  Kokkos::parallel_for("simple3d_warmup",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {depth, height, width}),
    KOKKOS_LAMBDA(int z, int y, int x) {
      int idx = z * height * width + y * width + x;
      d_simple(idx) = 1.0f / (1.0f + Kokkos::exp(-d_simple(idx)));
    });
  Kokkos::fence();

  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("simple3d",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {depth, height, width}),
      KOKKOS_LAMBDA(int z, int y, int x) {
        int idx = z * height * width + y * width + x;
        d_simple(idx) = 1.0f / (1.0f + Kokkos::exp(-d_simple(idx)));
      });
  }
  Kokkos::fence();
  auto t3 = std::chrono::steady_clock::now();
  double time_simple = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * 1e-3;

  printf("Average execution time (pitched vs simple): %f %f (us)\n",
         (float)(time_pitched / repeat), (float)(time_simple / repeat));
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const int w[] = {227, 256, 720, 768, 854, 1280, 1440, 1920, 2048, 3840, 4096};
    const int h[] = {227, 256, 480, 576, 480,  720, 1080, 1080, 1080, 2160, 2160};
    const int d[] = {1, 3};

    for (int i = 0; i < 11; i++)
      malloc2D(repeat, w[i], h[i]);

    for (int i = 0; i < 11; i++)
      for (int j = 0; j < 2; j++)
        malloc3D(repeat, w[i], h[i], d[j]);
  }
  Kokkos::finalize();
  return 0;
}
