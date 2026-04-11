#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>

#define DEGREE_TO_RADIAN (M_PI / 180.0)
#define EARTH_RADIUS_KM  6371.0

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {

  // Generate synthetic input using an LCG on the host
  Kokkos::View<double*> a_lat("a_lat", N);
  Kokkos::View<double*> a_lon("a_lon", N);
  Kokkos::View<double*> b_lat("b_lat", N);
  Kokkos::View<double*> b_lon("b_lon", N);
  Kokkos::View<double*> dist("dist", N);

  auto h_a_lat = Kokkos::create_mirror_view(a_lat);
  auto h_a_lon = Kokkos::create_mirror_view(a_lon);
  auto h_b_lat = Kokkos::create_mirror_view(b_lat);
  auto h_b_lon = Kokkos::create_mirror_view(b_lon);

  uint64_t seed = 12345ULL;
  auto lcg = [&]() -> double {
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    seed = (a * seed + c) % m;
    return (double)seed / (double)m;
  };

  for (int i = 0; i < N; i++) {
    h_a_lat(i) = lcg() * 180.0 - 90.0;   // [-90,  90]
    h_a_lon(i) = lcg() * 360.0 - 180.0;  // [-180, 180]
    h_b_lat(i) = lcg() * 180.0 - 90.0;
    h_b_lon(i) = lcg() * 360.0 - 180.0;
  }

  // Reference result computed on the host
  std::vector<double> expected(N);
  for (int i = 0; i < N; i++) {
    double ax = h_a_lon(i) * DEGREE_TO_RADIAN;
    double ay = h_a_lat(i) * DEGREE_TO_RADIAN;
    double bx = h_b_lon(i) * DEGREE_TO_RADIAN;
    double by = h_b_lat(i) * DEGREE_TO_RADIAN;
    double x = (bx - ax) / 2.0;
    double y = (by - ay) / 2.0;
    double sinysqrd = sin(y) * sin(y);
    double sinxsqrd = sin(x) * sin(x);
    double scale = cos(ay) * cos(by);
    expected[i] = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
  }

  Kokkos::deep_copy(a_lat, h_a_lat);
  Kokkos::deep_copy(a_lon, h_a_lon);
  Kokkos::deep_copy(b_lat, h_b_lat);
  Kokkos::deep_copy(b_lon, h_b_lon);

  Kokkos::fence();
  auto t_start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("haversine", N, KOKKOS_LAMBDA(int p) {
      double ax = a_lon(p) * DEGREE_TO_RADIAN;
      double ay = a_lat(p) * DEGREE_TO_RADIAN;
      double bx = b_lon(p) * DEGREE_TO_RADIAN;
      double by = b_lat(p) * DEGREE_TO_RADIAN;
      double x        = (bx - ax) / 2.0;
      double y        = (by - ay) / 2.0;
      double sinysqrd = sin(y) * sin(y);
      double sinxsqrd = sin(x) * sin(x);
      double scale    = cos(ay) * cos(by);
      dist(p) = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
    });
  }

  Kokkos::fence();
  auto t_end = std::chrono::steady_clock::now();
  auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9) / repeat);

  auto h_dist = Kokkos::create_mirror_view(dist);
  Kokkos::deep_copy(h_dist, dist);

  double max_error = 0.0;
  for (int i = 0; i < N; i++) {
    double err = fabs(h_dist(i) - expected[i]);
    if (err > max_error) max_error = err;
  }
  printf("The maximum error in distance is %f\n", max_error);

  } // end Kokkos scope
  Kokkos::finalize();
  return 0;
}
