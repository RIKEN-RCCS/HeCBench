// Kokkos port of coordinates (cuspatial) CUDA benchmark.
// Converts lon/lat degree coordinates to flat-Earth Cartesian using
// Kokkos::parallel_for. Results are verified against CPU reference.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
template <typename T>
struct lonlat_2d { T x, y; };  // longitude, latitude in degrees

template <typename T>
struct cartesian_2d { T x, y; };

// ---------------------------------------------------------------------------
// Transformation functor
// ---------------------------------------------------------------------------
template <typename T>
struct to_cartesian_functor {
  lonlat_2d<T> origin;

  KOKKOS_INLINE_FUNCTION
  cartesian_2d<T> operator()(const lonlat_2d<T>& p) const {
    constexpr T pi = T(3.14159265358979323846);
    T ox_rad = origin.x * pi / T(180);
    T oy_rad = origin.y * pi / T(180);
    T p_rad_x = p.x * pi / T(180);
    T p_rad_y = p.y * pi / T(180);
    return {(p_rad_x - ox_rad) * Kokkos::cos(oy_rad), p_rad_y - oy_rad};
  }
};

// ---------------------------------------------------------------------------
// Run one benchmark for type T
// ---------------------------------------------------------------------------
template <typename T>
bool run(int num_coords, int repeat) {
  using lonlat   = lonlat_2d<T>;
  using cartesian = cartesian_2d<T>;

  // Generate random lon/lat input on the host
  std::srand(42);
  Kokkos::View<lonlat*, Kokkos::HostSpace> h_input("h_input", num_coords);
  for (int i = 0; i < num_coords; ++i) {
    h_input(i).x = T(-180) + T(360) * std::rand() / RAND_MAX;  // lon [-180,180]
    h_input(i).y = T(-90)  + T(180) * std::rand() / RAND_MAX;  // lat [-90,90]
  }

  lonlat origin;
  origin.x = T(-73.9857);  // New York lon
  origin.y = T( 40.7484);  // New York lat

  // CPU reference
  to_cartesian_functor<T> functor{origin};
  Kokkos::View<cartesian*, Kokkos::HostSpace> h_ref("h_ref", num_coords);
  for (int i = 0; i < num_coords; ++i)
    h_ref(i) = functor(h_input(i));

  // Copy input to device
  Kokkos::View<lonlat*> d_input("d_input", num_coords);
  Kokkos::deep_copy(d_input, h_input);

  Kokkos::View<cartesian*> d_output("d_output", num_coords);

  // Warm-up
  Kokkos::parallel_for(
      "coordinates_warmup", Kokkos::RangePolicy<>(0, num_coords),
      KOKKOS_LAMBDA(const int i) {
        d_output(i) = functor(d_input(i));
      });
  Kokkos::fence();

  // Timed runs
  for (int r = 0; r < repeat; ++r) {
    Kokkos::parallel_for(
        "coordinates", Kokkos::RangePolicy<>(0, num_coords),
        KOKKOS_LAMBDA(const int i) {
          d_output(i) = functor(d_input(i));
        });
  }
  Kokkos::fence();

  // Verify
  auto h_output = Kokkos::create_mirror_view(d_output);
  Kokkos::deep_copy(h_output, d_output);

  constexpr T eps = (sizeof(T) == 4) ? T(1e-5) : T(1e-10);
  bool pass = true;
  for (int i = 0; i < num_coords; ++i) {
    if (std::fabs(h_output(i).x - h_ref(i).x) > eps ||
        std::fabs(h_output(i).y - h_ref(i).y) > eps) {
      printf("  MISMATCH at %d: got (%.10g, %.10g) expected (%.10g, %.10g)\n",
             i,
             (double)h_output(i).x, (double)h_output(i).y,
             (double)h_ref(i).x,   (double)h_ref(i).y);
      pass = false;
      break;
    }
  }
  return pass;
}

int main(int argc, char* argv[]) {
  int num_coords = (argc > 1) ? std::atoi(argv[1]) : 100000;
  int repeat     = (argc > 2) ? std::atoi(argv[2]) : 100;

  Kokkos::initialize(argc, argv);
  {
    printf("coordinates: num_coords=%d  repeat=%d\n", num_coords, repeat);

    bool ok_d = run<double>(num_coords, repeat);
    printf("double: %s\n", ok_d ? "PASS" : "FAIL");

    bool ok_f = run<float>(num_coords, repeat);
    printf("float:  %s\n", ok_f ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
