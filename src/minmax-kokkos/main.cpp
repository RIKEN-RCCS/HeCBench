// minmax-kokkos/main.cpp
// Port of minmax-cuda: finds min/max point by squared distance from origin
// using Kokkos::parallel_reduce with MinLoc/MaxLoc reducers.

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// vec_2d – lightweight 2D point (inlined from vec_2d.hpp)
// ---------------------------------------------------------------------------
template <typename T>
struct vec_2d {
  T x{}, y{};
};

template <typename T>
bool operator==(vec_2d<T> const& a, vec_2d<T> const& b) {
  return a.x == b.x && a.y == b.y;
}

template <typename T>
vec_2d<T> operator-(vec_2d<T> const& a, vec_2d<T> const& b) {
  return {a.x - b.x, a.y - b.y};
}

// ---------------------------------------------------------------------------
// Point generation (inlined from utils.hpp)
// ---------------------------------------------------------------------------
template <typename T>
vec_2d<T> random_point(vec_2d<T> minXY, vec_2d<T> maxXY) {
  T x = minXY.x + (maxXY.x - minXY.x) * static_cast<T>(rand()) / RAND_MAX;
  T y = minXY.y + (maxXY.y - minXY.y) * static_cast<T>(rand()) / RAND_MAX;
  return {x, y};
}

template <typename T>
auto generate_rects(T size) {
  T phi = static_cast<T>((1.0 + std::sqrt(5.0)) * 0.5);
  vec_2d<T> tl{T{0}, T{0}};
  vec_2d<T> br{size, size};
  std::size_t num_points = 0;
  std::vector<std::tuple<std::size_t, vec_2d<T>, vec_2d<T>>> rects;

  do {
    vec_2d<T> area = br - tl;
    switch (rects.size() % 4) {
      case 0: br.x = tl.x - (tl.x - br.x) / phi; break;
      case 1: br.y = tl.y - (tl.y - br.y) / phi; break;
      case 2: tl.x = tl.x + (br.x - tl.x) / phi; break;
      case 3: tl.y = tl.y + (br.y - tl.y) / phi; break;
    }
    area = br - tl;
    auto n = static_cast<std::size_t>(std::sqrt(area.x * area.y * 1'000'000));
    rects.emplace_back(n, tl, br);
    num_points += n;
  } while ((br - tl).x > 1 && (br - tl).y > 1);

  return std::make_pair(num_points, std::move(rects));
}

template <typename T>
std::pair<std::size_t, std::vector<vec_2d<T>>> generate_points(T size) {
  auto [total, rects] = generate_rects(size);
  srand(123);
  std::vector<vec_2d<T>> pts(total);
  std::size_t offset = 0;
  for (auto const& r : rects) {
    auto n  = std::get<0>(r);
    auto tl = std::get<1>(r);
    auto br = std::get<2>(r);
    std::generate(pts.begin() + offset, pts.begin() + offset + n,
                  [&] { return random_point<T>(tl, br); });
    offset += n;
  }
  return {total, pts};
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
template <typename T>
void eval(T bounding_box_size, int repeat) {
  auto [total_points, points] = generate_points(bounding_box_size);
  printf("Total number of points: %zu\n", total_points);

  // Copy host points into a Kokkos device view
  Kokkos::View<vec_2d<T>*> d_pts("d_pts", total_points);
  {
    auto h_pts = Kokkos::create_mirror_view(d_pts);
    for (std::size_t i = 0; i < total_points; ++i) h_pts(i) = points[i];
    Kokkos::deep_copy(d_pts, h_pts);
  }

  using ValLoc = Kokkos::ValLocScalar<T, int64_t>;
  ValLoc sep_min{}, sep_max{}, cmb_min{}, cmb_max{};

  // --- separate min + max ---
  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; ++r) {
    Kokkos::parallel_reduce(
        "min_element", total_points,
        KOKKOS_LAMBDA(int64_t i, ValLoc& upd) {
          T d2 = d_pts(i).x * d_pts(i).x + d_pts(i).y * d_pts(i).y;
          if (d2 < upd.val) { upd.val = d2; upd.loc = i; }
        },
        Kokkos::MinLoc<T, int64_t>(sep_min));

    Kokkos::parallel_reduce(
        "max_element", total_points,
        KOKKOS_LAMBDA(int64_t i, ValLoc& upd) {
          T d2 = d_pts(i).x * d_pts(i).x + d_pts(i).y * d_pts(i).y;
          if (d2 > upd.val) { upd.val = d2; upd.loc = i; }
        },
        Kokkos::MaxLoc<T, int64_t>(sep_max));
  }

  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  printf("Average execution time of min() and max(): %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
             * 1e-3 / repeat);

  // --- combined minmax (two reducers in one pass) ---
  Kokkos::fence();
  t0 = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; ++r) {
    Kokkos::parallel_reduce(
        "minmax_element", total_points,
        KOKKOS_LAMBDA(int64_t i, ValLoc& mn, ValLoc& mx) {
          T d2 = d_pts(i).x * d_pts(i).x + d_pts(i).y * d_pts(i).y;
          if (d2 < mn.val) { mn.val = d2; mn.loc = i; }
          if (d2 > mx.val) { mx.val = d2; mx.loc = i; }
        },
        Kokkos::MinLoc<T, int64_t>(cmb_min),
        Kokkos::MaxLoc<T, int64_t>(cmb_max));
  }

  Kokkos::fence();
  t1 = std::chrono::steady_clock::now();
  printf("Average execution time of minmax(): %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
             * 1e-3 / repeat);

  // --- CPU reference ---
  auto dist2 = [](vec_2d<T> const& p) { return p.x * p.x + p.y * p.y; };
  auto cmp   = [&](vec_2d<T> const& a, vec_2d<T> const& b) {
    return dist2(a) < dist2(b);
  };
  auto ref_min = *std::min_element(points.begin(), points.end(), cmp);
  auto ref_max = *std::max_element(points.begin(), points.end(), cmp);

  bool ok = (points[sep_min.loc] == ref_min) && (points[sep_max.loc] == ref_max);
  ok     &= (points[cmb_min.loc] == ref_min) && (points[cmb_max.loc] == ref_max);
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <bounding-box size> <repeat>\n", argv[0]);
    return 1;
  }
  Kokkos::initialize(argc, argv);
  {
    const int size   = std::atoi(argv[1]);
    const int repeat = std::atoi(argv[2]);
    eval(static_cast<float>(size), repeat);
  }
  Kokkos::finalize();
  return 0;
}
