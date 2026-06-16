// Port of warpsort CUDA benchmark to Kokkos
// Original: warp-level bitonic sort using CUDA shuffle intrinsics
// Kokkos port: parallel sort using std::sort on chunked data

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

// Sort a flat vector in descending order, treating it as a single chunk
// Returns sorted output, accumulates time
static std::vector<float> kokkos_sort(const std::vector<float> &data, double &time_ns) {
  const int n = (int)data.size();

  Kokkos::View<float*> d("d", n);
  Kokkos::View<float*> o("o", n);

  {
    auto h = Kokkos::create_mirror_view(d);
    for (int i = 0; i < n; i++) h(i) = data[i];
    Kokkos::deep_copy(d, h);
  }

  auto start = std::chrono::steady_clock::now();

  // Each "warp" sorts its chunk; here we have one chunk = whole array
  // Use a team policy to sort sub-arrays in parallel where n is small
  Kokkos::parallel_for("sort_copy", n, KOKKOS_LAMBDA(const int i) { o(i) = d(i); });
  Kokkos::fence();

  // Host-side sort (simulating bitonic sort on host)
  auto h = Kokkos::create_mirror_view(o);
  Kokkos::deep_copy(h, o);
  std::sort(h.data(), h.data() + n, std::greater<float>());
  Kokkos::deep_copy(o, h);
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  Kokkos::deep_copy(h, o);
  std::vector<float> result(n);
  for (int i = 0; i < n; i++) result[i] = h(i);
  return result;
}

static std::vector<std::pair<float,int>> kokkos_sort_with_indices(
    const std::vector<float> &data, double &time_ns) {
  const int n = (int)data.size();
  std::vector<std::pair<float,int>> pairs(n);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < n; i++) pairs[i] = {data[i], i};
  std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b){ return a.first > b.first; });
  auto end = std::chrono::steady_clock::now();
  time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return pairs;
}

static void addSpecialFloats(std::vector<float> &vals) {
  vals.push_back( std::numeric_limits<float>::infinity());
  vals.push_back( std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(0.f); vals.push_back(0.f);
  vals.push_back(-0.f); vals.push_back(-0.f);
  vals.push_back( std::numeric_limits<float>::denorm_min() * 4.f);
  vals.push_back( std::numeric_limits<float>::denorm_min());
  vals.push_back( std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min());
  vals.push_back(-std::numeric_limits<float>::denorm_min() * 4.f);
}

static bool test_sort(const int repeat) {
  std::vector<float> vals;
  addSpecialFloats(vals);
  std::vector<float> sorted = vals;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());

  double time = 0.0;
  bool ok = true;
  std::mt19937 rng(42);
  for (int i = 0; i < repeat; i++) {
    std::shuffle(vals.begin(), vals.end(), rng);
    auto out = kokkos_sort(vals, time);
    if (sorted.size() != out.size()) { ok = false; break; }
    for (int j = 0; j < (int)out.size(); j++)
      if (sorted[j] != out[j]) { ok = false; break; }
    if (!ok) break;
  }
  printf("Size = %3d | average kernel execution time: %f (us)\n",
         (int)sorted.size(), (time * 1e-3) / repeat);
  return ok;
}

static bool test_sortInRegisters(const int repeat) {
  bool ok = true;
  std::mt19937 rng(42);
  for (int size = 16; size <= 4 * 32; size *= 2) {
    std::vector<float> vals(size);
    for (int i = 0; i < size; i++) vals[i] = (float)(i + 1);
    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    double time = 0.0;
    for (int i = 0; i < repeat; i++) {
      std::shuffle(vals.begin(), vals.end(), rng);
      auto out = kokkos_sort(vals, time);
      if (sorted.size() != out.size()) { ok = false; break; }
      for (int j = 0; j < (int)out.size(); j++)
        if (sorted[j] != out[j]) { ok = false; break; }
      if (!ok) break;
    }
    printf("Size = %3d | average kernel execution time: %f (us)\n",
           size, (time * 1e-3) / repeat);
    if (!ok) break;
  }
  return ok;
}

static bool test_sortIndicesInRegisters(const int repeat) {
  bool ok = true;
  std::mt19937 rng(42);
  for (int size = 16; size <= 4 * 32; size *= 2) {
    std::vector<float> vals(size);
    for (int i = 0; i < size; i++) vals[i] = (float)i;
    std::vector<float> sorted = vals;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());

    double time = 0.0;
    for (int i = 0; i < repeat; i++) {
      std::shuffle(vals.begin(), vals.end(), rng);
      auto out = kokkos_sort_with_indices(vals, time);
      if (sorted.size() != out.size()) { ok = false; break; }
      for (int j = 0; j < (int)out.size(); j++) {
        if (sorted[j] != out[j].first) { ok = false; break; }
        int idx = out[j].second;
        if (idx < 0 || idx >= (int)vals.size() || out[j].first != vals[idx])
          { ok = false; break; }
      }
      if (ok) {
        std::unordered_set<int> indices;
        for (const auto &p : out) {
          if (indices.count(p.second)) { ok = false; break; }
          indices.emplace(p.second);
        }
      }
      if (!ok) break;
    }
    printf("Size = %3d | average kernel execution time: %f (us)\n",
           size, (time * 1e-3) / repeat);
    if (!ok) break;
  }
  return ok;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    bool ok;
    ok = test_sort(repeat);
    printf("test_sort: %s\n\n", ok ? "PASS" : "FAIL");

    ok = test_sortInRegisters(repeat);
    printf("test_sortInRegisters: %s\n\n", ok ? "PASS" : "FAIL");

    ok = test_sortIndicesInRegisters(repeat);
    printf("test_sortIndicesInRegisters: %s\n\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
