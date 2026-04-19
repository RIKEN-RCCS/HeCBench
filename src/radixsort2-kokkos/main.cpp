/*
 * Kokkos port of radixsort2-cuda benchmark.
 *
 * Original: Thrust sort and sort_by_key on device.
 * Port:
 *   - Keys-only:   Kokkos::sort on a device View
 *   - Key-value:   host-side std::sort on (key, value) pairs (permutation)
 * Supports unsigned int keys (default) and float keys (-float flag).
 * Measures throughput (MElements/s) and verifies the sorted result.
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Simple command-line helpers
// ---------------------------------------------------------------------------
static bool hasCmdFlag(int argc, char** argv, const char* flag)
{
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], flag) == 0)
      return true;
  return false;
}

static int getCmdInt(int argc, char** argv, const char* flag, int def)
{
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0)
      return atoi(argv[i + 1]);
  return def;
}

// ---------------------------------------------------------------------------
// Keys-only sort with Kokkos::sort
// ---------------------------------------------------------------------------
template <typename T>
static bool keysOnlySort(int numElements, int numIterations, bool floatKeys)
{
  printf("\nSorting %d %s keys (keys-only, Kokkos::sort)\n",
         numElements, floatKeys ? "float" : "unsigned int");

  std::mt19937 rng(19937);
  std::vector<T> h_keys(numElements);

  if (floatKeys) {
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& k : h_keys) k = dist(rng);
  } else {
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    for (auto& k : h_keys) k = (T)dist(rng);
  }

  Kokkos::View<T*> d_keys("d_keys", numElements);
  Kokkos::View<T*, Kokkos::HostSpace> h_view(h_keys.data(), numElements);

  double totalTime = 0.0;

  for (int iter = 0; iter < numIterations; ++iter) {
    Kokkos::deep_copy(d_keys, h_view); // reset to unsorted
    Kokkos::fence();

    auto t0 = std::chrono::steady_clock::now();
    Kokkos::sort(d_keys);
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();

    totalTime +=
        std::chrono::duration<double>(t1 - t0).count();
  }

  double avgTime = totalTime / numIterations;
  printf("Throughput = %.4f MElements/s, Time = %.5f s\n",
         1.0e-6 * numElements / avgTime, avgTime);

  // Verify: copy sorted result back and check order
  Kokkos::View<T*, Kokkos::HostSpace> h_sorted("h_sorted", numElements);
  Kokkos::deep_copy(h_sorted, d_keys);

  for (int i = 1; i < numElements; ++i) {
    if (h_sorted(i) < h_sorted(i - 1))
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Key-value sort: host-side std::sort on (key, index) pairs
// ---------------------------------------------------------------------------
template <typename T>
static bool keyValueSort(int numElements, int numIterations, bool floatKeys)
{
  printf("\nSorting %d %s keys with values (host std::sort)\n",
         numElements, floatKeys ? "float" : "unsigned int");

  std::mt19937 rng(19937);
  std::vector<T> h_keys_orig(numElements);

  if (floatKeys) {
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& k : h_keys_orig) k = dist(rng);
  } else {
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    for (auto& k : h_keys_orig) k = (T)dist(rng);
  }

  // values are indices 0..N-1
  std::vector<unsigned int> h_values_orig(numElements);
  for (int i = 0; i < numElements; ++i)
    h_values_orig[i] = (unsigned int)i;

  // Working copy of (key, value) pairs
  using Pair = std::pair<T, unsigned int>;
  std::vector<Pair> pairs(numElements);

  double totalTime = 0.0;

  for (int iter = 0; iter < numIterations; ++iter) {
    // Reset pairs
    for (int i = 0; i < numElements; ++i)
      pairs[i] = {h_keys_orig[i], h_values_orig[i]};

    auto t0 = std::chrono::steady_clock::now();
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.first < b.first; });
    auto t1 = std::chrono::steady_clock::now();

    totalTime +=
        std::chrono::duration<double>(t1 - t0).count();
  }

  double avgTime = totalTime / numIterations;
  printf("Throughput = %.4f MElements/s, Time = %.5f s\n",
         1.0e-6 * numElements / avgTime, avgTime);

  // Verify: keys are sorted
  for (int i = 1; i < numElements; ++i) {
    if (pairs[i].first < pairs[i - 1].first)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  printf("%s Starting...\n\n", argv[0]);

  const int  numElements  = getCmdInt(argc, argv, "-n",          1048576);
  const int  numIter      = getCmdInt(argc, argv, "-iterations", 100);
  const bool floatKeys    = hasCmdFlag(argc, argv, "-float");
  const bool keysOnly     = hasCmdFlag(argc, argv, "-keysonly");

  Kokkos::initialize(argc, argv);
  {
    bool ok = true;

    if (floatKeys) {
      ok &= keysOnlySort<float>(numElements, numIter, true);
      if (!keysOnly)
        ok &= keyValueSort<float>(numElements, numIter, true);
    } else {
      ok &= keysOnlySort<unsigned int>(numElements, numIter, false);
      if (!keysOnly)
        ok &= keyValueSort<unsigned int>(numElements, numIter, false);
    }

    printf("\n%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
