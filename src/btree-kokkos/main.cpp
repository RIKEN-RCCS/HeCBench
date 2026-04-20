// GPU hash map benchmark – Kokkos port of btree-cuda.
// Uses open-addressing (linear probing) as a portable substitute for the
// warp-cooperative GPU B-tree in the original CUDA code.
// Interface: insert(keys, values), search(queries) → results
// Usage: ./main <numKeys> <numQueries>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

static constexpr uint32_t EMPTY_KEY   = 0xFFFFFFFFu;
static constexpr uint32_t EMPTY_VALUE = 0u;

// ─── Insert kernel ────────────────────────────────────────────────────────────

void hash_insert(
    const Kokkos::View<uint32_t*>& d_keys,
    const Kokkos::View<uint32_t*>& d_values,
    Kokkos::View<uint32_t*>& tbl_keys,
    Kokkos::View<uint32_t*>& tbl_values,
    uint32_t numKeys,
    uint32_t table_size)
{
  Kokkos::parallel_for("insert", numKeys,
    KOKKOS_LAMBDA(int i) {
      uint32_t key = d_keys(i);
      uint32_t val = d_values(i);
      uint32_t h   = key % table_size;
      while (true) {
        uint32_t old = Kokkos::atomic_compare_exchange(
                         &tbl_keys(h), EMPTY_KEY, key);
        if (old == EMPTY_KEY || old == key) {
          tbl_values(h) = val;
          break;
        }
        h = (h + 1) % table_size;
      }
    });
}

// ─── Search kernel ────────────────────────────────────────────────────────────

void hash_search(
    const Kokkos::View<uint32_t*>& d_queries,
    const Kokkos::View<uint32_t*>& tbl_keys,
    const Kokkos::View<uint32_t*>& tbl_values,
    Kokkos::View<uint32_t*>& d_results,
    uint32_t numQueries,
    uint32_t table_size)
{
  Kokkos::parallel_for("search", numQueries,
    KOKKOS_LAMBDA(int i) {
      uint32_t key = d_queries(i);
      uint32_t h   = key % table_size;
      while (true) {
        if (tbl_keys(h) == key) {
          d_results(i) = tbl_values(h);
          break;
        }
        if (tbl_keys(h) == EMPTY_KEY) {
          d_results(i) = EMPTY_VALUE;
          break;
        }
        h = (h + 1) % table_size;
      }
    });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  uint32_t numKeys    = 1u << 20;
  uint32_t numQueries = numKeys;
  if (argc > 1) numKeys    = (uint32_t)std::atoi(argv[1]);
  if (argc > 2) numQueries = (uint32_t)std::atoi(argv[2]);

  printf("Building the table with %u keys\n",    numKeys);
  printf("Searching the table with %u queries\n", numQueries);

  std::mt19937 rng(std::random_device{}());

  // Prepare keys [0, numKeys) shuffled; values == keys
  std::vector<uint32_t> h_keys(numKeys);
  std::iota(h_keys.begin(), h_keys.end(), 0u);
  std::shuffle(h_keys.begin(), h_keys.end(), rng);

  std::vector<uint32_t> h_values(numKeys);
  for (uint32_t i = 0; i < numKeys; ++i) h_values[i] = h_keys[i];

  // Prepare query keys [0, 2*numKeys) shuffled
  std::vector<uint32_t> h_queries(numQueries);
  {
    std::vector<uint32_t> pool(numQueries * 2);
    std::iota(pool.begin(), pool.end(), 0u);
    std::shuffle(pool.begin(), pool.end(), rng);
    std::copy(pool.begin(), pool.begin() + numQueries, h_queries.begin());
  }

  Kokkos::initialize(argc, argv);
  {
    const uint32_t table_size = numKeys * 2;  // load factor ~0.5

    // Device buffers for input data
    Kokkos::View<uint32_t*> d_keys("d_keys",       numKeys);
    Kokkos::View<uint32_t*> d_values("d_values",   numKeys);
    Kokkos::View<uint32_t*> d_queries("d_queries", numQueries);
    Kokkos::View<uint32_t*> d_results("d_results", numQueries);

    // Hash table
    Kokkos::View<uint32_t*> tbl_keys("tbl_keys",     table_size);
    Kokkos::View<uint32_t*> tbl_values("tbl_values", table_size);

    // Upload host data
    {
      auto hk = Kokkos::create_mirror_view(d_keys);
      auto hv = Kokkos::create_mirror_view(d_values);
      auto hq = Kokkos::create_mirror_view(d_queries);
      for (uint32_t i = 0; i < numKeys;    ++i) { hk(i) = h_keys[i]; hv(i) = h_values[i]; }
      for (uint32_t i = 0; i < numQueries; ++i)   hq(i) = h_queries[i];
      Kokkos::deep_copy(d_keys,    hk);
      Kokkos::deep_copy(d_values,  hv);
      Kokkos::deep_copy(d_queries, hq);
    }

    // Initialise table with sentinel values
    Kokkos::deep_copy(tbl_keys,   EMPTY_KEY);
    Kokkos::deep_copy(tbl_values, EMPTY_VALUE);
    Kokkos::fence();

    auto t_start = std::chrono::high_resolution_clock::now();

    // Build
    hash_insert(d_keys, d_values, tbl_keys, tbl_values, numKeys, table_size);
    Kokkos::fence();

    auto t_build = std::chrono::high_resolution_clock::now();

    // Search
    hash_search(d_queries, tbl_keys, tbl_values, d_results, numQueries, table_size);
    Kokkos::fence();

    auto t_end = std::chrono::high_resolution_clock::now();

    double build_ms  = std::chrono::duration_cast<std::chrono::microseconds>(
                         t_build - t_start).count() / 1e3;
    double search_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                         t_end   - t_build).count() / 1e3;
    double total_ms  = build_ms + search_ms;

    printf("Build  time: %.3f ms\n",  build_ms);
    printf("Search time: %.3f ms\n",  search_ms);
    printf("Total  time: %.3f ms\n",  total_ms);

    // Verify
    auto h_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_results);
    int errors = 0;
    for (uint32_t i = 0; i < numQueries; ++i) {
      uint32_t q = h_queries[i];
      if (q < numKeys) {
        // Key was inserted; value == key
        if (h_results(i) != q) ++errors;
      } else {
        // Key was NOT inserted; should return 0
        if (h_results(i) != 0u) ++errors;
      }
    }
    if (errors == 0)
      printf("PASS\n");
    else
      printf("VERIFICATION FAILED: %d errors\n", errors);
  }
  Kokkos::finalize();
  return 0;
}
