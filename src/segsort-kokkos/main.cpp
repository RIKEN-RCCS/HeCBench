#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

using index_t  = int;
using offset_t = int;
using key_t    = int;
using val_t    = uint64_t;

// CPU reference: stable_sort per segment (keys + values)
template<class K, class V>
static void gold_segsort(std::vector<K> &key, std::vector<V> &val,
                          const std::vector<offset_t> &seg, index_t num_segs)
{
  using P = std::pair<K, V>;
  std::vector<P> pairs;
  for (index_t s = 0; s < num_segs; s++) {
    offset_t st = seg[s], ed = seg[s + 1];
    pairs.clear();
    pairs.reserve(ed - st);
    for (index_t j = st; j < ed; j++) pairs.emplace_back(key[j], val[j]);
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const P &a, const P &b) { return a.first < b.first; });
    for (index_t j = st; j < ed; j++) {
      key[j] = pairs[j - st].first;
      val[j] = pairs[j - st].second;
    }
  }
}

// CPU reference: stable_sort per segment (keys only)
template<class K>
static void gold_segsort_keys(std::vector<K> &key,
                               const std::vector<offset_t> &seg, index_t num_segs)
{
  for (index_t s = 0; s < num_segs; s++)
    std::stable_sort(key.begin() + seg[s], key.begin() + seg[s + 1],
                     [](K a, K b) { return a < b; });
}

// After sorting by key, sort values within equal-key ranges so that the
// comparison between GPU and CPU results is key-independent.
template<class K, class V>
static void sort_vals_of_same_key(const std::vector<K> &key, std::vector<V> &val,
                                   const std::vector<offset_t> &seg, index_t num_segs)
{
  for (index_t s = 0; s < num_segs; s++) {
    offset_t st = seg[s], ed = seg[s + 1];
    for (offset_t i = st; i < ed; ) {
      auto r  = std::equal_range(key.begin() + st, key.begin() + ed, key[i]);
      offset_t i_st = (offset_t)std::distance(key.begin(), r.first);
      offset_t i_ed = (offset_t)std::distance(key.begin(), r.second);
      std::sort(val.begin() + i_st, val.begin() + i_ed,
                [](V a, V b) { return a < b; });
      i += i_ed - i_st;
    }
  }
}

static void run_test(index_t num_segs, index_t avg_seg_size, bool keys_only)
{
  index_t num_elements = num_segs * avg_seg_size;

  // Random data
  std::mt19937 rng(42);
  std::uniform_int_distribution<key_t> key_dist(0, num_elements - 1);
  std::uniform_int_distribution<val_t> val_dist(0, (val_t)UINT64_MAX);

  std::vector<key_t> key(num_elements), key_gold(num_elements);
  std::vector<val_t> val(num_elements), val_gold(num_elements);
  std::vector<offset_t> seg(num_segs + 1);

  for (index_t i = 0; i < num_elements; i++) {
    key[i] = key_gold[i] = key_dist(rng);
    val[i] = val_gold[i] = val_dist(rng);
  }
  // Uniform segment boundaries
  for (index_t s = 0; s <= num_segs; s++) seg[s] = s * avg_seg_size;

  // CPU gold reference
  if (keys_only)
    gold_segsort_keys(key_gold, seg, num_segs);
  else
    gold_segsort(key_gold, val_gold, seg, num_segs);

  // Kokkos implementation: parallel_for over segments, std::stable_sort inside
  Kokkos::View<key_t*, Kokkos::HostSpace> h_key("h_key", num_elements);
  Kokkos::View<val_t*, Kokkos::HostSpace> h_val("h_val", num_elements);
  Kokkos::View<offset_t*, Kokkos::HostSpace> h_seg("h_seg", num_segs + 1);

  for (index_t i = 0; i < num_elements; i++) { h_key(i) = key[i]; h_val(i) = val[i]; }
  for (index_t s = 0; s <= num_segs; s++) h_seg(s) = seg[s];

  auto start = std::chrono::steady_clock::now();

  if (keys_only) {
    Kokkos::parallel_for("segsort_keys",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_segs),
        [&](const index_t s) {
          offset_t st = h_seg(s), ed = h_seg(s + 1);
          std::stable_sort(h_key.data() + st, h_key.data() + ed,
                           [](key_t a, key_t b) { return a < b; });
        });
  } else {
    // Sort key+value pairs per segment
    Kokkos::parallel_for("segsort_pairs",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_segs),
        [&](const index_t s) {
          using P = std::pair<key_t, val_t>;
          offset_t st = h_seg(s), ed = h_seg(s + 1);
          std::vector<P> pairs(ed - st);
          for (offset_t j = st; j < ed; j++)
            pairs[j - st] = {h_key(j), h_val(j)};
          std::stable_sort(pairs.begin(), pairs.end(),
                           [](const P &a, const P &b) { return a.first < b.first; });
          for (offset_t j = st; j < ed; j++) {
            h_key(j) = pairs[j - st].first;
            h_val(j) = pairs[j - st].second;
          }
        });
  }
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;

  // Check keys
  index_t key_errors = 0;
  for (index_t i = 0; i < num_elements; i++)
    if (h_key(i) != key_gold[i]) key_errors++;

  const char *label = keys_only ? "keys " : "values";
  printf("%s: num_segs=%d, seg_size=%6d, time=%8.3f ms, status: %s\n",
         label, num_segs, avg_seg_size, ms,
         key_errors == 0 ? "PASS" : "FAIL");

  if (!keys_only) {
    // For value check we need to sort vals of same key in both arrays
    std::vector<key_t>  kk(h_key.data(), h_key.data() + num_elements);
    std::vector<val_t>  vv(h_val.data(), h_val.data() + num_elements);
    sort_vals_of_same_key(kk, vv, seg, num_segs);
    sort_vals_of_same_key(key_gold, val_gold, seg, num_segs);
    index_t val_errors = 0;
    for (index_t i = 0; i < num_elements; i++)
      if (vv[i] != val_gold[i]) val_errors++;
    printf("values check: %s\n", val_errors == 0 ? "PASS" : "FAIL");
  }
}

int main()
{
  Kokkos::initialize();
  {
    const index_t num_segs = 100;
    for (index_t avg : {1000, 10000, 100000}) {
      run_test(num_segs, avg, false); // keys + values
      run_test(num_segs, avg, true);  // keys only
    }
  }
  Kokkos::finalize();
  return 0;
}
