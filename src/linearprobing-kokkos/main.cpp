#include <Kokkos_Core.hpp>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>

// Hash table constants
const uint32_t kHashTableCapacity = 64 * 1024 * 1024;
const uint32_t kNumKeyValues = kHashTableCapacity / 2;
const uint32_t kEmpty = 0xffffffff;

struct KeyValue {
  uint32_t key;
  uint32_t value;
};

// 32-bit Murmur3 hash
KOKKOS_INLINE_FUNCTION uint32_t hash(uint32_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k & (kHashTableCapacity - 1);
}

using KeyView   = Kokkos::View<uint32_t *>;
using ValueView = Kokkos::View<uint32_t *>;
using KVView    = Kokkos::View<KeyValue *>;

// ---- Insert ---------------------------------------------------------------
double insert_hashtable(KeyView d_keys, ValueView d_values,
                        const Kokkos::View<KeyValue *> d_kvs,
                        uint32_t num_kvs) {
  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
      "insert_hashtable",
      Kokkos::RangePolicy<>(0, num_kvs),
      KOKKOS_LAMBDA(const uint32_t tid) {
        uint32_t key   = d_kvs(tid).key;
        uint32_t value = d_kvs(tid).value;
        uint32_t slot  = hash(key);
        while (true) {
          // Atomically: if slot key is kEmpty, replace with our key
          uint32_t prev =
              Kokkos::atomic_compare_exchange(&d_keys(slot), kEmpty, key);
          if (prev == kEmpty || prev == key) {
            d_values(slot) = value;
            break;
          }
          slot = (slot + 1) & (kHashTableCapacity - 1);
        }
      });
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (double)time;
}

// ---- Delete ---------------------------------------------------------------
double delete_hashtable(KeyView d_keys, ValueView d_values,
                        const Kokkos::View<KeyValue *> d_kvs,
                        uint32_t num_kvs) {
  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
      "delete_hashtable",
      Kokkos::RangePolicy<>(0, num_kvs),
      KOKKOS_LAMBDA(const uint32_t tid) {
        uint32_t key  = d_kvs(tid).key;
        uint32_t slot = hash(key);
        while (true) {
          if (d_keys(slot) == key) {
            d_values(slot) = kEmpty;
            break;
          }
          if (d_keys(slot) == kEmpty) break;
          slot = (slot + 1) & (kHashTableCapacity - 1);
        }
      });
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (double)time;
}

// ---- Iterate --------------------------------------------------------------
std::vector<KeyValue> iterate_hashtable(KeyView d_keys, ValueView d_values,
                                        KVView d_iter_kvs) {
  Kokkos::View<uint32_t> d_size("kvs_size");
  Kokkos::deep_copy(d_size, 0u);

  auto start = std::chrono::steady_clock::now();

  Kokkos::parallel_for(
      "iterate_hashtable",
      Kokkos::RangePolicy<>(0, kHashTableCapacity),
      KOKKOS_LAMBDA(const uint32_t tid) {
        if (d_keys(tid) != kEmpty) {
          uint32_t value = d_values(tid);
          if (value != kEmpty) {
            uint32_t pos = Kokkos::atomic_fetch_add(&d_size(), 1u);
            d_iter_kvs(pos).key   = d_keys(tid);
            d_iter_kvs(pos).value = value;
          }
        }
      });
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (iterate): %f (s)\n", time * 1e-9f);

  uint32_t num_kvs = 0;
  Kokkos::deep_copy(num_kvs, d_size);

  // Copy result back to host
  auto h_iter_kvs = Kokkos::create_mirror_view(d_iter_kvs);
  Kokkos::deep_copy(h_iter_kvs, d_iter_kvs);

  std::vector<KeyValue> kvs(num_kvs);
  for (uint32_t i = 0; i < num_kvs; i++) kvs[i] = h_iter_kvs(i);
  return kvs;
}

// ---- Correctness check ----------------------------------------------------
void test_correctness(std::vector<KeyValue> insert_kvs,
                      std::vector<KeyValue> delete_kvs,
                      std::vector<KeyValue> kvs) {
  printf("Testing that there are no duplicate keys...\n");
  std::unordered_set<uint32_t> unique_keys;
  for (uint32_t i = 0; i < kvs.size(); i++) {
    if (i % 10000000 == 0)
      printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());
    if (unique_keys.count(kvs[i].key)) {
      printf("Duplicate key found in hashtable at slot %d\n", i);
      exit(-1);
    }
    unique_keys.insert(kvs[i].key);
  }

  printf("Building unordered_map from original list...\n");
  std::unordered_map<uint32_t, std::vector<uint32_t>> all_kvs_map;
  for (auto &kv : insert_kvs)
    all_kvs_map[kv.key].push_back(kv.value);
  for (auto &kv : delete_kvs)
    all_kvs_map.erase(kv.key);

  if (unique_keys.size() != all_kvs_map.size()) {
    printf("# of unique keys in hashtable is incorrect\n");
    exit(-1);
  }

  printf("Testing that each key/value in hashtable is in the original list...\n");
  for (uint32_t i = 0; i < kvs.size(); i++) {
    if (i % 10000000 == 0)
      printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());
    auto it = all_kvs_map.find(kvs[i].key);
    if (it == all_kvs_map.end()) { printf("Hashtable key not found\n"); exit(-1); }
    auto &vals = it->second;
    if (std::find(vals.begin(), vals.end(), kvs[i].value) == vals.end()) {
      printf("Hashtable value not found\n"); exit(-1);
    }
  }
  printf("Correctness check passed.\n");
}

// ---- std::unordered_map timing --------------------------------------------
void test_unordered_map(std::vector<KeyValue> insert_kvs,
                        std::vector<KeyValue> delete_kvs) {
  printf("Timing std::unordered_map...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  {
    std::unordered_map<uint32_t, uint32_t> m;
    for (auto &kv : insert_kvs) m[kv.key] = kv.value;
    for (auto &kv : delete_kvs) m.erase(kv.key);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
  printf("Total time for std::unordered_map: %f ms (%f million keys/second)\n",
         ms, kNumKeyValues / (ms / 1000.0) / 1e6);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of insert batches> <number of delete batches>\n", argv[0]);
    return 1;
  }
  const uint32_t num_insert_batches = atoi(argv[1]);
  const uint32_t num_delete_batches = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    uint32_t seed = 123;
    std::mt19937 rnd(seed);
    printf("Random number generator seed = %u\n", seed);
    printf("Initializing keyvalue pairs with random numbers...\n");

    std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);
    std::vector<KeyValue> insert_kvs(kNumKeyValues);
    for (auto &kv : insert_kvs) { kv.key = dis(rnd); kv.value = dis(rnd); }

    std::vector<KeyValue> all_kvs_copy = insert_kvs;
    std::shuffle(all_kvs_copy.begin(), all_kvs_copy.end(), rnd);
    std::vector<KeyValue> delete_kvs(all_kvs_copy.begin(),
                                     all_kvs_copy.begin() + kNumKeyValues / 2);

    printf("Testing insertion/deletion of %u/%u elements...\n",
           (uint32_t)insert_kvs.size(), (uint32_t)delete_kvs.size());

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Device hash table (separate key/value views)
    KeyView   d_keys("d_keys",   kHashTableCapacity);
    ValueView d_values("d_values", kHashTableCapacity);
    Kokkos::deep_copy(d_keys,   kEmpty);
    Kokkos::deep_copy(d_values, kEmpty);

    uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
    uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;

    // Device buffers for batch transfer
    Kokkos::View<KeyValue *> d_ins_kvs("d_ins_kvs", num_inserts_per_batch);
    Kokkos::View<KeyValue *> d_del_kvs("d_del_kvs", num_deletes_per_batch);
    Kokkos::View<KeyValue *> d_iter_kvs("d_iter_kvs", kNumKeyValues);

    auto h_ins_kvs = Kokkos::create_mirror_view(d_ins_kvs);
    auto h_del_kvs = Kokkos::create_mirror_view(d_del_kvs);

    double total_insert_time = 0.0;
    for (uint32_t i = 0; i < num_insert_batches; i++) {
      memcpy(h_ins_kvs.data(), insert_kvs.data() + i * num_inserts_per_batch,
             sizeof(KeyValue) * num_inserts_per_batch);
      Kokkos::deep_copy(d_ins_kvs, h_ins_kvs);
      total_insert_time += insert_hashtable(d_keys, d_values, d_ins_kvs, num_inserts_per_batch);
    }
    printf("Average kernel execution time (insert): %f (s)\n",
           (total_insert_time * 1e-9) / num_insert_batches);

    double total_delete_time = 0.0;
    for (uint32_t i = 0; i < num_delete_batches; i++) {
      memcpy(h_del_kvs.data(), delete_kvs.data() + i * num_deletes_per_batch,
             sizeof(KeyValue) * num_deletes_per_batch);
      Kokkos::deep_copy(d_del_kvs, h_del_kvs);
      total_delete_time += delete_hashtable(d_keys, d_values, d_del_kvs, num_deletes_per_batch);
    }
    printf("Average kernel execution time (delete): %f (s)\n",
           (total_delete_time * 1e-9) / num_delete_batches);

    std::vector<KeyValue> kvs = iterate_hashtable(d_keys, d_values, d_iter_kvs);

    auto wall_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start).count() / 1000.0;
    printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n",
           ms, kNumKeyValues / (ms / 1000.0) / 1e6);

    test_unordered_map(insert_kvs, delete_kvs);
    test_correctness(insert_kvs, delete_kvs, kvs);
  }
  Kokkos::finalize();
  printf("Success\n");
  return 0;
}
