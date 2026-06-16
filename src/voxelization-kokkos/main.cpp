// Port of voxelization CUDA benchmark to Kokkos
// Generates synthetic LiDAR point cloud and performs voxelization + feature extraction
// (replaces file I/O with synthetic data generation)

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

static const unsigned int MAX_POINTS_NUM  = 300000;
static const int          FEATURE_NUM     = 5;

// Params matching the original benchmark
static const float MIN_X = -54.f, MAX_X = 54.f;
static const float MIN_Y = -54.f, MAX_Y = 54.f;
static const float MIN_Z = -5.0f, MAX_Z = 3.0f;
static const float PIL_X = 0.075f, PIL_Y = 0.075f, PIL_Z = 0.2f;
static const int   MAX_VOXELS = 160000;
static const int   MAX_PPV    = 10;

KOKKOS_INLINE_FUNCTION
uint64_t hash_fn(uint64_t k) {
  k ^= k >> 16; k *= 0x85ebca6b; k ^= k >> 13; k *= 0xc2b2ae35; k ^= k >> 16;
  return k;
}

// Insert key into open-addressing hash table; returns slot or -1 if full
KOKKOS_INLINE_FUNCTION
void insertHash(uint32_t key, Kokkos::View<uint32_t*> ht, uint32_t ht_size,
                Kokkos::View<uint32_t*> voxel_count) {
  uint64_t hv  = hash_fn(key);
  uint32_t slot = hv % (ht_size / 2);
  uint32_t empty_key = 0xFFFFFFFFu;
  while (true) {
    uint32_t pre = Kokkos::atomic_compare_exchange(&ht(slot), empty_key, key);
    if (pre == empty_key) {
      ht(slot + ht_size / 2) = Kokkos::atomic_fetch_add(&voxel_count(0), 1u);
      break;
    } else if (pre == key) {
      break;
    }
    slot = (slot + 1) % (ht_size / 2);
  }
}

KOKKOS_INLINE_FUNCTION
uint32_t lookupHash(uint32_t key, const Kokkos::View<uint32_t*> &ht, uint32_t ht_size) {
  uint64_t hv  = hash_fn(key);
  uint32_t slot = hv % (ht_size / 2);
  while (true) {
    if (ht(slot) == key) return ht(slot + ht_size / 2);
    if (ht(slot) == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    slot = (slot + 1) % (ht_size / 2);
  }
}

static int getGridX() { return (int)std::round((MAX_X - MIN_X) / PIL_X); }
static int getGridY() { return (int)std::round((MAX_Y - MIN_Y) / PIL_Y); }
static int getGridZ() { return (int)std::round((MAX_Z - MIN_Z) / PIL_Z); }

static void generateVoxels(const Kokkos::View<float*> &d_points,
                            size_t points_size, int repeat) {
  int gx = getGridX(), gy = getGridY(), gz = getGridZ();
  uint32_t ht_size = (uint32_t)points_size * 2 * 2;

  Kokkos::View<uint32_t*> d_ht("ht", ht_size);
  Kokkos::View<float*>    d_vox_tmp("vox_tmp",
      (size_t)MAX_VOXELS * MAX_PPV * FEATURE_NUM);
  Kokkos::View<uint32_t*> d_vox_num("vox_num", MAX_VOXELS);
  Kokkos::View<uint32_t*> d_real_num("real_num", 1);
  Kokkos::View<float*>    d_features("features",
      (size_t)MAX_VOXELS * FEATURE_NUM);

  Kokkos::fence();

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // Reset
    Kokkos::deep_copy(d_ht, 0xFFFFFFFFu);
    Kokkos::deep_copy(d_vox_num, 0u);
    Kokkos::deep_copy(d_real_num, 0u);
    Kokkos::deep_copy(d_vox_tmp, 0.f);
    Kokkos::fence();

    // Build hash table (assign voxel IDs)
    Kokkos::parallel_for("build_hash", (int)points_size, KOKKOS_LAMBDA(const int idx) {
      float px = d_points(FEATURE_NUM * idx);
      float py = d_points(FEATURE_NUM * idx + 1);
      float pz = d_points(FEATURE_NUM * idx + 2);
      int vx = (int)floorf((px - MIN_X) / PIL_X);
      int vy = (int)floorf((py - MIN_Y) / PIL_Y);
      int vz = (int)floorf((pz - MIN_Z) / PIL_Z);
      if (vx < 0 || vx >= gx || vy < 0 || vy >= gy || vz < 0 || vz >= gz) return;
      uint32_t voff = (uint32_t)vz * gy * gx + (uint32_t)vy * gx + vx;
      insertHash(voff, d_ht, ht_size, d_real_num);
    });
    Kokkos::fence();

    // Scatter points to voxels
    Kokkos::parallel_for("voxelize", (int)points_size, KOKKOS_LAMBDA(const int idx) {
      float px = d_points(FEATURE_NUM * idx);
      float py = d_points(FEATURE_NUM * idx + 1);
      float pz = d_points(FEATURE_NUM * idx + 2);
      if (px < MIN_X || px >= MAX_X || py < MIN_Y || py >= MAX_Y ||
          pz < MIN_Z || pz >= MAX_Z) return;
      int vx = (int)floorf((px - MIN_X) / PIL_X);
      int vy = (int)floorf((py - MIN_Y) / PIL_Y);
      int vz = (int)floorf((pz - MIN_Z) / PIL_Z);
      if (vx < 0 || vx >= gx || vy < 0 || vy >= gy || vz < 0 || vz >= gz) return;
      uint32_t voff = (uint32_t)vz * gy * gx + (uint32_t)vy * gx + vx;
      uint32_t vid  = lookupHash(voff, d_ht, ht_size);
      if (vid >= (uint32_t)MAX_VOXELS) return;
      uint32_t cur = Kokkos::atomic_fetch_add(&d_vox_num(vid), 1u);
      if (cur < (uint32_t)MAX_PPV) {
        uint32_t dst = vid * (FEATURE_NUM * MAX_PPV) + cur * FEATURE_NUM;
        uint32_t src = idx * FEATURE_NUM;
        for (int f = 0; f < FEATURE_NUM; f++) d_vox_tmp(dst + f) = d_points(src + f);
      }
    });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the voxelization kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  auto h_rn = Kokkos::create_mirror_view(d_real_num);
  Kokkos::deep_copy(h_rn, d_real_num);
  std::cout << "valid_num: " << h_rn(0) << std::endl;

  uint32_t real_voxels = h_rn(0);

  // Feature extraction: average per voxel
  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("feat_extract", (int)real_voxels, KOKKOS_LAMBDA(const int vid) {
      uint32_t valid = d_vox_num(vid);
      if (valid > (uint32_t)MAX_PPV) valid = (uint32_t)MAX_PPV;
      int base = vid * MAX_PPV * FEATURE_NUM;
      for (int f = 0; f < FEATURE_NUM; f++) {
        float s = d_vox_tmp(base + f);
        for (int p = 1; p < (int)valid; p++) s += d_vox_tmp(base + p * FEATURE_NUM + f);
        d_features(vid * FEATURE_NUM + f) = (valid > 0) ? s / valid : 0.f;
      }
    });
    Kokkos::fence();
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the feature extraction kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <repeat> [points_count]\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int npoints = (argc >= 3) ? atoi(argv[2]) : 100000;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_points("points", (size_t)npoints * FEATURE_NUM);

    // Generate synthetic point cloud
    {
      auto h = Kokkos::create_mirror_view(d_points);
      std::mt19937 rng(123);
      std::uniform_real_distribution<float> dx(MIN_X, MAX_X);
      std::uniform_real_distribution<float> dy(MIN_Y, MAX_Y);
      std::uniform_real_distribution<float> dz(MIN_Z, MAX_Z);
      std::uniform_real_distribution<float> di(0.f, 1.f);
      for (int i = 0; i < npoints; i++) {
        h(i * FEATURE_NUM + 0) = dx(rng);
        h(i * FEATURE_NUM + 1) = dy(rng);
        h(i * FEATURE_NUM + 2) = dz(rng);
        h(i * FEATURE_NUM + 3) = di(rng);
        h(i * FEATURE_NUM + 4) = di(rng);
      }
      Kokkos::deep_copy(d_points, h);
    }

    std::cout << "find points num: " << npoints << std::endl;
    generateVoxels(d_points, npoints, repeat);
  }
  Kokkos::finalize();
  return 0;
}
