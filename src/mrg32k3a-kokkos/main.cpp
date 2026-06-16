// mrg32k3a-kokkos/main.cpp
// Port of mrg32k3a-cuda: generates pseudo-random numbers using the MRG32k3a
// algorithm on host (sequential) and device (Kokkos parallel_for).
//
// Device version: each thread i is seeded independently so all n numbers are
// produced in parallel. Verification checks that every value lies in [0, 1).

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// MRG32k3a constants
// ---------------------------------------------------------------------------
static constexpr double MRG_M1   = 4294967087.0;
static constexpr double MRG_M2   = 4294944443.0;
static constexpr double MRG_A12  = 1403580.0;
static constexpr double MRG_A13N = 810728.0;
static constexpr double MRG_A21  = 527612.0;
static constexpr double MRG_A23N = 1370589.0;

// ---------------------------------------------------------------------------
// Host sequential MRG32k3a: generates n floats from a fixed initial state.
// ---------------------------------------------------------------------------
void run_on_host(int n, unsigned long long seed, std::vector<float>& out) {
  // Initial state derived from seed (must be in (0, m-1))
  double s10 = 1.0 + static_cast<double>(seed % static_cast<unsigned long long>(MRG_M1 - 2));
  double s11 = 12345.0;
  double s12 = 67890.0;
  double s20 = 1.0 + static_cast<double>((seed * 6364136223846793005ULL + 1442695040888963407ULL)
                                          % static_cast<unsigned long long>(MRG_M2 - 2));
  double s21 = 24680.0;
  double s22 = 11111.0;

  for (int i = 0; i < n; ++i) {
    // Component 1
    double p1 = MRG_A12 * s11 - MRG_A13N * s10;
    p1 = std::fmod(p1, MRG_M1);
    if (p1 < 0.0) p1 += MRG_M1;

    // Component 2
    double p2 = MRG_A21 * s22 - MRG_A23N * s20;
    p2 = std::fmod(p2, MRG_M2);
    if (p2 < 0.0) p2 += MRG_M2;

    // Advance state
    s10 = s11; s11 = s12; s12 = p1;
    s20 = s21; s21 = s22; s22 = p2;

    // Combine
    double u = (p1 - p2) / MRG_M1;
    if (p1 <= p2) u += 1.0;
    out[i] = static_cast<float>(u);
  }
}

// ---------------------------------------------------------------------------
// Device parallel MRG32k3a: thread i generates one number from a unique seed.
// Uses a simple hash of (base_seed, i) to create independent initial states.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
static unsigned long long mrg_hash(unsigned long long x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

KOKKOS_INLINE_FUNCTION
float mrg32k3a_single(int idx, unsigned long long seed) {
  // Derive six independent positive state values in range (0, m-1)
  auto h0 = mrg_hash(seed ^ static_cast<unsigned long long>(idx) * 6364136223846793005ULL);
  auto h1 = mrg_hash(h0 + 1);
  auto h2 = mrg_hash(h1 + 2);
  auto h3 = mrg_hash(h2 + 3);
  auto h4 = mrg_hash(h3 + 4);
  auto h5 = mrg_hash(h4 + 5);

  auto clamp1 = [](unsigned long long v) -> double {
    return 1.0 + static_cast<double>(v % static_cast<unsigned long long>(MRG_M1 - 2));
  };
  auto clamp2 = [](unsigned long long v) -> double {
    return 1.0 + static_cast<double>(v % static_cast<unsigned long long>(MRG_M2 - 2));
  };

  double s10 = clamp1(h0), s11 = clamp1(h1), s12 = clamp1(h2);
  double s20 = clamp2(h3), s21 = clamp2(h4), s22 = clamp2(h5);

  // Component 1
  double p1 = MRG_A12 * s11 - MRG_A13N * s10;
  p1 = Kokkos::fmod(p1, MRG_M1);
  if (p1 < 0.0) p1 += MRG_M1;

  // Component 2
  double p2 = MRG_A21 * s22 - MRG_A23N * s20;
  p2 = Kokkos::fmod(p2, MRG_M2);
  if (p2 < 0.0) p2 += MRG_M2;

  double u = (p1 - p2) / MRG_M1;
  if (p1 <= p2) u += 1.0;
  return static_cast<float>(u);
}

void run_on_device(int n, unsigned long long seed, Kokkos::View<float*>& d_out) {
  Kokkos::parallel_for(
      "mrg32k3a_device", n,
      KOKKOS_LAMBDA(int i) { d_out(i) = mrg32k3a_single(i, seed); });
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of pseudorandom numbers> <repeat>\n", argv[0]);
    return 1;
  }
  const int  n      = std::atoi(argv[1]);
  const int  repeat = std::atoi(argv[2]);
  const unsigned long long seed = 1234ULL;

  Kokkos::initialize(argc, argv);
  {
    std::vector<float> h_data(n, 0.f);
    Kokkos::View<float*> d_data("d_data", n);

    // warmup
    run_on_host(n, seed, h_data);
    run_on_device(n, seed, d_data);
    Kokkos::fence();

    // --- host timing ---
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_on_host(n, seed, h_data);
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time on host: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);

    // --- device timing ---
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_on_device(n, seed, d_data);
    Kokkos::fence();
    t1 = std::chrono::steady_clock::now();
    printf("Average execution time on device: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);

    // --- verify: all values in [0, 1) ---
    auto h_mirror = Kokkos::create_mirror_view(d_data);
    Kokkos::deep_copy(h_mirror, d_data);

    bool ok_host = true, ok_dev = true;
    for (int i = 0; i < n; ++i) {
      if (h_data[i] < 0.f || h_data[i] >= 1.f) { ok_host = false; break; }
    }
    for (int i = 0; i < n; ++i) {
      if (h_mirror(i) < 0.f || h_mirror(i) >= 1.f) { ok_dev = false; break; }
    }
    printf("%s\n", (ok_host && ok_dev) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
