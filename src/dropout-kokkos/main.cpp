// Kokkos port of dropout-cuda
// curand replaced with a simple LCG-based per-element random number.
// Philox-style vectorised loops are merged into a simple parallel_for.

#include <cstdio>
#include <chrono>
#include <cstdint>
#include <Kokkos_Core.hpp>

// Simple LCG random in [0,1)
KOKKOS_INLINE_FUNCTION float lcg_rand(uint64_t& state)
{
  state = 6364136223846793005ULL * state + 1442695040888963407ULL;
  return (float)((state >> 33) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nelem  = atol(argv[1]);
  const int      repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*>   d_a("a",    nelem);
    Kokkos::View<float*>   d_b("b",    nelem);
    Kokkos::View<uint8_t*> d_mask("mask", nelem);

    // initialise input
    Kokkos::parallel_for(nelem, KOKKOS_LAMBDA(int64_t i) { d_a(i) = 0.1f; });
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();
    for (int p = 1; p <= repeat; p++) {
      float pa = (float)p / repeat;
      float scale = 1.f / pa;
      Kokkos::parallel_for(nelem, KOKKOS_LAMBDA(int64_t i) {
        uint64_t s = (uint64_t)(12345678ULL * (i + 1) + 87654321ULL * p);
        float r = lcg_rand(s);
        uint8_t keep = (r < pa) ? 1 : 0;
        d_b(i)    = d_a(i) * keep * scale;
        d_mask(i) = keep;
      });
    }
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time %lf (s)\n", time * 1e-9);
  }
  Kokkos::finalize();
  return 0;
}
