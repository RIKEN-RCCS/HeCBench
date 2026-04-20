/*
 * Kokkos port of atomicAggregate.
 * Original used warp-level atomic aggregation (__match_any_sync, __ballot_sync).
 * Kokkos port uses plain Kokkos::atomic_fetch_add, which is the portable equivalent.
 * The warp-coalescing optimization is not portable; we use simple atomics.
 */

#include <chrono>
#include <cstdio>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    const int nBlocks = 65536;
    const int blockSize = 256;
    const int N = nBlocks * blockSize;

    for (int ds = 32; ds >= 1; ds /= 2) {
      Kokkos::View<int*> d("d", ds);
      Kokkos::deep_copy(d, 0);

      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++) {
        Kokkos::parallel_for("atomicAgg", N, KOKKOS_LAMBDA(int tid) {
          int idx = tid % ds;
          Kokkos::atomic_fetch_add(&d(idx), 1);
        });
      }
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<float> time = end - start;
      printf("Total kernel time (%d locations): %f (s)\n", ds, time.count());

      auto h_d = Kokkos::create_mirror_view(d);
      Kokkos::deep_copy(h_d, d);

      int expected = (blockSize / ds) * nBlocks * repeat;
      bool ok = true;
      for (int i = 0; i < ds; i++) {
        if (h_d(i) != expected) { ok = false; break; }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }
  }
  Kokkos::finalize();
  return 0;
}
