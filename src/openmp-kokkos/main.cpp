/*
 * Kokkos port of the "openmp" array-increment benchmark.
 * The original uses multiple CPU threads each targeting a GPU.
 * This Kokkos version runs one parallel_for over the full array.
 *
 * correctResult: data[i] == i + sum(j%b for j in 0..repeat-1)
 *
 * Args: <repeat>
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

static bool correctResult(const int* data, const int n, const int b, const int repeat)
{
  for (int i = 0; i < n; i++) {
    int sum = 0;
    for (int j = 0; j < repeat; j++) sum += j % b;
    if (sum + i != data[i]) {
      printf("check: data[%d]=%d != expected=%d\n", i, data[i], sum + i);
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int b      = 3;

  const unsigned int nwords = 33554432u;  // same as original (32 M ints)

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_a("a", nwords);

    // Initialize host-side mirror
    auto h_a = Kokkos::create_mirror_view(d_a);
    for (unsigned int i = 0; i < nwords; i++) h_a(i) = (int)i;
    Kokkos::deep_copy(d_a, h_a);

    auto start = std::chrono::steady_clock::now();

    Kokkos::parallel_for("increment", Kokkos::RangePolicy<>(0, (int)nwords),
      KOKKOS_LAMBDA(int idx) {
        for (int i = 0; i < repeat; i++)
          d_a(idx) += i % b;
      });
    Kokkos::fence();

    auto end  = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Work took %f seconds\n", elapsed);

    // Copy back and verify
    Kokkos::deep_copy(h_a, d_a);
    bool ok = correctResult(h_a.data(), (int)nwords, b, repeat);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
