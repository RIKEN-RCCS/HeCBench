#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
  int n = 8388608;
  int reps = 100;
  int warmup = 5;
  int compaction_factor = 50;
  int remove_value = 0;

  int opt;
  while ((opt = getopt(argc, argv, "n:r:w:c:")) >= 0) {
    switch (opt) {
      case 'n': n = atoi(optarg); break;
      case 'r': reps = atoi(optarg); break;
      case 'w': warmup = atoi(optarg); break;
      case 'c': compaction_factor = atoi(optarg); break;
      default: break;
    }
  }

  const int keep_count = (int)((long long)n * compaction_factor / 100);

  // Build input on host: fill with remove_value, then set keep_count elements non-zero
  std::vector<int> h_input(n, remove_value);
  srand(42);
  // Randomly set keep_count positions to non-zero
  std::vector<int> positions(n);
  for (int i = 0; i < n; i++) positions[i] = i;
  for (int i = 0; i < keep_count; i++) {
    int j = i + rand() % (n - i);
    std::swap(positions[i], positions[j]);
    h_input[positions[i]] = (rand() % 254) + 1; // 1..255
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_input("d_input", n);
    Kokkos::View<int*> d_flags("d_flags", n);
    Kokkos::View<int*> d_scan("d_scan", n);
    Kokkos::View<int*> d_output("d_output", n);

    // Copy input to device
    {
      auto h = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < n; i++) h(i) = h_input[i];
      Kokkos::deep_copy(d_input, h);
    }

    const int rv = remove_value;
    double total_ms = 0.0;

    for (int iter = 0; iter < warmup + reps; iter++) {
      auto t0 = std::chrono::steady_clock::now();

      // Compute flags
      Kokkos::parallel_for("flags", n, KOKKOS_LAMBDA(const int i) {
        d_flags(i) = (d_input(i) != rv) ? 1 : 0;
      });
      Kokkos::fence();

      // Exclusive prefix scan using parallel_scan
      Kokkos::parallel_scan("scan", n, KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        const int val = d_flags(i);
        if (final) d_scan(i) = update;
        update += val;
      });
      Kokkos::fence();

      // Scatter non-removed elements
      Kokkos::parallel_for("scatter", n, KOKKOS_LAMBDA(const int i) {
        if (d_flags(i)) {
          d_output(d_scan(i)) = d_input(i);
        }
      });
      Kokkos::fence();

      auto t1 = std::chrono::steady_clock::now();
      if (iter >= warmup) {
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
    }

    printf("Total stream compaction time for %d iterations: %f (ms)\n", reps, total_ms);

    // Verification: get output count
    // The last element of d_scan + d_flags[last] gives total count
    auto h_scan = Kokkos::create_mirror_view(d_scan);
    auto h_flags = Kokkos::create_mirror_view(d_flags);
    Kokkos::deep_copy(h_scan, d_scan);
    Kokkos::deep_copy(h_flags, d_flags);
    int output_count = h_scan(n - 1) + h_flags(n - 1);

    if (output_count == keep_count)
      printf("Verification: PASS\n");
    else
      printf("Verification: FAIL (got %d, expected %d)\n", output_count, keep_count);
  }
  Kokkos::finalize();
  return 0;
}
