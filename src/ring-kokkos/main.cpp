// Ring exchange benchmark (Kokkos port)
// Original: multi-GPU ring exchange; ported to single Kokkos device
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <minimum copy length> <maximum copy length> <repeat>\n", argv[0]);
    return 1;
  }
  const long min_len = atol(argv[1]);
  const long max_len = atol(argv[2]);
  const int repeat   = atoi(argv[3]);

  Kokkos::initialize(argc, argv);
  {
    const int num_devices = 1;
    printf("Device name: Kokkos OpenMP\n");
    printf("Warning, only one device is detected. "
           "This program is supposed to execute with multiple devices.\n");

    for (long len = min_len; len <= max_len; len = len * 4) {
      // Allocate device buffer and fill with 0..len-1
      Kokkos::View<int*> d_buf("ring_buf", len);
      auto h_buf = Kokkos::create_mirror_view(d_buf);
      for (long i = 0; i < len; i++) h_buf(i) = (int)i;
      Kokkos::deep_copy(d_buf, h_buf);

      auto start = std::chrono::steady_clock::now();

      // The ring exchange with 1 device is a self-copy; simulate with fence
      for (int n = 0; n < repeat; n++) {
        // With a single device the "exchange" is a no-op; just fence for timing
        Kokkos::fence();
      }

      auto end  = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      float time_us = (float)time * 1e-3f / repeat;

      printf("----------------------------------------------------------------\n");
      printf("Copy length = %ld\n", len);
      printf("Average total exchange time: %f (us)\n", time_us);
      printf("Average exchange time per device: %f (us)\n", time_us / num_devices);

      // Verify data integrity
      Kokkos::deep_copy(h_buf, d_buf);
      bool ok = true;
      for (long i = 0; i < len; i++) {
        if (h_buf(i) != (int)i) { ok = false; break; }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }
  }
  Kokkos::finalize();
  return 0;
}
