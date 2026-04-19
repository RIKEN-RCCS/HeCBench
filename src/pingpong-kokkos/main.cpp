/*
 * Kokkos port of pingpong-cuda benchmark.
 *
 * Original: MPI + NCCL point-to-point send/recv bandwidth test between 2 GPUs
 *           over sizes 2^16 to 2^27 bytes.
 * Port: single-device bandwidth test using Kokkos::deep_copy between two Views
 *       of doubles.  Reports transfer size, average transfer time, and
 *       bandwidth for each size.
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    const int loop_count = 50;
    const int warmup     = 5;

    printf("%-30s %20s %20s\n",
           "Transfer size (B)", "Transfer Time (s)", "Bandwidth (GB/s)");

    for (int p = 16; p <= 27; ++p) {
      const long int N      = 1L << p;   // number of doubles
      const long int num_B  = N * (long int)sizeof(double);
      const double   num_GB = (double)num_B / 1.0e9;

      Kokkos::View<double*> d_src("src", N);
      Kokkos::View<double*> d_dst("dst", N);

      // Warm-up
      for (int w = 0; w < warmup; ++w)
        Kokkos::deep_copy(d_dst, d_src);
      Kokkos::fence();

      auto t0 = std::chrono::steady_clock::now();
      for (int j = 0; j < loop_count; ++j)
        Kokkos::deep_copy(d_dst, d_src);
      Kokkos::fence();
      auto t1 = std::chrono::steady_clock::now();

      double elapsed  = std::chrono::duration<double>(t1 - t0).count();
      double avg_time = elapsed / (double)loop_count;

      printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, "
             "Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time, num_GB / avg_time);
    }
  }
  Kokkos::finalize();
  return 0;
}
