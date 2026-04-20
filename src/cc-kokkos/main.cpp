// Single-rank AllReduce bandwidth benchmark using Kokkos.
// Simulates NCCL AllReduce: parallel_reduce (sum) + parallel_for (fill).

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int repeat = 5;
    if (argc > 1) repeat = std::atoi(argv[1]);

    const int nRanks = 1;

    // Sizes: 1M, 10M, 100M, 1000M floats
    const size_t sizes[] = {
      1024ULL * 1024,
      10ULL * 1024 * 1024,
      100ULL * 1024 * 1024,
      1000ULL * 1024 * 1024
    };

    printf("%-20s  %-25s  %-15s\n",
           "Transfer Size (B)", "Avg Transfer Time (s)", "Bandwidth (GB/s)");

    for (size_t size : sizes) {
      const size_t num_B  = sizeof(float) * size * nRanks;
      const double num_GB = (double)num_B / 1e9;

      Kokkos::View<float*> sendbuff("sendbuff", size);
      Kokkos::View<float*> recvbuff("recvbuff", size);

      // Initialise sendbuff to 1.0f
      Kokkos::parallel_for("init", size,
        KOKKOS_LAMBDA(int i) { sendbuff(i) = 1.0f; });
      Kokkos::fence();

      Kokkos::Timer timer;

      for (int r = 0; r < repeat; ++r) {
        // AllReduce: sum all elements (single rank → sum == value)
        double total = 0.0;
        Kokkos::parallel_reduce("reduce", size,
          KOKKOS_LAMBDA(int i, double& acc) { acc += sendbuff(i); },
          total);

        // Broadcast result back into recvbuff (fill with per-element "reduced" value)
        float reduced_val = (float)(total / size);  // == 1.0 * nRanks
        Kokkos::parallel_for("fill", size,
          KOKKOS_LAMBDA(int i) { recvbuff(i) = reduced_val; });
        Kokkos::fence();
      }

      double elapsed = timer.seconds();
      double avg_time = elapsed / repeat;
      double bandwidth = num_GB / avg_time;

      // Verify: recvbuff[0] should equal nRanks (1.0f)
      auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, recvbuff);
      if (std::fabs(h_recv(0) - (float)nRanks) > 1e-4f) {
        printf("VERIFICATION FAILED at size=%zu: got %.6f expected %.6f\n",
               size, (double)h_recv(0), (double)nRanks);
      }

      printf("%-20zu  %-25.6f  %-15.4f\n", num_B, avg_time, bandwidth);
    }
  }
  Kokkos::finalize();
  return 0;
}
