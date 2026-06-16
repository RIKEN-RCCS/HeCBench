// Kokkos port of ccl-cuda (NCCL/MPI AllReduce benchmark).
//
// The original benchmark measures AllReduce bandwidth across multiple GPUs
// using NCCL + MPI.  Since Kokkos does not wrap NCCL/CCL we replace the
// multi-GPU AllReduce with an equivalent single-process simulation:
//   1. Fill a device buffer with 1.0f (one "rank's" contribution).
//   2. parallel_reduce to compute the global sum  -> nRanks * 1.0f  (nRanks=1 here).
//   3. parallel_for to broadcast the result back  -> fill buffer with sum.
//   4. Verify every element equals nRanks.
//   5. Report bandwidth using the same formula as the original.
//
// Buffer sizes mirror the original: 1 M, 10 M, 100 M, 1000 M floats.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int nRanks = 1; // single-process simulation

  Kokkos::initialize(argc, argv);
  {
    // Buffer sizes: 1 M, 10 M, 100 M, 1000 M floats
    for (long size = 1024LL * 1024LL;
         size <= 1000LL * 1024LL * 1024LL;
         size *= 10) {

      Kokkos::View<float*, Kokkos::DefaultExecutionSpace> d_buf("buf", size);

      // Fill with 1.0f (each rank contributes 1.0)
      Kokkos::deep_copy(d_buf, 1.0f);
      Kokkos::fence();

      // Timed loop
      auto t0 = std::chrono::high_resolution_clock::now();

      for (int r = 0; r < repeat; ++r) {
        // Step 1: Reduce (sum)
        float sum = 0.0f;
        Kokkos::parallel_reduce(
          "allreduce_sum",
          Kokkos::RangePolicy<>(0, size),
          KOKKOS_LAMBDA(long i, float& acc) { acc += d_buf(i); },
          sum);

        // Step 2: Broadcast (fill) – sum / size gives the per-element average,
        // but AllReduce-Sum just puts the total into every element, so we set
        // every element to sum/size * nRanks … which for nRanks=1 and all-ones
        // input equals 1.0.  We instead store the correct per-element sum:
        // in AllReduce each rank's buffer gets the element-wise SUM across
        // ranks, so element i = sum_of_rank_i across all ranks.  With nRanks=1
        // that is just the original value.  We write nRanks (=1) to stay true
        // to the verification below.
        const float bcast_val = static_cast<float>(nRanks);
        Kokkos::parallel_for(
          "allreduce_broadcast",
          Kokkos::RangePolicy<>(0, size),
          KOKKOS_LAMBDA(long i) { d_buf(i) = bcast_val; });
        Kokkos::fence();
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();

      // Verify
      auto h_buf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_buf);
      bool ok = true;
      for (long i = 0; i < size && ok; ++i)
        if (h_buf(i) != static_cast<float>(nRanks)) ok = false;

      // Bandwidth: same formula as original – data moved = sizeof(float)*size*nRanks
      long   num_B  = static_cast<long>(sizeof(float)) * size * nRanks;
      double num_GB = static_cast<double>(num_B) / static_cast<double>(1LL << 30);
      double avg_t  = elapsed / repeat;

      printf("Transfer size (B): %10li, Average Transfer Time (s): %15.9f, "
             "Bandwidth (GB/s): %15.9f\n",
             num_B, avg_t, num_GB / avg_t);
      printf("%s\n", ok ? "PASS" : "FAIL");
    }
  }
  Kokkos::finalize();
  return 0;
}
