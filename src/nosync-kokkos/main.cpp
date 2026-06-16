// Port of nosync-cuda (Thrust nosync benchmark) to Kokkos.
//
// The CUDA version uses Thrust's par_nosync policy to demonstrate that an
// inclusive_scan can be overlapped with a concurrent reduce on the same data.
// The correctness check is: reduce_result - last_scan_element == 0
//
//   For sequence [0, 1, ..., n-1]:
//     reduce = n*(n-1)/2
//     inclusive_scan last element = n*(n-1)/2
//   => reduce_result - last_scan_element = 0
//
// The Kokkos port runs the same operations sequentially (Kokkos has no
// stream-based nosync policy) and verifies the same condition.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    int sum = -1;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::View<int *> d_vec("d_vec", n);
      Kokkos::View<int *> d_res("d_res", n);

      // Fill d_vec with [0, 1, ..., n-1]
      Kokkos::parallel_for(
          "sequence", n,
          KOKKOS_LAMBDA(int i) { d_vec(i) = i; });
      Kokkos::fence();

      // Inclusive prefix sum into d_res
      Kokkos::parallel_scan(
          "inclusive_scan", n,
          KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
            update += d_vec(i);
            if (final_pass)
              d_res(i) = update;
          });
      Kokkos::fence();

      // Total reduce of d_vec
      int reduce_result = 0;
      Kokkos::parallel_reduce(
          "reduce", n,
          KOKKOS_LAMBDA(int i, int &lsum) { lsum += d_vec(i); },
          reduce_result);
      Kokkos::fence();

      // Read last element of scan result on host
      auto h_res = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        d_res);
      int last_scan = h_res(n - 1);

      sum = reduce_result - last_scan;
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    std::cout << "Average execution time: " << (time * 1e-3f) / repeat
              << " (us)\n";

    std::cout << ((sum == 0) ? "PASS" : "FAIL") << "\n";
  }
  Kokkos::finalize();
  return 0;
}
