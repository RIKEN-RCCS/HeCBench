#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

struct SmallKernelArgs  { char args[16]; };
struct MediumKernelArgs { char args[256]; };
struct LargeKernelArgs  { char args[4096]; };

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  SmallKernelArgs  small_kernel_args  = {};
  MediumKernelArgs medium_kernel_args = {};
  LargeKernelArgs  large_kernel_args  = {};

  Kokkos::initialize(argc, argv);
  {
    // Warmup - small
    for (int i = 0; i < repeat; i++) {
      SmallKernelArgs args = small_kernel_args;
      Kokkos::parallel_for("small_warmup", 1, KOKKOS_LAMBDA(int idx) {
        volatile char* dummy = nullptr;
        if (dummy) *dummy = args.args[idx];
      });
      Kokkos::fence();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      SmallKernelArgs args = small_kernel_args;
      Kokkos::parallel_for("small", 1, KOKKOS_LAMBDA(int idx) {
        volatile char* dummy = nullptr;
        if (dummy) *dummy = args.args[idx];
      });
      Kokkos::fence();
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernelWithSmallArgs: %f (us)\n", (time * 1e-3f) / repeat);

    // Warmup - medium
    for (int i = 0; i < repeat; i++) {
      MediumKernelArgs args = medium_kernel_args;
      Kokkos::parallel_for("medium_warmup", 1, KOKKOS_LAMBDA(int idx) {
        volatile char* dummy = nullptr;
        if (dummy) *dummy = args.args[idx];
      });
      Kokkos::fence();
    }

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      MediumKernelArgs args = medium_kernel_args;
      Kokkos::parallel_for("medium", 1, KOKKOS_LAMBDA(int idx) {
        volatile char* dummy = nullptr;
        if (dummy) *dummy = args.args[idx];
      });
      Kokkos::fence();
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernelWithMediumArgs: %f (us)\n", (time * 1e-3f) / repeat);

    // Warmup - large
    for (int i = 0; i < repeat; i++) {
      LargeKernelArgs args = large_kernel_args;
      Kokkos::parallel_for("large_warmup", 1, KOKKOS_LAMBDA(int idx) {
        volatile char* dummy = nullptr;
        if (dummy) *dummy = args.args[idx];
      });
      Kokkos::fence();
    }

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      LargeKernelArgs args = large_kernel_args;
      Kokkos::parallel_for("large", 1, KOKKOS_LAMBDA(int idx) {
        volatile char* dummy = nullptr;
        if (dummy) *dummy = args.args[idx];
      });
      Kokkos::fence();
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernelWithLargeArgs: %f (us)\n", (time * 1e-3f) / repeat);
  }
  Kokkos::finalize();
  return 0;
}
