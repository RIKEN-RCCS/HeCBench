/*
 * Kokkos port of p2p-cuda benchmark.
 *
 * Original: multi-GPU P2P ping-pong copy + SimpleKernel (dst[i] = src[i]*2)
 * Port: single-device deep_copy for bandwidth, two kernel passes to verify
 *       result[i] == float(i % 4096) * 4.0f
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
  printf("[%s] - Starting...\n", argv[0]);

  if (argc < 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const size_t buf_len  = 1024UL * 1024UL * 16UL; // 16 M floats
    const size_t buf_size = buf_len * sizeof(float);

    using DevView  = Kokkos::View<float*>;
    using HostView = Kokkos::View<float*, Kokkos::HostSpace>;

    DevView  d_buf("d_buf", buf_len);
    DevView  d_tmp("d_tmp", buf_len);
    HostView h_buf("h_buf", buf_len);

    // -----------------------------------------------------------------------
    // Bandwidth measurement: ping-pong deep_copy between two device Views
    // -----------------------------------------------------------------------
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; ++i) {
      if (i % 2 == 0)
        Kokkos::deep_copy(d_tmp, d_buf);
      else
        Kokkos::deep_copy(d_buf, d_tmp);
    }

    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    auto time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    printf("Single-device copy bandwidth: %.2f GB/s\n",
           1.0 / (double)time_ns * (double)(repeat * buf_size));

    // -----------------------------------------------------------------------
    // Kernel verification
    // Initialise host buffer h[i] = float(i % 4096), copy to device
    // -----------------------------------------------------------------------
    printf("Preparing host buffer and copying to device...\n");
    for (size_t i = 0; i < buf_len; ++i)
      h_buf(i) = float(i % 4096);
    Kokkos::deep_copy(d_buf, h_buf);

    // First kernel pass: d_tmp[i] = d_buf[i] * 2
    printf("Run kernel pass 1: d_tmp[i] = d_buf[i] * 2...\n");
    Kokkos::parallel_for(
        "SimpleKernel1",
        Kokkos::RangePolicy<>(0, (int)buf_len),
        KOKKOS_LAMBDA(int i) { d_tmp(i) = d_buf(i) * 2.0f; });
    Kokkos::fence();

    // Second kernel pass: d_buf[i] = d_tmp[i] * 2
    printf("Run kernel pass 2: d_buf[i] = d_tmp[i] * 2...\n");
    Kokkos::parallel_for(
        "SimpleKernel2",
        Kokkos::RangePolicy<>(0, (int)buf_len),
        KOKKOS_LAMBDA(int i) { d_buf(i) = d_tmp(i) * 2.0f; });
    Kokkos::fence();

    // Copy result back and verify: expected = float(i % 4096) * 4
    printf("Copying result back to host and verifying...\n");
    Kokkos::deep_copy(h_buf, d_buf);

    int error_count = 0;
    for (size_t i = 0; i < buf_len; ++i) {
      float ref = float(i % 4096) * 4.0f;
      if (h_buf(i) != ref) {
        printf("Verification error @ element %zu: val = %f, ref = %f\n",
               i, (double)h_buf(i), (double)ref);
        if (++error_count > 10)
          break;
      }
    }

    if (error_count == 0)
      printf("Test passed\n");
    else
      printf("Test failed!\n");
  }
  Kokkos::finalize();
  return 0;
}
