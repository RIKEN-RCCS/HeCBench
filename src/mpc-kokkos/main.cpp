// MPC Delta Compression - Kokkos port (simplified)
//
// Implements the core idea of MPC compression:
//   1. Delta-XOR encode adjacent doubles (treating each double as uint64)
//   2. Count non-zero bytes in each delta value
//   3. Pack non-zero bytes (compute offsets, then scatter)
//   4. Report compression ratio and throughput
//
// Usage: ./main [repeat [n]]
//   repeat : number of timing iterations (default 1)
//   n      : number of double values to compress (default 1000000)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int repeat = (argc > 1) ? atoi(argv[1]) : 1;
        const int n      = (argc > 2) ? atoi(argv[2]) : 1000000;

        printf("MPC compression (simplified delta-XOR): n=%d doubles, repeat=%d\n", n, repeat);

        // Generate synthetic scientific-looking double data
        Kokkos::View<uint64_t*> d_input("input", n);
        {
            auto h = Kokkos::create_mirror_view(d_input);
            for (int i = 0; i < n; i++) {
                double val = sin(i * 0.001) * 1e6 + cos(i * 0.0013) * 1e4;
                memcpy(&h(i), &val, sizeof(double));
            }
            Kokkos::deep_copy(d_input, h);
        }

        // Work arrays
        Kokkos::View<uint64_t*> d_delta("delta",    n);
        Kokkos::View<int*>      d_nbytes("nbytes",  n); // non-zero byte count per element
        Kokkos::View<int*>      d_offset("offsets", n); // exclusive prefix sum of nbytes
        Kokkos::View<uint8_t*>  d_packed("packed",  (size_t)n * 8); // worst-case output

        double total_time = 0.0;
        int64_t packed_size = 0;

        for (int rep = 0; rep < repeat; rep++) {
            Kokkos::fence();
            Kokkos::Timer timer;

            // Step 1: Compute delta XOR and count non-zero bytes
            Kokkos::parallel_for(
                "delta_xor", n,
                KOKKOS_LAMBDA(int i) {
                    const uint64_t cur  = d_input(i);
                    const uint64_t prev = (i > 0) ? d_input(i - 1) : 0ULL;
                    const uint64_t delta = cur ^ prev;
                    d_delta(i) = delta;

                    int cnt = 0;
                    for (int b = 0; b < 8; b++)
                        if ((delta >> (8 * b)) & 0xFF) cnt++;
                    d_nbytes(i) = cnt;
                });

            // Step 2: Exclusive prefix sum to get output offsets
            // (serial scan on host - for a full port this would use
            //  Kokkos::Experimental::exclusive_scan or a parallel scan)
            {
                auto h_nb  = Kokkos::create_mirror_view(d_nbytes);
                auto h_off = Kokkos::create_mirror_view(d_offset);
                Kokkos::deep_copy(h_nb, d_nbytes);
                int acc = 0;
                for (int i = 0; i < n; i++) {
                    h_off(i) = acc;
                    acc += h_nb(i);
                }
                packed_size = acc;
                Kokkos::deep_copy(d_offset, h_off);
            }

            // Step 3: Pack non-zero bytes into output buffer
            Kokkos::parallel_for(
                "pack", n,
                KOKKOS_LAMBDA(int i) {
                    const uint64_t delta = d_delta(i);
                    int out = d_offset(i);
                    for (int b = 0; b < 8; b++) {
                        const uint8_t byte = (uint8_t)((delta >> (8 * b)) & 0xFF);
                        if (byte) d_packed(out++) = byte;
                    }
                });

            Kokkos::fence();
            total_time += timer.seconds();
        }

        const double avg_ms    = total_time * 1000.0 / repeat;
        const double ratio     = (double)(n * 8) / packed_size;
        const double throughput = (double)n * 8 / (total_time / repeat) / 1e9;

        printf("Input size      : %d doubles = %zu bytes\n", n, (size_t)n * 8);
        printf("Packed size     : %ld bytes\n", packed_size);
        printf("Compression ratio: %.3f:1\n", ratio);
        printf("Average time    : %.3f ms\n", avg_ms);
        printf("Throughput      : %.3f GB/s\n", throughput);
    }
    Kokkos::finalize();
    return 0;
}
