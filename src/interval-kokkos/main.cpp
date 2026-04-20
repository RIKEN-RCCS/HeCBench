/*
 * Port of interval-omp to Kokkos.
 *
 * Uses a Kokkos::parallel_for over THREADS work items to run the interval
 * Newton root-finding algorithm.  Each thread maintains a private local_stack
 * of intervals to process and writes confirmed roots into a global_stack
 * backed by a Kokkos::View (device memory).
 *
 * Device-callable interval arithmetic is provided by the Kokkos-annotated
 * headers in this directory (kokkos_gpu_interval.h, etc.).
 *
 * Optional CPU reference comparison requires Boost.Interval:
 *   compile with -DHAVE_BOOST and -I<boost_root>
 *
 * Usage: ./main <impl_choice> <repeat>
 *   impl_choice : 0 = naive (while-loop), 1 = optimised (top-in-registers)
 *   repeat      : number of timed iterations
 */

#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <cmath>

// The interval-omp/interval.h defines: BLOCK_SIZE, GRID_SIZE, THREADS,
// DEPTH_RESULT, typedef T (= double by default).
#include "kokkos_gpu_interval.h"   // brings in kokkos_interval_lib.h, rounded_arith.h, interval.h

#ifdef HAVE_BOOST
#  include "../interval-cuda/cpu_interval.h"
   using I_CPU = interval<T, policies::rounded_math<T>>;
#endif

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <impl_choice> <repeat>\n", argv[0]);
        return 1;
    }
    const int impl_choice = atoi(argv[1]);
    const int repeat       = atoi(argv[2]);

    switch (impl_choice) {
        case 0:  printf("GPU implementation 1 (naive)\n");      break;
        case 1:  printf("GPU implementation 2 (optimised)\n");  break;
        default: printf("GPU implementation 1 (naive)\n");      break;
    }

    Kokkos::initialize(argc, argv);
    {
        using I = interval_gpu<T>;

        // Device storage: result buffer (THREADS * DEPTH_RESULT intervals)
        // and per-thread result counts.
        Kokkos::View<I*>   d_buffer  ("d_buffer",   (size_t)THREADS * DEPTH_RESULT);
        Kokkos::View<int*> d_nresults("d_nresults",  THREADS);

        I initial_interval(T(0.01), T(4.0));
        std::cout << "Searching for roots in ["
                  << initial_interval.lower() << ", "
                  << initial_interval.upper() << "]...\n";

        long time_ns = 0;

        auto t0 = std::chrono::steady_clock::now();

        for (int it = 0; it < repeat; ++it) {
            auto buf     = d_buffer;   // capture Views by value
            auto nres    = d_nresults;
            const int ic = impl_choice;
            const I   iv = initial_interval;

            Kokkos::parallel_for("interval_newton",
                Kokkos::RangePolicy<>(0, THREADS),
                KOKKOS_LAMBDA(int thread_id) {
                    global_stack<I, DEPTH_RESULT, THREADS> result(
                        buf.data(), thread_id);

                    switch (ic) {
                        case 1:
                            newton_interval<T, THREADS, DEPTH_RESULT>(
                                result, iv, thread_id);
                            break;
                        default:
                            newton_interval_naive<T, THREADS, DEPTH_RESULT>(
                                result, iv, thread_id);
                            break;
                    }
                    nres(thread_id) = result.size();
                });
            Kokkos::fence();
        }

        double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        time_ns = (long)(elapsed * 1e9);

        // Copy results back to host
        auto h_nresults = Kokkos::create_mirror_view_and_copy(
                              Kokkos::HostSpace{}, d_nresults);
        // The host-readable intervals live in d_buffer; reinterpret as host
        auto h_buffer = Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace{}, d_buffer);

        std::cout << "Found " << h_nresults(0)
                  << " intervals that may contain the root(s)\n";
        std::cout.precision(15);
        for (int i = 0; i < h_nresults(0); ++i) {
            // Interleaved layout: slot i of thread 0 is at index THREADS*i + 0
            const I& iv0 = h_buffer(THREADS * i);
            std::cout << " i[" << i << "] = ["
                      << iv0.lower() << ", " << iv0.upper() << "]\n";
        }

        std::cout << "Number of equations solved: " << THREADS << "\n";
        std::cout << "Average execution time: "
                  << (time_ns * 1e-3 / repeat) << " us\n";

#ifdef HAVE_BOOST
        // ── CPU reference verification ──────────────────────────────────────
        std::vector<I_CPU> h_result_cpu((size_t)THREADS * DEPTH_RESULT);
        std::vector<int>   h_nresults_cpu(THREADS, 0);

        for (int tid = 0; tid < THREADS; ++tid) {
            global_stack_cpu<I_CPU, DEPTH_RESULT, THREADS> result_cpu(
                h_result_cpu.data(), tid);
            I_CPU iv_cpu(T(0.01), T(4.0));
            test_interval_newton_cpu<I_CPU>(result_cpu, iv_cpu, tid);
            h_nresults_cpu[tid] = result_cpu.size();
        }

        bool pass = true;
        for (int tid = 0; tid < THREADS && pass; ++tid)
            if (h_nresults_cpu[tid] != h_nresults(tid)) pass = false;
        std::cout << (pass ? "PASS" : "FAIL") << "\n";
#endif
    }
    Kokkos::finalize();
    return 0;
}
