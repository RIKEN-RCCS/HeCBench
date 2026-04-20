/*
 * Port of idivide-omp to Kokkos.
 *
 * Tests throughput and latency of integer division, including the int_fastdiv
 * fast-division class from ../idivide-cuda/fastdiv.h.
 *
 * On CUDA/HIP backends fastdiv.h already marks its functions __host__
 * __device__; on other backends they are plain inline functions, which
 * KOKKOS_LAMBDA can call directly.
 *
 * Usage: ./main <repeat>
 */

#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
#include <cstdio>

#include "../idivide-cuda/fastdiv.h"

#define NOW std::chrono::high_resolution_clock::now()

// ── throughput_test ───────────────────────────────────────────────────────────
template<typename divisor_type>
static void throughput_test(int n, divisor_type d1, divisor_type d2,
                             divisor_type d3, int dummy,
                             Kokkos::View<int*>& d_buf)
{
    Kokkos::parallel_for("throughput", n, KOKKOS_LAMBDA(int x) {
        int x1 = x / d1;
        int x2 = x / d2;
        int x3 = x / d3;
        int aggregate = x1 + x2 + x3;
        if ((aggregate & dummy) == 1) d_buf(0) = aggregate;
    });
    Kokkos::fence();
}

// ── latency_test ──────────────────────────────────────────────────────────────
template<typename divisor_type>
static void latency_test(int n,
                          divisor_type d1, divisor_type d2, divisor_type d3,
                          divisor_type d4, divisor_type d5, divisor_type d6,
                          divisor_type d7, divisor_type d8, divisor_type d9,
                          divisor_type d10,
                          int dummy,
                          Kokkos::View<int*>& d_buf)
{
    Kokkos::parallel_for("latency", n, KOKKOS_LAMBDA(int x) {
        x /= d1;  x /= d2;  x /= d3;  x /= d4;  x /= d5;
        x /= d6;  x /= d7;  x /= d8;  x /= d9;  x /= d10;
        if ((x & dummy) == 1) d_buf(0) = x;
    });
    Kokkos::fence();
}

// ── check: verify int_fastdiv gives same result as plain division ─────────────
static int check_fastdiv(int n, int divisor_val,
                         Kokkos::View<int*>& d_results)
{
    int_fastdiv fdiv(divisor_val);
    const int dval = divisor_val;

    // Zero the result buffer
    Kokkos::deep_copy(d_results, 0);

    Kokkos::parallel_for("check_fastdiv", n, KOKKOS_LAMBDA(int dividend) {
        // Positive dividend
        {
            int quotient      = dividend / dval;
            int fast_quotient = dividend / fdiv;
            if (quotient != fast_quotient) {
                int eid = Kokkos::atomic_fetch_add(&d_results(0), 1);
                if (eid == 0) {
                    d_results(1) = dividend;
                    d_results(2) = quotient;
                    d_results(3) = fast_quotient;
                }
            }
        }
        // Negative dividend
        {
            int neg = -dividend;
            int quotient      = neg / dval;
            int fast_quotient = neg / fdiv;
            if (quotient != fast_quotient) {
                int eid = Kokkos::atomic_fetch_add(&d_results(0), 1);
                if (eid == 0) {
                    d_results(1) = neg;
                    d_results(2) = quotient;
                    d_results(3) = fast_quotient;
                }
            }
        }
    });
    Kokkos::fence();

    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_results);
    return h(0);
}

// ── functional test ───────────────────────────────────────────────────────────
static int functional_test()
{
    const int blocks         = 256;
    const int divisor_count  = 100000;
    const int dividend_count = 1000000;
    const int grids          = (dividend_count + blocks - 1) / blocks;
    const int n              = grids * blocks;

    Kokkos::View<int*> d_results("results", 4);

    std::cout << "Running functional test on " << divisor_count
              << " divisors, with " << n << " dividends each\n";

    for (int d = 1; d < divisor_count; ++d) {
        for (int sign = 1; sign >= -1; sign -= 2) {
            int divisor = d * sign;
            int nerrors = check_fastdiv(n, divisor, d_results);
            if (nerrors > 0) {
                auto h = Kokkos::create_mirror_view_and_copy(
                             Kokkos::HostSpace{}, d_results);
                std::cout << h(0) << " wrong results, one for dividend "
                          << h(1) << ", correct=" << h(2)
                          << ", fast=" << h(3) << "\n";
                return 1;
            }
        }
    }
    return 0;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <repeat>\n";
        return 1;
    }
    const int repeat = atoi(argv[1]);

    Kokkos::initialize(argc, argv);
    {
        if (functional_test()) {
            Kokkos::finalize();
            return 1;
        }

        const int grids  = 32 * 1024;
        const int blocks = 256;
        const int n      = grids * blocks;

        Kokkos::View<int*> d_buf("buf", 1);

        // Warmup
        for (int i = 0; i < 100; i++) {
            throughput_test<int>         (n, 3, 5, 7, 0, d_buf);
            throughput_test<int_fastdiv> (n, 3, 5, 7, 0, d_buf);
        }

        std::cout << "THROUGHPUT TEST\n";

        std::cout << "Benchmarking plain division by constant... ";
        auto start = NOW;
        for (int i = 0; i < repeat; i++)
            throughput_test<int>(n, 3, 5, 7, 0, d_buf);
        auto end = NOW;
        std::chrono::duration<double> slow = end - start;
        std::cout << slow.count() << " seconds\n";

        std::cout << "Benchmarking fast division by constant... ";
        start = NOW;
        for (int i = 0; i < repeat; i++)
            throughput_test<int_fastdiv>(n, 3, 5, 7, 0, d_buf);
        end = NOW;
        std::chrono::duration<double> fast = end - start;
        std::cout << fast.count() << " seconds\n";
        std::cout << "Speedup = " << slow.count() / fast.count() << "\n";

        // Warmup latency
        for (int i = 0; i < 100; i++) {
            latency_test<int>        (n, 1,2,3,4,5,6,7,8,9,10, 0, d_buf);
            latency_test<int_fastdiv>(n, 1,2,3,4,5,6,7,8,9,10, 0, d_buf);
        }

        std::cout << "LATENCY TEST\n";

        std::cout << "Benchmarking plain division by constant... ";
        start = NOW;
        for (int i = 0; i < repeat; i++)
            latency_test<int>(n, 1,2,3,4,5,6,7,8,9,10, 0, d_buf);
        end = NOW;
        slow = end - start;
        std::cout << slow.count() << " seconds\n";

        std::cout << "Benchmarking fast division by constant... ";
        start = NOW;
        for (int i = 0; i < repeat; i++)
            latency_test<int_fastdiv>(n, 1,2,3,4,5,6,7,8,9,10, 0, d_buf);
        end = NOW;
        fast = end - start;
        std::cout << fast.count() << " seconds\n";
        std::cout << "Speedup = " << slow.count() / fast.count() << "\n";
    }
    Kokkos::finalize();
    return 0;
}
