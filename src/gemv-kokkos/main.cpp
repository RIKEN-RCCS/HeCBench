#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    unsigned int size = 512;
    unsigned int iter = 1;
    // block_dim_x/y are not used in Kokkos but accepted for CLI compatibility
    unsigned int block_dim_x = 32;
    unsigned int block_dim_y = 4;

    // Parse args before Kokkos::initialize so saved values survive argc/argv modification
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc)
            size = (unsigned int)atoi(argv[++i]);
        else if ((strcmp(argv[i], "--iter") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc)
            iter = (unsigned int)atoi(argv[++i]);
        else if ((strcmp(argv[i], "--block_x") == 0 || strcmp(argv[i], "-x") == 0) && i + 1 < argc)
            block_dim_x = (unsigned int)atoi(argv[++i]);
        else if ((strcmp(argv[i], "--block_y") == 0 || strcmp(argv[i], "-y") == 0) && i + 1 < argc)
            block_dim_y = (unsigned int)atoi(argv[++i]);
    }
    (void)block_dim_x;
    (void)block_dim_y;

    Kokkos::initialize(argc, argv);
    {
        printf("Testing GEMV with size=%u\n", size);

        const unsigned int n = size;
        Kokkos::View<float*> d_M("M", (size_t)n * n);
        Kokkos::View<float*> d_x("x", n);
        Kokkos::View<float*> d_y("y", n);

        // Initialise on host and copy to device
        {
            auto h_M = Kokkos::create_mirror_view(d_M);
            auto h_x = Kokkos::create_mirror_view(d_x);
            for (size_t i = 0; i < (size_t)n * n; i++) h_M(i) = 1.0f / (float)n;
            for (unsigned int i = 0; i < n; i++) h_x(i) = 1.0f;
            Kokkos::deep_copy(d_M, h_M);
            Kokkos::deep_copy(d_x, h_x);
        }

        // Warm-up
        Kokkos::parallel_for("gemv_warmup", Kokkos::RangePolicy<>(0, (int)n),
            KOKKOS_LAMBDA(const int row) {
                float sum = 0.0f;
                for (unsigned int j = 0; j < n; j++)
                    sum += d_M((size_t)row * n + j) * d_x(j);
                d_y(row) = sum;
            });
        Kokkos::fence();

        auto t_start = std::chrono::steady_clock::now();
        for (unsigned int it = 0; it < iter; it++) {
            Kokkos::parallel_for("gemv", Kokkos::RangePolicy<>(0, (int)n),
                KOKKOS_LAMBDA(const int row) {
                    float sum = 0.0f;
                    for (unsigned int j = 0; j < n; j++)
                        sum += d_M((size_t)row * n + j) * d_x(j);
                    d_y(row) = sum;
                });
        }
        Kokkos::fence();
        auto t_end = std::chrono::steady_clock::now();

        double total_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              t_end - t_start).count() * 1e-6;
        printf("Kernel time: %.3f ms\n", total_ms / (double)iter);
    }
    Kokkos::finalize();
    return 0;
}
