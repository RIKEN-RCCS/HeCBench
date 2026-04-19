// simpleMultiDevice ported to Kokkos (single device)
// Original: distributes float reduction across multiple GPUs.
// Kokkos port: single-device parallel_reduce over DATA_N floats.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

static const int DATA_N = 1048576 * 32;  // 33554432

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    // Generate random float data on the host before Kokkos init
    std::vector<float> h_data(DATA_N);
    srand(42);
    for (int i = 0; i < DATA_N; i++)
        h_data[i] = (float)rand() / (float)RAND_MAX;

    float  gpu_sum = 0.0f;
    double cpu_sum = 0.0;
    double diff    = 0.0;

    Kokkos::initialize(argc, argv);
    {
        printf("Starting simpleMultiDevice\n");
        printf("GPU device count: 1\n");
        printf("Generating input data of size %d ...\n\n", DATA_N);

        // Copy to device view
        Kokkos::View<double*> d_data("d_data", DATA_N);
        {
            auto h_view = Kokkos::create_mirror_view(d_data);
            for (int i = 0; i < DATA_N; i++) h_view(i) = (double)h_data[i];
            Kokkos::deep_copy(d_data, h_view);
        }

        printf("Computing with 1 GPUs...\n");

        double gpu_sum_d = 0.0;
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < repeat; k++) {
            double local_sum = 0.0;
            Kokkos::parallel_reduce("Reduce", DATA_N,
                KOKKOS_LAMBDA(const int i, double& lsum) {
                    lsum += d_data(i);
                }, local_sum);
            Kokkos::fence();
            gpu_sum_d = local_sum;
        }
        gpu_sum = (float)gpu_sum_d;
        auto t1 = std::chrono::steady_clock::now();
        double elapsed_us =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
            * 1e-3 / repeat;
        printf("  Average GPU Processing time: %f (us)\n\n", elapsed_us);

        // CPU reference (double precision)
        printf("Computing with Host CPU...\n\n");
        for (int i = 0; i < DATA_N; i++) cpu_sum += (double)h_data[i];

        printf("Comparing GPU and Host CPU results...\n");
        diff = fabs(cpu_sum - (double)gpu_sum) / fabs(cpu_sum);
        printf("  GPU sum: %f\n  CPU sum: %f\n", gpu_sum, (float)cpu_sum);
        printf("  Relative difference: %E \n\n", diff);
    }
    Kokkos::finalize();

    return (diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
