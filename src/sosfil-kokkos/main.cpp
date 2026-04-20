#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>

static constexpr int SOS_WIDTH  = 6;
static constexpr int MAX_SECTIONS = 32;  // compile-time upper bound for stack arrays

template <typename T>
void filtering(const int repeat, const int n_signals, const int n_samples,
               const int n_sections, const int zi_width)
{
    assert(n_sections <= MAX_SECTIONS);
    assert(n_samples >= n_sections);

    const int sos_size = n_sections * SOS_WIDTH;
    const int z_size   = n_signals * n_sections * zi_width;
    const int x_size   = n_signals * n_samples;

    Kokkos::View<T*, Kokkos::HostSpace> sos("sos", sos_size);
    Kokkos::View<T*, Kokkos::HostSpace> zi("zi",   z_size);
    Kokkos::View<T*, Kokkos::HostSpace> x_in("x_in", x_size);

    // Initialize: all-ones SOS, all-ones zi, sinusoidal signals
    for (int i = 0; i < n_sections; i++)
        for (int j = 0; j < SOS_WIDTH; j++)
            sos(i * SOS_WIDTH + j) = (T)1;

    for (int i = 0; i < z_size; i++) zi(i) = (T)1;

    for (int i = 0; i < n_signals; i++)
        for (int j = 0; j < n_samples; j++)
            x_in(i * n_samples + j) = (T)std::sin(2 * 3.14 * (i + 1 + j));

    // Raw pointers for capture in KOKKOS_LAMBDA
    T* sos_ptr = sos.data();
    T* zi_ptr  = zi.data();
    T* x_ptr   = x_in.data();

    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
        Kokkos::parallel_for("sosfil", n_signals, KOKKOS_LAMBDA(int sig) {
            // Local per-signal zi state (fixed-size stack allocation)
            T local_zi[MAX_SECTIONS * 2];

            // Load zi from global (read-only across repeats, not written back)
            for (int sec = 0; sec < n_sections; sec++) {
                for (int k = 0; k < zi_width; k++) {
                    local_zi[sec * zi_width + k] =
                        zi_ptr[sig * n_sections * zi_width + sec * zi_width + k];
                }
            }

            // Process all samples: cascade through sections in order
            for (int s = 0; s < n_samples; s++) {
                T x_n = x_ptr[sig * n_samples + s];
                for (int sec = 0; sec < n_sections; sec++) {
                    const T* b = sos_ptr + sec * SOS_WIDTH;
                    T temp = b[0] * x_n + local_zi[sec * zi_width + 0];
                    local_zi[sec * zi_width + 0] =
                        b[1] * x_n - b[4] * temp + local_zi[sec * zi_width + 1];
                    local_zi[sec * zi_width + 1] =
                        b[2] * x_n - b[5] * temp;
                    x_n = temp;
                }
                x_ptr[sig * n_samples + s] = x_n;
            }
        });
        Kokkos::fence();
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average kernel execution time %lf (s)\n", ns * 1e-9 / repeat);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    Kokkos::initialize(argc, argv);
    {
        const int numSections = 32;
        const int numSignals  = 8;
        const int numSamples  = 100000;
        const int zi_width    = 2;

        filtering<float> (repeat, numSignals, numSamples, numSections, zi_width);
        filtering<double>(repeat, numSignals, numSamples, numSections, zi_width);
    }
    Kokkos::finalize();
    return 0;
}
