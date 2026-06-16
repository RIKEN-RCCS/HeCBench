// Slit diffraction via 2D FFT (Kokkos port of si-cuda)
// Implements Cooley-Tukey radix-2 FFT applied row-wise then column-wise.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using KComplex = Kokkos::complex<double>;
// LayoutRight = row-major; rows are contiguous so &view(row,0) is a valid 1D pointer
using View2D = Kokkos::View<KComplex**, Kokkos::LayoutRight>;

// In-place 1D Cooley-Tukey radix-2 FFT (forward, n must be power of 2)
KOKKOS_INLINE_FUNCTION
void fft1d(KComplex* data, int n) {
    // Bit-reverse permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            KComplex tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        KComplex wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            KComplex w(1.0, 0.0);
            for (int jj = 0; jj < len / 2; jj++) {
                KComplex u = data[i + jj];
                KComplex v = data[i + jj + len / 2] * w;
                data[i + jj]           = u + v;
                data[i + jj + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Returns elapsed time in ms
double fft2d(View2D d, View2D scratch, int N) {
    auto t0 = std::chrono::steady_clock::now();

    // Row-wise FFT: each row is contiguous in LayoutRight
    Kokkos::parallel_for("FFT_rows", N, KOKKOS_LAMBDA(int row) {
        fft1d(&d(row, 0), N);
    });
    Kokkos::fence();

    // Column-wise FFT: copy each column into a scratch row, FFT, copy back
    Kokkos::parallel_for("FFT_cols", N, KOKKOS_LAMBDA(int col) {
        for (int k = 0; k < N; k++) scratch(col, k) = d(k, col);
        fft1d(&scratch(col, 0), N);
        for (int k = 0; k < N; k++) d(k, col) = scratch(col, k);
    });
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6;
}

// Reference direct 2D DFT (O(N^4), CPU-only, for small N verification)
void reference_dft(const double* input, double* output, int N) {
    for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    double angle = -2.0 * M_PI * ((double)(j * k + i * l)) / N;
                    re += input[j * N + i] * cos(angle);
                    im += input[j * N + i] * sin(angle);
                }
            }
            output[k * N + l] = re * re + im * im;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <transform_size_N> <repeat>\n", argv[0]);
        return 1;
    }
    const int N      = atoi(argv[1]);
    const int repeat = atoi(argv[2]);

    // N must be a power of 2 for Cooley-Tukey
    if (N <= 0 || (N & (N - 1)) != 0) {
        printf("Error: N must be a positive power of 2\n");
        return 1;
    }

    printf("Running FFT for %d x %d = %d = 2 ^ %d data points...\n",
           N, N, N * N, (int)(log2((double)(N * N))));

    Kokkos::initialize(argc, argv);
    {
        View2D d("data",    N, N);
        View2D scratch("scratch", N, N);

        auto h_d = Kokkos::create_mirror_view(d);

        const int slit_height = 4;
        const int slit_width  = 2;
        const int slit_dist   = 8;

        // Build double-slit input pattern
        std::vector<double> input_ref(N * N, 0.0);
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int di = (i - N / 2);  if (di < 0) di = -di;
                int dj = (j - N / 2);  if (dj < 0) dj = -dj;
                if (di >= slit_dist && di <= slit_dist + slit_width && dj <= slit_height) {
                    h_d(j, i)             = KComplex(1.0, 0.0);
                    input_ref[j * N + i]  = 1.0;
                } else {
                    h_d(j, i) = KComplex(0.0, 0.0);
                }
            }
        }
        Kokkos::deep_copy(d, h_d);

        // Save initial data in a separate view so repeated runs can reset correctly
        View2D d_init("data_init", N, N);
        Kokkos::deep_copy(d_init, h_d);

        // Warmup
        {
            View2D tmp("tmp_warm", N, N);
            View2D stmp("stmp_warm", N, N);
            Kokkos::deep_copy(tmp, d_init);
            fft2d(tmp, stmp, N);
        }

        // Timed runs
        double total_ms = 0.0;
        for (int r = 0; r < repeat; r++) {
            Kokkos::deep_copy(d, d_init);
            total_ms += fft2d(d, scratch, N);
        }
        printf("Average execution time of FFT: %lf ms\n", total_ms / repeat);

        // Copy result back and compute power spectrum
        Kokkos::deep_copy(h_d, d);
        std::vector<double> output(N * N);
        for (int idx = 0; idx < N * N; idx++) {
            double re = h_d(idx / N, idx % N).real();
            double im = h_d(idx / N, idx % N).imag();
            output[idx] = re * re + im * im;
        }

        // Verify against direct DFT reference (only for small N)
        bool ok = true;
        if (N <= 64) {
            std::vector<double> ref_output(N * N, 0.0);
            reference_dft(input_ref.data(), ref_output.data(), N);
            for (int i = 0; i < N * N; i++) {
                if (std::abs(output[i] - ref_output[i]) > 1e-3 * (1.0 + ref_output[i])) {
                    ok = false;
                    break;
                }
            }
        }

        printf("%s\n", ok ? "PASS" : "FAIL");
    }
    Kokkos::finalize();
    return 0;
}
