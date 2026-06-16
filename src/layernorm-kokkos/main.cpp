#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

static float rand_float() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// CPU reference implementation
static void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                                  const float* inp, const float* weight, const float* bias,
                                  int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* x = inp + i * C;
        float m = 0.0f;
        for (int j = 0; j < C; j++) m += x[j];
        m /= C;
        float v = 0.0f;
        for (int j = 0; j < C; j++) { float d = x[j] - m; v += d * d; }
        v /= C;
        float s = 1.0f / std::sqrt(v + 1e-5f);
        mean[i] = m;
        rstd[i] = s;
        float* o = out + i * C;
        for (int j = 0; j < C; j++)
            o[j] = (s * (x[j] - m)) * weight[j] + bias[j];
    }
}

int main(int argc, char** argv) {
    int kernel_num = 1;
    if (argc > 1) kernel_num = atoi(argv[1]);
    printf("Using kernel 1 (Kokkos)\n");

    Kokkos::initialize(argc, argv);
    {
        srand(0);
        const int B = 8, T = 1024, C = 768;
        const int N = B * T;

        // Host storage for initialization and reference
        std::vector<float> h_inp(N * C), h_weight(C), h_bias(C);
        std::vector<float> ref_out(N * C), ref_mean(N), ref_rstd(N);

        for (int i = 0; i < N * C; i++) h_inp[i]    = rand_float();
        for (int i = 0; i < C;     i++) h_weight[i] = rand_float();
        for (int i = 0; i < C;     i++) h_bias[i]   = rand_float();

        // CPU reference
        layernorm_forward_cpu(ref_out.data(), ref_mean.data(), ref_rstd.data(),
                              h_inp.data(), h_weight.data(), h_bias.data(), N, C);

        // Device views
        Kokkos::View<float*> inp("inp", N * C);
        Kokkos::View<float*> weight("weight", C);
        Kokkos::View<float*> bias("bias", C);
        Kokkos::View<float*> out("out", N * C);
        Kokkos::View<float*> d_mean("mean", N);
        Kokkos::View<float*> d_rstd("rstd", N);

        // Copy inputs to device views via host mirrors
        {
            auto h = Kokkos::create_mirror_view(inp);
            for (int i = 0; i < N * C; i++) h(i) = h_inp[i];
            Kokkos::deep_copy(inp, h);
        }
        {
            auto h = Kokkos::create_mirror_view(weight);
            for (int i = 0; i < C; i++) h(i) = h_weight[i];
            Kokkos::deep_copy(weight, h);
        }
        {
            auto h = Kokkos::create_mirror_view(bias);
            for (int i = 0; i < C; i++) h(i) = h_bias[i];
            Kokkos::deep_copy(bias, h);
        }

        // Kokkos layernorm kernel
        auto run_kernel = [&]() {
            Kokkos::parallel_for("layernorm_forward", N, KOKKOS_LAMBDA(int i) {
                float m = 0.0f;
                for (int j = 0; j < C; j++) m += inp(i * C + j);
                m /= C;
                float v = 0.0f;
                for (int j = 0; j < C; j++) {
                    float d = inp(i * C + j) - m;
                    v += d * d;
                }
                v /= C;
                float s = 1.0f / Kokkos::sqrt(v + 1e-5f);
                d_mean(i) = m;
                d_rstd(i) = s;
                for (int j = 0; j < C; j++)
                    out(i * C + j) = (s * (inp(i * C + j) - m)) * weight(j) + bias(j);
            });
            Kokkos::fence();
        };

        // Warmup
        run_kernel();

        // Correctness check
        printf("Checking correctness...\n");
        {
            auto h_out  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
            auto h_mean = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_mean);
            auto h_rstd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_rstd);

            int nfaults = 0;
            for (int i = 0; i < N * C && nfaults < 10; i++) {
                float tol = 1e-4f + std::fabs(ref_out[i]) * 1.19e-7f;
                if (std::fabs(h_out(i) - ref_out[i]) > tol) {
                    printf("Mismatch out[%d]: got %f expected %f\n", i, h_out(i), ref_out[i]);
                    nfaults++;
                }
            }
            for (int i = 0; i < N && nfaults < 10; i++) {
                if (std::fabs(h_mean(i) - ref_mean[i]) > 1e-4f) {
                    printf("Mismatch mean[%d]: got %f expected %f\n", i, h_mean(i), ref_mean[i]);
                    nfaults++;
                }
                if (std::fabs(h_rstd(i) - ref_rstd[i]) > 1e-4f) {
                    printf("Mismatch rstd[%d]: got %f expected %f\n", i, h_rstd(i), ref_rstd[i]);
                    nfaults++;
                }
            }
            if (nfaults > 0) { Kokkos::finalize(); return 1; }
        }

        printf("All results match. Starting benchmarks.\n");

        const int repeat = 2000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < repeat; n++) run_kernel();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;

        // Bytes: read inp (N*C), weight (C), bias (C), write out (N*C), mean (N), rstd (N)
        long bytes = (long)(2 * N * C + 2 * C + 2 * N) * sizeof(float);
        double gb_s = (bytes * 1e-9) / (ms * 1e-3);

        printf("Average kernel time: %.4f ms\n", ms);
        printf("Memory bandwidth: %.2f GB/s\n", gb_s);
    }
    Kokkos::finalize();
    return 0;
}
