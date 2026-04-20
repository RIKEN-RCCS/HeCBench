// MNIST CNN Kokkos port
// Data files required in ./data/:
//   train-images.idx3-ubyte, train-labels.idx1-ubyte
//   t10k-images.idx3-ubyte,  t10k-labels.idx1-ubyte
//
// Network architecture (same as CUDA version):
//   l_input : 28x28 input
//   l_c1    : conv 5x5, 6 features  -> 6x24x24
//   l_s1    : subsample 4x4         -> 6x6x6
//   l_f     : fully-connected       -> 10 outputs

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------
static const float dt        = 1.0e-1f;
static const float threshold = 1.0e-2f;

// -------------------------------------------------------------------------
// Kokkos layer descriptor
// -------------------------------------------------------------------------
struct Layer {
    int M, N, O;

    Kokkos::View<float*> output;
    Kokkos::View<float*> preact;
    Kokkos::View<float*> bias;
    Kokkos::View<float*> weight;
    Kokkos::View<float*> d_output;
    Kokkos::View<float*> d_preact;
    Kokkos::View<float*> d_weight;

    Layer(int M_, int N_, int O_, const char *tag)
        : M(M_), N(N_), O(O_),
          output (Kokkos::view_alloc(std::string(tag)+"_out",   Kokkos::WithoutInitializing), O_),
          preact (Kokkos::view_alloc(std::string(tag)+"_pre",   Kokkos::WithoutInitializing), O_),
          bias   (Kokkos::view_alloc(std::string(tag)+"_bias",  Kokkos::WithoutInitializing), N_),
          weight (Kokkos::view_alloc(std::string(tag)+"_wt",    Kokkos::WithoutInitializing), M_*N_),
          d_output(Kokkos::view_alloc(std::string(tag)+"_dout", Kokkos::WithoutInitializing), O_),
          d_preact(Kokkos::view_alloc(std::string(tag)+"_dpre", Kokkos::WithoutInitializing), O_),
          d_weight(Kokkos::view_alloc(std::string(tag)+"_dwt",  Kokkos::WithoutInitializing), M_*N_)
    {
        auto hb = Kokkos::create_mirror_view(bias);
        auto hw = Kokkos::create_mirror_view(weight);
        for (int i = 0; i < N_; i++) {
            hb(i) = 0.5f - (float)rand() / RAND_MAX;
            for (int j = 0; j < M_; j++)
                hw(i*M_ + j) = 0.5f - (float)rand() / RAND_MAX;
        }
        Kokkos::deep_copy(bias,   hb);
        Kokkos::deep_copy(weight, hw);
        clear();
        bp_clear();
    }

    void setOutput(float *data) {
        // data lives on host (O floats)
        auto hv = Kokkos::create_mirror_view(
            Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(data, O));
        Kokkos::deep_copy(output, hv);
    }

    void clear() {
        Kokkos::deep_copy(output, 0.0f);
        Kokkos::deep_copy(preact, 0.0f);
    }

    void bp_clear() {
        Kokkos::deep_copy(d_output, 0.0f);
        Kokkos::deep_copy(d_preact, 0.0f);
        Kokkos::deep_copy(d_weight, 0.0f);
    }
};

// -------------------------------------------------------------------------
// Sigmoid and error helpers
// -------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION float step_function(float v) {
    return 1.f / (1.f + Kokkos::Experimental::exp(-v));
}

// -------------------------------------------------------------------------
// Forward pass kernels
// -------------------------------------------------------------------------

// fp_preact_c1: accumulate conv results into c1.preact
static void fp_preact_c1(Layer &input, Layer &c1)
{
    // Layout: c1.weight[f][ky][kx], c1.preact[f][oy][ox]
    // input.output[iy][ix] (28x28, flattened)
    const int N = 5*5*6*24*24;
    auto w  = c1.weight;
    auto pa = c1.preact;
    auto in = input.output;

    Kokkos::parallel_for("fp_pre_c1", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int kx = idx % 5;  idx /= 5;
        int ky = idx % 5;  idx /= 5;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 24; idx /= 24;
        int ox = idx % 24;
        Kokkos::atomic_add(&pa[f*24*24 + oy*24 + ox],
                           w[f*5*5 + ky*5 + kx] * in[(oy+ky)*28 + (ox+kx)]);
    });
}

static void fp_bias_c1(Layer &c1)
{
    const int N = 6*24*24;
    auto pa   = c1.preact;
    auto bias = c1.bias;
    Kokkos::parallel_for("fp_bias_c1", N, KOKKOS_LAMBDA(int n) {
        int f  = n / (24*24);
        pa[n] += bias[f];
    });
}

static void apply_step_function(Layer &l)
{
    auto pre = l.preact;
    auto out = l.output;
    int  sz  = l.O;
    Kokkos::parallel_for("step_fn", sz, KOKKOS_LAMBDA(int i) {
        out[i] = step_function(pre[i]);
    });
}

static void fp_preact_s1(Layer &c1, Layer &s1)
{
    // s1.weight[0][ky][kx], s1.preact[f][oy][ox] <- c1.output[f][oy*4+ky][ox*4+kx]
    const int N = 4*4*6*6*6;
    auto w  = s1.weight;
    auto pa = s1.preact;
    auto in = c1.output;

    Kokkos::parallel_for("fp_pre_s1", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int kx = idx % 4;  idx /= 4;
        int ky = idx % 4;  idx /= 4;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 6;  idx /= 6;
        int ox = idx % 6;
        Kokkos::atomic_add(&pa[f*6*6 + oy*6 + ox],
                           w[ky*4 + kx] * in[f*24*24 + (oy*4+ky)*24 + (ox*4+kx)]);
    });
}

static void fp_bias_s1(Layer &s1)
{
    const int N = 6*6*6;
    auto pa   = s1.preact;
    auto bias = s1.bias;
    Kokkos::parallel_for("fp_bias_s1", N, KOKKOS_LAMBDA(int n) {
        pa[n] += bias[0];
    });
}

static void fp_preact_f(Layer &s1, Layer &lf)
{
    // lf.weight[out][f][y][x], lf.preact[out]
    const int N = 10*6*6*6;
    auto w  = lf.weight;
    auto pa = lf.preact;
    auto in = s1.output;

    Kokkos::parallel_for("fp_pre_f", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int o  = idx % 10; idx /= 10;
        int f  = idx % 6;  idx /= 6;
        int y  = idx % 6;  idx /= 6;
        int x  = idx % 6;
        Kokkos::atomic_add(&pa[o],
                           w[o*6*6*6 + f*6*6 + y*6 + x] * in[f*6*6 + y*6 + x]);
    });
}

static void fp_bias_f(Layer &lf)
{
    auto pa   = lf.preact;
    auto bias = lf.bias;
    Kokkos::parallel_for("fp_bias_f", 10, KOKKOS_LAMBDA(int i) {
        pa[i] += bias[i];
    });
}

// -------------------------------------------------------------------------
// Backward pass kernels
// -------------------------------------------------------------------------

static void makeError(Layer &lf, unsigned int Y)
{
    auto dp  = lf.d_preact;
    auto out = lf.output;
    Kokkos::parallel_for("makeError", 10, KOKKOS_LAMBDA(int i) {
        dp[i] = ((int)Y == i ? 1.0f : 0.0f) - out[i];
    });
}

static void bp_weight_f(Layer &lf, Layer &s1)
{
    const int N = 10*6*6*6;
    auto dw  = lf.d_weight;
    auto dp  = lf.d_preact;
    auto in  = s1.output;
    Kokkos::parallel_for("bp_wt_f", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int o = idx % 10; idx /= 10;
        int f = idx % 6;  idx /= 6;
        int y = idx % 6;  idx /= 6;
        int x = idx % 6;
        dw[o*6*6*6 + f*6*6 + y*6 + x] = dp[o] * in[f*6*6 + y*6 + x];
    });
}

static void bp_bias_f(Layer &lf)
{
    auto bias = lf.bias;
    auto dp   = lf.d_preact;
    Kokkos::parallel_for("bp_bias_f", 10, KOKKOS_LAMBDA(int i) {
        bias[i] += dt * dp[i];
    });
}

static void bp_output_s1(Layer &s1, Layer &lf)
{
    const int N = 10*6*6*6;
    auto dout = s1.d_output;
    auto wf   = lf.weight;
    auto dpf  = lf.d_preact;
    Kokkos::parallel_for("bp_out_s1", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int o = idx % 10; idx /= 10;
        int f = idx % 6;  idx /= 6;
        int y = idx % 6;  idx /= 6;
        int x = idx % 6;
        Kokkos::atomic_add(&dout[f*6*6 + y*6 + x],
                           wf[o*6*6*6 + f*6*6 + y*6 + x] * dpf[o]);
    });
}

static void bp_preact_s1(Layer &s1)
{
    const int N = 6*6*6;
    auto dp   = s1.d_preact;
    auto dout = s1.d_output;
    auto pre  = s1.preact;
    Kokkos::parallel_for("bp_pre_s1", N, KOKKOS_LAMBDA(int n) {
        float o = step_function(pre[n]);
        dp[n]   = dout[n] * o * (1.f - o);
    });
}

static void bp_weight_s1(Layer &s1, Layer &c1)
{
    const int N = 1*4*4*6*6*6;
    auto dw   = s1.d_weight;
    auto dp   = s1.d_preact;
    auto pc1  = c1.output;
    Kokkos::parallel_for("bp_wt_s1", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int kx = idx % 4;  idx /= 4;
        int ky = idx % 4;  idx /= 4;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 6;  idx /= 6;
        int ox = idx % 6;
        Kokkos::atomic_add(&dw[ky*4 + kx],
                           dp[f*6*6 + oy*6 + ox] *
                           pc1[f*24*24 + (oy*4+ky)*24 + (ox*4+kx)]);
    });
}

static void bp_bias_s1(Layer &s1)
{
    const int N = 6*6*6;
    const float d = 216.f;
    auto bias = s1.bias;
    auto dp   = s1.d_preact;
    // accumulate into bias[0]
    Kokkos::parallel_for("bp_bias_s1", N, KOKKOS_LAMBDA(int n) {
        Kokkos::atomic_add(&bias[0], dt * dp[n] / d);
    });
}

static void bp_output_c1(Layer &c1, Layer &s1)
{
    const int N = 1*4*4*6*6*6;
    auto dout = c1.d_output;
    auto ws1  = s1.weight;
    auto dps1 = s1.d_preact;
    Kokkos::parallel_for("bp_out_c1", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int kx = idx % 4;  idx /= 4;
        int ky = idx % 4;  idx /= 4;
        int f  = idx % 6;  idx /= 6;
        int oy = idx % 6;  idx /= 6;
        int ox = idx % 6;
        Kokkos::atomic_add(&dout[f*24*24 + (oy*4+ky)*24 + (ox*4+kx)],
                           ws1[ky*4 + kx] * dps1[f*6*6 + oy*6 + ox]);
    });
}

static void bp_preact_c1(Layer &c1)
{
    const int N = 6*24*24;
    auto dp   = c1.d_preact;
    auto dout = c1.d_output;
    auto pre  = c1.preact;
    Kokkos::parallel_for("bp_pre_c1", N, KOKKOS_LAMBDA(int n) {
        float o = step_function(pre[n]);
        dp[n]   = dout[n] * o * (1.f - o);
    });
}

static void bp_weight_c1(Layer &c1, Layer &input)
{
    const int N = 6*5*5*24*24;
    const float d = 576.f;
    auto dw  = c1.d_weight;
    auto dp  = c1.d_preact;
    auto inp = input.output;
    Kokkos::parallel_for("bp_wt_c1", N, KOKKOS_LAMBDA(int n) {
        int idx = n;
        int f  = idx % 6;  idx /= 6;
        int kx = idx % 5;  idx /= 5;
        int ky = idx % 5;  idx /= 5;
        int oy = idx % 24; idx /= 24;
        int ox = idx % 24;
        Kokkos::atomic_add(&dw[f*5*5 + ky*5 + kx],
                           dp[f*24*24 + oy*24 + ox] *
                           inp[(oy+ky)*28 + (ox+kx)] / d);
    });
}

static void bp_bias_c1(Layer &c1)
{
    const int N = 6*24*24;
    const float d = 576.f;
    auto bias = c1.bias;
    auto dp   = c1.d_preact;
    Kokkos::parallel_for("bp_bias_c1", N, KOKKOS_LAMBDA(int n) {
        int f = n / (24*24);
        Kokkos::atomic_add(&bias[f], dt * dp[n] / d);
    });
}

static void apply_grad(Kokkos::View<float*> wt, Kokkos::View<float*> dw, int sz)
{
    Kokkos::parallel_for("apply_grad", sz, KOKKOS_LAMBDA(int i) {
        wt[i] += dt * dw[i];
    });
}

// -------------------------------------------------------------------------
// Norm of d_preact on host (replacing snrm2 from CUDA cublas)
// -------------------------------------------------------------------------
static float compute_norm(Kokkos::View<float*> v, int n)
{
    auto hv = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(hv, v);
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += hv(i) * hv(i);
    return sqrtf(sum);
}

// -------------------------------------------------------------------------
// Forward + backward pass wrappers
// -------------------------------------------------------------------------
static void forward_pass(double data[28][28],
                         Layer &l_input, Layer &l_c1, Layer &l_s1, Layer &l_f)
{
    l_input.clear(); l_c1.clear(); l_s1.clear(); l_f.clear();

    float inp[28*28];
    for (int i = 0; i < 28; i++)
        for (int j = 0; j < 28; j++)
            inp[i*28+j] = (float)data[i][j];
    l_input.setOutput(inp);

    fp_preact_c1(l_input, l_c1);
    fp_bias_c1(l_c1);
    apply_step_function(l_c1);

    fp_preact_s1(l_c1, l_s1);
    fp_bias_s1(l_s1);
    apply_step_function(l_s1);

    fp_preact_f(l_s1, l_f);
    fp_bias_f(l_f);
    apply_step_function(l_f);
}

static void back_pass(Layer &l_input, Layer &l_c1, Layer &l_s1, Layer &l_f)
{
    bp_weight_f(l_f, l_s1);
    bp_bias_f(l_f);

    bp_output_s1(l_s1, l_f);
    bp_preact_s1(l_s1);
    bp_weight_s1(l_s1, l_c1);
    bp_bias_s1(l_s1);

    bp_output_c1(l_c1, l_s1);
    bp_preact_c1(l_c1);
    bp_weight_c1(l_c1, l_input);
    bp_bias_c1(l_c1);

    apply_grad(l_f.weight,  l_f.d_weight,  l_f.M  * l_f.N);
    apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
    apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    if (argc != 2) { printf("Usage: %s <iterations>\n", argv[0]); return 1; }
    const int iter = atoi(argv[1]);

    srand(123);

    // Load MNIST
    mnist_data *train_set = nullptr, *test_set = nullptr;
    unsigned int train_cnt = 0, test_cnt = 0;
    if (mnist_load("data/train-images.idx3-ubyte",
                   "data/train-labels.idx1-ubyte",
                   &train_set, &train_cnt) ||
        mnist_load("data/t10k-images.idx3-ubyte",
                   "data/t10k-labels.idx1-ubyte",
                   &test_set, &test_cnt)) {
        fprintf(stderr, "Failed to load MNIST data from ./data/\n");
        return 1;
    }

    Kokkos::initialize(const_cast<int&>(argc),
                       const_cast<char**>(argv));
    {
        // Build layers (same sizes as CUDA version)
        Layer l_input(0,       0,  28*28,  "inp");
        Layer l_c1   (5*5,     6,  24*24*6,"c1" );
        Layer l_s1   (4*4,     1,  6*6*6,  "s1" );
        Layer l_f    (6*6*6,  10,  10,     "f"  );

        auto t1 = std::chrono::high_resolution_clock::now();

        // ---- Training ----
        fprintf(stdout, "Learning\n");
        int remaining = iter;
        while (remaining < 0 || remaining-- > 0) {
            float err = 0.f;
            for (unsigned i = 0; i < train_cnt; i++) {
                forward_pass(train_set[i].data, l_input, l_c1, l_s1, l_f);

                l_f.bp_clear();
                l_s1.bp_clear();
                l_c1.bp_clear();

                makeError(l_f, train_set[i].label);
                err += compute_norm(l_f.d_preact, 10);

                back_pass(l_input, l_c1, l_s1, l_f);
            }
            err /= train_cnt;
            fprintf(stdout, "error: %e\n", err);
            if (err < threshold) {
                fprintf(stdout, "Training complete, error less than threshold\n\n");
                break;
            }
        }

        // ---- Testing ----
        fprintf(stdout, "Testing\n");
        int errors = 0;
        for (unsigned i = 0; i < test_cnt; i++) {
            forward_pass(test_set[i].data, l_input, l_c1, l_s1, l_f);

            auto hf = Kokkos::create_mirror_view(l_f.output);
            Kokkos::deep_copy(hf, l_f.output);
            unsigned int pred = 0;
            for (int k = 1; k < 10; k++)
                if (hf(k) > hf(pred)) pred = k;
            if (pred != test_set[i].label) errors++;
        }
        fprintf(stdout, "Error Rate: %.2f%%\n",
                100.0 * errors / (double)test_cnt);

        auto t2 = std::chrono::high_resolution_clock::now();
        double total = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        printf("Total time (learn + test) %.6f secs\n", total / 1e6);
    }
    Kokkos::finalize();
    return 0;
}
