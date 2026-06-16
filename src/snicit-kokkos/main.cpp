// SNICIT-kokkos: simplified sparse neural network inference (Kokkos port)
// The real SNICIT requires dataset files; this version uses synthetic data
// when the data path does not exist (or always, for benchmarking).
//
// Weight matrices are stored in CSC (compressed sparse column) format so that
// each output neuron j can independently sum its contributions:
//   output[b][j] = ReLU( sum_k(W[row_idx[k]] * input[b][row_idx[k]]) + bias[j] )
//   where k in [col_ptr[j], col_ptr[j+1])

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <sys/stat.h>

// --------------------------------------------------------------------------
// Sparse matrix (CSC) + bias for one layer
// --------------------------------------------------------------------------
struct LayerCSC {
    Kokkos::View<int*>   col_ptr;   // [num_neurons + 1]
    Kokkos::View<int*>   row_idx;   // [nnz]
    Kokkos::View<float*> vals;      // [nnz]
    Kokkos::View<float*> bias;      // [num_neurons]
};

// Build a synthetic CSC sparse matrix with the given density
LayerCSC make_synthetic_layer(int num_neurons, float density, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float>       gauss(0.0f, 1.0f);

    // Build host-side CSC
    std::vector<int>   h_col_ptr;
    std::vector<int>   h_row_idx;
    std::vector<float> h_vals;
    std::vector<float> h_bias(num_neurons);

    h_col_ptr.reserve(num_neurons + 1);
    h_col_ptr.push_back(0);

    for (int j = 0; j < num_neurons; j++) {
        for (int i = 0; i < num_neurons; i++) {
            if (uniform(rng) < density) {
                h_row_idx.push_back(i);
                h_vals.push_back(gauss(rng) * 0.1f);
            }
        }
        h_col_ptr.push_back((int)h_row_idx.size());
        h_bias[j] = gauss(rng) * 0.01f;
    }

    int nnz = (int)h_row_idx.size();

    LayerCSC layer;
    layer.col_ptr = Kokkos::View<int*>("col_ptr", num_neurons + 1);
    layer.row_idx = Kokkos::View<int*>("row_idx", nnz);
    layer.vals    = Kokkos::View<float*>("vals",   nnz);
    layer.bias    = Kokkos::View<float*>("bias",   num_neurons);

    auto h_cp = Kokkos::create_mirror_view(layer.col_ptr);
    auto h_ri = Kokkos::create_mirror_view(layer.row_idx);
    auto h_v  = Kokkos::create_mirror_view(layer.vals);
    auto h_b  = Kokkos::create_mirror_view(layer.bias);

    for (int j = 0; j <= num_neurons; j++) h_cp(j)      = h_col_ptr[j];
    for (int k = 0; k < nnz;          k++) h_ri(k)      = h_row_idx[k];
    for (int k = 0; k < nnz;          k++) h_v(k)       = h_vals[k];
    for (int j = 0; j < num_neurons;  j++) h_b(j)       = h_bias[j];

    Kokkos::deep_copy(layer.col_ptr, h_cp);
    Kokkos::deep_copy(layer.row_idx, h_ri);
    Kokkos::deep_copy(layer.vals,    h_v);
    Kokkos::deep_copy(layer.bias,    h_b);

    return layer;
}

// --------------------------------------------------------------------------
// Run one forward pass through all layers for a batch
// --------------------------------------------------------------------------
double run_inference(
    const std::vector<LayerCSC>& layers,
    Kokkos::View<float**>        buf0,   // [batch_size, num_neurons]  – input
    Kokkos::View<float**>        buf1,   // [batch_size, num_neurons]  – output
    int batch_size,
    int num_neurons,
    int num_layers)
{
    auto t0 = std::chrono::steady_clock::now();

    // Ping-pong between buf0 and buf1
    Kokkos::View<float**> cur = buf0;
    Kokkos::View<float**> nxt = buf1;

    for (int layer = 0; layer < num_layers; layer++) {
        auto col_ptr = layers[layer].col_ptr;
        auto row_idx = layers[layer].row_idx;
        auto vals    = layers[layer].vals;
        auto bias    = layers[layer].bias;

        Kokkos::parallel_for("SpMV_ReLU",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {batch_size, num_neurons}),
            KOKKOS_LAMBDA(int b, int j) {
                float sum = bias(j);
                int start = col_ptr(j);
                int end   = col_ptr(j + 1);
                for (int k = start; k < end; k++) {
                    sum += vals(k) * cur(b, row_idx(k));
                }
                nxt(b, j) = sum > 0.0f ? sum : 0.0f;
            }
        );
        Kokkos::fence();

        // swap buffers
        Kokkos::View<float**> tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6;
}

// --------------------------------------------------------------------------
// Simple argument parser (avoids CLI11 dependency)
// --------------------------------------------------------------------------
static std::string get_opt(int argc, char* argv[], const char* flag, const char* dflt) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}
static int get_opt_int(int argc, char* argv[], const char* flag, int dflt) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
    return dflt;
}

// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string benchmark  = get_opt(argc, argv, "-k", "A");
    std::string path       = get_opt(argc, argv, "-p", "../dataset");
    int         num_input  = get_opt_int(argc, argv, "-n", 10000);
    int         batch_size = get_opt_int(argc, argv, "-b", 10000);

    // Also accept long-form flags
    if (benchmark == "A" && get_opt(argc, argv, "--benchmark", "") != "")
        benchmark = get_opt(argc, argv, "--benchmark", "A");
    if (path == "../dataset" && get_opt(argc, argv, "--root_data_path", "") != "")
        path = get_opt(argc, argv, "--root_data_path", "../dataset");
    if (num_input == 10000) num_input = get_opt_int(argc, argv, "--num_input", 10000);
    if (batch_size == 10000) batch_size = get_opt_int(argc, argv, "--batch_size", 10000);

    int   num_hidden_neurons;
    int   num_layers;
    float density;

    if      (benchmark == "A") { num_hidden_neurons = 128; num_layers = 18; density = 0.6f; }
    else if (benchmark == "B") { num_hidden_neurons = 256; num_layers = 18; density = 0.6f; }
    else if (benchmark == "C") { num_hidden_neurons = 256; num_layers = 12; density = 0.5f; }
    else if (benchmark == "D") { num_hidden_neurons = 256; num_layers = 12; density = 0.5f; }
    else {
        fprintf(stderr, "Unknown benchmark '%s'. Use A, B, C, or D.\n", benchmark.c_str());
        return 1;
    }

    // Check whether the dataset path exists; warn and fall back to synthetic data
    struct stat st;
    bool use_synthetic = (stat(path.c_str(), &st) != 0);
    if (use_synthetic)
        printf("[Warning] Dataset path '%s' not found – using synthetic data.\n\n",
               path.c_str());

    printf("Benchmark: %s\n",          benchmark.c_str());
    printf("num_hidden_neurons = %d\n", num_hidden_neurons);
    printf("num_layers = %d\n",         num_layers);
    printf("num_inputs = %d\n",         num_input);
    printf("batch_size = %d\n",         batch_size);

    if (batch_size > num_input) batch_size = num_input;
    int num_batches = (num_input + batch_size - 1) / batch_size;

    Kokkos::initialize(argc, argv);
    {
        std::mt19937 rng(12345);

        // Build per-layer sparse weight matrices (synthetic)
        printf("Generating synthetic weight matrices...\n");
        std::vector<LayerCSC> layers;
        layers.reserve(num_layers);
        for (int l = 0; l < num_layers; l++)
            layers.push_back(make_synthetic_layer(num_hidden_neurons, density, rng));

        // Allocate ping-pong buffers
        Kokkos::View<float**> buf0("buf0", batch_size, num_hidden_neurons);
        Kokkos::View<float**> buf1("buf1", batch_size, num_hidden_neurons);

        // Fill buf0 with random binary-ish input
        {
            auto h_buf = Kokkos::create_mirror_view(buf0);
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            for (int b = 0; b < batch_size; b++)
                for (int j = 0; j < num_hidden_neurons; j++)
                    h_buf(b, j) = u(rng) > 0.5f ? 1.0f : 0.0f;
            Kokkos::deep_copy(buf0, h_buf);
        }

        // Warmup
        run_inference(layers, buf0, buf1, batch_size, num_hidden_neurons, num_layers);

        // Timed runs over all batches
        printf("Running inference...\n");
        double total_ms = 0.0;
        for (int bat = 0; bat < num_batches; bat++)
            total_ms += run_inference(
                layers, buf0, buf1, batch_size, num_hidden_neurons, num_layers);

        double avg_ms   = total_ms / num_batches;
        double thpt     = 1000.0 / avg_ms;   // batches per second

        printf("Average inference time per batch: %lf ms\n", avg_ms);
        printf("Throughput: %lf batches/sec\n", thpt);
    }
    Kokkos::finalize();
    return 0;
}
