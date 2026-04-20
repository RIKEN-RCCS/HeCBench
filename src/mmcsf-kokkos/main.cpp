#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// -------------------------------------------------------------------------
// Sparse 3-D tensor in COO format
// -------------------------------------------------------------------------
struct COOTensor {
    int ndims;
    std::vector<int> dims;
    std::vector<int> inds0, inds1, inds2;   // 0-indexed
    std::vector<float> vals;
    int nnz() const { return (int)vals.size(); }
};

// -------------------------------------------------------------------------
// Read a .tns file with the header format used by toy.tns:
//   Line 1: ndims
//   Line 2: dim0 dim1 ... dimN
//   Remaining: i0 i1 ... iN  value  (1-indexed)
// -------------------------------------------------------------------------
static bool load_tns(const std::string& fname, COOTensor& T)
{
    std::ifstream f(fname);
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname.c_str()); return false; }

    std::string line;

    // ndims
    if (!std::getline(f, line)) return false;
    T.ndims = std::stoi(line);

    // dimension sizes
    if (!std::getline(f, line)) return false;
    {
        std::istringstream ss(line);
        int d;
        while (ss >> d) T.dims.push_back(d);
    }

    // nonzero entries (1-indexed)
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        int i0, i1, i2; float val;
        if (!(ss >> i0 >> i1 >> i2 >> val)) continue;
        T.inds0.push_back(i0 - 1);
        T.inds1.push_back(i1 - 1);
        T.inds2.push_back(i2 - 1);
        T.vals.push_back(val);
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <input.tns> [R] [repeat]\n", argv[0]);
        return 1;
    }
    const char *tnsFile = argv[1];
    int R      = (argc > 2) ? atoi(argv[2]) : 32;
    int repeat = (argc > 3) ? atoi(argv[3]) : 1;

    // -----------------------------------------------------------------------
    // Load tensor
    // -----------------------------------------------------------------------
    COOTensor X;
    if (!load_tns(tnsFile, X)) return 1;
    const int nnz = X.nnz();

    if (X.ndims != 3) {
        fprintf(stderr, "Only 3-D tensors supported (ndims=%d)\n", X.ndims);
        return 1;
    }

    int dim0 = X.dims[0], dim1 = X.dims[1], dim2 = X.dims[2];
    printf("Tensor: %d x %d x %d  nnz=%d  R=%d  repeat=%d\n",
           dim0, dim1, dim2, nnz, R, repeat);

    Kokkos::initialize(argc, argv);
    {
        // -----------------------------------------------------------------------
        // Move COO data to device
        // -----------------------------------------------------------------------
        Kokkos::View<int*>   d_i0("i0",   nnz);
        Kokkos::View<int*>   d_i1("i1",   nnz);
        Kokkos::View<int*>   d_i2("i2",   nnz);
        Kokkos::View<float*> d_v ("vals", nnz);
        {
            auto hi0 = Kokkos::create_mirror_view(d_i0);
            auto hi1 = Kokkos::create_mirror_view(d_i1);
            auto hi2 = Kokkos::create_mirror_view(d_i2);
            auto hv  = Kokkos::create_mirror_view(d_v);
            for (int n = 0; n < nnz; n++) {
                hi0(n) = X.inds0[n];
                hi1(n) = X.inds1[n];
                hi2(n) = X.inds2[n];
                hv (n) = X.vals[n];
            }
            Kokkos::deep_copy(d_i0, hi0);
            Kokkos::deep_copy(d_i1, hi1);
            Kokkos::deep_copy(d_i2, hi2);
            Kokkos::deep_copy(d_v,  hv);
        }

        // -----------------------------------------------------------------------
        // Factor matrices: U0 (output), U1, U2 (inputs)
        // Initialise with random values in [0, 1); U0 starts at zero each iter.
        // -----------------------------------------------------------------------
        Kokkos::View<float*> U0("U0", dim0 * R);
        Kokkos::View<float*> U1("U1", dim1 * R);
        Kokkos::View<float*> U2("U2", dim2 * R);

        srand(42);
        {
            auto hU1 = Kokkos::create_mirror_view(U1);
            auto hU2 = Kokkos::create_mirror_view(U2);
            for (int i = 0; i < dim1 * R; i++) hU1(i) = (float)rand() / RAND_MAX;
            for (int i = 0; i < dim2 * R; i++) hU2(i) = (float)rand() / RAND_MAX;
            Kokkos::deep_copy(U1, hU1);
            Kokkos::deep_copy(U2, hU2);
        }

        // -----------------------------------------------------------------------
        // MTTKRP for mode 0: U0[i*R+r] += val * U1[j*R+r] * U2[k*R+r]
        // -----------------------------------------------------------------------
        double totalTime = 0.0;

        for (int rep = 0; rep < repeat; rep++) {
            // Reset U0
            Kokkos::deep_copy(U0, 0.0f);
            Kokkos::fence();

            auto t0 = std::chrono::steady_clock::now();

            Kokkos::parallel_for("mttkrp_mode0",
                Kokkos::RangePolicy<>(0, nnz),
                KOKKOS_LAMBDA(int n) {
                    int   i   = d_i0[n];
                    int   j   = d_i1[n];
                    int   k   = d_i2[n];
                    float val = d_v [n];
                    for (int r = 0; r < R; r++) {
                        Kokkos::atomic_add(&U0[i * R + r],
                                           val * U1[j * R + r] * U2[k * R + r]);
                    }
                }
            );
            Kokkos::fence();

            auto t1 = std::chrono::steady_clock::now();
            totalTime += std::chrono::duration<double>(t1 - t0).count();
        }

        printf("Average MTTKRP time: %.6f s\n", totalTime / repeat);
        printf("Throughput: %.3f GFLOP/s\n",
               (2.0 * nnz * R) / (totalTime / repeat) * 1e-9);

        // Print a few values of U0 for sanity check
        auto hU0 = Kokkos::create_mirror_view(U0);
        Kokkos::deep_copy(hU0, U0);
        printf("U0[0..%d]:", (int)std::min(R, 4) - 1);
        for (int r = 0; r < std::min(R, 4); r++)
            printf(" %.4f", hU0(r));
        printf("\n");
    }
    Kokkos::finalize();
    return 0;
}
