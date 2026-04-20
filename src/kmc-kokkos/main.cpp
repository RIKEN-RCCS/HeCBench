/*
KMC kernel matrix computation for SVM (cuBLAS SGEMV replacement).
Kokkos port (OpenMP backend).

For each support vector trvei:
  1. dp[i] = sum_k tva[i*len_tv + k] * vtm[k]   (matrix-vector multiply)
  2. v_f_g[i] = exp(-gamma * (tv_sq[trvei] + tv_sq[i] - 2*dp[i]))

Usage: ./main [ntv [len_tv [repeat]]]
  ntv    = number of training vectors (default 100)
  len_tv = feature dimension          (default 50)
  repeat = benchmark repetitions      (default 10)
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[])
{
    int ntv    = 100;
    int len_tv = 50;
    int repeat = 10;

    if (argc > 1) ntv    = atoi(argv[1]);
    if (argc > 2) len_tv = atoi(argv[2]);
    if (argc > 3) repeat = atoi(argv[3]);

    const float gamma_val = 0.5f;

    Kokkos::initialize(argc, argv);
    {
        using ViewF1 = Kokkos::View<float*>;
        using ViewF2 = Kokkos::View<float**>;
        using ViewD1 = Kokkos::View<double*>;

        // Training matrix: tva[i][k] = feature k of vector i
        // Transposed: tr_ar[k][i]  (column-major for BLAS, row-major here as tr_ar(k,i))
        ViewF2 d_tva  ("tva",   ntv, len_tv);
        ViewF1 d_vtm  ("vtm",   len_tv);       // current query vector
        ViewF1 d_dp   ("dp",    ntv);           // dot products
        ViewD1 d_tv_sq("tv_sq", ntv);           // squared norms
        ViewD1 d_vfg  ("vfg",   ntv);           // kernel values (output)

        // Initialize random training data
        srand(42);
        {
            auto h_tva   = Kokkos::create_mirror_view(d_tva);
            auto h_tv_sq = Kokkos::create_mirror_view(d_tv_sq);
            for (int i = 0; i < ntv; i++) {
                double sq = 0.0;
                for (int k = 0; k < len_tv; k++) {
                    float val = (float)rand() / (float)RAND_MAX;
                    h_tva(i, k) = val;
                    sq += (double)val * (double)val;
                }
                h_tv_sq(i) = sq;
            }
            Kokkos::deep_copy(d_tva,   h_tva);
            Kokkos::deep_copy(d_tv_sq, h_tv_sq);
        }

        double total_inner_ns = 0.0; // accumulated time for matvec calls
        double total_outer_ns = 0.0; // total offload time per repeat

        for (int rep = 0; rep < repeat; rep++) {

            auto t_offload_start = std::chrono::steady_clock::now();
            double rep_inner_ns = 0.0;

            for (int trvei = 0; trvei < ntv; trvei++) {

                // Copy current training vector into vtm
                const int tv_idx = trvei;
                Kokkos::parallel_for("copyVtm", len_tv, KOKKOS_LAMBDA(int k) {
                    d_vtm(k) = d_tva(tv_idx, k);
                });
                Kokkos::fence();

                // ---- Timed region: matrix-vector multiply ----
                auto t1 = std::chrono::steady_clock::now();

                Kokkos::parallel_for("matvec", ntv, KOKKOS_LAMBDA(int i) {
                    float s = 0.0f;
                    for (int k = 0; k < len_tv; k++)
                        s += d_tva(i, k) * d_vtm(k);
                    d_dp(i) = s;
                });
                Kokkos::fence();

                auto t2 = std::chrono::steady_clock::now();
                rep_inner_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
                // ---- End timed region ----

                // Compute kernel values (post-processing, not timed per original)
                const int trv = trvei;
                Kokkos::parallel_for("kernel", ntv, KOKKOS_LAMBDA(int ic) {
                    d_vfg(ic) = Kokkos::exp(-(double)gamma_val *
                                 (d_tv_sq(trv) + d_tv_sq(ic) - 2.0 * (double)d_dp(ic)));
                });
                Kokkos::fence();
            }

            auto t_offload_end = std::chrono::steady_clock::now();
            double offload_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    t_offload_end - t_offload_start).count();

            total_inner_ns += rep_inner_ns;
            total_outer_ns += offload_ns;
        }

        // Average over repeats
        double avg_inner_ns  = total_inner_ns / repeat;
        double avg_outer_ns  = total_outer_ns / repeat;

        // Average per-vector offload time (matching original: time / ntv in us)
        printf("Average kernel matrix offload time: %lf (us)\n",
               (avg_inner_ns * 1e-3) / ntv);
        printf("Total kernel matrix execution time: %lf (us)\n",
               avg_outer_ns * 1e-3);
    }
    Kokkos::finalize();
    return 0;
}
