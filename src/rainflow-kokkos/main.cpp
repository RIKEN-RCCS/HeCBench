#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// Define double3 since we don't have CUDA headers
struct double3 {
    double x, y, z;
};

// -----------------------------------------------------------------------
// Device-callable helpers
// -----------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void Extrema(const double* history, const int history_length,
             double* result, int& result_length)
{
    result[0] = history[0];
    int eidx = 0;
    for (int i = 1; i < history_length - 1; i++) {
        if ((history[i] > result[eidx] && history[i] > history[i + 1]) ||
            (history[i] < result[eidx] && history[i] < history[i + 1]))
            result[++eidx] = history[i];
    }
    result[++eidx] = history[history_length - 1];
    result_length  = eidx + 1;
}

KOKKOS_INLINE_FUNCTION
void Execute(const double* history, const int history_length,
             double* extrema, int* points, double3* results,
             int* results_length)
{
    int extrema_length = 0;
    Extrema(history, history_length, extrema, extrema_length);

    int pidx = -1, eidx = -1, ridx = -1;

    for (int i = 0; i < extrema_length; i++) {
        points[++pidx] = ++eidx;
        double xRange, yRange;
        while (pidx >= 2 &&
               (xRange = fabs(extrema[points[pidx - 1]] - extrema[points[pidx]])) >=
               (yRange = fabs(extrema[points[pidx - 2]] - extrema[points[pidx - 1]])))
        {
            double yMean = 0.5 * (extrema[points[pidx - 2]] + extrema[points[pidx - 1]]);

            if (pidx == 2) {
                results[++ridx] = {0.5, yRange, yMean};
                points[0] = points[1];
                points[1] = points[2];
                pidx = 1;
            } else {
                results[++ridx] = {1.0, yRange, yMean};
                points[pidx - 2] = points[pidx];
                pidx -= 2;
            }
        }
    }

    for (int i = 0; i <= pidx - 1; i++) {
        double range = fabs(extrema[points[i]] - extrema[points[i + 1]]);
        double mean  = 0.5 * (extrema[points[i]] + extrema[points[i + 1]]);
        results[++ridx] = {0.5, range, mean};
    }

    *results_length = ridx + 1;
}

// -----------------------------------------------------------------------
// CPU reference (uses separate buffers to avoid aliasing with device data)
// -----------------------------------------------------------------------

static void ref_Extrema(const double* history, int history_length,
                        double* result, int& result_length)
{
    result[0] = history[0];
    int eidx  = 0;
    for (int i = 1; i < history_length - 1; i++) {
        if ((history[i] > result[eidx] && history[i] > history[i + 1]) ||
            (history[i] < result[eidx] && history[i] < history[i + 1]))
            result[++eidx] = history[i];
    }
    result[++eidx] = history[history_length - 1];
    result_length  = eidx + 1;
}

static void ref_Execute(const double* history, int history_length,
                        double* extrema, int* points, double3* results,
                        int* results_length)
{
    int extrema_length = 0;
    ref_Extrema(history, history_length, extrema, extrema_length);

    int pidx = -1, eidx = -1, ridx = -1;

    for (int i = 0; i < extrema_length; i++) {
        points[++pidx] = ++eidx;
        double xRange, yRange;
        while (pidx >= 2 &&
               (xRange = fabs(extrema[points[pidx - 1]] - extrema[points[pidx]])) >=
               (yRange = fabs(extrema[points[pidx - 2]] - extrema[points[pidx - 1]])))
        {
            double yMean = 0.5 * (extrema[points[pidx - 2]] + extrema[points[pidx - 1]]);
            if (pidx == 2) {
                results[++ridx] = {0.5, yRange, yMean};
                points[0] = points[1];
                points[1] = points[2];
                pidx = 1;
            } else {
                results[++ridx] = {1.0, yRange, yMean};
                points[pidx - 2] = points[pidx];
                pidx -= 2;
            }
        }
    }
    for (int i = 0; i <= pidx - 1; i++) {
        double range = fabs(extrema[points[i]] - extrema[points[i + 1]]);
        double mean  = 0.5 * (extrema[points[i]] + extrema[points[i + 1]]);
        results[++ridx] = {0.5, range, mean};
    }
    *results_length = ridx + 1;
}

static void reference(const double* history, const int* history_lengths,
                      double* extrema, int* points, double3* results,
                      int* result_length, int num_history)
{
    for (int i = 0; i < num_history; i++) {
        const int offset         = history_lengths[i];
        const int history_length = history_lengths[i + 1] - offset;
        ref_Execute(history + offset, history_length,
                    extrema + offset, points + offset,
                    results + offset, result_length + i);
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const int num_history = atoi(argv[1]);
    const int repeat      = atoi(argv[2]);

    int *history_lengths     = (int*)    malloc((num_history + 1) * sizeof(int));
    int *result_lengths      = (int*)    malloc(num_history       * sizeof(int));
    int *ref_result_lengths  = (int*)    malloc(num_history       * sizeof(int));

    srand(123);

    const int scale    = 100;
    size_t total_length = 0;
    int n;
    for (n = 0; n < num_history; n++) {
        history_lengths[n] = (int)total_length;
        total_length += (size_t)((rand() % 10 + 1) * scale);
    }
    history_lengths[n] = (int)total_length;

    printf("Total history length = %zu\n", total_length);

    double  *history = (double*)  malloc(total_length * sizeof(double));
    double  *extrema = (double*)  malloc(total_length * sizeof(double));
    double3 *results = (double3*) malloc(total_length * sizeof(double3));
    int     *points  = (int*)     malloc(total_length * sizeof(int));

    // Reference work arrays (separate from device work arrays)
    double  *ref_extrema = (double*)  malloc(total_length * sizeof(double));
    double3 *ref_results = (double3*) malloc(total_length * sizeof(double3));
    int     *ref_points  = (int*)     malloc(total_length * sizeof(int));

    for (size_t i = 0; i < total_length; i++)
        history[i] = rand() / (double)RAND_MAX;

    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<int*>     d_history_lengths("history_lengths", num_history + 1);
        Kokkos::View<double*>  d_history        ("history",         total_length);
        Kokkos::View<double*>  d_extrema        ("extrema",         total_length);
        Kokkos::View<double3*> d_results        ("results",         total_length);
        Kokkos::View<int*>     d_points         ("points",          total_length);
        Kokkos::View<int*>     d_result_lengths ("result_lengths",  num_history);

        {
            auto hhl = Kokkos::create_mirror_view(d_history_lengths);
            auto hh  = Kokkos::create_mirror_view(d_history);
            for (int i = 0; i <= num_history; i++) hhl(i) = history_lengths[i];
            for (size_t i = 0; i < total_length; i++) hh(i) = history[i];
            Kokkos::deep_copy(d_history_lengths, hhl);
            Kokkos::deep_copy(d_history, hh);
        }

        auto start = std::chrono::steady_clock::now();

        for (n = 0; n < repeat; n++) {
            Kokkos::parallel_for("rainflow", num_history,
                KOKKOS_LAMBDA(int i) {
                    const int offset         = d_history_lengths[i];
                    const int history_length = d_history_lengths[i + 1] - offset;
                    Execute(d_history.data() + offset,
                            history_length,
                            d_extrema.data() + offset,
                            d_points.data()  + offset,
                            d_results.data() + offset,
                            d_result_lengths.data() + i);
                });
            Kokkos::fence();
        }

        auto end  = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

        // Copy result_lengths back
        {
            auto h = Kokkos::create_mirror_view(d_result_lengths);
            Kokkos::deep_copy(h, d_result_lengths);
            for (int i = 0; i < num_history; i++) result_lengths[i] = h(i);
        }
    }
    Kokkos::finalize();

    reference(history, history_lengths,
              ref_extrema, ref_points, ref_results,
              ref_result_lengths, num_history);

    int error = memcmp(ref_result_lengths, result_lengths, num_history * sizeof(int));
    printf("%s\n", error ? "FAIL" : "PASS");

    free(history);
    free(history_lengths);
    free(extrema);
    free(points);
    free(results);
    free(result_lengths);
    free(ref_result_lengths);
    free(ref_extrema);
    free(ref_results);
    free(ref_points);
    return 0;
}
