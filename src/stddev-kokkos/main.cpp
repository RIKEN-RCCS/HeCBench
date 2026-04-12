/*
 * Standard deviation of matrix columns.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// CPU reference
template <typename T>
void stddev_ref(T *std_out, const T *data, int D, int N, bool sample) {
  for (int j = 0; j < D; j++) {
    T sum = T(0);
    for (int i = 0; i < N; i++)
      sum += data[i * D + j] * data[i * D + j];
    int sampleSize = sample ? N - 1 : N;
    std_out[j] = sqrtf(sum / sampleSize);
  }
}

template <typename T>
void stddev(Kokkos::View<T*> std_out,
            Kokkos::View<const T*> data,
            int D, int N, bool sample)
{
  // Zero out output
  Kokkos::parallel_for("stddev_zero", D, KOKKOS_LAMBDA(int i) {
    std_out(i) = T(0);
  });
  Kokkos::fence();

  // Accumulate sum of squares per column
  Kokkos::parallel_for("stddev_accum",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, D}),
      KOKKOS_LAMBDA(int i, int j) {
        T val = data(i * D + j);
        Kokkos::atomic_add(&std_out(j), val * val);
      });
  Kokkos::fence();

  int sampleSize = sample ? N - 1 : N;
  Kokkos::parallel_for("stddev_sqrt", D, KOKKOS_LAMBDA(int i) {
    std_out(i) = Kokkos::sqrt(std_out(i) / sampleSize);
  });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns (multiple of 32)\n");
    printf("N: number of rows (at least 1)\n");
    return 1;
  }
  const int D      = atoi(argv[1]);
  const int N      = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const bool sample = true;

  float *data    = (float*) malloc((long)D * N * sizeof(float));
  float *std_h   = (float*) malloc(D * sizeof(float));
  float *std_ref = (float*) malloc(D * sizeof(float));

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++)
      data[i * D + j] = rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_data("d_data", (long)D * N);
    Kokkos::View<float*> d_std("d_std", D);

    auto h_data = Kokkos::create_mirror_view(d_data);
    for (long i = 0; i < (long)D * N; i++) h_data(i) = data[i];
    Kokkos::deep_copy(d_data, h_data);

    // Warmup
    stddev<float>(d_std, Kokkos::View<const float*>(d_data), D, N, sample);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      stddev<float>(d_std, Kokkos::View<const float*>(d_data), D, N, sample);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);

    auto h_std = Kokkos::create_mirror_view(d_std);
    Kokkos::deep_copy(h_std, d_std);
    for (int i = 0; i < D; i++) std_h[i] = h_std(i);
  }
  Kokkos::finalize();

  stddev_ref<float>(std_ref, data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - std_h[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(data);
  free(std_h);
  free(std_ref);
  return 0;
}
