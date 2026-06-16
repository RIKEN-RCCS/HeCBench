/*
 * Threadfence / global reduction demo.
 * In Kokkos, we replace the custom last-block detection pattern
 * with a standard parallel_reduce which is semantically equivalent.
 *
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N      = atoi(argv[2]);

  float* h_array = (float*) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) h_array[i] = -1.f;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_array("d_array", N);
    auto h_a = Kokkos::create_mirror_view(d_array);
    for (int i = 0; i < N; i++) h_a(i) = h_array[i];
    Kokkos::deep_copy(d_array, h_a);

    bool ok = true;
    double total_time = 0.0;

    for (int n = 0; n < repeat; n++) {
      float result = 0.f;

      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_reduce("threadfence_sum", N,
          KOKKOS_LAMBDA(int i, float &sum) {
            sum += d_array(i);
          }, result);
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      if (result != -1.f * N) {
        ok = false;
        break;
      }
    }

    if (ok)
      printf("Average kernel execution time: %f (ms)\n", (total_time * 1e-6f) / repeat);

    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(h_array);
  return 0;
}
