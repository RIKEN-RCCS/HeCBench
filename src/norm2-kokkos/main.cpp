#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define max(a, b) (a < b ? b : a)

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = max(1, atoi(argv[1]));

  Kokkos::initialize(argc, argv);
  {
    bool ok = true;

    float* h_result = (float*) malloc(repeat * sizeof(float));
    if (h_result == nullptr) {
      printf("output on host allocation failed\n");
      Kokkos::finalize();
      return 1;
    }

    for (int n = 512*1024; n <= 1024*1024*512; n = n * 2) {
      float *a = (float*) malloc(n * sizeof(float));
      if (a == nullptr) {
        printf("input on host allocation failed\n");
        break;
      }

      double gold = 0.0;
      for (int i = 0; i < n; i++) {
        a[i] = (float)((i+1) % 7);
        gold += (double)a[i] * a[i];
      }
      gold = sqrt(gold);

      Kokkos::View<float*> d_a("d_a", n);
      auto h_a = Kokkos::create_mirror_view(d_a);
      for (int i = 0; i < n; i++) h_a(i) = a[i];
      Kokkos::deep_copy(d_a, h_a);
      Kokkos::fence();

      auto kstart = std::chrono::steady_clock::now();

      for (int j = 0; j < repeat; j++) {
        double sum = 0.0;
        Kokkos::parallel_reduce("nrm2", n,
          KOKKOS_LAMBDA(const int i, double& lsum) {
            double t = d_a(i);
            lsum += t * t;
          }, sum);
        Kokkos::fence();
        h_result[j] = (float)sqrt(sum);
      }

      auto kend = std::chrono::steady_clock::now();
      auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
      printf("#elements = %.2f M: average kokkos nrm2 execution time = %f (us), performance = %f (Gop/s)\n",
             n / (1024.f * 1024.f),
             (ktime * 1e-3f) / repeat,
             1.f * (2*n+1) * repeat / ktime);

      for (int j = 0; j < repeat; j++) {
        if (fabsf((float)gold - h_result[j]) > 1e-3f) {
          printf("FAIL at iteration %d: gold=%f actual=%f for %d elements\n",
                 j, (float)gold, h_result[j], n);
          ok = false;
          break;
        }
      }

      free(a);
    }

    free(h_result);
    if (ok) printf("PASS\n");
  }
  Kokkos::finalize();
  return 0;
}
