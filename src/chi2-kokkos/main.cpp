#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>
#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 7) {
    printf("Usage: %s <rows> <cols> <cases> <controls> <threads> <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int rows     = atoi(argv[1]);
  unsigned int cols     = atoi(argv[2]);
  int ncases            = atoi(argv[3]);
  int ncontrols         = atoi(argv[4]);
  int nthreads          = atoi(argv[5]);
  int repeat            = atoi(argv[6]);

  printf("Individuals=%d SNPs=%d cases=%d controls=%d nthreads=%d\n",
         rows, cols, ncases, ncontrols, nthreads);

  size_t size = (size_t)rows * (size_t)cols;
  printf("Size of the data = %lu\n", size);

  unsigned char* dataT      = (unsigned char*)malloc(size);
  float*         h_results  = (float*)malloc(cols * sizeof(float));
  float*         cpu_results = (float*)malloc(cols * sizeof(float));

  if (!dataT || !h_results || !cpu_results) {
    printf("ERROR: Memory for data not allocated.\n");
    free(dataT); free(h_results); free(cpu_results);
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> distrib(0, 2);
  for (size_t i = 0; i < size; i++) dataT[i] = distrib(gen) + '0';

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<unsigned char*> d_data("data", size);
    Kokkos::View<float*>         d_results("results", cols);

    {
      auto h_data = Kokkos::create_mirror_view(d_data);
      for (size_t i = 0; i < size; i++) h_data[i] = dataT[i];
      Kokkos::deep_copy(d_data, h_data);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("chi2", (int)cols,
        KOKKOS_LAMBDA(int i) {
          unsigned char y;
          int m, n;
          unsigned int p = 0;
          int cases[3]    = {1, 1, 1};
          int controls[3] = {1, 1, 1};
          int tot_cases = 1, tot_controls = 1, total = 1;
          float chisquare = 0.0f;
          float exp_[3], Conexpected[3], Cexpected[3];
          float numerator1, numerator2;

          for (m = 0; m < ncases; m++) {
            y = d_data[(size_t)m * (size_t)cols + i];
            if      (y == '0') cases[0]++;
            else if (y == '1') cases[1]++;
            else if (y == '2') cases[2]++;
          }
          for (n = ncases; n < ncases + ncontrols; n++) {
            y = d_data[(size_t)n * (size_t)cols + i];
            if      (y == '0') controls[0]++;
            else if (y == '1') controls[1]++;
            else if (y == '2') controls[2]++;
          }
          for (p = 0; p < 3; p++) {
            tot_cases    += cases[p];
            tot_controls += controls[p];
          }
          total = tot_cases + tot_controls;
          for (p = 0; p < 3; p++) {
            exp_[p]         = (float)cases[p] + controls[p];
            Cexpected[p]    = tot_cases    * exp_[p] / total;
            Conexpected[p]  = tot_controls * exp_[p] / total;
            numerator1      = (float)cases[p]    - Cexpected[p];
            numerator2      = (float)controls[p] - Conexpected[p];
            chisquare      += numerator1*numerator1/Cexpected[p]
                            + numerator2*numerator2/Conexpected[p];
          }
          d_results[i] = chisquare;
        });
      Kokkos::fence();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time = %f (s)\n", time * 1e-9f / repeat);

    {
      auto mirror = Kokkos::create_mirror_view(d_results);
      Kokkos::deep_copy(mirror, d_results);
      for (unsigned int i = 0; i < cols; i++) h_results[i] = mirror[i];
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_kernel(rows, cols, ncases, ncontrols, dataT, cpu_results);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();
    printf("Host execution time = %f (s)\n", cpu_time * 1e-9f);

    int error = 0;
    for (unsigned int k = 0; k < cols; k++)
      if (fabs(cpu_results[k] - h_results[k]) > 1e-4) error++;
    printf("%s\n", error ? "FAIL" : "PASS");
  }
  Kokkos::finalize();

  free(dataT); free(h_results); free(cpu_results);
  return 0;
}
