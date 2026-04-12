#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "reference.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  float *subject = (float *)malloc(sizeof(float) * N);
  float *lower_bound = (float *)malloc(sizeof(float) * N);
  float *upper_bound = (float *)malloc(sizeof(float) * N);
  float *lb = (float *)malloc(sizeof(float) * (N - M + 1));
  float *lb_h = (float *)malloc(sizeof(float) * (N - M + 1));
  float *avgs = (float *)malloc(sizeof(float) * (N - M + 1));
  float *stds = (float *)malloc(sizeof(float) * (N - M + 1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i)
    avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i)
    stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i)
    upper_bound[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i)
    lower_bound[i] = (float)rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float *> d_subject("subject", N);
    Kokkos::View<float *> d_avgs("avgs", N - M + 1);
    Kokkos::View<float *> d_stds("stds", N - M + 1);
    Kokkos::View<float *> d_lower("lower_bound", M);
    Kokkos::View<float *> d_upper("upper_bound", M);
    Kokkos::View<float *> d_lb("lb", N - M + 1);

    {
      auto hv = Kokkos::create_mirror_view(d_subject);
      for (int i = 0; i < N; i++) hv(i) = subject[i];
      Kokkos::deep_copy(d_subject, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_avgs);
      for (int i = 0; i < N - M + 1; i++) hv(i) = avgs[i];
      Kokkos::deep_copy(d_avgs, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_stds);
      for (int i = 0; i < N - M + 1; i++) hv(i) = stds[i];
      Kokkos::deep_copy(d_stds, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_lower);
      for (int i = 0; i < M; i++) hv(i) = lower_bound[i];
      Kokkos::deep_copy(d_lower, hv);
    }
    {
      auto hv = Kokkos::create_mirror_view(d_upper);
      for (int i = 0; i < M; i++) hv(i) = upper_bound[i];
      Kokkos::deep_copy(d_upper, hv);
    }

    const int numElements = N - M + 1;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(
          "keogh", numElements, KOKKOS_LAMBDA(const int idx) {
            float residues = 0.f;
            float avg = d_avgs(idx);
            float std = d_stds(idx);
            for (int i = 0; i < M; ++i) {
              float value = (d_subject(idx + i) - avg) / std;
              float lower = value - d_lower(i);
              float upper = value - d_upper(i);
              residues +=
                  upper * upper * (upper > 0) + lower * lower * (lower < 0);
            }
            d_lb(idx) = residues;
          });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("Average kernel execution time: %f (s)\n",
           (time * 1e-9f) / repeat);

    auto h_lb = Kokkos::create_mirror_view(d_lb);
    Kokkos::deep_copy(h_lb, d_lb);
    for (int i = 0; i < numElements; i++) lb[i] = h_lb(i);

    reference(subject, avgs, stds, lb_h, lower_bound, upper_bound, M, N);
    bool ok = true;
    for (int i = 0; i < N - M + 1; i++) {
      if (fabs(lb[i] - lb_h[i]) > 1e-3) {
        printf("%d %f %f\n", i, lb[i], lb_h[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower_bound);
  free(upper_bound);
  return 0;
}
