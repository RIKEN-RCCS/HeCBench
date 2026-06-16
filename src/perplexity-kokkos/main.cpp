#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <Kokkos_Core.hpp>
#include "reference.cpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points> <perplexity> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int p      = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const int n_nbrs = 4 * p;
  const int max_iter = 100;
  const float tol    = 1e-8f;

  srand(123);
  std::vector<float> distance(n * n_nbrs);
  for (int i = 0; i < n * n_nbrs; i++)
    distance[i] = rand() / (float)RAND_MAX;

  std::vector<float> data(n * n_nbrs, 0.f);
  std::vector<float> h_data(n * n_nbrs, 0.f);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_distance("distance", n * n_nbrs);
    Kokkos::View<float*> d_data("data", n * n_nbrs);

    {
      auto md = Kokkos::create_mirror_view(d_distance);
      for (int i = 0; i < n * n_nbrs; i++) md[i] = distance[i];
      Kokkos::deep_copy(d_distance, md);
    }

    const float desired_entropy = logf((float)p);
    double time_total = 0.0;

    for (int r = 0; r < repeat; r++) {
      Kokkos::deep_copy(d_data, 0.f);

      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for("perplexity", n,
        KOKKOS_LAMBDA(int i) {
          float beta_min = -INFINITY, beta_max = INFINITY;
          float beta = 1.0f;
          const int ik = i * n_nbrs;

          for (int step = 0; step < max_iter; step++) {
            float sum_Pi = FLT_EPSILON;
            for (int j = 0; j < n_nbrs; j++) {
              d_data[ik+j] = expf(-d_distance[ik+j] * beta);
              sum_Pi += d_data[ik+j];
            }
            float sum_disti_Pi = 0.f;
            float div = 1.0f / sum_Pi;
            for (int j = 0; j < n_nbrs; j++) {
              d_data[ik+j] *= div;
              sum_disti_Pi += d_distance[ik+j] * d_data[ik+j];
            }
            float entropy      = logf(sum_Pi) + beta * sum_disti_Pi;
            float entropy_diff = entropy - desired_entropy;
            if (fabsf(entropy_diff) <= tol) break;
            if (entropy_diff > 0) {
              beta_min = beta;
              beta = isinf(beta_max) ? beta * 2.0f : (beta + beta_max) * 0.5f;
            } else {
              beta_max = beta;
              beta = isinf(beta_min) ? beta * 0.5f : (beta + beta_min) * 0.5f;
            }
          }
        });
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    printf("Average kernel execution time: %f (s)\n", (time_total * 1e-9f) / repeat);

    {
      auto md = Kokkos::create_mirror_view(d_data);
      Kokkos::deep_copy(md, d_data);
      for (int i = 0; i < n * n_nbrs; i++) data[i] = md[i];
    }
  }
  Kokkos::finalize();

  reference<int,float>(distance.data(), h_data.data(), (float)p, max_iter, tol, n, n_nbrs);

  bool ok = true;
  for (int i = 0; i < n * n_nbrs; i++) {
    if (fabsf(data[i] - h_data[i]) > 1e-3f) {
      printf("mismatch at %d: got %f expected %f\n", i, data[i], h_data[i]);
      ok = false; break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
