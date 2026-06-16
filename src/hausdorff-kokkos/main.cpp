#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "reference.h"

inline float hd(const float2 ap, const float2 bp) {
  return (ap.x - bp.x) * (ap.x - bp.x) + (ap.y - bp.y) * (ap.y - bp.y);
}

void computeDistance(Kokkos::View<float2 *> d_Apoints,
                     Kokkos::View<float2 *> d_Bpoints, float &distance,
                     const int numA, const int numB) {
  float result = -1.f;
  Kokkos::parallel_reduce(
      "computeDistance", numA,
      KOKKOS_LAMBDA(const int i, float &lmax) {
        float d = FLT_MAX;
        float2 p = d_Apoints(i);
        for (int j = 0; j < numB; j++) {
          float dx = p.x - d_Bpoints(j).x;
          float dy = p.y - d_Bpoints(j).y;
          float tt = dx * dx + dy * dy;
          if (tt < d) d = tt;
        }
        if (d > lmax) lmax = d;
      },
      Kokkos::Max<float>(result));
  distance = result;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points in space A>", argv[0]);
    printf(" <number of points in space B> <repeat>\n");
    return 1;
  }
  const int num_Apoints = atoi(argv[1]);
  const int num_Bpoints = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  float2 *h_Apoints = (float2 *)malloc(sizeof(float2) * num_Apoints);
  float2 *h_Bpoints = (float2 *)malloc(sizeof(float2) * num_Bpoints);

  srand(123);
  for (int i = 0; i < num_Apoints; i++) {
    h_Apoints[i].x = (float)rand() / (float)RAND_MAX;
    h_Apoints[i].y = (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < num_Bpoints; i++) {
    h_Bpoints[i].x = (float)rand() / (float)RAND_MAX;
    h_Bpoints[i].y = (float)rand() / (float)RAND_MAX;
  }

  float h_distance[2] = {-1.f, -1.f};

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float2 *> d_Apoints("Apoints", num_Apoints);
    Kokkos::View<float2 *> d_Bpoints("Bpoints", num_Bpoints);

    auto h_A = Kokkos::create_mirror_view(d_Apoints);
    auto h_B = Kokkos::create_mirror_view(d_Bpoints);
    for (int i = 0; i < num_Apoints; i++) h_A(i) = h_Apoints[i];
    for (int i = 0; i < num_Bpoints; i++) h_B(i) = h_Bpoints[i];
    Kokkos::deep_copy(d_Apoints, h_A);
    Kokkos::deep_copy(d_Bpoints, h_B);

    double time = 0.0;

    for (int i = 0; i < repeat; i++) {
      h_distance[0] = -1.f;
      h_distance[1] = -1.f;

      auto start = std::chrono::steady_clock::now();

      computeDistance(d_Apoints, d_Bpoints, h_distance[0], num_Apoints,
                      num_Bpoints);
      computeDistance(d_Bpoints, d_Apoints, h_distance[1], num_Bpoints,
                      num_Apoints);
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count();
    }

    printf("Average execution time of kernels: %f (ms)\n",
           (time * 1e-6f) / repeat);

    printf("Verifying the result may take a while..\n");
    float r_distance =
        hausdorff_distance(h_Apoints, h_Bpoints, num_Apoints, num_Bpoints);
    float t_distance = std::max(h_distance[0], h_distance[1]);

    bool error = (fabsf(t_distance - r_distance)) > 1e-3f;
    printf("%s\n", error ? "FAIL" : "PASS");
  }
  Kokkos::finalize();

  free(h_Apoints);
  free(h_Bpoints);
  return 0;
}
