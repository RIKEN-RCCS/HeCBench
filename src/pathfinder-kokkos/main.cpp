// Kokkos port of PathFinder benchmark
// Original OMP target source: src/pathfinder-omp/main.cpp
// PathFinder uses dynamic programming to find minimum weight path from bottom to top.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define M_SEED   9
#define HALO     1
#define IN_RANGE(x, min, max)  ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

double get_time() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  int rows, cols, pyramid_height;

  if (argc == 4) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  } else {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    return 0;
  }

  int *data = new int[rows * cols];
  srand(M_SEED);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      data[i * cols + j] = rand() % 10;

  const int size = rows * cols;
  Kokkos::initialize(argc, argv);
  {
    // Copy wall (rows 1..rows-1) to device
    Kokkos::View<int*> d_wall("d_wall", size - cols);
    Kokkos::View<int*> d_src("d_src", cols);
    Kokkos::View<int*> d_result("d_result", cols);

    auto h_wall = Kokkos::create_mirror_view(d_wall);
    auto h_src  = Kokkos::create_mirror_view(d_src);

    for (int j = 0; j < cols; j++) h_src(j) = data[j];
    for (int i = 0; i < size - cols; i++) h_wall(i) = data[cols + i];
    Kokkos::deep_copy(d_wall, h_wall);
    Kokkos::deep_copy(d_src, h_src);

    double kstart = 0.0;
    double offload_start = get_time();

    for (int t = 0; t < rows - 1; t += pyramid_height) {
      if (t == pyramid_height) kstart = get_time();
      int iteration = MIN(pyramid_height, rows - t - 1);

      // For each pyramid level, update src→result
      // On CPU, we process each iteration sequentially (dependency between levels)
      // Copy src to result for first iteration
      Kokkos::deep_copy(d_result, d_src);

      for (int iter = 0; iter < iteration; iter++) {
        const int wall_row = t + iter;
        // Read from d_result into d_src after update
        Kokkos::View<int*> d_prev = (iter % 2 == 0) ? d_result : d_src;
        Kokkos::View<int*> d_next = (iter % 2 == 0) ? d_src    : d_result;

        Kokkos::parallel_for("pathfinder_step",
          Kokkos::RangePolicy<>(0, cols),
          KOKKOS_LAMBDA(const int j) {
            int prev_left  = (j > 0)       ? d_prev(j - 1) : d_prev(j);
            int prev_mid   = d_prev(j);
            int prev_right = (j < cols - 1) ? d_prev(j + 1) : d_prev(j);
            int shortest = MIN(prev_left, MIN(prev_mid, prev_right));
            d_next(j) = shortest + d_wall(wall_row * cols + j);
          });
        Kokkos::fence();
      }

      // result is in d_src if odd iterations, d_result if even
      if (iteration % 2 == 1) Kokkos::deep_copy(d_result, d_src);
      Kokkos::deep_copy(d_src, d_result);
    }

    double kend = get_time();
    printf("Total kernel execution time: %lf (s)\n", kend - kstart);

    double offload_end = get_time();
    printf("Device offloading time = %lf(s)\n", offload_end - offload_start);
  }
  Kokkos::finalize();

  delete[] data;
  return EXIT_SUCCESS;
}
