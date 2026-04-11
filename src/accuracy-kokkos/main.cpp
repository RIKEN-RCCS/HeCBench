#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>
#include "reference.h"

#define NUM_THREADS 256

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 5) {
      printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int nrows = atoi(argv[1]);
    const int ndims = atoi(argv[2]);
    const int top_k = atoi(argv[3]);
    const int repeat = atoi(argv[4]);

    const int data_size = nrows * ndims;

    int *label = (int*) malloc (nrows * sizeof(int));

    srand(123);
    for (int i = 0; i < nrows; i++)
      label[i] = rand() % ndims;

    float *data = (float*) malloc (data_size * sizeof(float));

    std::default_random_engine g (123);
    std::uniform_real_distribution<float> distr (0.f, 1.f);
    for (int i = 0; i < data_size; i++) {
      data[i] = distr(g);
    }

    int count_ref = reference(nrows, ndims, top_k, data, label);

    // Create device views
    Kokkos::View<int*> d_label("label", nrows);
    Kokkos::View<float*> d_data("data", data_size);
    Kokkos::View<int[1]> d_count("count");

    // Copy data to device
    auto h_label = Kokkos::View<int*, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(label, nrows);
    auto h_data = Kokkos::View<float*, Kokkos::HostSpace,
                    Kokkos::MemoryUnmanaged>(data, data_size);
    Kokkos::deep_copy(d_label, h_label);
    Kokkos::deep_copy(d_data, h_data);

    for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

      printf("Grid size is %d\n", ngrid);

      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++) {
        Kokkos::deep_copy(d_count, 0);

        Kokkos::parallel_for(
          Kokkos::TeamPolicy<>(ngrid, NUM_THREADS),
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int nrows_l = nrows;
            const int ndims_l = ndims;
            const int top_k_l = top_k;
            for (int row = team.league_rank(); row < nrows_l; row += team.league_size()) {
              const int label_data = d_label(row);
              const float label_pred = d_data(row * ndims_l + label_data);
              int ngt = 0;
              Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, ndims_l),
                [&](const int col, int& local_ngt) {
                  const float pred = d_data(row * ndims_l + col);
                  if (pred > label_pred || (pred == label_pred && col <= label_data)) {
                    ++local_ngt;
                  }
                }, ngt);
              Kokkos::single(Kokkos::PerTeam(team), [&]() {
                if (ngt <= top_k_l) {
                  Kokkos::atomic_add(&d_count(0), 1);
                }
              });
            }
          });
      }

      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

      auto h_count = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_count);
      bool ok = (h_count(0) == count_ref);
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    free(label);
    free(data);
  }
  Kokkos::finalize();

  return 0;
}
