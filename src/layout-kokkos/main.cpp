/*
 * AoS vs SoA data layout benchmark.
 * Ported to Kokkos from the OMP target version.
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define TREE_NUM 4096
#define TREE_SIZE 4096

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]);
  const int treeSize   = TREE_SIZE;
  const int treeNumber = TREE_NUM;

  if (iterations < 1) { std::cout << "Iterations cannot be 0 or negative. Exiting..\n"; return -1; }

  const int elements = treeSize * treeNumber;

  int *data      = (int*) malloc(elements   * sizeof(int));
  int *output    = (int*) malloc(treeNumber * sizeof(int));
  int *reference = (int*) malloc(treeNumber * sizeof(int));

  memset(reference, 0, treeNumber * sizeof(int));
  for (int i = 0; i < treeNumber; i++)
    for (int j = 0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_data("d_data", elements);
    Kokkos::View<int*> d_output("d_output", treeNumber);
    auto h_data = Kokkos::create_mirror_view(d_data);

    // AoS test
    for (int i = 0; i < treeNumber; i++)
      for (int j = 0; j < treeSize; j++)
        h_data(j + i * treeSize) = j + i * treeSize;
    Kokkos::deep_copy(d_data, h_data);

    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; n++) {
      Kokkos::parallel_for("layout_aos", treeNumber, KOKKOS_LAMBDA(int gid) {
        int res = 0;
        for (int i = 0; i < treeSize; i++)
          res += d_data(i + gid * treeSize);  // AoS: trees[gid].apples[i]
        d_output(gid) = res;
      });
      Kokkos::fence();
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time (AoS): "
              << (time * 1e-3f) / iterations << " (us)\n";

    auto h_output = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_output, d_output);
    for (int i = 0; i < treeNumber; i++) output[i] = h_output(i);

    bool fail = false;
    for (int i = 0; i < treeNumber; i++) {
      if (output[i] != reference[i]) { fail = true; break; }
    }
    std::cout << (fail ? "FAIL" : "PASS") << "\n";

    // SoA test
    for (int i = 0; i < treeNumber; i++)
      for (int j = 0; j < treeSize; j++)
        h_data(i + j * treeNumber) = j + i * treeSize;  // applesOnTrees[j].trees[i] = j+i*treeSize
    Kokkos::deep_copy(d_data, h_data);

    start = std::chrono::steady_clock::now();
    for (int n = 0; n < iterations; n++) {
      Kokkos::parallel_for("layout_soa", treeNumber, KOKKOS_LAMBDA(int gid) {
        int res = 0;
        for (int i = 0; i < treeSize; i++)
          res += d_data(gid + i * treeNumber);  // SoA: applesOnTrees[i].trees[gid]
        d_output(gid) = res;
      });
      Kokkos::fence();
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time (SoA): "
              << (time * 1e-3f) / iterations << " (us)\n";

    Kokkos::deep_copy(h_output, d_output);
    for (int i = 0; i < treeNumber; i++) output[i] = h_output(i);

    fail = false;
    for (int i = 0; i < treeNumber; i++) {
      if (output[i] != reference[i]) { fail = true; break; }
    }
    std::cout << (fail ? "FAIL" : "PASS") << "\n";
  }
  Kokkos::finalize();

  free(data);
  free(output);
  free(reference);
  return 0;
}
