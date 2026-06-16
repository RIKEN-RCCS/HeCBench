#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>
#include "utils.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: ./%s <list size> <0 or 1> <repeat>\n", argv[0]);
    printf("0 and 1 indicate an ordered list and a random list, respectively\n");
    exit(-1);
  }

  int elems = atoi(argv[1]);
  int setRandomList = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  std::vector<int> next(elems);
  std::vector<int> rank(elems);
  std::vector<long> list(elems);
  std::vector<long> d_res(elems);
  std::vector<long> h_res(elems);

  if (setRandomList)
    random_list(next);
  else
    ordered_list(next);

  for (int i = 0; i < elems; i++) {
    rank[i] = next[i] == NIL ? 0 : 1;
  }

  for (int i = 0; i < elems; i++)
    list[i] = ((long)next[i] << 32) | rank[i];

  // Number of pointer-doubling iterations needed: O(log2(elems))
  int num_iters = (int)ceil(log2((double)elems)) + 1;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<long *> d_plist("plist", elems);
    Kokkos::View<long *> d_tmp("plist_tmp", elems);

    double time = 0.0;

    for (int rep = 0; rep <= repeat; rep++) {
      // Reinitialize plist to original state
      auto h_init = Kokkos::create_mirror_view(d_plist);
      for (int i = 0; i < elems; i++) h_init(i) = list[i];
      Kokkos::deep_copy(d_plist, h_init);

      auto start = std::chrono::steady_clock::now();

      // Double-buffered pointer jumping: avoids the need for intra-kernel
      // barriers by separating read and write phases across kernel launches.
      for (int iter = 0; iter < num_iters; iter++) {
        Kokkos::parallel_for(
            "wyllie", elems, KOKKOS_LAMBDA(const int index) {
              long node = d_plist(index);
              long next_idx = node >> 32;
              if (next_idx != (long)NIL) {
                long next_node = d_plist(next_idx);
                long next_next = next_node >> 32;
                if (next_next != (long)NIL) {
                  long temp = (node & MASK) + (next_node & MASK);
                  temp |= (next_next << 32);
                  d_tmp(index) = temp;
                } else {
                  d_tmp(index) = node;
                }
              } else {
                d_tmp(index) = node;
              }
            });
        Kokkos::fence();
        Kokkos::deep_copy(d_plist, d_tmp);
      }

      auto end = std::chrono::steady_clock::now();
      if (rep > 0)
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();
    }

    printf("Average kernel execution time: %f (ms)\n",
           (time * 1e-6f) / repeat);

    auto h_result = Kokkos::create_mirror_view(d_plist);
    Kokkos::deep_copy(h_result, d_plist);
    for (int i = 0; i < elems; i++) d_res[i] = h_result(i) & MASK;
  }
  Kokkos::finalize();

  // Verify: compute distance from the *end* of the list
  h_res[0] = elems - 1;
  int i = 0;
  for (int r = 1; r < elems; r++) {
    h_res[next[i]] = elems - 1 - r;
    i = next[i];
  }

#ifdef DEBUG
  printf("Ranks:\n");
  for (i = 0; i < elems; i++) {
    printf("%d: %ld %ld\n", i, h_res[i], d_res[i]);
  }
#endif

  printf("%s\n", (h_res == d_res) ? "PASS" : "FAIL");
  return 0;
}
