#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <Kokkos_Core.hpp>

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems  = atoi(argv[1]);
  const int block_size = atoi(argv[2]);  // unused in Kokkos version but kept for API compat
  const int repeat     = atoi(argv[3]);
  (void)block_size;

  std::vector<int> input(num_elems);
  for (int i = 0; i < num_elems; i++)
    input[i] = i - num_elems / 2;

  std::mt19937 g;
  g.seed(19937);
  std::shuffle(input.begin(), input.end(), g);

  std::vector<int> output(num_elems, 0);
  int nres = 0;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_input("input", num_elems);
    Kokkos::View<int*> d_output("output", num_elems);

    auto h_input = Kokkos::create_mirror_view(d_input);
    for (int i = 0; i < num_elems; i++) h_input(i) = input[i];
    Kokkos::deep_copy(d_input, h_input);

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      int count = 0;
      Kokkos::parallel_scan("filter", num_elems,
        KOKKOS_LAMBDA(int i, int& partial, bool is_final) {
          bool pred = (d_input(i) > 0);
          if (is_final && pred)
            d_output(partial) = d_input(i);
          partial += (pred ? 1 : 0);
        }, count);
      Kokkos::fence();
      nres = count;
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %lf (ms)\n", (time * 1e-6) / repeat);

    auto h_output = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_output, d_output);
    for (int i = 0; i < num_elems; i++) output[i] = h_output(i);
  }
  Kokkos::finalize();

  // Generate host reference output
  std::vector<int> h_output(num_elems);
  int h_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0)
      h_output[h_flt_count++] = input[i];
  }

  // Verify (sort both before comparing since ordering may differ)
  std::sort(h_output.begin(), h_output.begin() + h_flt_count);
  std::sort(output.begin(), output.begin() + nres);

  bool equal = (h_flt_count == nres) &&
               std::equal(h_output.begin(), h_output.begin() + h_flt_count, output.begin());

  printf("\nFilter using shared memory %s \n", equal ? "PASS" : "FAIL");
  return 0;
}
