/*
 * Popcount (population count) benchmark.
 * Multiple implementations of counting set bits.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define m1  0x5555555555555555UL
#define m2  0x3333333333333333UL
#define m4  0x0f0f0f0f0f0f0f0fUL
#define h01 0x0101010101010101UL

// Reference implementation
int popcount_ref(unsigned long x) {
  int count;
  for (count = 0; x; count++) x &= x - 1;
  return count;
}

void checkResults(const unsigned long *d, const int *r, int length) {
  int error = 0;
  for (int i = 0; i < length; i++) {
    if (popcount_ref(d[i]) != r[i]) { error = 1; break; }
  }
  printf("%s\n", error ? "Fail" : "Success");
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  unsigned long *data   = (unsigned long*) malloc(length * sizeof(unsigned long));
  int           *result = (int*)           malloc(length * sizeof(int));

  srand(2);
  for (int i = 0; i < length; i++) {
    unsigned long t = (unsigned long)rand() << 32;
    data[i] = t | rand();
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<unsigned long*> d_data("d_data", length);
    Kokkos::View<int*>           d_result("d_result", length);

    auto h_data = Kokkos::create_mirror_view(d_data);
    for (int i = 0; i < length; i++) h_data(i) = data[i];
    Kokkos::deep_copy(d_data, h_data);

    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("popcount1", length, KOKKOS_LAMBDA(int i) {
        unsigned long x = d_data(i);
        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        x += x >> 8;
        x += x >> 16;
        x += x >> 32;
        d_result(i) = (int)(x & 0x7f);
      });
      Kokkos::fence();
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pc1): %f (us)\n", (time * 1e-3) / repeat);

    auto h_result = Kokkos::create_mirror_view(d_result);
    Kokkos::deep_copy(h_result, d_result);
    for (int i = 0; i < length; i++) result[i] = h_result(i);
    checkResults(data, result, length);

    // __builtin_popcountll equivalent: use standard popcount
    start = std::chrono::steady_clock::now();
    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("popcount2", length, KOKKOS_LAMBDA(int i) {
        unsigned long x = d_data(i);
        // Manual Hamming weight (popcount64)
        x = x - ((x >> 1) & m1);
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        d_result(i) = (int)((x * h01) >> 56);
      });
      Kokkos::fence();
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (pc2): %f (us)\n", (time * 1e-3) / repeat);

    Kokkos::deep_copy(h_result, d_result);
    for (int i = 0; i < length; i++) result[i] = h_result(i);
    checkResults(data, result, length);
  }
  Kokkos::finalize();

  free(data);
  free(result);
  return 0;
}
