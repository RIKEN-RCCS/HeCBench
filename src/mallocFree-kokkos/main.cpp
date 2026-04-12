/*
   Memory allocation benchmark using Kokkos::View.
   Ported from mallocFree-sycl.
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define NUM_SIZE 19
#define NUM_ITER 500

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <total_global_mem_in_bytes>\n", argv[0]);
    return 1;
  }

  const size_t totalGlobalMem = atol(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    size_t size[NUM_SIZE] = {0};
    int num = NUM_SIZE;
    for (int i = 0; i < num; i++) {
      size[i] = 1 << (i + 6);  // 64 bytes to 16MB
      if ((NUM_ITER + 1) * size[i] > totalGlobalMem) {
        num = i;
        break;
      }
    }

    printf("==== Evaluate Kokkos::View device allocation and free ====\n");
    for (int i = 0; i < num; i++) {
      size_t nelems = size[i] / sizeof(int);
      auto start = std::chrono::steady_clock::now();
      for (int j = 0; j < NUM_ITER; j++) {
        Kokkos::View<int*> v("v", nelems);
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("View allocation+deallocation(%zu) takes %lf us\n",
             size[i], time * 1e-3 / NUM_ITER);
    }

    printf("\n==== Evaluate Kokkos::View host mirror allocation and free ====\n");
    for (int i = 0; i < num; i++) {
      size_t nelems = size[i] / sizeof(int);
      auto start = std::chrono::steady_clock::now();
      for (int j = 0; j < NUM_ITER; j++) {
        Kokkos::View<int*, Kokkos::HostSpace> v("v_host", nelems);
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("HostSpace allocation+deallocation(%zu) takes %lf us\n",
             size[i], time * 1e-3 / NUM_ITER);
    }
  }
  Kokkos::finalize();
  return 0;
}
