#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>

inline size_t shrRoundUp(int group_size, size_t global_size) {
  size_t r = global_size % group_size;
  if (r == 0) return global_size;
  else return global_size + group_size - r;
}

template <typename T>
void dot(const size_t iNumElements, const int iNumIterations) {
  int szLocalWorkSize = 256;
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;

  // Allocate and initialise host arrays
  T *srcA = (T*)malloc(src_size * sizeof(T));
  T *srcB = (T*)malloc(src_size * sizeof(T));

  for (size_t i = 0; i < iNumElements; i++) {
    srcA[i] = (i < iNumElements / 2) ? (T)-1 : (T)1;
    srcB[i] = (T)-1;
  }
  for (size_t i = iNumElements; i < src_size; i++)
    srcA[i] = srcB[i] = (T)0;

  // Upload to device
  Kokkos::View<T*> d_srcA("srcA", src_size);
  Kokkos::View<T*> d_srcB("srcB", src_size);
  {
    auto h_srcA = Kokkos::create_mirror_view(d_srcA);
    auto h_srcB = Kokkos::create_mirror_view(d_srcB);
    for (size_t i = 0; i < src_size; i++) { h_srcA(i) = srcA[i]; h_srcB(i) = srcB[i]; }
    Kokkos::deep_copy(d_srcA, h_srcA);
    Kokkos::deep_copy(d_srcB, h_srcB);
  }

  T dst = (T)0;

  auto start = std::chrono::steady_clock::now();

  for (int it = 0; it < iNumIterations; it++) {
    T local_dst = (T)0;
    // Each iteration processes src_size/4 groups of 4 elements
    const size_t n_groups = src_size / 4;
    Kokkos::parallel_reduce("dot_product", n_groups,
      KOKKOS_LAMBDA(size_t iGID, T &val) {
        size_t iInOffset = iGID * 4;
        val += d_srcA(iInOffset  ) * d_srcB(iInOffset  )
             + d_srcA(iInOffset+1) * d_srcB(iInOffset+1)
             + d_srcA(iInOffset+2) * d_srcB(iInOffset+2)
             + d_srcA(iInOffset+3) * d_srcB(iInOffset+3);
      }, local_dst);
    Kokkos::fence();
    dst = local_dst;
  }

  auto end = std::chrono::steady_clock::now();
  auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("Average kernel execution time %f (ms)\n", (time_ns * 1e-6f) / iNumIterations);
  printf("%s\n\n", dst == (T)0 ? "PASS" : "FAIL");

  free(srcA);
  free(srcB);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements   = atol(argv[1]);
  const int    iNumIterations = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    dot<float> (iNumElements, iNumIterations);
    dot<double>(iNumElements, iNumIterations);
  }
  Kokkos::finalize();
  return EXIT_SUCCESS;
}
