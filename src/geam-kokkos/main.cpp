// Kokkos port of geam-cuda (matrix transpose via cublasSgeam/cublasDgeam)
// cuBLAS GEAM replaced by a naive parallel_for transpose kernel.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <Kokkos_Core.hpp>

template <typename T>
static void run_transpose(int nrow, int ncol, int repeat)
{
  const size_t size = (size_t)nrow * ncol;

  T* matrix  = (T*)malloc(size * sizeof(T));
  T* matrixT = (T*)malloc(size * sizeof(T));  // host reference

  for (size_t i = 0; i < size; i++) matrix[i] = (T)(rand() % 13);

  // CPU reference
  for (int i = 0; i < nrow; i++)
    for (int j = 0; j < ncol; j++)
      matrixT[(size_t)j * nrow + i] = matrix[(size_t)i * ncol + j];

  Kokkos::View<T*> d_matrix  ("matrix",  size);
  Kokkos::View<T*> d_matrixT ("matrixT", size);

  {
    auto h = Kokkos::create_mirror_view(d_matrix);
    for (size_t i = 0; i < size; i++) h(i) = matrix[i];
    Kokkos::deep_copy(d_matrix, h);
  }

  const int warmup = 4;
  double time_ns = 0.0;

  for (int i = 0; i < warmup + repeat; i++) {
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    Kokkos::parallel_for(size, KOKKOS_LAMBDA(size_t idx) {
      int r = (int)(idx / ncol);
      int c = (int)(idx % ncol);
      d_matrixT[(size_t)c * nrow + r] = d_matrix[idx];
    });

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    if (i >= warmup)
      time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Device: average matrix transpose time = %f (ms)\n",
         time_ns * 1e-6 / repeat);

  auto h = Kokkos::create_mirror_view(d_matrixT);
  Kokkos::deep_copy(h, d_matrixT);

  int error = memcmp(h.data(), matrixT, size * sizeof(T));
  printf("%s\n", error ? "FAIL" : "PASS");

  free(matrix);
  free(matrixT);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage %s <matrix row> <matrix col> <repeat>\n", argv[0]);
    return 1;
  }

  int nrow   = atoi(argv[1]);
  int ncol   = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  if (nrow <= 0 || ncol <= 0 || repeat < 0) {
    printf("Error: invalid inputs\n"); return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    printf("---- FP32 transpose (%d x %d) ----\n", nrow, ncol);
    run_transpose<float>(nrow, ncol, repeat);

    printf("---- FP64 transpose (%d x %d) ----\n", nrow, ncol);
    run_transpose<double>(nrow, ncol, repeat);
  }
  Kokkos::finalize();
  return 0;
}
