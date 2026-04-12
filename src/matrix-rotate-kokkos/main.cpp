#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <Kokkos_Core.hpp>

void rotate_matrix_parallel(Kokkos::View<float *> d_matrix, const int n,
                             const int repeat) {
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "rotate_matrix", n / 2, KOKKOS_LAMBDA(const int layer) {
          int first = layer;
          int last = n - 1 - layer;
          for (int i = first; i < last; ++i) {
            int offset = i - first;

            float top = d_matrix(first * n + i);
            d_matrix(first * n + i) = d_matrix((last - offset) * n + first);
            d_matrix((last - offset) * n + first) =
                d_matrix(last * n + (last - offset));
            d_matrix(last * n + (last - offset)) = d_matrix(i * n + last);
            d_matrix(i * n + last) = top;
          }
        });
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
}

void rotate_matrix_serial(float *matrix, const int n) {
  for (int layer = 0; layer < n / 2; ++layer) {
    int first = layer;
    int last = n - 1 - layer;
    for (int i = first; i < last; ++i) {
      int offset = i - first;
      float top = matrix[first * n + i];
      matrix[first * n + i] = matrix[(last - offset) * n + first];
      matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
      matrix[last * n + (last - offset)] = matrix[i * n + last];
      matrix[i * n + last] = top;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  float *serial_res = (float *)aligned_alloc(1024, n * n * sizeof(float));
  float *parallel_res = (float *)aligned_alloc(1024, n * n * sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      serial_res[i * n + j] = parallel_res[i * n + j] = i * n + j;

  for (int i = 0; i < repeat; i++) rotate_matrix_serial(serial_res, n);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float *> d_matrix("matrix", n * n);
    auto h_matrix = Kokkos::create_mirror_view(d_matrix);
    for (int i = 0; i < n * n; i++) h_matrix(i) = parallel_res[i];
    Kokkos::deep_copy(d_matrix, h_matrix);

    rotate_matrix_parallel(d_matrix, n, repeat);

    Kokkos::deep_copy(h_matrix, d_matrix);
    for (int i = 0; i < n * n; i++) parallel_res[i] = h_matrix(i);
  }
  Kokkos::finalize();

  bool ok = true;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (serial_res[i * n + j] != parallel_res[i * n + j]) {
        ok = false;
        break;
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(serial_res);
  free(parallel_res);
  return 0;
}
