// Kokkos port of spaxpby-cuda
// Sparse vector + dense vector: Y[i] = beta*Y[i] + alpha*X[sparse]

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static void init_matrix(float *matrix, size_t num_rows, size_t num_cols, size_t nnz) {
  size_t n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (size_t i = 0; i < n; i++) d[i] = (float)i;
  for (size_t i = n; i > 0; i--) {
    size_t a = i - 1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (size_t i = 0; i < num_rows; i++)
    for (size_t j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= (float)nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("The function computes the sum of a sparse vector and a dense vector\n");
    printf("Y[i] = beta*Y[i] for all i, then Y[X_indices[k]] += alpha*X_values[k]\n");
    printf("Usage %s <M> <N> <nnz> <repeat>\n", argv[0]);
    return 1;
  }

  size_t m = atol(argv[1]);
  size_t n = atol(argv[2]);
  size_t nnz = atol(argv[3]);
  int repeat = atoi(argv[4]);

  const size_t size = m * n;
  float *hA       = (float*)malloc(size * sizeof(float));
  float *hB       = (float*)malloc(size * sizeof(float));
  float *hY       = (float*)malloc(size * sizeof(float));
  float *hA_values  = (float*)malloc(nnz * sizeof(float));
  size_t *hA_indices = (size_t*)malloc(nnz * sizeof(size_t));

  printf("Initializing input matrices..\n");
  init_matrix(hA, m, n, nnz);

  size_t k = 0;
  for (size_t i = 0; i < size; i++) {
    if (hA[i] != 0.0f) { hA_indices[k] = i; hA_values[k] = hA[i]; k++; }
  }
  init_matrix(hB, m, n, size);
  printf("Done\n");

  const float alpha = 1.0f, beta = 1.0f;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<size_t*> dX_indices("dX_indices", nnz);
    Kokkos::View<float*>  dX_values("dX_values", nnz);
    Kokkos::View<float*>  dY("dY", size);

    {
      auto h_idx = Kokkos::create_mirror_view(dX_indices);
      auto h_xv  = Kokkos::create_mirror_view(dX_values);
      auto h_y   = Kokkos::create_mirror_view(dY);
      for (size_t i = 0; i < nnz; i++) { h_idx(i) = hA_indices[i]; h_xv(i) = hA_values[i]; }
      for (size_t i = 0; i < size; i++) h_y(i) = hB[i];
      Kokkos::deep_copy(dX_indices, h_idx);
      Kokkos::deep_copy(dX_values, h_xv);
      Kokkos::deep_copy(dY, h_y);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      // Y[i] *= beta
      Kokkos::parallel_for("axpby_scale", size,
        KOKKOS_LAMBDA(const size_t j) { dY(j) *= beta; });
      // Y[idx] += alpha * X
      Kokkos::parallel_for("axpby_scatter", nnz,
        KOKKOS_LAMBDA(const size_t j) {
          Kokkos::atomic_add(&dY(dX_indices(j)), alpha * dX_values(j));
        });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SPAXPBY : %f (us)\n", (time * 1e-3f) / repeat);

    // Copy result back
    auto h_y = Kokkos::create_mirror_view(dY);
    Kokkos::deep_copy(h_y, dY);
    for (size_t i = 0; i < size; i++) hY[i] = h_y(i);
  }
  Kokkos::finalize();

  // Reference computation
  printf("Computing the reference results..\n");
  for (int iter = 0; iter < repeat; iter++) {
    for (size_t i = 0; i < size; i++)
      hB[i] = alpha * hA[i] + beta * hB[i];
  }
  printf("Done\n");

  int correct = 1;
  for (size_t i = 0; i < size; i++) {
    if (fabsf(hY[i] - hB[i]) > 1e-2f) { correct = 0; break; }
  }
  printf("%s\n", correct ? "axpby_example test PASSED" : "axpby_example test FAILED: wrong result");

  free(hA); free(hB); free(hY); free(hA_values); free(hA_indices);
  return 0;
}
