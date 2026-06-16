/*
 * Dense embedding kernel.
 * Ported to Kokkos from the OMP target version.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

template <typename T>
void reference(const T* input, const T* dense, T* output,
               int embedding_dim, int batch_size, const int* offset)
{
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] =
            input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows       = atoi(argv[1]);
  const int batch_size  = atoi(argv[2]);
  const int repeat      = atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  printf("Number of rows in the embedding table: %d\n", nrows);
  printf("Batch size: %d\n", batch_size);

  const int embed_dims[] = {768, 2048, 12288};

  Kokkos::initialize(argc, argv);
  {
    for (size_t n = 0; n < sizeof(embed_dims) / sizeof(int); n++) {
      int ncols = embed_dims[n];
      printf("\nEmbedding dimension: %d\n", ncols);

      int input_size = nrows * ncols;
      int dense_size = batch_size * ncols;

      float *input       = (float*) malloc(input_size * sizeof(float));
      float *dense       = (float*) malloc(dense_size * sizeof(float));
      float *output_k1   = (float*) malloc(input_size * sizeof(float));
      float *output_ref  = (float*) malloc(input_size * sizeof(float));
      int   *offset      = (int*)   malloc((batch_size + 1) * sizeof(int));

      std::default_random_engine g(123);
      std::uniform_real_distribution<float> distr(0.f, 1.f);
      for (int i = 0; i < input_size; i++) input[i] = distr(g);
      for (int i = 0; i < dense_size; i++) dense[i] = distr(g);

      // Build offsets: divide nrows evenly over batch_size
      offset[0] = 0;
      for (int i = 1; i <= batch_size; i++)
        offset[i] = (int)(1LL * nrows * i / batch_size) * ncols / nrows;
      // Use actual nrows per batch
      offset[0] = 0;
      for (int i = 1; i <= batch_size; i++)
        offset[i] = offset[i-1] + nrows / batch_size;
      // Scale offsets by ncols
      // Actually offset tracks rows, not individual elements
      // Simplify: offset[i] = i * (nrows / batch_size)
      for (int i = 0; i <= batch_size; i++)
        offset[i] = i * (nrows / batch_size);

      Kokkos::View<float*> d_input("d_input", input_size);
      Kokkos::View<float*> d_dense("d_dense", dense_size);
      Kokkos::View<float*> d_output("d_output", input_size);
      Kokkos::View<int*>   d_offset("d_offset", batch_size + 1);

      auto h_input  = Kokkos::create_mirror_view(d_input);
      auto h_dense  = Kokkos::create_mirror_view(d_dense);
      auto h_offset = Kokkos::create_mirror_view(d_offset);
      for (int i = 0; i < input_size; i++) h_input(i) = input[i];
      for (int i = 0; i < dense_size; i++) h_dense(i) = dense[i];
      for (int i = 0; i <= batch_size; i++) h_offset(i) = offset[i];
      Kokkos::deep_copy(d_input,  h_input);
      Kokkos::deep_copy(d_dense,  h_dense);
      Kokkos::deep_copy(d_offset, h_offset);

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        Kokkos::parallel_for("dense_embedding",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {batch_size, ncols}),
            KOKKOS_LAMBDA(int batch_idx, int idx) {
              const int off   = d_offset(batch_idx);
              const int range = d_offset(batch_idx + 1) - off;
              const float dense_elem = d_dense(batch_idx * ncols + idx);
              for (int nested_idx = idx; nested_idx < range; nested_idx += ncols) {
                d_output(off + nested_idx) = d_input(off + nested_idx) + dense_elem;
              }
            });
        Kokkos::fence();
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of dense embedding kernel: %f (us)\n",
             (time * 1e-3f) / repeat);

      auto h_output = Kokkos::create_mirror_view(d_output);
      Kokkos::deep_copy(h_output, d_output);
      for (int i = 0; i < input_size; i++) output_k1[i] = h_output(i);

      reference(input, dense, output_ref, ncols, batch_size, offset);

      bool ok = true;
      for (int i = 0; i < input_size; i++) {
        if (fabsf(output_k1[i] - output_ref[i]) > 1e-5f) { ok = false; break; }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");

      free(input); free(dense); free(output_k1); free(output_ref); free(offset);
    }
  }
  Kokkos::finalize();
  return 0;
}
