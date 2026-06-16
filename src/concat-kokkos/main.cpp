/*
 * Tensor concatenation along a dimension.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

/* Convert 3-dim tensor index into flat vector index */
KOKKOS_INLINE_FUNCTION
int flat_3dim(int id1, int id2, int id3, int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

// CPU reference
void concat_cpu(const float *inp1, const float *inp2, float *output,
                int sz0, int sz2, int sz1_1, int sz1_2)
{
  for (int idx0 = 0; idx0 < sz0; idx0++) {
    for (int idx1 = 0; idx1 < sz1_1 + sz1_2; idx1++) {
      for (int idx2 = 0; idx2 < sz2; idx2++) {
        float *dst_ptr = output + flat_3dim(idx0, idx1, idx2, sz1_1 + sz1_2, sz2);
        const float *src_ptr;
        int sz1;
        int idx1_t = idx1;
        if (idx1_t < sz1_1) {
          sz1 = sz1_1;
          src_ptr = inp1;
        } else {
          idx1_t -= sz1_1;
          sz1 = sz1_2;
          src_ptr = inp2;
        }
        src_ptr += flat_3dim(idx0, idx1_t, idx2, sz1, sz2);
        *dst_ptr = *src_ptr;
      }
    }
  }
}

void concat(Kokkos::View<const float*> inp1,
            Kokkos::View<const float*> inp2,
            Kokkos::View<float*>       output,
            int sz0, int sz2, int sz1_1, int sz1_2)
{
  int nele = sz0 * sz2 * (sz1_1 + sz1_2);

  Kokkos::parallel_for("concat", nele, KOKKOS_LAMBDA(int idx) {
    int idx_save = idx;
    int idx2 = idx % sz2;
    idx = idx / sz2;
    int idx1 = idx % (sz1_1 + sz1_2);
    int idx0 = idx / (sz1_1 + sz1_2);

    const float *src_ptr;
    int sz1;
    int idx1_t = idx1;
    if (idx1_t < sz1_1) {
      sz1 = sz1_1;
      src_ptr = inp1.data();
    } else {
      idx1_t -= sz1_1;
      sz1 = sz1_2;
      src_ptr = inp2.data();
    }
    src_ptr += flat_3dim(idx0, idx1_t, idx2, sz1, sz2);
    output(idx_save) = *src_ptr;
  });
  Kokkos::fence();
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    for (int nhead = 6; nhead <= 48; nhead *= 2) {
      srand(nhead);

      const int seq_len    = 1024;
      const int batch_size = 8;
      const int hidden_dim = nhead * 128;
      const int head_dim   = hidden_dim / nhead;

      const int sl1 = rand() % (seq_len - 1) + 1;
      const int sl2 = seq_len - sl1;
      const int beam_size = 8;

      printf("\n");
      printf("num_head = %d\t", nhead);
      printf("seq_len = %d\t", seq_len);
      printf("batch_size = %d\t", batch_size);
      printf("hidden_dimension = %d\t", hidden_dim);
      printf("beam_size = %d\n", beam_size);

      const size_t inp1_size = (size_t)batch_size * beam_size * hidden_dim * sl1;
      const size_t inp2_size = (size_t)batch_size * beam_size * hidden_dim * sl2;
      const size_t outp_size = (size_t)batch_size * beam_size * hidden_dim * seq_len;

      float *inp1     = (float*) malloc(inp1_size * sizeof(float));
      float *inp2     = (float*) malloc(inp2_size * sizeof(float));
      float *outp_ref = (float*) malloc(outp_size * sizeof(float));

      for (size_t i = 0; i < inp1_size; i++) inp1[i] = (float)(rand() % (int)inp1_size);
      for (size_t i = 0; i < inp2_size; i++) inp2[i] = (float)(rand() % (int)inp2_size);

      Kokkos::View<float*> d_inp1("d_inp1", inp1_size);
      Kokkos::View<float*> d_inp2("d_inp2", inp2_size);
      Kokkos::View<float*> d_outp("d_outp", outp_size);

      auto h_inp1 = Kokkos::create_mirror_view(d_inp1);
      auto h_inp2 = Kokkos::create_mirror_view(d_inp2);
      for (size_t i = 0; i < inp1_size; i++) h_inp1(i) = inp1[i];
      for (size_t i = 0; i < inp2_size; i++) h_inp2(i) = inp2[i];
      Kokkos::deep_copy(d_inp1, h_inp1);
      Kokkos::deep_copy(d_inp2, h_inp2);

      float size_bytes = 2.f * outp_size * sizeof(float) * 1e-9f;
      printf("Total device memory usage (GB) = %.2f\n", size_bytes);

      // warmup and verify
      concat(d_inp1, d_inp2, d_outp,
             batch_size * beam_size * nhead, head_dim, sl1, sl2);

      auto h_outp = Kokkos::create_mirror_view(d_outp);
      Kokkos::deep_copy(h_outp, d_outp);

      concat_cpu(inp1, inp2, outp_ref,
                 batch_size * beam_size * nhead, head_dim, sl1, sl2);

      int error = 0;
      for (size_t i = 0; i < outp_size; i++) {
        if (fabsf(h_outp(i) - outp_ref[i]) > 1e-5f) { error = 1; break; }
      }
      printf("%s\n", error ? "FAIL" : "PASS");

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        concat(d_inp1, d_inp2, d_outp,
               batch_size * beam_size * nhead, head_dim, sl1, sl2);
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      float avg_time = (time * 1e-3f) / repeat;
      printf("Average kernel execution time: %f (us)\n", avg_time);
      printf("Average kernel throughput : %f (GB/s)\n",
             size_bytes / (avg_time * 1e-6f));

      free(inp1);
      free(inp2);
      free(outp_ref);
    }
  }
  Kokkos::finalize();
  return 0;
}
