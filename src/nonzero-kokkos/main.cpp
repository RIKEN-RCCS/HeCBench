// Reference: https://pytorch.org/docs/stable/generated/torch.nonzero.html
//
// Port of nonzero-cuda to Kokkos.
// Uses parallel_reduce to count non-zeros, parallel_scan to collect their
// flat indices, and parallel_for to write (row, col) multi-dim indices.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#define MAX_DIMS 2

// ---- write_indices ----------------------------------------------------------
// Given d_out[0..nzero-1] holding flat indices of non-zero elements, rewrite
// them in-place as multi-dimensional indices:
//   d_out[dim * nzero + i]  = index along dimension `dim` for non-zero i
// This matches the layout produced by the CUDA reference.
template <typename index_t>
void write_indices(Kokkos::View<int64_t *> d_out,
                   const index_t sizes[MAX_DIMS],
                   int ndim,
                   index_t nzero)
{
  // Capture dimension sizes by value inside the lambda
  index_t s0 = sizes[0];
  index_t s1 = sizes[1];

  Kokkos::parallel_for(
      "write_indices", nzero, KOKKOS_LAMBDA(int64_t idx) {
        int64_t idx_flat = d_out[idx];
        // dim 1 (column): idx_flat % s1
        d_out[idx + 1 * nzero] = idx_flat % s1;
        // dim 0 (row):    idx_flat / s1
        d_out[idx + 0 * nzero] = (idx_flat / s1) % s0;
      });
}

// ---- nonzero ----------------------------------------------------------------
template <typename scalar_t>
void nonzero(int nrows, int ncols, int repeat)
{
  const int in_ndims = MAX_DIMS;
  int64_t in_sizes[MAX_DIMS] = {nrows, ncols};

  int64_t num_items = (int64_t)nrows * ncols;

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> dist(-1, 1);

  std::vector<scalar_t> h_in(num_items);

  bool ok = true;
  long sum_time = 0;
  long idx_time = 0;

  for (int n = 0; n < repeat; n++) {

    int64_t r_nzeros = 0;
    do {
      r_nzeros = 0;
      for (int64_t i = 0; i < num_items; i++) {
        h_in[i] = static_cast<scalar_t>(dist(gen));
        if (h_in[i] != static_cast<scalar_t>(0))
          r_nzeros++;
      }
    } while (r_nzeros == 0);

    // Device view for input
    Kokkos::View<scalar_t *> d_in("d_in", num_items);
    {
      auto h_in_view = Kokkos::create_mirror_view(d_in);
      for (int64_t i = 0; i < num_items; i++)
        h_in_view(i) = h_in[i];
      Kokkos::deep_copy(d_in, h_in_view);
    }

    // ---- Count non-zeros (parallel_reduce) ----------------------------------
    int64_t h_nzeros = 0;
    auto start = std::chrono::steady_clock::now();

    Kokkos::parallel_reduce(
        "count_nonzero", num_items,
        KOKKOS_LAMBDA(int64_t i, int64_t & lsum) {
          if (d_in(i) != static_cast<scalar_t>(0))
            lsum++;
        },
        h_nzeros);
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    sum_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    if (h_nzeros != r_nzeros) {
      printf(
          "Number of non-zero elements mismatch: %ld != %ld (expected)\n",
          (long)h_nzeros, (long)r_nzeros);
      ok = false;
    } else {
      int64_t d_out_size = h_nzeros * in_ndims;
      Kokkos::View<int64_t *> d_out("d_out", d_out_size);

      // ---- Collect flat indices (parallel_scan) ----------------------------
      start = std::chrono::steady_clock::now();

      // We need a scan that writes the position of each non-zero into d_out.
      // Kokkos::parallel_scan gives each thread its exclusive prefix count,
      // so we can store the flat index directly.
      Kokkos::parallel_scan(
          "collect_nonzero_indices", num_items,
          KOKKOS_LAMBDA(int64_t i, int64_t & update, bool final_pass) {
            bool nz = (d_in(i) != static_cast<scalar_t>(0));
            if (final_pass && nz)
              d_out(update) = i;
            if (nz)
              update++;
          });
      Kokkos::fence();

      // ---- Write multi-dim indices (parallel_for) --------------------------
      write_indices<int64_t>(d_out, in_sizes, in_ndims, h_nzeros);
      Kokkos::fence();

      end = std::chrono::steady_clock::now();
      idx_time +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();

      // ---- Verify -----------------------------------------------------------
      auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);

      int64_t cnt_nzero = 0;
      for (int64_t i = 0; i < h_nzeros; i++) {
        int64_t row = h_out(i);
        int64_t col = h_out(h_nzeros + i);
        if (h_in[in_sizes[1] * row + col] != static_cast<scalar_t>(0))
          cnt_nzero++;
      }
      ok = (cnt_nzero == h_nzeros);
    }

    if (!ok)
      break;
  }

  printf("Average time for sum reduction: %lf (us)\n",
         sum_time * 1e-3 / repeat);
  printf("Average time for write index operations: %lf (us)\n",
         idx_time * 1e-3 / repeat);
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n",
           argv[0]);
    return 1;
  }

  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  if (nrows <= 0) nrows = 1;
  if (ncols <= 0) ncols = 1;

  Kokkos::initialize(argc, argv);
  {
    // Warmup + timed run (matches CUDA reference which loops twice)
    for (int w = 0; w < 2; w++) {
      printf("=========== Data type is I8 ==========\n");
      nonzero<int8_t>(nrows, ncols, repeat);

      printf("=========== Data type is I16 ==========\n");
      nonzero<int16_t>(nrows, ncols, repeat);

      printf("=========== Data type is I32 ==========\n");
      nonzero<int32_t>(nrows, ncols, repeat);

      printf("=========== Data type is FP32 ==========\n");
      nonzero<float>(nrows, ncols, repeat);

      printf("=========== Data type is FP64 ==========\n");
      nonzero<double>(nrows, ncols, repeat);
    }
  }
  Kokkos::finalize();
  return 0;
}
