#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>

// Perform a blocked-to-striped rearrangement over n elements.
//
// Within each block of `items_per_block = block_size * items_per_thread`
// consecutive elements starting at position k:
//   Input  (blocked):  thread t owns positions k + t*IPT .. k + t*IPT + (IPT-1)
//   Output (striped):  thread t owns positions k + t, k + BLK+t, k + 2*BLK+t, ...
//
// Reading the output sequentially: out[k + p] = in[k + (p%IPT)*block_size + p/IPT]
//
// Equivalently for each output index idx:
//   k   = block-start
//   p   = idx - k
//   t   = p / IPT   (thread that "owned" this slot in blocked layout)
//   j   = p % IPT   (item index within that thread)
//   src = k + j*block_size + t
//   out[idx] = in[src]  (if src < n, else 0)

void blocked_to_striped(Kokkos::View<const int*> d_in,
                        Kokkos::View<int*>       d_out,
                        const int n,
                        const int block_size,
                        const int items_per_thread)
{
  const int items_per_block = block_size * items_per_thread;

  Kokkos::parallel_for("blockexchange",
    Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(const int idx) {
      const int k   = (idx / items_per_block) * items_per_block;
      const int p   = idx - k;
      const int t   = p / items_per_thread;   // thread index in blocked layout
      const int j   = p % items_per_thread;   // item index within thread
      const int src = k + j * block_size + t;
      d_out(idx) = (src < n) ? d_in(src) : 0;
    });
  Kokkos::fence();
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4) {
      printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int nrows  = atoi(argv[1]);
    const int ncols  = atoi(argv[2]);
    const int repeat = atoi(argv[3]);

    const int n = nrows * ncols;
    constexpr int block_size        = 256;
    constexpr int items_per_thread  = 4;
    const int     items_per_block   = block_size * items_per_thread;

    // Device views
    Kokkos::View<int*> d_A("d_A", n);
    Kokkos::View<int*> d_out("d_out", n);

    // Fill input: A[i] = i
    Kokkos::parallel_for("init", n,
      KOKKOS_LAMBDA(const int i) { d_A(i) = i; });
    Kokkos::fence();

    Kokkos::View<const int*> d_A_const = d_A;

    // Benchmark
    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++)
      blocked_to_striped(d_A_const, d_out, n, block_size, items_per_thread);
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    long long time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel: %f (us)\n", (time_ns * 1e-3) / repeat);

    // Copy back and verify (only when items_per_block divides n)
    auto h_out = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out, d_out);

    if (n % items_per_block == 0) {
      bool ok = true;
      // For each block k, and each item-slot j (0..IPT-1):
      //   elements at positions k + j*1, k + j*1 + IPT, k + j*1 + 2*IPT, ...
      //   i.e. out[k + j + m*IPT] for m = 0..block_size-1 should differ by 1
      // Equivalent check from CUDA: for each base i = k+j (j=0..IPT-1),
      //   out[i + m*IPT] and out[i + (m+1)*IPT] differ by 1.
    outer:
      for (int k = 0; k < n; k += items_per_block) {
        int i = k;
        for (int j = 0; j < items_per_thread; j++, i++) {
          for (int m = 0; m < block_size - 1; m++) {
            if (i + (m + 1) * items_per_thread < n) {
              int diff = h_out(i + (m + 1) * items_per_thread) -
                         h_out(i + m * items_per_thread);
              if (diff != 1) {
                printf("Error at index %d (diff=%d)\n",
                       i + (m + 1) * items_per_thread, diff);
                ok = false;
                goto outer;
              }
            }
          }
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    } else {
      printf("n is not a multiple of items_per_block; skipping verification.\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
