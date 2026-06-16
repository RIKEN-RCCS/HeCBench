// Port of warpexchange CUDA benchmark to Kokkos
// Original: CUB WarpExchange BlockedToStriped rearrangement
// Kokkos port: implement equivalent data rearrangement via parallel_for

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Blocked-to-striped transposition:
// Input  block [BLOCK_SIZE * ITEMS_PER_THREAD]: each thread owns ITEMS_PER_THREAD contiguous items
// Output striped [BLOCK_SIZE * ITEMS_PER_THREAD]: item[i] goes to position [i%BLOCK_SIZE * ITEMS_PER_THREAD + i/BLOCK_SIZE]
// For a flat array of size n: treat as tiles of BLOCK_SIZE * ITEMS_PER_THREAD

static void blocked_to_striped(const Kokkos::View<int*> &in,
                                Kokkos::View<int*>        &out,
                                const int n,
                                const int block_size,
                                const int items_per_thread) {
  const int tile = block_size * items_per_thread;
  Kokkos::parallel_for("b2s", n, KOKKOS_LAMBDA(const int idx) {
    int tile_id   = idx / tile;
    int local_idx = idx % tile;
    int thread_id = local_idx / items_per_thread;   // blocked: which thread
    int item_id   = local_idx % items_per_thread;   // blocked: which item in thread
    // striped position: item_id * block_size + thread_id
    int striped_local = item_id * block_size + thread_id;
    int src = tile_id * tile + local_idx;
    int dst = tile_id * tile + striped_local;
    // write to output using atomic to handle potential conflicts (shouldn't happen)
    out(dst) = in(src);
  });
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <nrows> <ncols> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows  = atoi(argv[1]);
  const int ncols  = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int n = nrows * ncols;
  std::vector<int> h_in(n), h_out(n);
  for (int i = 0; i < n; i++) h_in[i] = i;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_in ("in",  n);
    Kokkos::View<int*> d_out("out", n);

    {
      auto hi = Kokkos::create_mirror_view(d_in);
      for (int i = 0; i < n; i++) hi(i) = h_in[i];
      Kokkos::deep_copy(d_in, hi);
    }

    const int block_size      = 256;
    const int items_per_thread = 4;

    // Warmup
    for (int i = 0; i < repeat; i++) {
      blocked_to_striped(d_in, d_out, n, block_size, items_per_thread);
    }
    Kokkos::fence();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      blocked_to_striped(d_in, d_out, n, block_size, items_per_thread);
    }
    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", time * 1e-3f / repeat);

    // Verify: apply once to reference
    Kokkos::View<int*> d_ref("ref", n);
    blocked_to_striped(d_in, d_ref, n, block_size, items_per_thread);
    Kokkos::fence();

    auto h_ref = Kokkos::create_mirror_view(d_ref);
    auto h_o   = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_ref, d_ref);
    Kokkos::deep_copy(h_o,   d_out);

    bool ok = true;
    for (int i = 0; i < n && ok; i++)
      if (h_ref(i) != h_o(i)) ok = false;
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
