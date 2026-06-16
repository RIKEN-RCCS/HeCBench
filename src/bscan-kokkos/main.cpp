#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>

// Binary predicate used by both the scan and the CPU reference.
KOKKOS_INLINE_FUNCTION bool valid(int x) { return x > 0; }

// Run an exclusive prefix scan over N elements where each element
// contributes 1 if valid(x) else 0.
//
// The CUDA version launches grid_size independent blocks, each scanning N
// elements in parallel.  In the Kokkos port we run a single parallel_scan
// over N elements and scale the reported throughput by grid_size to give a
// comparable bandwidth figure.
template <int N>
void bscan(const int repeat)
{
  Kokkos::View<int*> d_in ("d_in",  N);
  Kokkos::View<int*> d_out("d_out", N);
  auto h_in  = Kokkos::create_mirror_view(d_in);
  auto h_out = Kokkos::create_mirror_view(d_out);

  int  h_in_arr[N], h_out_arr[N], ref_out[N];
  bool ok          = true;
  double total_ns  = 0.0;
  int    valid_count = 0;

  // Matches CUDA: grid_size = 12*7*8*9*10
  constexpr size_t grid_size = 12ULL * 7 * 8 * 9 * 10;

  srand(123);

  for (int iter = 0; iter < repeat; iter++) {
    // Generate random input (same RNG sequence as CUDA version)
    for (int i = 0; i < N; i++) {
      h_in_arr[i] = rand() % N - N / 2;
      if (valid(h_in_arr[i])) valid_count++;
      h_in(i) = h_in_arr[i];
    }
    Kokkos::deep_copy(d_in, h_in);

    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();

    Kokkos::parallel_scan("bscan",
      Kokkos::RangePolicy<>(0, N),
      KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        const int val = valid(d_in(i)) ? 1 : 0;
        if (final) d_out(i) = update;   // exclusive prefix stored before accumulation
        update += val;
      });

    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    total_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // Verify result
    Kokkos::deep_copy(h_out, d_out);
    for (int i = 0; i < N; i++) h_out_arr[i] = h_out(i);

    ref_out[0] = 0;
    ok &= (h_out_arr[0] == ref_out[0]);
    for (int i = 1; i < N; i++) {
      ref_out[i] = ref_out[i - 1] + (h_in_arr[i - 1] > 0 ? 1 : 0);
      ok &= (ref_out[i] == h_out_arr[i]);
    }
    if (!ok) break;
  }

  printf("Block size = %d, ratio of valid elements = %f, verify = %s\n",
         N, valid_count * 1.f / (N * repeat), ok ? "PASS" : "FAIL");

  if (ok) {
    printf("Average execution time: %f (us)\n", (total_ns * 1e-3) / repeat);
    // Scale throughput by grid_size to match the CUDA multi-block measurement.
    printf("Billion elements per second: %f\n\n",
           (double)grid_size * N * repeat / total_ns);
  }
}

int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    bscan<32>(repeat);
    bscan<64>(repeat);
    bscan<128>(repeat);
    bscan<256>(repeat);
    bscan<512>(repeat);
    bscan<1024>(repeat);
  }
  Kokkos::finalize();
  return 0;
}
