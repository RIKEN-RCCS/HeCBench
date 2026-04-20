// Port of streamPriority CUDA benchmark to Kokkos
// Original: NVIDIA CUDA sample demonstrating stream priorities
// Kokkos port: simulates two workloads (heavy + light) and times the light one

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EACH_SIZE (256 * 1024)

static void mem_init(int *buf, size_t n) {
  for (size_t i = 0; i < n; i++)
    buf[i] = (int)i;
}

// heavy: iterative write to simulate busy work (simulates wait=true)
static void heavy_copy(Kokkos::View<int*> dst,
                        Kokkos::View<int*> src,
                        size_t n) {
  Kokkos::parallel_for("heavy_copy", n, KOKKOS_LAMBDA(const int i) {
    int v = src(i);
    // busy-wait loop (scaled down to keep runtime reasonable)
    for (int k = 0; k < 4; k++) {
      if (v > 0) v--;
    }
    dst(i) = src(i);
  });
}

static void light_copy(Kokkos::View<int*> dst,
                        Kokkos::View<int*> src,
                        size_t n) {
  Kokkos::parallel_for("light_copy", n, KOKKOS_LAMBDA(const int i) {
    dst(i) = src(i);
  });
}

// Returns elapsed time in nanoseconds for the "high priority" (light) work
static long eval(bool use_priority) {
  const size_t size = 1UL << 29; // 512 MiB of ints
  const size_t nelem = size / sizeof(int);

  std::vector<int> h_src_low(nelem), h_src_hi(nelem);
  mem_init(h_src_low.data(), nelem);
  mem_init(h_src_hi.data(), nelem);

  // Device views
  Kokkos::View<int*> d_src_low("src_low", nelem);
  Kokkos::View<int*> d_src_hi("src_hi",  nelem);
  Kokkos::View<int*> d_dst_low("dst_low", nelem);
  Kokkos::View<int*> d_dst_hi("dst_hi",  nelem);

  // Host mirror copies
  auto h_vl = Kokkos::create_mirror_view(d_src_low);
  auto h_vh = Kokkos::create_mirror_view(d_src_hi);
  for (size_t i = 0; i < nelem; i++) { h_vl(i) = h_src_low[i]; h_vh(i) = h_src_hi[i]; }
  Kokkos::deep_copy(d_src_low, h_vl);
  Kokkos::deep_copy(d_src_hi,  h_vh);

  const size_t chunk = EACH_SIZE / sizeof(int);
  const size_t nchunks = nelem / chunk;

  // Warmup
  for (size_t c = 0; c < nchunks; c++) {
    auto src_l = Kokkos::subview(d_src_low, Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    auto dst_l = Kokkos::subview(d_dst_low, Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    auto src_h = Kokkos::subview(d_src_hi,  Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    auto dst_h = Kokkos::subview(d_dst_hi,  Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    heavy_copy(dst_l, src_l, chunk);
    light_copy(dst_h, src_h, chunk);
  }
  Kokkos::fence();

  auto start = std::chrono::steady_clock::now();
  for (size_t c = 0; c < nchunks; c++) {
    auto src_l = Kokkos::subview(d_src_low, Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    auto dst_l = Kokkos::subview(d_dst_low, Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    auto src_h = Kokkos::subview(d_src_hi,  Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    auto dst_h = Kokkos::subview(d_dst_hi,  Kokkos::pair<size_t,size_t>(c*chunk,(c+1)*chunk));
    if (use_priority) {
      // Simulate priority: run light work first
      light_copy(dst_h, src_h, chunk);
      heavy_copy(dst_l, src_l, chunk);
    } else {
      heavy_copy(dst_l, src_l, chunk);
      light_copy(dst_h, src_h, chunk);
    }
  }
  // Time until the light (hi priority) work is done
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();

  // Verify
  auto h_dl = Kokkos::create_mirror_view(d_dst_low);
  auto h_dh = Kokkos::create_mirror_view(d_dst_hi);
  Kokkos::deep_copy(h_dl, d_dst_low);
  Kokkos::deep_copy(h_dh, d_dst_hi);

  bool ok = true;
  for (size_t i = 0; i < nelem && ok; i++) {
    if (h_dl(i) != h_src_low[i] || h_dh(i) != h_src_hi[i]) ok = false;
  }
  if (!ok) fprintf(stderr, "Verification failed!\n");

  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    printf("Starting [%s]...\n", argv[0]);
    printf("Stream priority range: low: 0 to high: -1 (simulated)\n");

    auto t1 = eval(true);
    printf("Elapsed time of kernel launched to high priority stream: %.3lf ms\n", t1 * 1e-6);

    auto t2 = eval(false);
    printf("Elapsed time of kernel launched to no-priority stream: %.3lf ms\n", t2 * 1e-6);
  }
  Kokkos::finalize();
  return 0;
}
