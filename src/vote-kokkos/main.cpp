// Port of vote-cuda benchmark to Kokkos
// Original: CUDA warp vote intrinsics (__any_sync, __all_sync)
// Kokkos port: simulate warp-vote semantics using team-level reductions

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define VOTE_DATA_GROUP 4

static void genVoteTestPattern(unsigned int *pat, int size) {
  for (int i = 0; i < size / 4; i++) pat[i] = 0;
  for (int i = 2*size/8; i < 4*size/8; i++) pat[i] = (i & 1) ? i : 0;
  for (int i = 2*size/4; i < 3*size/4; i++) pat[i] = (i & 1) ? 0 : i;
  for (int i = 3*size/4; i < size;     i++) pat[i] = 0xffffffff;
}

// Simulate VoteAny across groups of warp_size threads:
// result[i] = any non-zero value in the same warp group
static void vote_any_kernel(const unsigned int *input, unsigned int *result,
                            int total, int warp_size, int repeat) {
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("vote_any", total, KOKKOS_LAMBDA(const int tx) {
      int warp_base = (tx / warp_size) * warp_size;
      // serial scan to simulate __any_sync
      unsigned int any_val = 0;
      for (int k = warp_base; k < warp_base + warp_size; k++) {
        if (input[k]) { any_val = 1; break; }
      }
      result[tx] = any_val;
    });
    Kokkos::fence();
  }
}

// Simulate VoteAll across groups of warp_size threads:
static void vote_all_kernel(const unsigned int *input, unsigned int *result,
                            int total, int warp_size, int repeat) {
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("vote_all", total, KOKKOS_LAMBDA(const int tx) {
      int warp_base = (tx / warp_size) * warp_size;
      unsigned int all_val = 1;
      for (int k = warp_base; k < warp_base + warp_size; k++) {
        if (!input[k]) { all_val = 0; break; }
      }
      result[tx] = all_val;
    });
    Kokkos::fence();
  }
}

// Simulate VoteAny kernel 3
static void vote_any3_kernel(bool *info, int warp_size, int total, int repeat) {
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("vote_any3", total, KOKKOS_LAMBDA(const int tx) {
      bool *offs = info + tx * 3;
      int warp_base = (tx / warp_size) * warp_size;

      bool any_upper = false;
      for (int k = warp_base; k < warp_base + warp_size; k++) {
        if (k >= (total * 3) / 2) { any_upper = true; break; }
      }
      offs[0] = any_upper;
      offs[1] = (tx >= (total * 3) / 2);

      bool all_upper = true;
      for (int k = warp_base; k < warp_base + warp_size; k++) {
        if (k < (total * 3) / 2) { all_upper = false; break; }
      }
      if (all_upper) offs[2] = true;
    });
    Kokkos::fence();
  }
}

static int checkErrors1(const unsigned int *h, int start, int end, int ws, const char *t) {
  int sum = 0;
  for (int i = start; i < end; i++) sum += h[i];
  if (sum > 0) { printf("\t<%s>[%d-%d] %d FAILED\n", t, start, end-1, sum); }
  return (sum > 0);
}
static int checkErrors2(const unsigned int *h, int start, int end, int ws, const char *t) {
  int sum = 0;
  for (int i = start; i < end; i++) sum += h[i];
  if (sum != ws) { printf("\t<%s>[%d-%d] - FAILED\n", t, start, end-1); }
  return (sum != ws);
}

static int checkResultsVoteAnyKernel1(const unsigned int *h, int size, int ws) {
  int ec = 0;
  ec += checkErrors1(h, 0,                   VOTE_DATA_GROUP*ws/4,   ws, "Vote.Any");
  ec += checkErrors2(h, VOTE_DATA_GROUP*ws/4, 2*VOTE_DATA_GROUP*ws/4, ws, "Vote.Any");
  ec += checkErrors2(h, 2*VOTE_DATA_GROUP*ws/4, 3*VOTE_DATA_GROUP*ws/4, ws, "Vote.Any");
  ec += checkErrors2(h, 3*VOTE_DATA_GROUP*ws/4, 4*VOTE_DATA_GROUP*ws/4, ws, "Vote.Any");
  printf((ec == 0) ? "\tOK\n" : "\tERROR\n");
  return ec;
}
static int checkResultsVoteAllKernel2(const unsigned int *h, int size, int ws) {
  int ec = 0;
  ec += checkErrors1(h, 0,                     VOTE_DATA_GROUP*ws/4,     ws, "Vote.All");
  ec += checkErrors1(h, VOTE_DATA_GROUP*ws/4,   2*VOTE_DATA_GROUP*ws/4,   ws, "Vote.All");
  ec += checkErrors1(h, 2*VOTE_DATA_GROUP*ws/4, 3*VOTE_DATA_GROUP*ws/4,   ws, "Vote.All");
  ec += checkErrors2(h, 3*VOTE_DATA_GROUP*ws/4, 4*VOTE_DATA_GROUP*ws/4,   ws, "Vote.All");
  printf((ec == 0) ? "\tOK\n" : "\tERROR\n");
  return ec;
}
static int checkResultsVoteAnyKernel3(const bool *hinfo, int size) {
  int ec = 0;
  for (int i = 0; i < size * 3; i++) {
    switch (i % 3) {
      case 0: if (hinfo[i] != (i >= size * 1)) ec++; break;
      case 1: if (hinfo[i] != (i >= size * 3 / 2)) ec++; break;
      case 2: if (hinfo[i] != (i >= size * 2)) ec++; break;
    }
  }
  printf((ec == 0) ? "\tOK\n" : "\tERROR\n");
  return ec;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <warp_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int warp_size = atoi(argv[1]);
  const int repeat    = atoi(argv[2]);
  const int total     = VOTE_DATA_GROUP * warp_size;

  std::vector<unsigned int> h_input(total), h_result(total);
  genVoteTestPattern(h_input.data(), total);

  int error_count[3] = {0, 0, 0};

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<unsigned int*> d_input ("input",  total);
    Kokkos::View<unsigned int*> d_result("result", total);
    Kokkos::View<bool*>         d_info  ("info",   warp_size * 3 * 3);

    {
      auto hi = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < total; i++) hi(i) = h_input[i];
      Kokkos::deep_copy(d_input, hi);
    }

    printf("\tRunning <<Vote.Any>> kernel1 ...\n");
    // Warmup
    vote_any_kernel(d_input.data(), d_result.data(), total, warp_size, repeat);

    auto start = std::chrono::steady_clock::now();
    vote_any_kernel(d_input.data(), d_result.data(), total, warp_size, repeat);
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    printf("\tkernel execution time: %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-9f);

    {
      auto hr = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(hr, d_result);
      for (int i = 0; i < total; i++) h_result[i] = hr(i);
    }
    error_count[0] = checkResultsVoteAnyKernel1(h_result.data(), total, warp_size);

    printf("\tRunning <<Vote.All>> kernel2 ...\n");
    vote_all_kernel(d_input.data(), d_result.data(), total, warp_size, repeat);

    start = std::chrono::steady_clock::now();
    vote_all_kernel(d_input.data(), d_result.data(), total, warp_size, repeat);
    Kokkos::fence();
    end = std::chrono::steady_clock::now();
    printf("\tkernel execution time: %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-9f);

    {
      auto hr = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(hr, d_result);
      for (int i = 0; i < total; i++) h_result[i] = hr(i);
    }
    error_count[1] = checkResultsVoteAllKernel2(h_result.data(), total, warp_size);

    printf("\tRunning <<Vote.Any>> kernel3 ...\n");
    Kokkos::deep_copy(d_info, false);
    vote_any3_kernel(d_info.data(), warp_size, warp_size * 3, repeat);

    start = std::chrono::steady_clock::now();
    Kokkos::deep_copy(d_info, false);
    vote_any3_kernel(d_info.data(), warp_size, warp_size * 3, repeat);
    Kokkos::fence();
    end = std::chrono::steady_clock::now();
    printf("\tkernel execution time: %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-9f);

    {
      auto hi = Kokkos::create_mirror_view(d_info);
      Kokkos::deep_copy(hi, d_info);
      std::vector<bool> hinfo(warp_size * 3 * 3);
      for (int i = 0; i < warp_size * 3 * 3; i++) hinfo[i] = hi(i);
      error_count[2] = checkResultsVoteAnyKernel3(hinfo.data(), warp_size * 3);
    }
  }
  Kokkos::finalize();

  return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0)
         ? EXIT_SUCCESS : EXIT_FAILURE;
}
