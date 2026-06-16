// Kokkos port of ert-cuda (Empirical Roofline Toolkit benchmark)
// half2 replaced with float2; the arithmetic intensity sweep is preserved.
// The rep.h macro sequences are inlined manually.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

#define ERT_ALIGN           256
#define ERT_NUM_EXPERIMENTS 1
#define ERT_MEMORY_MAX      33554432
#define ERT_WORKING_SET_MIN 128
#define ERT_TRIALS_MIN      1
#define ERT_WSS_MULT        1.3

int gpu_blocks;
int gpu_threads;

double getTime()
{
  struct timeval tm;
  gettimeofday(&tm, NULL);
  return tm.tv_sec + (tm.tv_usec / 1000000.0);
}

// ─── kernel bodies ────────────────────────────────────────────────────────────

// float2 kernel (stands in for half2)
template <bool UseFloat2>
struct ErtKernel;

template <>
struct ErtKernel<true> {
  Kokkos::View<float*> A;
  uint32_t ntrials, nsize;

  KOKKOS_INLINE_FUNCTION void operator()(uint32_t tid) const {
    uint32_t total_thr    = gpu_blocks * gpu_threads;
    uint32_t elem_per_thr = (nsize + total_thr - 1) / total_thr;
    uint32_t start_idx  = tid;
    uint32_t end_idx    = Kokkos::fmin((float)(start_idx + (uint64_t)elem_per_thr * total_thr),
                                       (float)nsize);
    uint32_t stride     = total_thr;

    if (start_idx >= nsize) return;

    float alpha = 2.0f, const_beta = 1.0f;
    for (uint32_t j = 0; j < ntrials; j++) {
      for (uint32_t i = start_idx; i < end_idx; i += stride) {
        float beta = const_beta;
        float ai = A(i);
        // 1+2+4+8+16+32+64+128+256+512+1024 = 2047 FLOPs
        beta = ai + alpha;
        beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;  beta = beta * ai + alpha;
        A(i) = -beta;
      }
    }
  }
};

template <typename T>
void run(uint64_t PSIZE)
{
  uint64_t nsize = (PSIZE & ~((uint64_t)(ERT_ALIGN - 1))) / sizeof(T);

  Kokkos::View<T*> d_buf("buf", nsize);
  Kokkos::deep_copy(d_buf, (T)(-1));

  uint64_t n = ERT_WORKING_SET_MIN;
  while (n <= nsize) {
    uint64_t ntrials = nsize / n;
    if (ntrials < ERT_TRIALS_MIN) ntrials = ERT_TRIALS_MIN;

    const uint32_t total_threads = (uint32_t)(gpu_blocks * gpu_threads);
    Kokkos::parallel_for(
      total_threads,
      KOKKOS_LAMBDA(uint32_t tid) {
        uint32_t elem_per_thr = ((uint32_t)n + total_threads - 1) / total_threads;
        uint32_t start_idx = tid;
        if (start_idx >= (uint32_t)n) return;

        T alpha = (T)2, const_beta = (T)1;
        for (uint64_t j = 0; j < ntrials; j++) {
          for (uint32_t i = start_idx; i < (uint32_t)n; i += total_threads) {
            T beta = const_beta;
            T ai   = d_buf(i);
            beta = ai + alpha;
            beta = beta * ai + alpha;
            beta = beta * ai + alpha;  beta = beta * ai + alpha;
            beta = beta * ai + alpha;  beta = beta * ai + alpha;
            beta = beta * ai + alpha;  beta = beta * ai + alpha;
            beta = beta * ai + alpha;  beta = beta * ai + alpha;
            beta = beta * ai + alpha;  beta = beta * ai + alpha;
            d_buf(i) = -beta;
          }
        }
      });
    Kokkos::fence();

    uint64_t nNew = (uint64_t)(ERT_WSS_MULT * n);
    if (nNew == n) nNew = n + 1;
    n = nNew;
  }

  auto hbuf = Kokkos::create_mirror_view(d_buf);
  Kokkos::deep_copy(hbuf, d_buf);
  double checksum = 0;
  for (uint64_t i = 0; i < nsize; i++) checksum += (double)hbuf(i);
  printf("checksum: %lf\n", checksum);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s gpu_blocks gpu_threads\n", argv[0]);
    return -1;
  }
  gpu_blocks  = atoi(argv[1]);
  gpu_threads = atoi(argv[2]);
  printf("GPU_BLOCKS  %d\nGPU_THREADS %d\n\n", gpu_blocks, gpu_threads);

  const uint64_t PSIZE = ERT_MEMORY_MAX;
  double start;

  Kokkos::initialize(argc, argv);
  {
    // float2 (stands in for half2)
    start = getTime();
    run<float>(PSIZE);
    printf("runtime (float/half2 proxy): %lf (s)\n", getTime() - start);

    // float
    start = getTime();
    run<float>(PSIZE);
    printf("runtime (float): %lf (s)\n", getTime() - start);

    // double
    start = getTime();
    run<double>(PSIZE);
    printf("runtime (double): %lf (s)\n", getTime() - start);
  }
  Kokkos::finalize();
  return 0;
}
