#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "reference.h"

template <typename T>
void ChannelShuffleNCHWKernel(Kokkos::View<T*> X, Kokkos::View<T*> Y,
    int N, int G, int K, int HxW)
{
  const int C = G * K;
  Kokkos::parallel_for("NCHW",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{N,C,HxW}),
    KOKKOS_LAMBDA(int n, int c, int s) {
      Y[(n*C+c)*HxW+s] = X[(n*C+(c%G)*K+c/G)*HxW+s];
    });
}

template <typename T>
void ChannelShuffleNHWCKernel(Kokkos::View<T*> X, Kokkos::View<T*> Y,
    int O, int G, int K)
{
  const int C = G * K;
  Kokkos::parallel_for("NHWC",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{O,C}),
    KOKKOS_LAMBDA(int o, int i) {
      Y[o*C+i] = X[o*C+(i%G)*K+i/G];
    });
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <group size> <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int G      = atoi(argv[1]);
  const int W      = atoi(argv[2]);
  const int H      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  Kokkos::initialize(argc, argv);
  {
    for (int N = 1; N <= 64; N *= 4) {
      for (int C = 32; C <= 512; C *= 4) {
        printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);
        if (C % G != 0) continue;

        const int numel = N * C * W * H;
        const int K     = C / G;
        const int HxW   = W * H;
        const int O     = N * HxW;

        float *h_X     = (float*)malloc(numel * sizeof(float));
        float *h_Y     = (float*)malloc(numel * sizeof(float));
        float *h_Y_ref = (float*)malloc(numel * sizeof(float));
        for (int i = 0; i < numel; i++) h_X[i] = (float)i / numel;

        Kokkos::View<float*> d_X("X", numel), d_Y("Y", numel);
        {
          auto mX = Kokkos::create_mirror_view(d_X);
          for (int i = 0; i < numel; i++) mX[i] = h_X[i];
          Kokkos::deep_copy(d_X, mX);
        }

        // NHWC
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; r++)
          ChannelShuffleNHWCKernel<float>(d_X, d_Y, O, G, K);
        Kokkos::fence();
        auto t1 = std::chrono::steady_clock::now();
        long nhwc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();

        {
          auto mY = Kokkos::create_mirror_view(d_Y);
          Kokkos::deep_copy(mY, d_Y);
          for (int i = 0; i < numel; i++) h_Y[i] = mY[i];
        }
        long dummy = 0;
        ChannelShuffleNHWC_cpu(h_X, N, C, G, numel, h_Y_ref, dummy, 1);
        if (memcmp(h_Y, h_Y_ref, numel * sizeof(float)))
          printf("Failed to pass channel shuffle (NHWC) check\n");
        else
          printf("Average time of channel shuffle (NHWC): %f (ms)\n", (nhwc_ns * 1e-6f) / repeat);

        // NCHW
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; r++)
          ChannelShuffleNCHWKernel<float>(d_X, d_Y, N, G, K, HxW);
        Kokkos::fence();
        t1 = std::chrono::steady_clock::now();
        long nchw_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();

        {
          auto mY = Kokkos::create_mirror_view(d_Y);
          Kokkos::deep_copy(mY, d_Y);
          for (int i = 0; i < numel; i++) h_Y[i] = mY[i];
        }
        ChannelShuffleNCHW_cpu(h_X, N, C, G, numel, h_Y_ref, dummy, 1);
        if (memcmp(h_Y, h_Y_ref, numel * sizeof(float)))
          printf("Failed to pass channel shuffle (NCHW) check\n");
        else
          printf("Average time of channel shuffle (NCHW): %f (ms)\n", (nchw_ns * 1e-6f) / repeat);

        free(h_X); free(h_Y); free(h_Y_ref);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
