/*
  3D Convolution (direct).
  Ported to Kokkos from the OMP target version.

  Reference: Chapter 16 in Programming Massively Parallel Processors
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

template <typename T>
void reference(const T *X, const T *W, T *Y,
               int N, int M, int C, int K,
               int Hin, int Win, int Hout, int Wout)
{
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++)
      for (int h = 0; h < Hout; h++)
        for (int w = 0; w < Wout; w++) {
          Y[OI(n,m,h,w)] = 0;
          for (int c = 0; c < C; c++)
            for (int p = 0; p < K; p++)
              for (int q = 0; q < K; q++)
                Y[OI(n,m,h,w)] += X[II(n,c,h+p,w+q)] * W[WI(m,c,p,q)];
        }
}

template <typename T>
void conv3D(int N, int C, int M, int Win, int Hin, int K, int repeat)
{
  const int Hout = Hin - K + 1;
  const int Wout = Win - K + 1;

  size_t X_size = (size_t)N * C * Hin * Win;
  size_t W_size = (size_t)M * C * K * K;
  size_t Y_size = (size_t)N * M * Hout * Wout;

  T *X = (T*)malloc(X_size * sizeof(T));
  T *Wf = (T*)malloc(W_size * sizeof(T));
  T *Y = (T*)malloc(Y_size * sizeof(T));
  T *Y_ref = (T*)malloc(Y_size * sizeof(T));

  srand(123);
  for (size_t i = 0; i < W_size; i++) Wf[i] = (T)(rand() % 31);
  for (size_t i = 0; i < X_size; i++) X[i]  = (T)(rand() % 13);
  for (size_t i = 0; i < Y_size; i++) Y[i] = Y_ref[i] = (T)(-1);

  printf("input dimensions: C=%d Win=%d Hin=%d\n", C, Win, Hin);
  printf("output dimensions: M=%d Wout=%d Hout=%d\n", M, Wout, Hout);

  Kokkos::View<T*> d_X("d_X", X_size);
  Kokkos::View<T*> d_W("d_W", W_size);
  Kokkos::View<T*> d_Y("d_Y", Y_size);

  auto h_X = Kokkos::create_mirror_view(d_X);
  auto h_W = Kokkos::create_mirror_view(d_W);
  for (size_t i = 0; i < X_size; i++) h_X(i) = X[i];
  for (size_t i = 0; i < W_size; i++) h_W(i) = Wf[i];
  Kokkos::deep_copy(d_X, h_X);
  Kokkos::deep_copy(d_W, h_W);

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("conv3D",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {N,M,Hout,Wout}),
        KOKKOS_LAMBDA(int n, int m, int h, int w) {
          T s = 0;
          for (int c = 0; c < C; c++)
            for (int p = 0; p < K; p++)
              for (int q = 0; q < K; q++)
                s += d_X(II(n,c,h+p,w+q)) * d_W(WI(m,c,p,q));
          d_Y(OI(n,m,h,w)) = s;
        });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  auto h_Y = Kokkos::create_mirror_view(d_Y);
  Kokkos::deep_copy(h_Y, d_Y);
  for (size_t i = 0; i < Y_size; i++) Y[i] = h_Y(i);

  reference(X, Wf, Y_ref, N, M, C, K, Hin, Win, Hout, Wout);

  bool ok = true;
  for (size_t i = 0; i < Y_size && ok; i++) {
    if (fabsf((float)(Y[i] - Y_ref[i])) > 1e-3f) {
      printf("Mismatch at %zu: %f vs %f\n", i, (float)Y[i], (float)Y_ref[i]);
      ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(X); free(Wf); free(Y); free(Y_ref);
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("Usage: %s <batch size:N> <input channels:C> <output feature maps:M>", argv[0]);
    printf(" <input width:Win> <input height:Hin> <kernel size:K> <repeat>\n");
    return 1;
  }

  const int N      = atoi(argv[1]);
  const int C      = atoi(argv[2]);
  const int M      = atoi(argv[3]);
  const int Win    = atoi(argv[4]);
  const int Hin    = atoi(argv[5]);
  const int K      = atoi(argv[6]);
  const int repeat = atoi(argv[7]);

  Kokkos::initialize(argc, argv);
  {
    printf("3D convolution (FP32)\n");
    conv3D<float>(N, C, M, Win, Hin, K, repeat);
  }
  Kokkos::finalize();
  return 0;
}
