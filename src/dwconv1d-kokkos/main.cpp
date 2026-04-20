// Kokkos port of dwconv1d-cuda (RWKV time-mixing causal 1D depthwise convolution)
// Formula: out[b][c][t] = eps + sum_{u=0}^{t} w[c][T-1-(t-u)] * k[b][c][u]
// Tile/vectorisation optimisations from the CUDA version are straightened into
// simple parallel_for loops.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

// Forward: out(b,c,t) = eps + sum_{u<=t} w[c][T-1-(t-u)] * k[b][c][u]
void timex_forward(
    Kokkos::View<const float*> w,  // (C*T)
    Kokkos::View<const float*> k,  // (B*C*T)
    Kokkos::View<float*>       x,  // (B*C*T)
    float eps, int B, int C, int T, int repeat)
{
  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(B * C, KOKKOS_LAMBDA(int bc) {
      int b = bc / C;
      int c = bc % C;
      for (int t = 0; t < T; t++) {
        float s = eps;
        for (int u = 0; u <= t; u++) {
          s += w[c * T + (T - 1 - (t - u))] * k[b * C * T + c * T + u];
        }
        x[b * C * T + c * T + t] = s;
      }
    });
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("timex forward average time: %f (us)\n", time * 1e-3f / repeat);
}

// Backward: compute gw and gk
void timex_backward(
    Kokkos::View<const float*> w,   // (C*T)
    Kokkos::View<const float*> k,   // (B*C*T)
    Kokkos::View<const float*> gwk, // (B*C*T) gradient of loss wrt out
    Kokkos::View<float*>       gw,  // (B*C*T)
    Kokkos::View<float*>       gk,  // (B*C*T)
    int B, int C, int T, int repeat)
{
  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(B * C, KOKKOS_LAMBDA(int bc) {
      int b = bc / C;
      int c = bc % C;

      // gw[b,c,t] = sum_{u>=t} gwk[b,c,u] * k[b,c,u - (T-1-t)]
      for (int t = 0; t < T; t++) {
        float sg = 0.f;
        for (int u = t; u < T; u++) {
          int k_idx = u - (T - 1 - t);
          if (k_idx >= 0 && k_idx < T)
            sg += gwk[b * C * T + c * T + u] * k[b * C * T + c * T + k_idx];
        }
        gw[b * C * T + c * T + t] = sg;
      }

      // gk[b,c,u] = sum_{t>=u} gwk[b,c,t] * w[c][T-1-(t-u)]
      for (int u = 0; u < T; u++) {
        float sk = 0.f;
        for (int t = u; t < T; t++) {
          sk += gwk[b * C * T + c * T + t] * w[c * T + (T - 1 - (t - u))];
        }
        gk[b * C * T + c * T + u] = sk;
      }
    });
  }

  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("timex backward average time: %f (us)\n", time * 1e-3f / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <B> <C> <T> <repeat>\n", argv[0]);
    return 1;
  }

  const int B      = atoi(argv[1]);
  const int C      = atoi(argv[2]);
  const int T      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const float eps  = 0.1f;

  const int wSz = C * T;
  const int kSz = B * C * T;

  std::vector<float> h_w(wSz), h_k(kSz), h_x(kSz), h_gw(kSz), h_gk(kSz), h_gwk(kSz);
  srand(42);
  for (auto& v : h_w)   v = rand() / (float)RAND_MAX;
  for (auto& v : h_k)   v = rand() / (float)RAND_MAX;
  for (auto& v : h_gwk) v = rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_w  ("w",   wSz);
    Kokkos::View<float*> d_k  ("k",   kSz);
    Kokkos::View<float*> d_x  ("x",   kSz);
    Kokkos::View<float*> d_gw ("gw",  kSz);
    Kokkos::View<float*> d_gk ("gk",  kSz);
    Kokkos::View<float*> d_gwk("gwk", kSz);

    auto cp = [](auto& dst, const std::vector<float>& src) {
      auto hv = Kokkos::create_mirror_view(dst);
      for (int i = 0; i < (int)src.size(); i++) hv(i) = src[i];
      Kokkos::deep_copy(dst, hv);
    };
    cp(d_w, h_w); cp(d_k, h_k); cp(d_gwk, h_gwk);

    timex_forward(d_w, d_k, d_x, eps, B, C, T, repeat);
    timex_backward(d_w, d_k, d_gwk, d_gw, d_gk, B, C, T, repeat);

    // Checksum
    auto hx = Kokkos::create_mirror_view(d_x);
    Kokkos::deep_copy(hx, d_x);
    float sum = 0.f;
    for (int i = 0; i < kSz; i++) sum += hx(i);
    printf("Forward checksum: %f\n", sum / kSz);
  }
  Kokkos::finalize();
  return 0;
}
