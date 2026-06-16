#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

template <typename T>
void ref_nchw(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* sum,
    T* sumsq)
{
  for (int c = 0; c < C; c++) {
    T m_val = 0, v_val = 0;
    for (int n = 0; n < N; n++) {
      for (int hw = 0; hw < HxW; hw++) {
        const int index = (n * C + c) * HxW + hw;
        m_val += X[index];
        v_val += X[index] * X[index];
      }
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
void ref_nhwc(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* sum,
    T* sumsq)
{
  for (int c = 0; c < C; c++) {
    T m_val = 0, v_val = 0;
    for (int i = 0; i < N * HxW; i++) {
      const int index = i * C + c;
      m_val += X[index];
      v_val += X[index] * X[index];
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
bool check(int size, T* d, T* h) {
  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (abs(d[i] - h[i]) > 1) {
      ok = false;
      break;
    }
  }
  return ok;
}

template <typename T>
void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    Kokkos::View<const T*> d_X,
    Kokkos::View<T*> d_sum,
    Kokkos::View<T*> d_sumsq)
{
  Kokkos::parallel_for("ChannelSumNCHW", C,
    KOKKOS_LAMBDA(const int c) {
      T m_val = 0, v_val = 0;
      for (int n = 0; n < N; n++) {
        for (int hw = 0; hw < HxW; hw++) {
          const int index = (n * C + c) * HxW + hw;
          m_val += d_X(index);
          v_val += d_X(index) * d_X(index);
        }
      }
      d_sum(c) = m_val;
      d_sumsq(c) = v_val;
    });
  Kokkos::fence();
}

template <typename T>
void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    Kokkos::View<const T*> d_X,
    Kokkos::View<T*> d_sum,
    Kokkos::View<T*> d_sumsq)
{
  Kokkos::parallel_for("ChannelSumNHWC", C,
    KOKKOS_LAMBDA(const int c) {
      T m_val = 0, v_val = 0;
      for (int i = 0; i < N * HxW; i++) {
        const int index = i * C + c;
        m_val += d_X(index);
        v_val += d_X(index) * d_X(index);
      }
      d_sum(c) = m_val;
      d_sumsq(c) = v_val;
    });
  Kokkos::fence();
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int W = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  Kokkos::initialize(argc, argv);
  {
    for (int N = 1; N <= 64; N = N * 4) {
      for (int C = 32; C <= 512; C = C * 4) {

        printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

        int numel = N * C * W * H;

        scalar_t* h_X    = (scalar_t*) malloc(numel * sizeof(scalar_t));
        scalar_t* h_sum  = (scalar_t*) malloc(C * sizeof(scalar_t));
        scalar_t* h_sumsq = (scalar_t*) malloc(C * sizeof(scalar_t));
        scalar_t* r_sum  = (scalar_t*) malloc(C * sizeof(scalar_t));

        srand(numel);
        for (int i = 0; i < numel; i++) h_X[i] = rand() % 256;

        Kokkos::View<scalar_t*> d_X("d_X", numel);
        Kokkos::View<scalar_t*> d_sum("d_sum", C);
        Kokkos::View<scalar_t*> d_sumsq("d_sumsq", C);

        auto h_d_X = Kokkos::create_mirror_view(d_X);
        for (int i = 0; i < numel; i++) h_d_X(i) = h_X[i];
        Kokkos::deep_copy(d_X, h_d_X);
        Kokkos::fence();

        long time;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          ChannelSumNHWC<scalar_t>(N, C, W * H,
            Kokkos::View<const scalar_t*>(d_X), d_sum, d_sumsq);
        }
        auto end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        auto h_sum_mir = Kokkos::create_mirror_view(d_sum);
        Kokkos::deep_copy(h_sum_mir, d_sum);
        for (int i = 0; i < C; i++) h_sum[i] = h_sum_mir(i);

        ref_nhwc(N, C, W * H, h_X, r_sum, h_sumsq);
        bool ok = check(C, h_sum, r_sum);
        printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
        printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL");

        start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; i++) {
          ChannelSumNCHW<scalar_t>(N, C, W * H,
            Kokkos::View<const scalar_t*>(d_X), d_sum, d_sumsq);
        }
        end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        Kokkos::deep_copy(h_sum_mir, d_sum);
        for (int i = 0; i < C; i++) h_sum[i] = h_sum_mir(i);

        ref_nchw(N, C, W * H, h_X, r_sum, h_sumsq);
        ok = check(C, h_sum, r_sum);
        printf("Average time of channel sum (nchw): %f (ms)\n", (time * 1e-6f) / repeat);
        printf("Verification %s for channel sum (nchw)\n", ok ? "PASS" : "FAIL");

        free(h_X);
        free(h_sum);
        free(h_sumsq);
        free(r_sum);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
