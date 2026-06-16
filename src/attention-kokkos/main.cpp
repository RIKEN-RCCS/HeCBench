// Kokkos port of the attention benchmark.
// Implements single-query attention: output = softmax(key * query^T) * value
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include "reference.h"

// Compute attention using Kokkos.
// Layout follows CUDA original: key[n][d], value[n][d], query[d]
float* attention_device(
    const float* key,
    const float* value,
    const float* query,
    const int n, const int d,
    const int repeat)
{
  // Device views
  Kokkos::View<float*> d_key   ("key",   n * d);
  Kokkos::View<float*> d_value ("value", n * d);
  Kokkos::View<float*> d_query ("query", d);
  Kokkos::View<float*> d_dp    ("dp",    n);   // dot products
  Kokkos::View<float*> d_score ("score", n);
  Kokkos::View<float*> d_out   ("out",   d);

  // Copy host → device
  {
    auto hk = Kokkos::View<const float*, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(key,   n * d);
    auto hv = Kokkos::View<const float*, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(value, n * d);
    auto hq = Kokkos::View<const float*, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(query, d);
    Kokkos::deep_copy(d_key,   hk);
    Kokkos::deep_copy(d_value, hv);
    Kokkos::deep_copy(d_query, hq);
  }

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int k = 0; k < repeat; k++) {
    // ---- Kernel 1: dot products + exp_sum ----
    float exp_sum = 0.f;
    Kokkos::parallel_reduce(
      "attn_k1",
      n,
      KOKKOS_LAMBDA(const int i, float& lsum) {
        float dp = 0.f;
        for (int j = 0; j < d; j++)
          dp += d_key(i * d + j) * d_query(j);
        d_dp(i) = dp;
        lsum += Kokkos::exp(dp);
      },
      exp_sum
    );
    Kokkos::fence();

    // ---- Kernel 2: softmax scores ----
    const float inv_sum = 1.f / exp_sum;
    Kokkos::parallel_for(
      "attn_k2",
      n,
      KOKKOS_LAMBDA(const int i) {
        d_score(i) = Kokkos::exp(d_dp(i)) * inv_sum;
      }
    );
    Kokkos::fence();

    // ---- Kernel 3: weighted sum over values ----
    Kokkos::parallel_for(
      "attn_k3",
      d,
      KOKKOS_LAMBDA(const int j) {
        float sum = 0.f;
        for (int i = 0; i < n; i++)
          sum += d_score(i) * d_value(i * d + j);
        d_out(j) = sum;
      }
    );
    Kokkos::fence();
  }

  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);

  // Copy result back
  float* output = (float*) malloc(d * sizeof(float));
  auto hout = Kokkos::View<float*, Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(output, d);
  Kokkos::deep_copy(hout, d_out);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <rows n> <columns d> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);

  float* key   = (float*) malloc(n * d * sizeof(float));
  float* value = (float*) malloc(n * d * sizeof(float));
  float* query = (float*) malloc(d * sizeof(float));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
  for (int i = 0; i < n * d; i++) {
    key[i]   = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);

  Kokkos::initialize(argc, argv);
  float* dout = attention_device(key, value, query, n, d, r);
  Kokkos::finalize();

  float rmse = 0.f;
  for (int i = 0; i < d; i++)
    rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
  printf("RMSE = %f\n", sqrtf(rmse / d));

  free(key); free(value); free(query);
  free(dout); free(hout);
  return 0;
}
