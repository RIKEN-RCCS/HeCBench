#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// Three approximations of the Langevin function coth(x) - 1/x

void k0(Kokkos::View<const float*> a, Kokkos::View<float*> o, const int n) {
  Kokkos::parallel_for("langevin_k0", n, KOKKOS_LAMBDA(int t) {
    float x = a(t);
    o(t) = Kokkos::cosh(x) / Kokkos::sinh(x) - 1.f / x;
  });
  Kokkos::fence();
}

void k1(Kokkos::View<const float*> a, Kokkos::View<float*> o, const int n) {
  Kokkos::parallel_for("langevin_k1", n, KOKKOS_LAMBDA(int t) {
    float x = a(t);
    o(t) = 1.f / Kokkos::tanh(x) - 1.f / x;
  });
  Kokkos::fence();
}

/*
Copyright (c) 2018-2021, Norbert Juffa
  All rights reserved.
  ...polynomial approximation of coth(x) - 1/x for small x...
*/
void k2(Kokkos::View<const float*> a, Kokkos::View<float*> o, const int n) {
  Kokkos::parallel_for("langevin_k2", n, KOKKOS_LAMBDA(int t) {
    float x = a(t);
    float s, r;
    s = x * x;
    r =              7.70960469e-8f;
    r = r * s + (-1.65101926e-6f);
    r = r * s +  2.03457112e-5f;
    r = r * s + (-2.10521728e-4f);
    r = r * s +  2.11580913e-3f;
    r = r * s + (-2.22220998e-2f);
    r = r * s +  8.33333284e-2f;
    r = r * x +  0.25f * x;
    o(t) = r;
  });
  Kokkos::fence();
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage %s <n> <repeat>\n", argv[0]);
    return 1;
  }

  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const size_t size = sizeof(float) * n;

  float *a, *o, *o0, *o1, *o2;
  a  = (float*) malloc(size);
  o  = (float*) malloc(size);
  o0 = (float*) malloc(size);
  o1 = (float*) malloc(size);
  o2 = (float*) malloc(size);

  // the range [-1.8, -0.00001)
  for (int i = 0; i < n; i++) {
    a[i] = -1.8f + i * (1.79999f / n);
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_a("d_a", n);
    Kokkos::View<float*> d_o0("d_o0", n);
    Kokkos::View<float*> d_o1("d_o1", n);
    Kokkos::View<float*> d_o2("d_o2", n);

    auto h_a = Kokkos::create_mirror_view(d_a);
    for (int i = 0; i < n; i++) h_a(i) = a[i];
    Kokkos::deep_copy(d_a, h_a);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) k0(d_a, d_o0, n);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of k0: %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) k1(d_a, d_o1, n);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of k1: %f (s)\n", (time * 1e-9f) / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) k2(d_a, d_o2, n);
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of k2: %f (s)\n", (time * 1e-9f) / repeat);

    auto h_o0 = Kokkos::create_mirror_view(d_o0);
    auto h_o1 = Kokkos::create_mirror_view(d_o1);
    auto h_o2 = Kokkos::create_mirror_view(d_o2);
    Kokkos::deep_copy(h_o0, d_o0);
    Kokkos::deep_copy(h_o1, d_o1);
    Kokkos::deep_copy(h_o2, d_o2);
    for (int i = 0; i < n; i++) {
      o0[i] = h_o0(i);
      o1[i] = h_o1(i);
      o2[i] = h_o2(i);
    }
  }
  Kokkos::finalize();

  // https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions
  // Taylor series reference: L(x) = x/3 - x^3/45 + 2*x^5/945 - x^7/4725 + ...
  for (int i = 0; i < n; i++) {
    float x = a[i];
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;
    o[i] = x * (1.f/3.f - 1.f/45.f * x2 + 2.f/945.f * x4 - 1.f/4725.f * x6);
  }

  float e[3] = {0, 0, 0};
  for (int i = 0; i < n; i++) {
    e[0] += (o[i] - o0[i]) * (o[i] - o0[i]);
    e[1] += (o[i] - o1[i]) * (o[i] - o1[i]);
    e[2] += (o[i] - o2[i]) * (o[i] - o2[i]);
  }

  printf("\nError statistics for the kernels:\n");
  for (int i = 0; i < 3; i++) {
    printf("%f ", sqrtf(e[i]));
  }
  printf("\n");

  free(a);
  free(o);
  free(o0);
  free(o1);
  free(o2);
  return 0;
}
