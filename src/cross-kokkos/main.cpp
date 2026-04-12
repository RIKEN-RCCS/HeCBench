/*
 * Cross product of 3D vectors.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

template <typename T>
void cross_kernel(int numel, T* out, const T* x1, const T* x2,
                  int ostride, int x1stride, int x2stride)
{
  Kokkos::parallel_for("cross1", numel, KOKKOS_LAMBDA(int i) {
    T* out_row       = out + 3 * i;
    const T* x1_row  = x1 + 3 * i;
    const T* x2_row  = x2 + 3 * i;

    out_row[0 * ostride] = x1_row[1 * x1stride] * x2_row[2 * x2stride]
                         - x1_row[2 * x1stride] * x2_row[1 * x2stride];
    out_row[1 * ostride] = x1_row[2 * x1stride] * x2_row[0 * x2stride]
                         - x1_row[0 * x1stride] * x2_row[2 * x2stride];
    out_row[2 * ostride] = x1_row[0 * x1stride] * x2_row[1 * x2stride]
                         - x1_row[1 * x1stride] * x2_row[0 * x2stride];
  });
  Kokkos::fence();
}

template <typename T>
void cross3_kernel(int numel, T* out, const T* x1, const T* x2)
{
  Kokkos::parallel_for("cross3", numel, KOKKOS_LAMBDA(int i) {
    T* out_row       = out + 3 * i;
    const T* x1_row  = x1 + 3 * i;
    const T* x2_row  = x2 + 3 * i;

    T x1_c0 = x1_row[0], x1_c1 = x1_row[1], x1_c2 = x1_row[2];
    T x2_c0 = x2_row[0], x2_c1 = x2_row[1], x2_c2 = x2_row[2];

    out_row[0] = x1_c1 * x2_c2 - x1_c2 * x2_c1;
    out_row[1] = x1_c2 * x2_c0 - x1_c0 * x2_c2;
    out_row[2] = x1_c0 * x2_c1 - x1_c1 * x2_c0;
  });
  Kokkos::fence();
}

template <typename T>
void eval(int nrows, int repeat) {
  int num_elems  = nrows * 3;
  size_t size_bytes = num_elems * sizeof(T);

  T *a = (T*) malloc(size_bytes);
  T *b = (T*) malloc(size_bytes);
  T *o = (T*) malloc(size_bytes);
  T *o2 = (T*) malloc(size_bytes);
  T *o3 = (T*) malloc(size_bytes);

  std::default_random_engine g(123);
  std::uniform_real_distribution<T> distr(-2.f, 2.f);
  for (int i = 0; i < num_elems; i++) { a[i] = distr(g); b[i] = distr(g); }

  Kokkos::View<T*> d_a("d_a", num_elems);
  Kokkos::View<T*> d_b("d_b", num_elems);
  Kokkos::View<T*> d_o("d_o", num_elems);
  Kokkos::View<T*> d_o2("d_o2", num_elems);
  Kokkos::View<T*> d_o3("d_o3", num_elems);

  auto h_a = Kokkos::create_mirror_view(d_a);
  auto h_b = Kokkos::create_mirror_view(d_b);
  for (int i = 0; i < num_elems; i++) { h_a(i) = a[i]; h_b(i) = b[i]; }
  Kokkos::deep_copy(d_a, h_a);
  Kokkos::deep_copy(d_b, h_b);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    cross_kernel<T>(nrows, d_o.data(), d_a.data(), d_b.data(), 1, 1, 1);
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of cross1 kernel: %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    cross_kernel<T>(nrows, d_o2.data(), d_a.data(), d_b.data(), 1, 1, 1);
  end = std::chrono::steady_clock::now();
  printf("Average execution time of cross2 kernel: %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    cross3_kernel<T>(nrows, d_o3.data(), d_a.data(), d_b.data());
  end = std::chrono::steady_clock::now();
  printf("Average execution time of cross3 kernel: %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-3f) / repeat);

  auto h_o  = Kokkos::create_mirror_view(d_o);
  auto h_o2 = Kokkos::create_mirror_view(d_o2);
  auto h_o3 = Kokkos::create_mirror_view(d_o3);
  Kokkos::deep_copy(h_o,  d_o);
  Kokkos::deep_copy(h_o2, d_o2);
  Kokkos::deep_copy(h_o3, d_o3);
  for (int i = 0; i < num_elems; i++) { o[i] = h_o(i); o2[i] = h_o2(i); o3[i] = h_o3(i); }

  bool ok = true;
  for (int i = 0; i < num_elems; i++) {
    if (fabs(o[i] - o2[i]) > 1e-3 || fabs(o[i] - o3[i]) > 1e-3) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(a); free(b); free(o); free(o2); free(o3);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows  = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Kokkos::initialize(argc, argv);
  {
    printf("=========== Data type is FP32 ==========\n");
    eval<float>(nrows, repeat);
    printf("=========== Data type is FP64 ==========\n");
    eval<double>(nrows, repeat);
  }
  Kokkos::finalize();
  return 0;
}
