#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <Kokkos_Core.hpp>

#define P_x 5
#define P_y 1
#define MODULUS 17
#define A_COEFF 2

KOKKOS_INLINE_FUNCTION
int ext_euclidian_alg(int a, int b, int& x, int& y)
{
  x = 1; y = 0;
  int x1 = 0, y1 = 1, a1 = a, b1 = b;
  int s, t;
  while (b1) {
    int q = a1 / b1;
    s = x1; t = x - q * x1;
    x = s; x1 = t;
    s = y1; t = y - q * y1;
    y = s; y1 = t;
    s = b1; t = a1 - q * b1;
    a1 = s; b1 = t;
  }
  return a1;
}

KOKKOS_INLINE_FUNCTION
unsigned int make_positive(int a, unsigned int m)
{
  while (a < 0) a += m;
  return (unsigned int)a % m;
}

KOKKOS_INLINE_FUNCTION
int find_inverse(int a, unsigned int m)
{
  int t, s;
  ext_euclidian_alg(a, (int)m, t, s);
  return (int)make_positive(t, m);
}

KOKKOS_INLINE_FUNCTION
void point_addition(unsigned int m, int x1, int y1, int x2, int y2, int* x3, int* y3)
{
  int temp = (int)make_positive(x2 - x1, m);
  int slope = (int)make_positive((y2 - y1) * find_inverse(temp, m), m);
  *x3 = (int)make_positive(slope * slope - x1 - x2, m);
  *y3 = (int)make_positive(slope * (x1 - *x3) - y1, m);
}

KOKKOS_INLINE_FUNCTION
void point_doubling(unsigned int m, int a, int x1, int y1, int* x3, int* y3)
{
  int slope = (3 * x1 * x1 + a) * find_inverse(2 * y1, m);
  *x3 = (int)make_positive(slope * slope - 2 * x1, m);
  *y3 = (int)make_positive(slope * (x1 - *x3) - y1, m);
}

KOKKOS_INLINE_FUNCTION
int first_set_bit(int n)
{
  for (int i = (int)(sizeof(int) * 8) - 1; i >= 0; --i)
    if ((1 << i) & n) return i;
  return 0;
}

KOKKOS_INLINE_FUNCTION
void make_pk_fast(int sk, int P_x, int P_y, int* T_x, int* T_y, unsigned int m, int a)
{
  *T_x = P_x;
  *T_y = P_y;
  for (int i = first_set_bit(sk) - 1; i >= 0; --i) {
    point_doubling(m, a, *T_x, *T_y, T_x, T_y);
    if ((1 << i) & sk)
      point_addition(m, *T_x, *T_y, P_x, P_y, T_x, T_y);
  }
}

KOKKOS_INLINE_FUNCTION
void make_pk_slow(int sk, int P_x, int P_y, int* T_x, int* T_y, unsigned int m, int a)
{
  point_doubling(m, a, P_x, P_y, T_x, T_y);
  sk -= 2;
  while (sk > 0) {
    point_addition(m, *T_x, *T_y, P_x, P_y, T_x, T_y);
    --sk;
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <positive number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_pk = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int* pk_slow_x = (int*)malloc(num_pk * sizeof(int));
  int* pk_slow_y = (int*)malloc(num_pk * sizeof(int));
  int* pk_fast_x = (int*)malloc(num_pk * sizeof(int));
  int* pk_fast_y = (int*)malloc(num_pk * sizeof(int));

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> d_slow_x("slow_x", num_pk);
    Kokkos::View<int*> d_slow_y("slow_y", num_pk);
    Kokkos::View<int*> d_fast_x("fast_x", num_pk);
    Kokkos::View<int*> d_fast_y("fast_y", num_pk);

    // slow kernel timing
    auto start_slow = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("k_slow", num_pk, KOKKOS_LAMBDA(int i) {
        make_pk_slow(18, P_x, P_y, &d_slow_x(i), &d_slow_y(i), MODULUS, A_COEFF);
      });
      Kokkos::fence();
    }
    auto end_slow = std::chrono::steady_clock::now();
    double elapsed_slow = std::chrono::duration<double>(end_slow - start_slow).count();
    printf("Average time (slow kernel): %f s\n", elapsed_slow / repeat);

    // fast kernel timing
    auto start_fast = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("k_fast", num_pk, KOKKOS_LAMBDA(int i) {
        make_pk_fast(18, P_x, P_y, &d_fast_x(i), &d_fast_y(i), MODULUS, A_COEFF);
      });
      Kokkos::fence();
    }
    auto end_fast = std::chrono::steady_clock::now();
    double elapsed_fast = std::chrono::duration<double>(end_fast - start_fast).count();
    printf("Average time (fast kernel): %f s\n", elapsed_fast / repeat);

    // Copy results back
    auto h_slow_x = Kokkos::create_mirror_view(d_slow_x);
    auto h_slow_y = Kokkos::create_mirror_view(d_slow_y);
    auto h_fast_x = Kokkos::create_mirror_view(d_fast_x);
    auto h_fast_y = Kokkos::create_mirror_view(d_fast_y);
    Kokkos::deep_copy(h_slow_x, d_slow_x);
    Kokkos::deep_copy(h_slow_y, d_slow_y);
    Kokkos::deep_copy(h_fast_x, d_fast_x);
    Kokkos::deep_copy(h_fast_y, d_fast_y);

    for (int i = 0; i < num_pk; i++) {
      pk_slow_x[i] = h_slow_x(i);
      pk_slow_y[i] = h_slow_y(i);
      pk_fast_x[i] = h_fast_x(i);
      pk_fast_y[i] = h_fast_y(i);
    }
  }
  Kokkos::finalize();

  bool fail_pk_x = (memcmp(pk_slow_x, pk_fast_x, num_pk * sizeof(int)) != 0);
  bool fail_pk_y = (memcmp(pk_slow_y, pk_fast_y, num_pk * sizeof(int)) != 0);
  printf("%s\n", (fail_pk_x || fail_pk_y) ? "FAIL" : "PASS");

  free(pk_slow_x);
  free(pk_slow_y);
  free(pk_fast_x);
  free(pk_fast_y);
  return 0;
}
