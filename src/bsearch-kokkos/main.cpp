/*
 * Binary search benchmark with multiple variants.
 * Ported to Kokkos from the OMP target version.
 */

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <Kokkos_Core.hpp>

#ifndef Real_t
#define Real_t float
#endif

// bs1: standard binary search
template <typename T>
void bs(size_t aSize, size_t zSize,
        Kokkos::View<T*> d_a, Kokkos::View<T*> d_z, Kokkos::View<size_t*> d_r,
        size_t n, int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("bs1", zSize, KOKKOS_LAMBDA(int i) {
      T z = d_z(i);
      size_t low = 0, high = n;
      while (high - low > 1) {
        size_t mid = low + (high - low) / 2;
        if (z < d_a(mid)) high = mid;
        else low = mid;
      }
      d_r(i) = low;
    });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs1) " << (time * 1e-9f) / repeat << " (s)\n";
}

// bs2: bit-manipulation approach
template <typename T>
void bs2(size_t aSize, size_t zSize,
         Kokkos::View<T*> d_a, Kokkos::View<T*> d_z, Kokkos::View<size_t*> d_r,
         size_t n, int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("bs2", zSize, KOKKOS_LAMBDA(int i) {
      unsigned nbits = 0;
      size_t nn = n;
      while (nn >> nbits) nbits++;
      size_t k = (size_t)1 << (nbits - 1);
      T z = d_z(i);
      size_t idx = (d_a(k) <= z) ? k : 0;
      while (k >>= 1) {
        size_t r = idx | k;
        if (r < n && d_a(r) <= z) idx = r;
      }
      d_r(i) = idx;
    });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs2) " << (time * 1e-9f) / repeat << " (s)\n";
}

// bs3: Eytzinger-layout binary search (we use a simple right-to-left scan)
template <typename T>
void bs3(size_t aSize, size_t zSize,
         Kokkos::View<T*> d_a, Kokkos::View<T*> d_z, Kokkos::View<size_t*> d_r,
         size_t n, int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("bs3", zSize, KOKKOS_LAMBDA(int i) {
      T z = d_z(i);
      size_t low = 0, high = n;
      while (low < high) {
        size_t mid = (low + high) / 2;
        if (d_a(mid) < z) low = mid + 1;
        else high = mid;
      }
      d_r(i) = (low > 0) ? low - 1 : 0;
    });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs3) " << (time * 1e-9f) / repeat << " (s)\n";
}

// bs4: branch-free binary search
template <typename T>
void bs4(size_t aSize, size_t zSize,
         Kokkos::View<T*> d_a, Kokkos::View<T*> d_z, Kokkos::View<size_t*> d_r,
         size_t n, int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    Kokkos::parallel_for("bs4", zSize, KOKKOS_LAMBDA(int i) {
      T z = d_z(i);
      size_t base = 0, len = n;
      while (len > 1) {
        size_t half = len >> 1;
        base += (d_a(base + half) <= z) ? half : 0;
        len -= half;
      }
      d_r(i) = base;
    });
    Kokkos::fence();
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device execution time (bs4) " << (time * 1e-9f) / repeat << " (s)\n";
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <size of a> <size of z> <repeat>\n";
    return 1;
  }
  const size_t aSize = atol(argv[1]);
  const size_t zSize = atol(argv[2]);
  const int    repeat = atoi(argv[3]);
  const size_t N = aSize - 1;

  Real_t *a = (Real_t*) malloc(aSize * sizeof(Real_t));
  Real_t *z = (Real_t*) malloc(zSize * sizeof(Real_t));
  size_t *r = (size_t*) malloc(zSize * sizeof(size_t));

  for (size_t i = 0; i < aSize; i++) a[i] = (Real_t)i;
  srand(123);
  for (size_t i = 0; i < zSize; i++) z[i] = (Real_t)(rand() % N);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<Real_t*> d_a("d_a", aSize);
    Kokkos::View<Real_t*> d_z("d_z", zSize);
    Kokkos::View<size_t*> d_r("d_r", zSize);

    auto h_a = Kokkos::create_mirror_view(d_a);
    auto h_z = Kokkos::create_mirror_view(d_z);
    for (size_t i = 0; i < aSize; i++) h_a(i) = a[i];
    for (size_t i = 0; i < zSize; i++) h_z(i) = z[i];
    Kokkos::deep_copy(d_a, h_a);
    Kokkos::deep_copy(d_z, h_z);

    bs<Real_t> (aSize, zSize, d_a, d_z, d_r, N, repeat);
    bs2<Real_t>(aSize, zSize, d_a, d_z, d_r, N, repeat);
    bs3<Real_t>(aSize, zSize, d_a, d_z, d_r, N, repeat);
    bs4<Real_t>(aSize, zSize, d_a, d_z, d_r, N, repeat);
  }
  Kokkos::finalize();

  free(a); free(z); free(r);
  return 0;
}
