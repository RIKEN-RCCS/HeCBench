#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

// CAS-based atomic min for integral types
template <typename T>
KOKKOS_INLINE_FUNCTION
T atomic_min_cas(T* address, T val) {
  T ret = *address;
  while (val < ret) {
    T old = ret;
    ret = Kokkos::atomic_compare_exchange(address, old, val);
    if (ret == old) break;
  }
  return ret;
}

// CAS-based atomic max for integral types
template <typename T>
KOKKOS_INLINE_FUNCTION
T atomic_max_cas(T* address, T val) {
  T ret = *address;
  while (val > ret) {
    T old = ret;
    ret = Kokkos::atomic_compare_exchange(address, old, val);
    if (ret == old) break;
  }
  return ret;
}

// CAS-based atomic add for integral types
template <typename T>
KOKKOS_INLINE_FUNCTION
T atomic_add_cas(T* address, T val) {
  T old, newval, ret = *address;
  do {
    old    = ret;
    newval = old + val;
    ret    = Kokkos::atomic_compare_exchange(address, old, newval);
  } while (ret != old);
  return ret;
}

// Double min via CAS on the bit representation
KOKKOS_INLINE_FUNCTION
double atomic_min_double(double* address, double val) {
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long val_bits;
  memcpy(&val_bits, &val, sizeof(double));
  unsigned long long ret = *ptr;
  double ret_d;
  memcpy(&ret_d, &ret, sizeof(double));
  while (val < ret_d) {
    unsigned long long old_bits = ret;
    ret = Kokkos::atomic_compare_exchange(ptr, old_bits, val_bits);
    if (ret == old_bits) break;
    memcpy(&ret_d, &ret, sizeof(double));
  }
  double result;
  memcpy(&result, &ret, sizeof(double));
  return result;
}

// Double max via CAS on the bit representation
KOKKOS_INLINE_FUNCTION
double atomic_max_double(double* address, double val) {
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long val_bits;
  memcpy(&val_bits, &val, sizeof(double));
  unsigned long long ret = *ptr;
  double ret_d;
  memcpy(&ret_d, &ret, sizeof(double));
  while (val > ret_d) {
    unsigned long long old_bits = ret;
    ret = Kokkos::atomic_compare_exchange(ptr, old_bits, val_bits);
    if (ret == old_bits) break;
    memcpy(&ret_d, &ret, sizeof(double));
  }
  double result;
  memcpy(&result, &ret, sizeof(double));
  return result;
}

// Double add via CAS on the bit representation
KOKKOS_INLINE_FUNCTION
double atomic_add_double(double* address, double val) {
  unsigned long long* ptr = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old_bits, new_bits, ret = *ptr;
  do {
    old_bits = ret;
    double old_d;
    memcpy(&old_d, &old_bits, sizeof(double));
    double new_d = old_d + val;
    memcpy(&new_bits, &new_d, sizeof(double));
    ret = Kokkos::atomic_compare_exchange(ptr, old_bits, new_bits);
  } while (ret != old_bits);
  double result;
  memcpy(&result, &ret, sizeof(double));
  return result;
}

// ---- Test functions -------------------------------------------------------

template <typename T>
void testMin(T* h_ptr, const int repeat, const char* name) {
  Kokkos::View<T*> d_val("d_val_min", 1);
  auto hv = Kokkos::create_mirror_view(d_val);
  hv(0) = *h_ptr;
  Kokkos::deep_copy(d_val, hv);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomicMin", NUM_BLOCKS * BLOCK_SIZE,
      KOKKOS_LAMBDA(int idx) {
        T i = (T)(idx + 1);
        atomic_min_cas(&d_val(0), i);
      });
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  Kokkos::deep_copy(hv, d_val);
  *h_ptr = hv(0);
}

// Specialisation for double (uses bit-cast CAS)
template <>
void testMin<double>(double* h_ptr, const int repeat, const char* name) {
  Kokkos::View<double*> d_val("d_val_min_d", 1);
  auto hv = Kokkos::create_mirror_view(d_val);
  hv(0) = *h_ptr;
  Kokkos::deep_copy(d_val, hv);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomicMinD", NUM_BLOCKS * BLOCK_SIZE,
      KOKKOS_LAMBDA(int idx) {
        double i = (double)(idx + 1);
        atomic_min_double(&d_val(0), i);
      });
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  Kokkos::deep_copy(hv, d_val);
  *h_ptr = hv(0);
}

template <typename T>
void testMax(T* h_ptr, const int repeat, const char* name) {
  Kokkos::View<T*> d_val("d_val_max", 1);
  auto hv = Kokkos::create_mirror_view(d_val);
  hv(0) = *h_ptr;
  Kokkos::deep_copy(d_val, hv);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomicMax", NUM_BLOCKS * BLOCK_SIZE,
      KOKKOS_LAMBDA(int idx) {
        T i = (T)(idx + 1);
        atomic_max_cas(&d_val(0), i);
      });
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  Kokkos::deep_copy(hv, d_val);
  *h_ptr = hv(0);
}

template <>
void testMax<double>(double* h_ptr, const int repeat, const char* name) {
  Kokkos::View<double*> d_val("d_val_max_d", 1);
  auto hv = Kokkos::create_mirror_view(d_val);
  hv(0) = *h_ptr;
  Kokkos::deep_copy(d_val, hv);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomicMaxD", NUM_BLOCKS * BLOCK_SIZE,
      KOKKOS_LAMBDA(int idx) {
        double i = (double)(idx + 1);
        atomic_max_double(&d_val(0), i);
      });
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  Kokkos::deep_copy(hv, d_val);
  *h_ptr = hv(0);
}

template <typename T>
void testAdd(T* h_ptr, const int repeat, const char* name) {
  Kokkos::View<T*> d_val("d_val_add", 1);
  auto hv = Kokkos::create_mirror_view(d_val);
  hv(0) = *h_ptr;
  Kokkos::deep_copy(d_val, hv);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomicAdd", NUM_BLOCKS * BLOCK_SIZE,
      KOKKOS_LAMBDA(int idx) {
        T i = (T)(idx + 1);
        atomic_add_cas(&d_val(0), i);
      });
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  Kokkos::deep_copy(hv, d_val);
  *h_ptr = hv(0);
}

template <>
void testAdd<double>(double* h_ptr, const int repeat, const char* name) {
  Kokkos::View<double*> d_val("d_val_add_d", 1);
  auto hv = Kokkos::create_mirror_view(d_val);
  hv(0) = *h_ptr;
  Kokkos::deep_copy(d_val, hv);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("atomicAddD", NUM_BLOCKS * BLOCK_SIZE,
      KOKKOS_LAMBDA(int idx) {
        double i = (double)(idx + 1);
        atomic_add_double(&d_val(0), i);
      });
  }

  Kokkos::fence();
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  Kokkos::deep_copy(hv, d_val);
  *h_ptr = hv(0);
}

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    unsigned long long res_u64[3] = {ULONG_MAX, 0, 0};
    long long          res_s64[3] = {LONG_MAX, LONG_MIN, 0};
    double             res_f64[3] = {DBL_MAX, DBL_MIN, 0.0};

    testMin<unsigned long long>(res_u64,     repeat, "U64");
    testMin<long long>         (res_s64,     repeat, "S64");
    testMin<double>            (res_f64,     repeat, "F64");

    testMax<unsigned long long>(res_u64 + 1, repeat, "U64");
    testMax<long long>         (res_s64 + 1, repeat, "S64");
    testMax<double>            (res_f64 + 1, repeat, "F64");

    // Add kernels are slow — run only once
    testAdd<unsigned long long>(res_u64 + 2, 1, "U64");
    testAdd<long long>         (res_s64 + 2, 1, "S64");
    testAdd<double>            (res_f64 + 2, 1, "F64");

    const unsigned long long bound = (unsigned long long)NUM_BLOCKS * BLOCK_SIZE;
    unsigned long long sum = 0;
    for (unsigned long long i = 1; i <= bound; i++) sum += i;

    bool error = false;
    if (res_u64[0] != 1ULL || res_s64[0] != 1LL || res_f64[0] != 1.0) {
      error = true;
      printf("atomic min results: %llu %lld %lf\n",
             res_u64[0], res_s64[0], res_f64[0]);
    }
    if (res_u64[1] != bound ||
        res_s64[1] != (long long)bound ||
        res_f64[1] != (double)bound) {
      error = true;
      printf("atomic max results: %llu %lld %lf\n",
             res_u64[1], res_s64[1], res_f64[1]);
    }
    if (res_u64[2] != sum ||
        res_s64[2] != (long long)sum ||
        res_f64[2] != (double)sum) {
      error = true;
      printf("atomic add results: %llu %lld %lf\n",
             res_u64[2], res_s64[2], res_f64[2]);
    }
    printf("%s\n", error ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return 0;
}
