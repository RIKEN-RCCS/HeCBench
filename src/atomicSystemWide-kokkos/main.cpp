/*
 * Kokkos port of atomicSystemWide benchmark.
 * Original used CUDA atomicAdd_system, atomicExch_system etc. on unified memory.
 * Kokkos provides standard atomics (not system-wide); we use those here.
 * The verify() logic is identical to the original.
 */

#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <Kokkos_Core.hpp>

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

int verify(int *atom_arr, const int len, const int loop_num) {
  int val = 0;
  for (int i = 0; i < len * loop_num; i++) val += 10;
  if (val != atom_arr[0]) { printf("atomicAdd failed val=%d data=%d\n", val, atom_arr[0]); return 0; }

  bool found = false;
  for (int i = 0; i < len; i++) if (i == atom_arr[1]) { found = true; break; }
  if (!found) { printf("atomicExch failed\n"); return 0; }

  val = -(1 << 8);
  for (int i = 0; i < len; i++) val = MYMAX(val, i);
  if (val != atom_arr[2]) { printf("atomicMax failed\n"); return 0; }

  val = 1 << 8;
  for (int i = 0; i < len; i++) val = MYMIN(val, i);
  if (val != atom_arr[3]) { printf("atomicMin failed\n"); return 0; }

  found = false;
  for (int i = 0; i < len; i++) if (i == atom_arr[6]) { found = true; break; }
  if (!found) { printf("atomicCAS failed\n"); return 0; }

  val = 0xff;
  for (int i = 0; i < len; i++) val &= (2 * i + 7);
  if (val != atom_arr[7]) { printf("atomicAnd failed\n"); return 0; }

  val = 0;
  for (int i = 0; i < len; i++) val |= (1 << i);
  if (val != atom_arr[8]) { printf("atomicOr failed\n"); return 0; }

  val = 0xff;
  for (int i = 0; i < len; i++) val ^= i;
  if (val != atom_arr[9]) { printf("atomicXor failed\n"); return 0; }

  return 1;
}

int main(int argc, char** argv) {
  if (argc != 2) { printf("Usage: %s <loop count>\n", argv[0]); return 1; }
  const int loop_num = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const unsigned int numThreads = 256;
    const unsigned int numBlocks  = 64;
    const unsigned int numData    = 10;
    const int N = numThreads * numBlocks;

    Kokkos::View<int*> d_arr("atom_arr", numData);
    auto h_arr = Kokkos::create_mirror_view(d_arr);
    for (unsigned i = 0; i < numData; i++) h_arr(i) = 0;
    h_arr(7) = h_arr(9) = 0xff;
    Kokkos::deep_copy(d_arr, h_arr);

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    // GPU kernel
    Kokkos::parallel_for("atomicKernel", N, KOKKOS_LAMBDA(int tid) {
      for (int i = 0; i < loop_num; i++) {
        Kokkos::atomic_fetch_add(&d_arr(0), 10);
        Kokkos::atomic_exchange(&d_arr(1), tid);
        Kokkos::atomic_fetch_max(&d_arr(2), tid);
        Kokkos::atomic_fetch_min(&d_arr(3), tid);
        // CAS: swap d_arr[6] from tid-1 to tid
        int expected = tid - 1;
        Kokkos::atomic_compare_exchange(&d_arr(6), expected, tid);
        Kokkos::atomic_fetch_and(&d_arr(7), 2 * tid + 7);
        Kokkos::atomic_fetch_or(&d_arr(8), 1 << tid);
        Kokkos::atomic_fetch_xor(&d_arr(9), tid);
      }
    });
    Kokkos::fence();

    // CPU side (simulating the CPU kernel from original)
    Kokkos::deep_copy(h_arr, d_arr);
    for (unsigned i = N; i < 2*N; i++) {
      for (int j = 0; j < loop_num; j++) {
        __sync_fetch_and_add(&h_arr(0), 10);
        __sync_lock_test_and_set(&h_arr(1), (int)i);
        int old, exp;
        do { exp = h_arr(2); old = __sync_val_compare_and_swap(&h_arr(2), exp, MYMAX(exp, (int)i)); } while (old != exp);
        do { exp = h_arr(3); old = __sync_val_compare_and_swap(&h_arr(3), exp, MYMIN(exp, (int)i)); } while (old != exp);
        __sync_val_compare_and_swap(&h_arr(6), (int)i-1, (int)i);
        __sync_fetch_and_and(&h_arr(7), 2*(int)i+7);
        __sync_fetch_and_or(&h_arr(8), 1<<(int)i);
        __sync_fetch_and_xor(&h_arr(9), (int)i);
      }
    }
    Kokkos::deep_copy(d_arr, h_arr);
    Kokkos::fence();

    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Execution time of atomic kernels on host and device: %f (s)\n", ns * 1e-9f);

    Kokkos::deep_copy(h_arr, d_arr);
    int testResult = verify(h_arr.data(), 2 * N, loop_num);
    printf("systemWideAtomics completed, returned %s\n", testResult ? "OK" : "ERROR!");
  }
  Kokkos::finalize();
  return 0;
}
