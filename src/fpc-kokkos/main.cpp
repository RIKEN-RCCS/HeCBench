#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef unsigned long ulong;

static ulong* convertBuffer2Array(char *cbuffer, unsigned size, unsigned step) {
  ulong *values = NULL;
  posix_memalign((void**)&values, 1024, sizeof(ulong) * size / step);
  for (unsigned i = 0; i < size / step; i++) values[i] = 0;
  for (unsigned i = 0; i < size; i += step)
    for (unsigned j = 0; j < step; j++)
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8 * j);
  return values;
}

KOKKOS_INLINE_FUNCTION unsigned my_abs(int x) {
  unsigned t = (unsigned)(x >> 31);
  return (unsigned)((x ^ (int)t) - (int)t);
}

static unsigned FPCCompress(ulong *values, unsigned size) {
  unsigned compressable = 0;
  for (unsigned i = 0; i < size; i++) {
    if (values[i] == 0) { compressable += 1; continue; }
    if (my_abs((int)(values[i])) <= 0xFF) { compressable += 1; continue; }
    if (my_abs((int)(values[i])) <= 0xFFFF) { compressable += 2; continue; }
    if (((values[i]) & 0xFFFF) == 0) { compressable += 2; continue; }
    if (my_abs((int)((values[i]) & 0xFFFF)) <= 0xFF &&
        my_abs((int)((values[i] >> 16) & 0xFFFF)) <= 0xFF) { compressable += 2; continue; }
    unsigned byte0 = (values[i]) & 0xFF;
    unsigned byte1 = (values[i] >> 8) & 0xFF;
    unsigned byte2 = (values[i] >> 16) & 0xFF;
    unsigned byte3 = (values[i] >> 24) & 0xFF;
    if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) { compressable += 1; continue; }
    compressable += 4;
  }
  return compressable;
}

// Compute compressed increment for a single value (fpc logic)
KOKKOS_INLINE_FUNCTION unsigned compute_inc_fpc(ulong value) {
  if (value == 0) return 1u;
  if (my_abs((int)(value)) <= 0xFF) return 1u;
  if (my_abs((int)(value)) <= 0xFFFF) return 2u;
  if ((value & 0xFFFF) == 0) return 2u;
  if (my_abs((int)(value & 0xFFFF)) <= 0xFF &&
      my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) return 2u;
  if (((value & 0xFF) == ((value >> 8) & 0xFF)) &&
      ((value & 0xFF) == ((value >> 16) & 0xFF)) &&
      ((value & 0xFF) == ((value >> 24) & 0xFF))) return 1u;
  return 4u;
}

// fpc2 variant: evaluate all predicates, pick first matching
KOKKOS_INLINE_FUNCTION unsigned compute_inc_fpc2(ulong value) {
  // f1: value == 0
  if (value == 0) return 1u;
  // f2: |value| <= 0xFF
  if (my_abs((int)(value)) <= 0xFF) return 1u;
  // f3: |value| <= 0xFFFF
  if (my_abs((int)(value)) <= 0xFFFF) return 2u;
  // f4: lower 16 bits == 0
  if ((value & 0xFFFF) == 0) return 2u;
  // f5: both 16-bit halves <= 0xFF
  if (my_abs((int)(value & 0xFFFF)) <= 0xFF &&
      my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) return 2u;
  // f6: all 4 bytes equal
  if (((value & 0xFF) == ((value >> 8) & 0xFF)) &&
      ((value & 0xFF) == ((value >> 16) & 0xFF)) &&
      ((value & 0xFF) == ((value >> 24) & 0xFF))) return 1u;
  // f7: else
  return 4u;
}

static void run_fpc(const ulong* values, unsigned values_size, unsigned* cmp_size_hw) {
  Kokkos::View<ulong*> d_values("d_values", values_size);
  {
    auto h = Kokkos::create_mirror_view(d_values);
    for (unsigned i = 0; i < values_size; i++) h(i) = values[i];
    Kokkos::deep_copy(d_values, h);
  }

  unsigned total = 0;
  Kokkos::parallel_reduce("fpc",
    Kokkos::RangePolicy<>(0, (int)values_size),
    KOKKOS_LAMBDA(int i, unsigned& sum) {
      sum += compute_inc_fpc(d_values(i));
    }, total);
  Kokkos::fence();
  *cmp_size_hw = total;
}

static void run_fpc2(const ulong* values, unsigned values_size, unsigned* cmp_size_hw) {
  Kokkos::View<ulong*> d_values("d_values", values_size);
  {
    auto h = Kokkos::create_mirror_view(d_values);
    for (unsigned i = 0; i < values_size; i++) h(i) = values[i];
    Kokkos::deep_copy(d_values, h);
  }

  unsigned total = 0;
  Kokkos::parallel_reduce("fpc2",
    Kokkos::RangePolicy<>(0, (int)values_size),
    KOKKOS_LAMBDA(int i, unsigned& sum) {
      sum += compute_inc_fpc2(d_values(i));
    }, total);
  Kokkos::fence();
  *cmp_size_hw = total;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <work-group-size> <repeat>\n", argv[0]);
    return 1;
  }
  const int wgs    = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int step  = 4;
  const size_t size = (size_t)wgs * wgs * wgs;
  char* cbuffer = (char*)malloc(size * step);

  srand(2);
  for (size_t i = 0; i < size * step; i++)
    cbuffer[i] = (char)(0xFF << (rand() % 256));

  ulong *values = convertBuffer2Array(cbuffer, (unsigned)size, step);
  unsigned values_size = (unsigned)(size / step);

  unsigned cmp_size = FPCCompress(values, values_size);
  unsigned cmp_size_hw;

  Kokkos::initialize(argc, argv);
  {
    bool ok = true;

    // Warmup fpc
    run_fpc(values, values_size, &cmp_size_hw);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
      run_fpc(values, values_size, &cmp_size_hw);
      if (cmp_size_hw != cmp_size) {
        printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
        ok = false;
        break;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("fpc: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

    // Warmup fpc2
    run_fpc2(values, values_size, &cmp_size_hw);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
      run_fpc2(values, values_size, &cmp_size_hw);
      if (cmp_size_hw != cmp_size) {
        printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
        ok = false;
        break;
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("fpc2: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();

  free(values);
  free(cbuffer);
  return 0;
}
