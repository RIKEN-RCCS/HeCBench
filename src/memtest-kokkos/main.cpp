#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ─── Constants ───────────────────────────────────────────────────────────────
#define TYPE            unsigned long
#define MAX_ERR_RECORD_COUNT 10
#define BLOCKSIZE       (1024*1024)

// ─── Kernel 0: walk-with-bitmask write/read ──────────────────────────────────

void kernel0_write(Kokkos::View<char*> d_mem, unsigned long size) {
  unsigned long n = size / BLOCKSIZE;
  char* ptr = d_mem.data();

  Kokkos::parallel_for("k0_write", n, KOKKOS_LAMBDA(int i) {
    unsigned long* start_p = (unsigned long*)(ptr + (unsigned long)i * BLOCKSIZE);
    unsigned long* end_p   = (unsigned long*)(ptr + (unsigned long)(i + 1) * BLOCKSIZE);
    unsigned long* p       = start_p;
    unsigned int pattern   = 1;
    unsigned int mask      = 8;

    *p = pattern;
    pattern = (pattern << 1);
    while (p < end_p) {
      p = (unsigned long*)(((unsigned long)start_p) | mask);
      if (p == start_p) { mask <<= 1; if (mask == 0) break; continue; }
      if (p >= end_p) break;
      *p = pattern;
      pattern <<= 1;
      mask <<= 1;
      if (mask == 0) break;
    }
  });
}

void kernel0_read(Kokkos::View<const char*> d_mem, unsigned long size,
                  Kokkos::View<unsigned int*>  err_count,
                  Kokkos::View<unsigned long*> err_addr,
                  Kokkos::View<unsigned long*> err_expect,
                  Kokkos::View<unsigned long*> err_current,
                  Kokkos::View<unsigned long*> err_second_read) {
  unsigned long n = size / BLOCKSIZE;
  const char* ptr = d_mem.data();

  Kokkos::parallel_for("k0_read", n, KOKKOS_LAMBDA(int i) {
    unsigned long* start_p = (unsigned long*)(ptr + (unsigned long)i * BLOCKSIZE);
    unsigned long* end_p   = (unsigned long*)(ptr + (unsigned long)(i + 1) * BLOCKSIZE);
    unsigned long* p       = start_p;
    unsigned int pattern   = 1;
    unsigned int mask      = 8;

    if (*p != pattern) {
      unsigned int idx = Kokkos::atomic_fetch_add(&err_count(0), 1u) % MAX_ERR_RECORD_COUNT;
      err_addr(idx)        = (unsigned long)p;
      err_expect(idx)      = (unsigned long)pattern;
      err_current(idx)     = (unsigned long)(*p);
      err_second_read(idx) = (unsigned long)(*p);
    }
    pattern <<= 1;
    while (p < end_p) {
      p = (unsigned long*)(((unsigned long)start_p) | mask);
      if (p == start_p) { mask <<= 1; if (mask == 0) break; continue; }
      if (p >= end_p) break;
      if (*p != pattern) {
        unsigned int idx = Kokkos::atomic_fetch_add(&err_count(0), 1u) % MAX_ERR_RECORD_COUNT;
        err_addr(idx)        = (unsigned long)p;
        err_expect(idx)      = (unsigned long)pattern;
        err_current(idx)     = (unsigned long)(*p);
        err_second_read(idx) = (unsigned long)(*p);
      }
      pattern <<= 1;
      mask    <<= 1;
      if (mask == 0) break;
    }
  });
}

// ─── Kernel 1: address-as-value write/read ───────────────────────────────────

void kernel1_write(Kokkos::View<char*> d_mem, unsigned long size) {
  unsigned long* buf = (unsigned long*)d_mem.data();
  unsigned long  n   = size / sizeof(unsigned long);

  Kokkos::parallel_for("k1_write", (int)n, KOKKOS_LAMBDA(int i) {
    buf[i] = (unsigned long)(buf + i);
  });
}

void kernel1_read(Kokkos::View<const char*> d_mem, unsigned long size,
                  Kokkos::View<unsigned int*>  err_count,
                  Kokkos::View<unsigned long*> err_addr,
                  Kokkos::View<unsigned long*> err_expect,
                  Kokkos::View<unsigned long*> err_current,
                  Kokkos::View<unsigned long*> err_second_read) {
  const unsigned long* buf = (const unsigned long*)d_mem.data();
  unsigned long        n   = size / sizeof(unsigned long);

  Kokkos::parallel_for("k1_read", (int)n, KOKKOS_LAMBDA(int i) {
    if (buf[i] != (unsigned long)(buf + i)) {
      unsigned int idx = Kokkos::atomic_fetch_add(&err_count(0), 1u) % MAX_ERR_RECORD_COUNT;
      err_addr(idx)        = (unsigned long)(buf + i);
      err_expect(idx)      = (unsigned long)(buf + i);
      err_current(idx)     = (unsigned long)buf[i];
      err_second_read(idx) = (unsigned long)buf[i];
    }
  });
}

// ─── Moving-inversion kernels ─────────────────────────────────────────────────

void kernel_write(Kokkos::View<char*> d_mem, unsigned long size, TYPE p1) {
  TYPE*         buf = (TYPE*)d_mem.data();
  unsigned long n   = size / sizeof(TYPE);

  Kokkos::parallel_for("k_write", (int)n, KOKKOS_LAMBDA(int i) {
    buf[i] = p1;
  });
}

void kernel_read_write(Kokkos::View<char*> d_mem, unsigned long size,
                       TYPE p1, TYPE p2,
                       Kokkos::View<unsigned int*>  err_count,
                       Kokkos::View<unsigned long*> err_addr,
                       Kokkos::View<unsigned long*> err_expect,
                       Kokkos::View<unsigned long*> err_current,
                       Kokkos::View<unsigned long*> err_second_read) {
  TYPE*         buf = (TYPE*)d_mem.data();
  unsigned long n   = size / sizeof(TYPE);

  Kokkos::parallel_for("k_rw", (int)n, KOKKOS_LAMBDA(int i) {
    TYPE localp = buf[i];
    if (localp != p1) {
      unsigned int idx = Kokkos::atomic_fetch_add(&err_count(0), 1u) % MAX_ERR_RECORD_COUNT;
      err_addr(idx)        = (unsigned long)(buf + i);
      err_expect(idx)      = (unsigned long)p1;
      err_current(idx)     = (unsigned long)localp;
      err_second_read(idx) = (unsigned long)buf[i];
    }
    buf[i] = p2;
  });
}

void kernel_read(Kokkos::View<const char*> d_mem, unsigned long size, TYPE p1,
                 Kokkos::View<unsigned int*>  err_count,
                 Kokkos::View<unsigned long*> err_addr,
                 Kokkos::View<unsigned long*> err_expect,
                 Kokkos::View<unsigned long*> err_current,
                 Kokkos::View<unsigned long*> err_second_read) {
  const TYPE*   buf = (const TYPE*)d_mem.data();
  unsigned long n   = size / sizeof(TYPE);

  Kokkos::parallel_for("k_read", (int)n, KOKKOS_LAMBDA(int i) {
    TYPE localp = buf[i];
    if (localp != p1) {
      unsigned int idx = Kokkos::atomic_fetch_add(&err_count(0), 1u) % MAX_ERR_RECORD_COUNT;
      err_addr(idx)        = (unsigned long)(buf + i);
      err_expect(idx)      = (unsigned long)p1;
      err_current(idx)     = (unsigned long)localp;
      err_second_read(idx) = (unsigned long)buf[i];
    }
  });
}

// ─── Kernel 5: alternating bit patterns ──────────────────────────────────────

void kernel5_init(Kokkos::View<char*> d_mem, unsigned long size) {
  unsigned int* buf = (unsigned int*)d_mem.data();
  unsigned long n   = size / 64;

  Kokkos::parallel_for("k5_init", (int)n, KOKKOS_LAMBDA(int i) {
    unsigned int p1 = 1u << (i % 32);
    unsigned int p2 = ~p1;
    buf[i * 16 +  0] = p1; buf[i * 16 +  1] = p1;
    buf[i * 16 +  2] = p2; buf[i * 16 +  3] = p2;
    buf[i * 16 +  4] = p1; buf[i * 16 +  5] = p1;
    buf[i * 16 +  6] = p2; buf[i * 16 +  7] = p2;
    buf[i * 16 +  8] = p1; buf[i * 16 +  9] = p1;
    buf[i * 16 + 10] = p2; buf[i * 16 + 11] = p2;
    buf[i * 16 + 12] = p1; buf[i * 16 + 13] = p1;
    buf[i * 16 + 14] = p2; buf[i * 16 + 15] = p2;
  });
}

void kernel5_move(Kokkos::View<char*> d_mem, unsigned long size) {
  unsigned long n          = size / BLOCKSIZE;
  unsigned int  half_count = BLOCKSIZE / sizeof(unsigned int) / 2;
  char*         ptr        = d_mem.data();

  Kokkos::parallel_for("k5_move", (int)n, KOKKOS_LAMBDA(int i) {
    unsigned int* mybuf     = (unsigned int*)(ptr + (unsigned long)i * BLOCKSIZE);
    unsigned int* mybuf_mid = (unsigned int*)(ptr + (unsigned long)i * BLOCKSIZE + BLOCKSIZE / 2);

    for (int j = 0; j < (int)half_count; j++)
      mybuf_mid[j] = mybuf[j];
    for (int j = 0; j < (int)half_count - 8; j++)
      mybuf[j + 8] = mybuf_mid[j];
    for (int j = 0; j < 8; j++)
      mybuf[j] = mybuf_mid[half_count - 8 + j];
  });
}

void kernel5_check(Kokkos::View<const char*> d_mem, unsigned long size,
                   Kokkos::View<unsigned int*>  err_count,
                   Kokkos::View<unsigned long*> err_addr,
                   Kokkos::View<unsigned long*> err_expect,
                   Kokkos::View<unsigned long*> err_current,
                   Kokkos::View<unsigned long*> err_second_read) {
  const unsigned int* buf = (const unsigned int*)d_mem.data();
  unsigned long       n   = size / (2 * sizeof(unsigned int));

  Kokkos::parallel_for("k5_check", (int)n, KOKKOS_LAMBDA(int i) {
    if (buf[2 * i] != buf[2 * i + 1]) {
      unsigned int idx = Kokkos::atomic_fetch_add(&err_count(0), 1u) % MAX_ERR_RECORD_COUNT;
      err_addr(idx)        = (unsigned long)(buf + 2 * i);
      err_expect(idx)      = (unsigned long)buf[2 * i + 1];
      err_current(idx)     = (unsigned long)buf[2 * i];
      err_second_read(idx) = (unsigned long)buf[2 * i];
    }
  });
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

static void check(Kokkos::View<unsigned int*> err_count) {
  auto h = Kokkos::create_mirror_view(err_count);
  Kokkos::deep_copy(h, err_count);
  printf("%s", h(0) ? "x" : ".");
  // reset
  Kokkos::deep_copy(err_count, 0u);
}

static void moving_inversion(Kokkos::View<unsigned int*>  err_count,
                              Kokkos::View<unsigned long*> err_addr,
                              Kokkos::View<unsigned long*> err_expect,
                              Kokkos::View<unsigned long*> err_current,
                              Kokkos::View<unsigned long*> err_second_read,
                              Kokkos::View<char*>          d_mem,
                              unsigned long mem_size, unsigned long p1) {
  auto d_mem_c = Kokkos::View<const char*>(d_mem);
  unsigned long p2 = ~p1;

  kernel_write(d_mem, mem_size, p1);

  for (int i = 0; i < 10; i++) {
    kernel_read_write(d_mem, mem_size, p1, p2,
                      err_count, err_addr, err_expect, err_current, err_second_read);
    p1 = p2;
    p2 = ~p1;
  }

  kernel_read(d_mem_c, mem_size, p1,
              err_count, err_addr, err_expect, err_current, err_second_read);
  check(err_count);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // 2 GB device memory
  const unsigned long mem_size = 2UL * 1024 * 1024 * 1024;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<char*>          d_mem("d_mem", mem_size);
    Kokkos::View<unsigned int*>  err_count("err_count", 1);
    Kokkos::View<unsigned long*> err_addr("err_addr",   MAX_ERR_RECORD_COUNT);
    Kokkos::View<unsigned long*> err_expect("err_expect", MAX_ERR_RECORD_COUNT);
    Kokkos::View<unsigned long*> err_current("err_current", MAX_ERR_RECORD_COUNT);
    Kokkos::View<unsigned long*> err_second_read("err_second_read", MAX_ERR_RECORD_COUNT);

    Kokkos::deep_copy(err_count, 0u);

    auto d_mem_c = Kokkos::View<const char*>(d_mem);

    // ── Test 0: bitmask walk ─────────────────────────────────────────────────
    printf("\ntest0: ");
    for (int i = 0; i < repeat; i++) {
      kernel0_write(d_mem, mem_size);
      Kokkos::fence();
      kernel0_read(d_mem_c, mem_size,
                   err_count, err_addr, err_expect, err_current, err_second_read);
      Kokkos::fence();
    }
    check(err_count);

    // ── Test 1: address-as-value ─────────────────────────────────────────────
    printf("\ntest1: ");
    for (int i = 0; i < repeat; i++) {
      kernel1_write(d_mem, mem_size);
      Kokkos::fence();
      kernel1_read(d_mem_c, mem_size,
                   err_count, err_addr, err_expect, err_current, err_second_read);
      Kokkos::fence();
    }
    check(err_count);

    // ── Test 2: moving inversions with 0x00.../0xFF... ───────────────────────
    printf("\ntest2: ");
    for (int i = 0; i < repeat; i++) {
      moving_inversion(err_count, err_addr, err_expect, err_current, err_second_read,
                       d_mem, mem_size, 0UL);
      moving_inversion(err_count, err_addr, err_expect, err_current, err_second_read,
                       d_mem, mem_size, ~0UL);
    }

    // ── Test 3: moving inversions with 0x8080.../0x7f7f... ──────────────────
    printf("\ntest3: ");
    for (int i = 0; i < repeat; i++) {
      unsigned long p1 = 0x8080808080808080UL;
      moving_inversion(err_count, err_addr, err_expect, err_current, err_second_read,
                       d_mem, mem_size, p1);
      moving_inversion(err_count, err_addr, err_expect, err_current, err_second_read,
                       d_mem, mem_size, ~p1);
    }

    // ── Test 4: moving inversions with random pattern ────────────────────────
    printf("\ntest4: ");
    srand(123);
    for (int i = 0; i < repeat; i++) {
      unsigned long p1 = (unsigned long)rand();
      p1 = (p1 << 32) | (unsigned long)rand();
      moving_inversion(err_count, err_addr, err_expect, err_current, err_second_read,
                       d_mem, mem_size, p1);
    }

    // ── Test 5: alternating bit patterns ─────────────────────────────────────
    printf("\ntest5: ");
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      kernel5_init(d_mem, mem_size);
      Kokkos::fence();
      kernel5_move(d_mem, mem_size);
      Kokkos::fence();
      kernel5_check(d_mem_c, mem_size,
                    err_count, err_addr, err_expect, err_current, err_second_read);
      Kokkos::fence();
    }
    auto t1 = std::chrono::steady_clock::now();
    check(err_count);
    printf("\nAverage kernel execution time (test5): %f (s)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
           * 1e-9f / repeat);
  }
  Kokkos::finalize();
  return 0;
}
