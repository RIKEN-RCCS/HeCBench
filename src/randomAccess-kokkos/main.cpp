/*
 * Random Access (HPCC RandomAccess) benchmark.
 * Ported to Kokkos from the OMP target version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

typedef unsigned long long u64Int;
typedef long long s64Int;

#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#define NUPDATE (4 * TableSize)

KOKKOS_INLINE_FUNCTION
u64Int HPCC_starts(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int)temp < 0 ? POLY : 0);
  }

  for (i = 62; i >= 0; i--)
    if ((n >> i) & 1) break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j = 0; j < 64; j++)
      if ((ran >> j) & 1) temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
  }
  return ran;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  u64Int logTableSize, TableSize;
  double totalMem = 1024.0 * 1024.0 * 512.0;
  totalMem /= sizeof(u64Int);
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; /* EMPTY */

  printf("Table size = %llu\n", TableSize);
  fprintf(stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize, TableSize);
  fprintf(stdout, "Number of updates = %llu\n", (u64Int)NUPDATE);

  u64Int *Table = (u64Int*) malloc(TableSize * sizeof(u64Int));
  if (!Table) {
    fprintf(stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<u64Int*> d_Table("d_Table", TableSize);
    Kokkos::View<u64Int*> d_ran("d_ran", 128);

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      // Initialize the table
      Kokkos::parallel_for("init_table", (int)TableSize, KOKKOS_LAMBDA(int i) {
        d_Table(i) = (u64Int)i;
      });
      Kokkos::fence();

      // Update the table using 128 parallel streams
      Kokkos::parallel_for("update_table", 128, KOKKOS_LAMBDA(int j) {
        u64Int ran = HPCC_starts((NUPDATE / 128) * j);
        for (u64Int i = 0; i < NUPDATE / 128; i++) {
          ran = (ran << 1) ^ ((s64Int)ran < 0 ? POLY : 0);
          Kokkos::atomic_fetch_xor(&d_Table(ran & (TableSize - 1)), ran);
        }
      });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    auto h_Table = Kokkos::create_mirror_view(d_Table);
    Kokkos::deep_copy(h_Table, d_Table);
    for (u64Int i = 0; i < TableSize; i++) Table[i] = h_Table(i);
  }
  Kokkos::finalize();

  // Validation
  u64Int temp = 0x1;
  for (u64Int i = 0; i < (u64Int)NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
    Table[temp & (TableSize - 1)] ^= temp;
  }

  u64Int errors = 0;
  for (u64Int i = 0; i < TableSize; i++)
    if (Table[i] != i) errors++;

  fprintf(stdout, "Found %llu errors in %llu locations (%s).\n",
          errors, TableSize, (errors <= 0.01 * TableSize) ? "passed" : "failed");

  free(Table);
  return (errors <= 0.01 * TableSize) ? 0 : 1;
}
