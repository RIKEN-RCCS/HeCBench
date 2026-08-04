/*
 * Tridiagonal solvers - Kokkos port
 * Ported from tridiagonal-omp
 * Implements PCR, CR (Cyclic), and Sweep solvers using Kokkos
 */

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "tridiagonal.h"
#include "shrUtils.h"
#include "file_read_write.h"
#include "test_gen_result_check.h"
#include "cpu_solvers.h"
#include "pcr_small_systems.h"
#include "cyclic_small_systems.h"
#include "sweep_small_systems.h"

bool             useLmem = false;
bool             useVec4 = false;
int              SWEEP_BLOCK_SIZE = 256;

// ─────────────────────────── run() ───────────────────────────────────────────

void run(int system_size, int num_systems)
{
  double time_spent_gpu[3];
  double time_spent_cpu[1];

  const unsigned int mem_size = sizeof(float) * num_systems * system_size;

  float* a  = (float*)malloc(mem_size);
  float* b  = (float*)malloc(mem_size);
  float* c  = (float*)malloc(mem_size);
  float* d  = (float*)malloc(mem_size);
  float* x1 = (float*)malloc(mem_size);
  float* x2 = (float*)malloc(mem_size);

  for (int i = 0; i < num_systems; i++)
    test_gen_cyclic(&a[i * system_size], &b[i * system_size], &c[i * system_size],
                    &d[i * system_size], &x1[i * system_size], system_size, 0);

  shrLog("  Num_systems = %d, system_size = %d\n", num_systems, system_size);

  time_spent_cpu[0] = serial_small_systems(a, b, c, d, x2, system_size, num_systems);

  shrLog("\n----- CPU  solvers -----\n");
  shrLog("  CPU Time =    %.5f s\n", time_spent_cpu[0]);
  shrLog("  Throughput =  %.4f systems/sec\n",
         (float)num_systems / (time_spent_cpu[0] * 1000.0));

  shrLog("\n----- optimized GPU solvers -----\n\n");

  time_spent_gpu[0] = pcr_small_systems(a, b, c, d, x1, system_size, num_systems, 0);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-pcrsmall-base, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[0] = pcr_small_systems(a, b, c, d, x1, system_size, num_systems, 1);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-pcrsmall-optimized, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[1] = cyclic_small_systems(a, b, c, d, x1, system_size, num_systems, 0);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-cyclicsmall-base, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[1] = cyclic_small_systems(a, b, c, d, x1, system_size, num_systems, 1);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-cyclicsmall-optimized, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  if (!useVec4) {
    time_spent_gpu[2] = sweep_small_systems(a, b, c, d, x1, system_size, num_systems, false);
    shrLogEx(LOGBOTH | MASTER, 0,
      "Tridiagonal-sweepsmall-noreorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
      (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems);
    compare_small_systems(x1, x2, system_size, num_systems);
  }

  time_spent_gpu[2] = sweep_small_systems(a, b, c, d, x1, system_size, num_systems, true);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-sweepsmall-reorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  free(a); free(b); free(c); free(d); free(x1); free(x2);
}

// ─────────────────────────── main() ──────────────────────────────────────────

int main(int argc, const char** argv)
{
  shrSetLogFileName("oclTridiagonal.txt");
  shrLog("%s Starting...\n\n", argv[0]);

  int num_systems = 128 * 128;
  int system_size = SYSTEM_SIZE;

  if (shrCheckCmdLineFlag(argc, argv, "num_systems")) {
    char* ctaList; char* ctaStr;
    shrGetCmdLineArgumentstr(argc, argv, "num_systems", &ctaList);
    ctaStr = strtok(ctaList, " ,.-");
    num_systems = atoi(ctaStr);
  }

  if (shrCheckCmdLineFlag(argc, argv, "system_size")) {
    char* ctaList; char* ctaStr;
    shrGetCmdLineArgumentstr(argc, argv, "system_size", &ctaList);
    ctaStr = strtok(ctaList, " ,.-");
    system_size = atoi(ctaStr);
    if (system_size > 128) {
      shrLog("system size must be no more than 128\n");
      return -1;
    }
  }

  if (shrCheckCmdLineFlag(argc, argv, "lmem"))  useLmem = true;
  if (shrCheckCmdLineFlag(argc, argv, "vec4"))  useVec4 = true;

  if (shrCheckCmdLineFlag(argc, argv, "sweep-cta")) {
    char* ctaList; char* ctaStr;
    shrGetCmdLineArgumentstr(argc, argv, "sweep-cta", &ctaList);
    ctaStr = strtok(ctaList, " ,.-");
    SWEEP_BLOCK_SIZE = atoi(ctaStr);
  }

  if (useVec4) shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE / 4);
  else         shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE);

  Kokkos::initialize(argc, const_cast<char**>(argv));
  {
    run(system_size, num_systems);
  }
  Kokkos::finalize();

  return 0;
}
