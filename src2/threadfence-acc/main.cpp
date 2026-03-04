#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <openacc.h>

#include <stdint.h>
#include <string.h>

/*----------------------------*/
/* check benchmark results */
static uint64_t fnv1a_hash_u32(const uint32_t* p, size_t n_u32) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n_u32; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ull;
  }
  return h;
}

static uint64_t hash_float_array(const float* a, int n) {
  return fnv1a_hash_u32((const uint32_t*)a, (size_t)n);
}

static void dump_result_summary(
    const char* tag, int iter, int teams, int blocks, int N,
    const unsigned int* count_host, const float* result_host)
{
  const float r0 = result_host[0];
  const float r1 = (teams > 1) ? result_host[1] : 0.0f;
  const float rL = result_host[teams - 1];
  const uint64_t h = hash_float_array(result_host, teams);

  printf("[%s] iter=%d teams=%d blocks=%d N=%d  count=%u  "
         "r0=% .9e r1=% .9e rLast=% .9e  hash=0x%016llx\n",
         tag, iter, teams, blocks, N,
         count_host[0], r0, r1, rL, (unsigned long long)h);
}
/*----------------------------*/

// OpenACC version: kernel-splitting to emulate OpenMP barriers / synchronization points.
// - Phase 1: per-team partial sums into result[0..teams-1] using gangs=vectors mapping.
// - Phase 2: one device "serial" region sums result[0..teams-1] into result[0].
// - Phase 3: reset count (kept for interface equivalence; not used for "last block" detection here).

static void sum_acc(
    const int teams,
    const int blocks,
    const float* __restrict array,
    const int N,
    unsigned int* __restrict count,
    float* __restrict result)
{
  // -----------------------------
  // Phase 0: initialize count on device (kept for interface equivalence)
  // -----------------------------
  #pragma acc serial present(count[0:1])
  {
    count[0] = 0;
  }

  // -----------------------------
  // Phase 1: per-team partial sums
  //   gangs  = teams
  //   vectors= blocks
  // -----------------------------
  #pragma acc parallel loop gang num_gangs(teams) vector_length(blocks) \
          present(array[0:N], result[0:teams])
  for (int bid = 0; bid < teams; ++bid) {

    float partial = 0.0f;

    #pragma acc loop vector reduction(+:partial)
    for (int lid = 0; lid < blocks; ++lid) {
      const int gid = bid * blocks + lid;
      if (gid < N) partial += array[gid];
    }

    result[bid] = partial;
  }

  // -----------------------------
  // Phase 2: global reduction of partial sums
  //   Separate device serial region acts as a synchronization point (kernel-splitting).
  // -----------------------------
  #pragma acc serial present(result[0:teams])
  {
    float total = 0.0f;
    for (int i = 0; i < teams; ++i) total += result[i];
    result[0] = total; // final sum
  }

  // -----------------------------
  // Phase 3: reset count (kept for interface equivalence)
  // -----------------------------
  #pragma acc serial present(count[0:1])
  {
    count[0] = 0;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N      = atoi(argv[2]);

  const int blocks = 256;
  const int grids  = (N + blocks - 1) / blocks; // teams

  float* h_array   = (float*) malloc (N * sizeof(float));
  float* h_result  = (float*) malloc (grids * sizeof(float));
  unsigned int* h_count = (unsigned int*) malloc (sizeof(unsigned int));
  h_count[0] = 0;

  bool ok = true;
  double time = 0.0;

  for (int i = 0; i < N; i++) h_array[i] = -1.f;

  //OpenACC data region
  #pragma acc data copyin(h_array[0:N], h_count[0:1]) create(h_result[0:grids])
  {
    for (int n = 0; n < repeat; n++) {

      auto start = std::chrono::steady_clock::now();

      sum_acc(grids, blocks, h_array, N, h_count, h_result);

      /*check benchmark results*/
      if (n < 5) {
				#pragma acc update self(h_result[0:grids], h_count[0:1])
				dump_result_summary("ACC", n, grids, blocks, N, h_count, h_result);
			} else {
				#pragma acc update self(h_result[0:1], h_count[0:1])
			}

			if (h_result[0] != -1.f * N || h_count[0] != 0) {
				ok = false;
				printf("[ACC] FAIL check: r0=%g expected=%g count=%u\n",
							 h_result[0], -1.f*(float)N, h_count[0]);
				break;
			}
			/*--------------------------*/

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      //result[0] only is needed on host for check
      #pragma acc update self(h_result[0:1])

      if (h_result[0] != -1.f * N) {
        ok = false;
        break;
      }
    }
  }

  if (ok) printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  free(h_count);
  free(h_result);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
