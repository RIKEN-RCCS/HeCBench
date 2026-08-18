#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#include <stdint.h>
#include <string.h>

//--------------------------------
//check benchmark results
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
//--------------------------------

void sum (
    const int teams,
    const int blocks,
    const float*__restrict array,
    const int N,
    unsigned int *__restrict count,
    volatile float*__restrict result)
{
  #pragma omp target teams num_teams(teams) thread_limit(blocks)
  {
    bool isLastBlockDone;
    float partialSum;
    #pragma omp parallel 
    {
      // Each block sums a subset of the input array.
      unsigned int bid = omp_get_team_num();
      unsigned int num_blocks = teams;
      unsigned int block_size = blocks;
      unsigned int lid = omp_get_thread_num();
      unsigned int gid = bid * block_size + lid;

      if (lid == 0) partialSum = 0;
      #pragma omp barrier

      if (gid < N) {
        #pragma omp atomic update
        partialSum += array[gid];
      }

      #pragma omp barrier

      if (lid == 0) {

        // Thread 0 of each block stores the partial sum
        // to global memory. The compiler will use 
        // a store operation that bypasses the L1 cache
        // since the "result" variable is declared as
        // volatile. This ensures that the threads of
        // the last block will read the correct partial
        // sums computed by all other blocks.
        result[bid] = partialSum;

        // Thread 0 signals that it is done.
        unsigned int value;
        #pragma omp atomic capture
        value = (*count)++;

        // Thread 0 determines if its block is the last
        // block to be done.
        isLastBlockDone = (value == (num_blocks - 1));
      }

      // Synchronize to make sure that each thread reads
      // the correct value of isLastBlockDone.
      #pragma omp barrier

      if (isLastBlockDone) {

        // The last block sums the partial sums
        // stored in result[0 .. num_blocks-1]
        if (lid == 0) partialSum = 0;
        #pragma omp barrier

        for (int i = lid; i < num_blocks; i += block_size) {
          #pragma omp atomic update
          partialSum += result[i];
        }

        #pragma omp barrier

        if (lid == 0) {

          // Thread 0 of last block stores the total sum
          // to global memory and resets the count
          // varialble, so that the next kernel call
          // works properly.
          result[0] = partialSum;
          *count = 0;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);

  const int blocks = 256;
  const int grids = (N + blocks - 1) / blocks;

  float* h_array = (float*) malloc (N * sizeof(float));

  float* h_result = (float*) malloc (grids * sizeof(float));

  unsigned int* h_count = (unsigned int*) malloc (sizeof(unsigned int));
  h_count[0] = 0;

  bool ok = true;
  double time = 0.0;

  for (int i = 0; i < N; i++) h_array[i] = -1.f;
  
  #pragma omp target data map (to: h_array[0:N], h_count[0:1]) \
                          map (alloc: h_result[0:grids])
  {
    for (int n = 0; n < repeat; n++) {
  
      //#pragma omp target update to (h_array[0:N])
  
      auto start = std::chrono::steady_clock::now();

      sum (grids, blocks, h_array, N, h_count, h_result);

      /*check benchmark results*/
			if (n < 5) {
				#pragma omp target update from (h_result[0:grids], h_count[0:1])
				dump_result_summary("OMP", n, grids, blocks, N, h_count, h_result);
			} else {
				#pragma omp target update from (h_result[0:1], h_count[0:1])
			}

			if (h_result[0] != -1.f * N || h_count[0] != 0) {
				ok = false;
				printf("[OMP] FAIL check: r0=%g expected=%g count=%u\n",
							 h_result[0], -1.f*(float)N, h_count[0]);
				break;
			}
			/*---------------------------------*/

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
      #pragma omp target update from (h_result[0:1])
  
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
