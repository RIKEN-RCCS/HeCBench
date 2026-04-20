/**
 * shmem_kernels (Kokkos port): shared memory bandwidth microbenchmark.
 * Ported from shmembench-omp.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define VECTOR_SIZE    (1024 * 1024)
#define TOTAL_ITERATIONS 1024
#define BLOCK_SIZE     256

struct float4 {
  float x, y, z, w;
};

int main(int argc, char* argv[]) {
  printf("Shared memory bandwidth microbenchmark\n");

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE * sizeof(double);
  printf("Buffer sizes: %dMB\n", datasize / (1024 * 1024));

  double* c = (double*)malloc(datasize);
  memset(c, 0, sizeof(double) * VECTOR_SIZE);

  Kokkos::initialize(argc, argv);
  {
    const int TOTAL_BLOCKS = VECTOR_SIZE / BLOCK_SIZE;

    Kokkos::View<double*> d_c("c", VECTOR_SIZE);

    using scratch_t = Kokkos::View<float4*,
                                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                   Kokkos::MemoryUnmanaged>;
    size_t scratch_bytes = scratch_t::shmem_size(BLOCK_SIZE * 6);

    using team_policy = Kokkos::TeamPolicy<>;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n; iter++) {
      Kokkos::parallel_for("shmembench",
        team_policy(TOTAL_BLOCKS / 4, BLOCK_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const team_policy::member_type& team) {
          int tid       = team.team_rank();
          int blk       = team.team_size();
          int gid       = team.league_rank();
          int globaltid = gid * blk + tid;

          scratch_t shm_buffer(team.team_scratch(0), BLOCK_SIZE * 6);

          // Init
          auto init_val = [=](int i) KOKKOS_INLINE_FUNCTION -> float4 {
            return {(float)i, (float)i + 11, (float)i + 19, (float)i + 23};
          };

          shm_buffer(tid + 0 * blk) = init_val(tid);
          shm_buffer(tid + 1 * blk) = init_val(tid + 1);
          shm_buffer(tid + 2 * blk) = init_val(tid + 3);
          shm_buffer(tid + 3 * blk) = init_val(tid + 7);
          shm_buffer(tid + 4 * blk) = init_val(tid + 13);
          shm_buffer(tid + 5 * blk) = init_val(tid + 17);

          team.team_barrier();

          for (int j = 0; j < TOTAL_ITERATIONS; j++) {
            // swap pairs
            float4 tmp;
            tmp = shm_buffer(tid + 1 * blk);
            shm_buffer(tid + 1 * blk) = shm_buffer(tid + 0 * blk);
            shm_buffer(tid + 0 * blk) = tmp;

            tmp = shm_buffer(tid + 3 * blk);
            shm_buffer(tid + 3 * blk) = shm_buffer(tid + 2 * blk);
            shm_buffer(tid + 2 * blk) = tmp;

            tmp = shm_buffer(tid + 5 * blk);
            shm_buffer(tid + 5 * blk) = shm_buffer(tid + 4 * blk);
            shm_buffer(tid + 4 * blk) = tmp;

            team.team_barrier();

            tmp = shm_buffer(tid + 2 * blk);
            shm_buffer(tid + 2 * blk) = shm_buffer(tid + 1 * blk);
            shm_buffer(tid + 1 * blk) = tmp;

            tmp = shm_buffer(tid + 4 * blk);
            shm_buffer(tid + 4 * blk) = shm_buffer(tid + 3 * blk);
            shm_buffer(tid + 3 * blk) = tmp;

            team.team_barrier();
          }

          // Reduce all 6 float4 values into a double
          float4 v0 = shm_buffer(tid + 0 * blk);
          float4 v1 = shm_buffer(tid + 1 * blk);
          float4 v2 = shm_buffer(tid + 2 * blk);
          float4 v3 = shm_buffer(tid + 3 * blk);
          float4 v4 = shm_buffer(tid + 4 * blk);
          float4 v5 = shm_buffer(tid + 5 * blk);

          float4 r = {v0.x + v1.x + v2.x + v3.x + v4.x + v5.x,
                      v0.y + v1.y + v2.y + v3.y + v4.y + v5.y,
                      v0.z + v1.z + v2.z + v3.z + v4.z + v5.z,
                      v0.w + v1.w + v2.w + v3.w + v4.w + v5.w};

          // Pack float4 into two doubles in d_c
          float4* g_data = reinterpret_cast<float4*>(&d_c(0));
          g_data[globaltid] = r;
        });
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    double time_shmem_128b =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)n;
    printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);

    {
      auto h_c = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(c, VECTOR_SIZE);
      Kokkos::deep_copy(h_c, d_c);
    }

    double sum = 0;
    for (long i = 0; i < VECTOR_SIZE; i++) sum += c[i];
    if (sum != 21256458760384741137729978368.00)
      printf("checksum failed\n");

    printf("Memory throughput\n");
    const long long operations_bytes  = (6LL + 4 * 5 * TOTAL_ITERATIONS + 6) * VECTOR_SIZE * sizeof(float);
    const long long operations_128bit = (6LL + 4 * 5 * TOTAL_ITERATIONS + 6) * VECTOR_SIZE / 4;

    printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n",
           (double)operations_bytes  / time_shmem_128b,
           (double)operations_128bit / time_shmem_128b);
  }
  Kokkos::finalize();

  free(c);
  return 0;
}
