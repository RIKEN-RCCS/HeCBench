/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <Kokkos_Core.hpp>

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

struct char4 { char x; char y; char z; char w; };

/* CPU implementation */
void CPU(int * data, int * distance) {
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] +=
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    /* host data */
    int *data;
    char *data_char;
    int *cpu_distance, *gpu_distance;

    /* used to time CPU and GPU implementations */
    double start_cpu, stop_cpu;
    double start_gpu, stop_gpu;
    double elapsedTime;
    struct timeval tp;
    struct timezone tzp;
    /* verification result */
    int status;

    /* seed RNG */
    srand(2);

    /* allocate host memory */
    data = (int *)malloc(INSTANCES * ATTRIBUTES * sizeof(int));
    data_char = (char *)malloc(INSTANCES * ATTRIBUTES * sizeof(char));
    cpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));
    gpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));

    /* randomly initialize host data */
    for (int i = 0; i < ATTRIBUTES; i++) {
      for (int j = 0; j < INSTANCES; j++) {
        data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
      }
    }

    /* CPU */
    bzero(cpu_distance, INSTANCES * INSTANCES * sizeof(int));
    gettimeofday(&tp, &tzp);
    start_cpu = tp.tv_sec * 1000000 + tp.tv_usec;
    CPU(data, cpu_distance);
    gettimeofday(&tp, &tzp);
    stop_cpu = tp.tv_sec * 1000000 + tp.tv_usec;
    elapsedTime = stop_cpu - start_cpu;
    printf("CPU time: %f (us)\n", elapsedTime);

    // Create Kokkos device views
    Kokkos::View<char*> d_data_char("d_data_char", INSTANCES * ATTRIBUTES);
    Kokkos::View<int*> d_gpu_distance("d_gpu_distance", INSTANCES * INSTANCES);

    // Copy data to device
    auto h_data_char = Kokkos::create_mirror_view(d_data_char);
    for (int i = 0; i < INSTANCES * ATTRIBUTES; i++)
      h_data_char(i) = data_char[i];
    Kokkos::deep_copy(d_data_char, h_data_char);

    // Kernel 1: Register-based (atomic)
    elapsedTime = 0;
    for (int n = 0; n < iterations; n++) {
      bzero(gpu_distance, INSTANCES * INSTANCES * sizeof(int));

      // Copy zeroed distance to device
      auto h_gpu_distance = Kokkos::create_mirror_view(d_gpu_distance);
      for (int i = 0; i < INSTANCES * INSTANCES; i++)
        h_gpu_distance(i) = 0;
      Kokkos::deep_copy(d_gpu_distance, h_gpu_distance);

      gettimeofday(&tp, &tzp);
      start_gpu = tp.tv_sec * 1000000 + tp.tv_usec;

      typedef Kokkos::TeamPolicy<> team_policy;
      typedef Kokkos::TeamPolicy<>::member_type member_type;

      Kokkos::parallel_for("kernel1",
        team_policy(INSTANCES * INSTANCES, THREADS),
        KOKKOS_LAMBDA(const member_type &team) {
          int idx = team.team_rank();
          int gx = team.league_rank() % INSTANCES;
          int gy = team.league_rank() / INSTANCES;

          for (int i = 4 * idx; i < ATTRIBUTES; i += THREADS * 4) {
            char4 j;
            j.x = d_data_char(i + ATTRIBUTES * gx);
            j.y = d_data_char(i + 1 + ATTRIBUTES * gx);
            j.z = d_data_char(i + 2 + ATTRIBUTES * gx);
            j.w = d_data_char(i + 3 + ATTRIBUTES * gx);

            char4 k;
            k.x = d_data_char(i + ATTRIBUTES * gy);
            k.y = d_data_char(i + 1 + ATTRIBUTES * gy);
            k.z = d_data_char(i + 2 + ATTRIBUTES * gy);
            k.w = d_data_char(i + 3 + ATTRIBUTES * gy);

            char count = 0;

            if (j.x ^ k.x)
              count++;
            if (j.y ^ k.y)
              count++;
            if (j.z ^ k.z)
              count++;
            if (j.w ^ k.w)
              count++;

            Kokkos::atomic_add(&d_gpu_distance(INSTANCES * gx + gy), (int)count);
          }
        }
      );
      Kokkos::fence();

      gettimeofday(&tp, &tzp);
      stop_gpu = tp.tv_sec * 1000000 + tp.tv_usec;
      elapsedTime += stop_gpu - start_gpu;

      // Copy result back
      Kokkos::deep_copy(h_gpu_distance, d_gpu_distance);
      for (int i = 0; i < INSTANCES * INSTANCES; i++)
        gpu_distance[i] = h_gpu_distance(i);
    }

    printf("Average kernel execution time (w/o shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    if (status != 0) printf("FAIL\n");
    else printf("PASS\n");

    // Kernel 2: Shared memory (scratch pad)
    elapsedTime = 0;
    typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
    typedef Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchView;
    int scratch_size = ScratchView::shmem_size(THREADS);

    for (int n = 0; n < iterations; n++) {
      bzero(gpu_distance, INSTANCES * INSTANCES * sizeof(int));

      auto h_gpu_distance = Kokkos::create_mirror_view(d_gpu_distance);
      for (int i = 0; i < INSTANCES * INSTANCES; i++)
        h_gpu_distance(i) = 0;
      Kokkos::deep_copy(d_gpu_distance, h_gpu_distance);

      gettimeofday(&tp, &tzp);
      start_gpu = tp.tv_sec * 1000000 + tp.tv_usec;

      typedef Kokkos::TeamPolicy<> team_policy;
      typedef Kokkos::TeamPolicy<>::member_type member_type;

      Kokkos::parallel_for("kernel2",
        team_policy(INSTANCES * INSTANCES, THREADS).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const member_type &team) {
          int idx = team.team_rank();
          int gx = team.league_rank() % INSTANCES;
          int gy = team.league_rank() / INSTANCES;

          ScratchView dist(team.team_scratch(0), THREADS);

          dist(idx) = 0;
          team.team_barrier();

          for (int i = 4 * idx; i < ATTRIBUTES; i += THREADS * 4) {
            char4 j;
            j.x = d_data_char(i + ATTRIBUTES * gx);
            j.y = d_data_char(i + 1 + ATTRIBUTES * gx);
            j.z = d_data_char(i + 2 + ATTRIBUTES * gx);
            j.w = d_data_char(i + 3 + ATTRIBUTES * gx);

            char4 k;
            k.x = d_data_char(i + ATTRIBUTES * gy);
            k.y = d_data_char(i + 1 + ATTRIBUTES * gy);
            k.z = d_data_char(i + 2 + ATTRIBUTES * gy);
            k.w = d_data_char(i + 3 + ATTRIBUTES * gy);

            char count = 0;

            if (j.x ^ k.x)
              count++;
            if (j.y ^ k.y)
              count++;
            if (j.z ^ k.z)
              count++;
            if (j.w ^ k.w)
              count++;

            dist(idx) += count;
          }

          team.team_barrier();

          if (idx == 0) {
            for (int i = 1; i < THREADS; i++) {
              dist(0) += dist(i);
            }
            d_gpu_distance(INSTANCES * gy + gx) = dist(0);
          }
        }
      );
      Kokkos::fence();

      gettimeofday(&tp, &tzp);
      stop_gpu = tp.tv_sec * 1000000 + tp.tv_usec;
      elapsedTime += stop_gpu - start_gpu;

      // Copy result back
      auto h_gpu_distance2 = Kokkos::create_mirror_view(d_gpu_distance);
      Kokkos::deep_copy(h_gpu_distance2, d_gpu_distance);
      for (int i = 0; i < INSTANCES * INSTANCES; i++)
        gpu_distance[i] = h_gpu_distance2(i);
    }

    printf("Average kernel execution time (w/ shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    if (status != 0) printf("FAIL\n");
    else printf("PASS\n");

    free(cpu_distance);
    free(gpu_distance);
    free(data_char);
    free(data);
  }
  Kokkos::finalize();
  return 0;
}
