/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <openacc.h>
#include <sys/time.h>

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

struct char4 { char x; char y; char z; char w; };


/* CPU implementation */
void CPU(int * data, int * distance) {
  /* compare all pairs of instances, accessing the attributes in
     row-major order */
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
  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  gettimeofday(&tp, &tzp);
  start_cpu = tp.tv_sec*1000000+tp.tv_usec;
  CPU(data, cpu_distance);
  gettimeofday(&tp, &tzp);
  stop_cpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_cpu - start_cpu;
  printf("CPU time: %f (us)\n",elapsedTime);

  #pragma acc data copyin( data_char[0:INSTANCES * ATTRIBUTES]) \
                          create( gpu_distance[0:INSTANCES * INSTANCES ])
  {
    for (int n = 0; n < iterations; n++) {
      /* register-based kernel */
      bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
      #pragma acc update device(gpu_distance[0:INSTANCES * INSTANCES])
  
      gettimeofday(&tp, &tzp);
      start_gpu = tp.tv_sec*1000000+tp.tv_usec;
  
      #pragma acc parallel num_gangs(INSTANCES*INSTANCES) vector_length(THREADS)
      {
        #pragma acc loop gang
        for (int gang_id = 0; gang_id < INSTANCES*INSTANCES; gang_id++) {
          int gx = gang_id % INSTANCES;
          int gy = gang_id / INSTANCES;

          #pragma acc loop vector
          for(int idx = 0; idx < THREADS; idx++) {
            for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
              char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
              char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);

              /* use a local variable (stored in register) to hold intermediate
                 values. This reduces writes to global memory */
              char count = 0;

              if(j.x ^ k.x)
                count++;
              if(j.y ^ k.y)
                count++;
              if(j.z ^ k.z)
                count++;
              if(j.w ^ k.w)
                count++;

              /* atomic write to global memory */
              #pragma acc atomic update
              gpu_distance[ INSTANCES*gx + gy ] += count;
            }
          }
        }
      }
  
      gettimeofday(&tp, &tzp);
      stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
      elapsedTime += stop_gpu - start_gpu;
    }
  
    #pragma acc update host(gpu_distance[0:INSTANCES * INSTANCES])
  
    printf("Average kernel execution time (w/o shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    if (status != 0) printf("FAIL\n");
    else printf("PASS\n");
  
    elapsedTime = 0; 
    for (int n = 0; n < iterations; n++) {
      /* shared memory GPU kernel */
      bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
      #pragma acc update device(gpu_distance[0:INSTANCES * INSTANCES])
  
      gettimeofday(&tp, &tzp);
      start_gpu = tp.tv_sec*1000000+tp.tv_usec;
  
      #pragma acc parallel num_gangs(INSTANCES*INSTANCES) vector_length(THREADS)
      {
        #pragma acc loop gang
        for (int gang_id = 0; gang_id < INSTANCES*INSTANCES; gang_id++) {
          int gx = gang_id % INSTANCES;
          int gy = gang_id / INSTANCES;
          int dist[THREADS];

          #pragma acc loop vector
          for(int idx = 0; idx < THREADS; idx++) {
            dist[idx] = 0;
          }

          #pragma acc loop vector
          for(int idx = 0; idx < THREADS; idx++) {
            for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
              char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
              char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);

              /* use a local variable (stored in register) to hold intermediate
                 values. This reduces writes to global memory */
              char count = 0;

              if(j.x ^ k.x)
                count++;
              if(j.y ^ k.y)
                count++;
              if(j.z ^ k.z)
                count++;
              if(j.w ^ k.w)
                count++;

              dist[idx] += count;
            }
          }

          /* Synchronize threads to make sure all have completed their updates
             of the shared array. Since the distances for each thread are read
             by thread 0 below, this must be ensured. Above, it was not
             necessary because each thread was accessing its own memory
          */

          /* Perform balanced tree reduction across the shared memory */
          for (int stride = THREADS/2; stride > 0; stride /= 2) {
            #pragma acc loop vector
            for(int idx = 0; idx < THREADS; idx++) {
              if (idx < stride) {
                dist[idx] += dist[idx + stride];
              }
            }
          }

          /* Thread 0 will then write the output to global memory. Note that
             this does not need to be performed atomically, because only one
             thread per block is writing to global memory, and each block
             corresponds to a unique memory address.
          */
          gpu_distance[INSTANCES*gy + gx] = dist[0];
        }
      }
  
      gettimeofday(&tp, &tzp);
      stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
      elapsedTime += stop_gpu - start_gpu;
    }
  
    #pragma acc update host(gpu_distance[0:INSTANCES * INSTANCES])
  
    printf("Average kernel execution time (w/ shared memory): %f (us)\n", elapsedTime / iterations);
    status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
    if (status != 0) printf("FAIL\n");
    else printf("PASS\n");
  }

  free(cpu_distance);
  free(gpu_distance);
  free(data_char);
  free(data);
  return status;
}
