#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <openacc.h>

#define BLOCK_SIZE 256
#include "reference.h"

template <typename T>
void BlockRangeAtomicOnGlobalMem(T* data, int n)
{
  #pragma acc parallel loop vector_length(BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++) {
    #pragma acc atomic update
    data[i % BLOCK_SIZE]++;  //arbitrary number to add
  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(T* data, int n)
{
  #pragma acc parallel loop vector_length(BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++) {
    #pragma acc atomic update
    data[i & 0x1F]++; //arbitrary number to add
  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n)
{
  #pragma acc parallel loop vector_length(BLOCK_SIZE)
  for ( unsigned int i = 0; i < n; i++) {
    #pragma acc atomic update
    data[offset]++;    //arbitrary number to add
  }
}

template <typename T>
void BlockRangeAtomicOnSharedMem(T* data, int n)
{
  unsigned int gridDim_x = (n / BLOCK_SIZE);
  
#pragma acc parallel num_gangs(gridDim_x) vector_length(BLOCK_SIZE) \
  copy(data[0:BLOCK_SIZE])
  {
    T smem_data[BLOCK_SIZE];

#pragma acc loop gang
    for (unsigned int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++) {
    
#pragma acc loop vector
      for (unsigned int threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      
	smem_data[threadIdx_x] = 0;

	unsigned int blockDim_x = BLOCK_SIZE;
	unsigned int tid = (blockIdx_x * blockDim_x) + threadIdx_x;

	for (unsigned int i = tid; i < n; i += blockDim_x * gridDim_x) {
#pragma acc atomic update
	  smem_data[threadIdx_x]++;
	}

	if (blockIdx_x == gridDim_x ) {
	  data[threadIdx_x] = smem_data[threadIdx_x];
	}
      }
    }
  }
}

template <typename T>
void WarpRangeAtomicOnSharedMem(T* data, int n)
{
unsigned int gridDim_x = n / BLOCK_SIZE;

#pragma acc parallel num_gangs(gridDim_x) vector_length(BLOCK_SIZE) copy(data[0:32])
 {
   T smem_data[32];

#pragma acc loop gang
   for (unsigned int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++) {
    
#pragma acc loop vector
     for (unsigned int v = 0; v < BLOCK_SIZE; v++) {
       if (v < 32) smem_data[v] = 0;
     }

#pragma acc loop vector
     for (unsigned int threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      
       unsigned int blockDim_x = BLOCK_SIZE;
       unsigned int tid = (blockIdx_x * blockDim_x) + threadIdx_x;

       for (unsigned int i = tid; i < n; i += blockDim_x * gridDim_x) {
#pragma acc atomic update
	 smem_data[i & 0x1F]++;
       }

       if (blockIdx_x == (gridDim_x) && threadIdx_x < 32) {
	 data[threadIdx_x] = smem_data[threadIdx_x];
       }
     }
   }
 }
}

template <typename T>
void SingleRangeAtomicOnSharedMem(T* data, int offset, int n)
{
  unsigned int gridDim_x = n / BLOCK_SIZE;

#pragma acc parallel num_gangs(gridDim_x) vector_length(BLOCK_SIZE) \
  copy(data[0:BLOCK_SIZE])
  {
    T smem_data[BLOCK_SIZE];

#pragma acc loop gang
    for (unsigned int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++) {

#pragma acc loop vector
      for (unsigned int v = 0; v < BLOCK_SIZE; v++) {
	smem_data[v] = 0;
      }

#pragma acc loop vector
      for (unsigned int threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
	unsigned int blockDim_x = BLOCK_SIZE;
	unsigned int tid = (blockIdx_x * blockDim_x) + threadIdx_x;

	for (unsigned int i = tid; i < n; i += blockDim_x * gridDim_x) {
# pragma acc atomic update
	  smem_data[offset]++;
	}

	if (blockIdx_x == gridDim_x && threadIdx_x == 0) {
	  data[0] = smem_data[0];
	}
      }
    }
  }
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
  size_t data_size = sizeof(T) * t;

  T* data = (T*) malloc (data_size);
  T* h_data = (T*) malloc (data_size);
  T* r_data = (T*) malloc (data_size);
  int fail;

  for(int i=0; i<t; i++) {
    data[i] = h_data[i] = i%1024+1;
  }

#pragma acc data create(data[0:t])
  {
    #pragma acc update device (data[0:t])
    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      BlockRangeAtomicOnGlobalMem<T>(data, n);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);

    #pragma acc update host (data[0:t])
    memcpy(r_data, h_data, data_size);
    for(int i=0; i<repeat; i++)
      BlockRangeAtomicOnGlobalMem_ref<T>(r_data, n);
    fail = memcmp(data, r_data, data_size);
    printf("%s\n", fail ? "FAIL" : "PASS");

    memcpy(data, h_data, data_size);
    #pragma acc update device (data[0:t])
    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      WarpRangeAtomicOnGlobalMem<T>(data, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);

    memcpy(r_data, h_data, data_size);
    #pragma acc update host (data[0:t])
    for(int i=0; i<repeat; i++)
      WarpRangeAtomicOnGlobalMem_ref<T>(r_data, n);
    fail = memcmp(data, r_data, data_size);
    printf("%s\n", fail ? "FAIL" : "PASS");

    memcpy(data, h_data, data_size);
    #pragma acc update device (data[0:t])
    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      SingleRangeAtomicOnGlobalMem<T>(data, i % BLOCK_SIZE, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);

    memcpy(r_data, h_data, data_size);
    #pragma acc update host (data[0:t])
    for(int i=0; i<repeat; i++)
      SingleRangeAtomicOnGlobalMem_ref<T>(r_data, i % BLOCK_SIZE, n);
    fail = memcmp(data, r_data, data_size);
    printf("%s\n", fail ? "FAIL" : "PASS");

    memcpy(data, h_data, data_size);
    #pragma acc update device (data[0:t])
    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      BlockRangeAtomicOnSharedMem<T>(data, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
            time * 1e-3f / repeat);

    #pragma acc update host (data[0:t])
    fail = memcmp(h_data, data, data_size);
    printf("%s\n", fail ? "FAIL" : "PASS");

    memcpy(data, h_data, data_size);
    #pragma acc update device (data[0:t])
    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      WarpRangeAtomicOnSharedMem<T>(data, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
            time * 1e-3f / repeat);

    #pragma acc update host (data[0:t])
    fail = memcmp(h_data, data, data_size);
    printf("%s\n", fail ? "FAIL" : "PASS");

    memcpy(data, h_data, data_size);
    #pragma acc update device (data[0:t])
    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      SingleRangeAtomicOnSharedMem<T>(data, i % BLOCK_SIZE, n);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
            time * 1e-3f / repeat);

    #pragma acc update host (data[0:t])
    fail = memcmp(h_data, data, data_size);
    printf("%s\n", fail ? "FAIL" : "PASS");

  }
  free(data);
  free(h_data);
  free(r_data);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3*4*7*8*9*256; // number of threads
  const int len = 1024; // data array length
  
  printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat); 

  printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat); 

  printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat); 

  return 0;
}
