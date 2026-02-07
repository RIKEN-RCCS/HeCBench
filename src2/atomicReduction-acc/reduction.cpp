/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cmath>


template<int THREADS>
void run_reduction1(int arrayLength, int* array, int& sum, int blocks, int N) {
#pragma acc data present(array[0:arrayLength])
  {
    for(int n = 0; n < N; n++) {
      sum = 0;
#pragma acc update device(sum)
#pragma acc parallel loop num_gangs(blocks) vector_length(THREADS) reduction(+:sum)
      for (int i = 0; i < arrayLength; i++) {
	sum += array[i];
      }
    }
#pragma acc update self(sum)
  }
}

template<int threads>
void run_reduction2(int arrayLength, int* array, int& sum, int blocks, int N) {
#pragma acc data present(array[0:arrayLength])
  {
    for(int n = 0; n < N; n++) {
      sum = 0;
#pragma acc update device(sum)
#pragma acc parallel loop num_gangs(blocks/2) vector_length(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength/2; i++) {
	sum += array[i*2] + array[i*2+1];
      }
    }
#pragma acc update self(sum)
  }
}

template<int threads>
void run_reduction3(int arrayLength, int* array, int& sum, int blocks, int N) {
#pragma acc data present(array[0:arrayLength])
  {
    for(int n = 0; n < N; n++) {
      sum = 0;
#pragma acc update device(sum)
#pragma acc parallel loop num_gangs(blocks/4) vector_length(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength/4; i++) {
	sum += array[i*4] + array[i*4+1] + array[i*4+2] + array[i*4+3];
      }
    }
#pragma acc update self(sum)
  }
}

template<int threads>
void run_reduction4(int arrayLength, int* array, int& sum, int blocks, int N) {
#pragma acc data present(array[0:arrayLength])
  {
    for(int n = 0; n < N; n++) {
      sum = 0;
#pragma acc update device(sum)
#pragma acc parallel loop num_gangs(blocks/8) vector_length(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength/8; i++) { 
	sum += array[i*8] + array[i*8+1] + array[i*8+2] + array[i*8+3] + 
	  array[i*8+4] + array[i*8+5] + array[i*8+6] + array[i*8+7];
      }
    }
#pragma acc update self(sum)
  }
}

template<int threads>
void run_reduction5(int arrayLength, int* array, int& sum, int blocks, int N) {
#pragma acc data present(array[0:arrayLength])
  {
    for(int n = 0; n < N; n++) {
      sum = 0;
#pragma acc update device(sum)
#pragma acc parallel loop num_gangs(blocks/16) vector_length(threads) reduction(+:sum)
      for (int i = 0; i < arrayLength/16; i++) { 
	sum += array[i*16] + array[i*16+1] + array[i*16+2] + array[i*16+3] + 
	  array[i*16+4] + array[i*16+5] + array[i*16+6] + array[i*16+7] +
	  array[i*16+8] + array[i*16+9] + array[i*16+10] + array[i*16+11] +
	  array[i*16+12] + array[i*16+13] + array[i*16+14] + array[i*16+15];
      }
    }
#pragma acc update self(sum)
  }
}


int main(int argc, char** argv)
{
  int arrayLength = 52428800;
  int block_sizes[] = {128, 256, 512, 1024};
  int N = 100;

  if (argc == 3) {
    arrayLength=atoi(argv[1]);
    N=atoi(argv[2]);
  }

  std::cout << "Array size: " << arrayLength*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
  std::cout << "Repeat the kernel execution: " << N << " times" << std::endl;

  int* array=(int*)malloc(arrayLength*sizeof(int));
  int checksum =0;
  for(int i=0;i<arrayLength;i++) {
    array[i]=rand()%2;
    checksum+=array[i];
  }

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  float GB=(float)arrayLength*sizeof(int)*N;
  int sum;

#pragma acc data copyin(array[0:arrayLength]) create(sum)
  {
    // warmup
    run_reduction1<256>(arrayLength, array, sum, 2048, N);

    for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
      const int threads = block_sizes[k];
      const int blocks=std::min((arrayLength+threads-1)/threads,2048);

      // start timing
      t1 = std::chrono::high_resolution_clock::now();

      switch(threads) {
      case 128:  run_reduction1<128>(arrayLength, array, sum, blocks, N); break;
      case 256:  run_reduction1<256>(arrayLength, array, sum, blocks, N); break;
      case 512:  run_reduction1<512>(arrayLength, array, sum, blocks, N); break;      
      case 1024:  run_reduction1<1024>(arrayLength, array, sum, blocks, N); break;      
      }

      t2 = std::chrono::high_resolution_clock::now();
      double times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;


      //      printf("%d %d\n", sum, checksum);
      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();

      switch(threads) {
      case 128:  run_reduction2<128>(arrayLength, array, sum, blocks, N); break;
      case 256:  run_reduction2<256>(arrayLength, array, sum, blocks, N); break;
      case 512:  run_reduction2<512>(arrayLength, array, sum, blocks, N); break;      
      case 1024:  run_reduction2<1024>(arrayLength, array, sum, blocks, N); break;      
      }

      t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();


      switch(threads) {
      case 128:  run_reduction3<128>(arrayLength, array, sum, blocks, N); break;
      case 256:  run_reduction3<256>(arrayLength, array, sum, blocks, N); break;
      case 512:  run_reduction3<512>(arrayLength, array, sum, blocks, N); break;      
      case 1024:  run_reduction3<1024>(arrayLength, array, sum, blocks, N); break;      
      }

      t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();

      switch(threads) {
      case 128:  run_reduction4<128>(arrayLength, array, sum, blocks, N); break;
      case 256:  run_reduction4<256>(arrayLength, array, sum, blocks, N); break;
      case 512:  run_reduction4<512>(arrayLength, array, sum, blocks, N); break;      
      case 1024:  run_reduction4<1024>(arrayLength, array, sum, blocks, N); break;      
      }

      t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;

      t1 = std::chrono::high_resolution_clock::now();

      switch(threads) {
      case 128:  run_reduction5<128>(arrayLength, array, sum, blocks, N); break;
      case 256:  run_reduction5<256>(arrayLength, array, sum, blocks, N); break;
      case 512:  run_reduction5<512>(arrayLength, array, sum, blocks, N); break;      
      case 1024:  run_reduction5<1024>(arrayLength, array, sum, blocks, N); break;      
      }

      t2 = std::chrono::high_resolution_clock::now();
      times =  std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
      std::cout << "Thread block size: " << threads << ", ";
      std::cout << "The average performance of reduction is "<< 1.0E-09 * GB/times<<" GBytes/sec"<<std::endl;

      if(sum==checksum)
        std::cout<<"VERIFICATION: PASS"<<std::endl<<std::endl;
      else
        std::cout<<"VERIFICATION: FAIL!!"<<std::endl<<std::endl;
    }
  }
  free(array);
  return 0;
}
