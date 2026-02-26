#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <openacc.h>
#include "reference.h"

// limits of integration
#define A 0
#define B 15

// row size is related to accuracy
#define ROW_SIZE 17
#define EPS      1e-7

#pragma acc routine seq
inline double f(double x)
{
  return exp(x)*sin(x);
}

#pragma acc routine seq
inline unsigned int getFirstSetBitPos(int n)
{
  return log2((float)(n&-n))+1;
}

int main( int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of work-groups> ", argv[0]);
    printf("<work-group size> <repeat>\n");
    return 1;
  }
  const int nwg = atoi(argv[1]);
  const int wgs = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  double *result = (double*) malloc (sizeof(double) * nwg);
  double d_sum = 0.0;
  const double base_a = A;
  const double base_b = B;

  //double d_sum;
  //double a = A;
  //double b = B;
  #pragma acc data create(result[0:nwg])
  {
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      #pragma acc parallel num_gangs(nwg) vector_length(wgs) copyin(base_a, base_b)
      {
        double smem[ROW_SIZE * 64];
	#pragma acc loop gang
        for (int blockIdx_x = 0; blockIdx_x < nwg; blockIdx_x++)
        {
          double diff = (base_b - base_a) / nwg;
          double local_a = base_a + blockIdx_x * diff;
          double local_b = base_a + (blockIdx_x + 1) * diff;
          double step = (local_b - local_a) / (1 << (ROW_SIZE - 1));
          int max_eval = (1 << (ROW_SIZE - 1));

          //int threadIdx_x = omp_get_thread_num();
          //int blockIdx_x = omp_get_team_num();
          //int gridDim_x = omp_get_gang();
          //int blockDim_x = omp_get_vector();
          //double diff = (b-a)/gridDim_x, step;
          //int k;
          //int max_eval = (1<<(ROW_SIZE-1));
          //b = a + (blockIdx_x+1)*diff;
          //a += blockIdx_x*diff;
	  #pragma acc loop vector
          for (int threadIdx_x = 0; threadIdx_x < wgs; threadIdx_x++) {

          //step = (b-a)/max_eval;

          double local_col[ROW_SIZE];  // specific to the row size
          for(int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;
          if(!threadIdx_x)
          {
            //k = blockDim_x;
            local_col[0] = f(local_a) + f(local_b);
          }
          //else
          //  k = threadIdx_x;

          for(int k = threadIdx_x; k < max_eval; k += wgs)
            if (k > 0) 
              local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0*f(local_a + step*k);

          for(int i = 0; i < ROW_SIZE; i++)
            smem[ROW_SIZE*threadIdx_x + i] = local_col[i];
	  }
          //#pragma omp barrier

	  #pragma acc loop vector
          for (int v = 0; v < wgs; v++) {
          if(v < ROW_SIZE)
          {
            double sum = 0.0;
            for(int i = v; i < wgs*ROW_SIZE; i+=ROW_SIZE)
              sum += smem[i];
            smem[v] = sum;
          }
          }
	  // auto barrier

	  #pragma acc loop vector
          for (int v = 0; v < wgs; v++) {
          if(!v)
          {
            //double *table = local_col;
            double table[ROW_SIZE];
            table[0] = smem[0];

            for(int k = 1; k < ROW_SIZE; k++)
              table[k] = table[k-1] + smem[k];

            for(int k = 0; k < ROW_SIZE; k++)  
              table[k]*= (local_b-local_a)/(1<<(k+1));

            for(int col = 0 ; col < ROW_SIZE-1 ; col++)
              for(int row = ROW_SIZE-1; row > col; row--)
                table[row] = table[row] + (table[row] - table[row-1])/((1<<(2*col+1))-1);

            result[blockIdx_x] = table[ROW_SIZE-1];
          }
	  }
        }
      }
      #pragma acc update self (result[0:nwg])
      d_sum = 0.0;
      for(int k = 0; k < nwg; k++) d_sum += result[k];
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);
  }

  // verify
  double ref_sum = reference(f, A, B, ROW_SIZE, EPS);
  printf("%s\n", (fabs(d_sum - ref_sum) > EPS) ? "FAIL" : "PASS");

  free(result);
  return 0;
}
