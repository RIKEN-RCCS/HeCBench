#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <chrono>
#include <openacc.h>
#include "common.h"

#define BLOCK_SIZE 16

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

int main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
        input_file = optarg;
        break;
      case 'v':
        do_verify = 1;
        break;
      case 's':
        matrix_dim = atoi(optarg);
        if (matrix_dim <= 0) {
          printf("Matrix dimension must be positive!\n");
          exit(EXIT_FAILURE);
        }
        if (matrix_dim % 16 != 0) {
          printf("Matrix dimension of %d not supported by the benchmark\n", matrix_dim);
          exit(EXIT_FAILURE);
        }
        printf("Generate input matrix internally, size =%d\n", matrix_dim);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
            argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }  

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } 

  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }
  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  /* beginning of timing point */
  stopwatch_start(&sw);

  #pragma acc data copy( m[0:(size_t)matrix_dim*matrix_dim])
  {
  int offset;
  int i=0;
  
  auto start = std::chrono::steady_clock::now();

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
    offset = i;  // add the offset 
    #pragma acc parallel num_gangs(1) vector_length(BLOCK_SIZE) present(m)
    {
      float shadow[BLOCK_SIZE * BLOCK_SIZE];

      #pragma acc loop vector
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        size_t array_offset = offset*(size_t)matrix_dim+offset;
        for(int i=0; i < BLOCK_SIZE; i++){
          shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
          array_offset += matrix_dim;
        }
      }

      for(int i=0; i < BLOCK_SIZE-1; i++) {

        #pragma acc loop vector
        for (int tx = 0; tx < BLOCK_SIZE; tx++) {
          if (tx>i){
            for(int j=0; j < i; j++)
              shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
            shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
          }
        }

        #pragma acc loop vector
        for (int tx = 0; tx < BLOCK_SIZE; tx++) {
          if (tx>i){
            for(int j=0; j < i+1; j++)
              shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
          }
        }
      }

      #pragma acc loop vector
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        size_t array_offset = (offset+1)*(size_t)matrix_dim+offset;
        for(int i=1; i < BLOCK_SIZE; i++){
          m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
          array_offset += matrix_dim;
        }
      }
    }

    {
      int num_peri = (matrix_dim-i)/BLOCK_SIZE-1;
      #pragma acc parallel loop gang num_gangs(num_peri) vector_length(2*BLOCK_SIZE) present(m)
      for (int bx = 0; bx < num_peri; bx++) {
        float dia[BLOCK_SIZE * BLOCK_SIZE];
        float peri_row[BLOCK_SIZE * BLOCK_SIZE];
        float peri_col[BLOCK_SIZE * BLOCK_SIZE];

        #pragma acc loop vector
        for (int tx = 0; tx < 2*BLOCK_SIZE; tx++) {
          size_t array_offset;
          int idx;

          if (tx < BLOCK_SIZE) {
            idx = tx;
            array_offset = offset*(size_t)matrix_dim+offset;
            for (int i=0; i < BLOCK_SIZE/2; i++){
              dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
              array_offset += matrix_dim;
            }

            array_offset = offset*(size_t)matrix_dim+offset;
            for (int i=0; i < BLOCK_SIZE; i++) {
              peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
              array_offset += matrix_dim;
            }

          } else {
            idx = tx-BLOCK_SIZE;

            array_offset = (offset+BLOCK_SIZE/2)*(size_t)matrix_dim+offset;
            for (int i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
              dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
              array_offset += matrix_dim;
            }

            array_offset = (offset+(bx+1)*BLOCK_SIZE)*(size_t)matrix_dim+offset;
            for (int i=0; i < BLOCK_SIZE; i++) {
              peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
              array_offset += matrix_dim;
            }
          }
        }

        #pragma acc loop vector
        for (int tx = 0; tx < 2*BLOCK_SIZE; tx++) {
          if (tx < BLOCK_SIZE) { //peri-row
            int idx=tx;
            for(int i=1; i < BLOCK_SIZE; i++){
              for (int j=0; j < i; j++)
                peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
            }
          } else { //peri-col
            int idx=tx - BLOCK_SIZE;
            for(int i=0; i < BLOCK_SIZE; i++){
              for(int j=0; j < i; j++)
                peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
              peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
            }
          }
        }

        #pragma acc loop vector
        for (int tx = 0; tx < 2*BLOCK_SIZE; tx++) {
          size_t array_offset;
          if (tx < BLOCK_SIZE) { //peri-row
            int idx=tx;
            array_offset = (offset+1)*(size_t)matrix_dim+offset;
            for(int i=1; i < BLOCK_SIZE; i++){
              m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
              array_offset += matrix_dim;
            }
          } else { //peri-col
            int idx=tx - BLOCK_SIZE;
            array_offset = (offset+(bx+1)*BLOCK_SIZE)*(size_t)matrix_dim+offset;
            for(int i=0; i < BLOCK_SIZE; i++){
              m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
              array_offset += matrix_dim;
            }
          }
        }
      }
    }

    {
      int num_int = (matrix_dim-i)/BLOCK_SIZE-1;
      #pragma acc parallel loop gang num_gangs(num_int*num_int) vector_length(BLOCK_SIZE*BLOCK_SIZE) present(m)
      for (int blk = 0; blk < num_int*num_int; blk++) {
        float peri_row[BLOCK_SIZE * BLOCK_SIZE];
        float peri_col[BLOCK_SIZE * BLOCK_SIZE];

        int  bx = blk % num_int;
        int  by = blk / num_int;

        int global_row_id = offset + (by+1)*BLOCK_SIZE;
        int global_col_id = offset + (bx+1)*BLOCK_SIZE;

        #pragma acc loop vector
        for (int tid = 0; tid < BLOCK_SIZE*BLOCK_SIZE; tid++) {
          int  tx = tid % BLOCK_SIZE;
          int  ty = tid / BLOCK_SIZE;

          peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*(size_t)matrix_dim+global_col_id+tx];
          peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*(size_t)matrix_dim+offset+tx];
        }

        #pragma acc loop vector
        for (int tid = 0; tid < BLOCK_SIZE*BLOCK_SIZE; tid++) {
          int  tx = tid % BLOCK_SIZE;
          int  ty = tid / BLOCK_SIZE;

          float sum = 0;
          for (int i=0; i < BLOCK_SIZE; i++)
            sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
          m[(global_row_id+ty)*(size_t)matrix_dim+global_col_id+tx] -= sum;
        }
      }
    }
  } // for

  offset = i;  // add the offset
  #pragma acc parallel num_gangs(1) vector_length(BLOCK_SIZE) present(m)
  {
    float shadow[BLOCK_SIZE * BLOCK_SIZE];

    #pragma acc loop vector
    for (int tx = 0; tx < BLOCK_SIZE; tx++) {
      size_t array_offset = offset*(size_t)matrix_dim+offset;
      for(int i=0; i < BLOCK_SIZE; i++){
        shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
        array_offset += matrix_dim;
      }
    }

    for(int i=0; i < BLOCK_SIZE-1; i++) {
      #pragma acc loop vector
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        if (tx>i) {
          for(int j=0; j < i; j++)
            shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
          shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
        }
      }

      #pragma acc loop vector
      for (int tx = 0; tx < BLOCK_SIZE; tx++) {
        if (tx>i){
          for(int j=0; j < i+1; j++)
            shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
        }
      }
    }

    #pragma acc loop vector
    for (int tx = 0; tx < BLOCK_SIZE; tx++) {
      size_t array_offset = (offset+1)*(size_t)matrix_dim+offset;
      for(int i=1; i < BLOCK_SIZE; i++){
        m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
        array_offset += matrix_dim;
      }
    }
   }

   auto end = std::chrono::steady_clock::now();
   auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
   printf("Total kernel execution time : %f (s)\n", time * 1e-9f);
  } // #pragma acc parallel loop  data map

  /* end of timing point */
  stopwatch_stop(&sw);
  printf("Device offloading time (s): %lf\n", get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);
}
