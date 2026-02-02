#include <stdlib.h>
#include <chrono>
#include <openacc.h>
#include "mv.h"

// dense matrix vector multiply
long mv_dense_parallel(const int repeat,
                       const int bs,
                       const size_t num_rows,
                       const REAL* x,
                             REAL* matrix,
                             REAL* y)
{
  long time;

#pragma acc data copyin(matrix[0:num_rows*num_rows],	\
			x[0:num_rows])			\
  copyout(y[0:num_rows])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      #pragma acc parallel loop vector_length(bs)
      for (size_t i = 0; i < num_rows; i++) {
        REAL temp = 0;
        for (size_t j = 0; j < num_rows; j++) {
          if (matrix[i * num_rows + j] != (REAL)0) 
            temp += matrix[i * num_rows + j] * x[j];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}

// sparse matrix vector multiply using the CSR format
long mv_csr_parallel(const int repeat,
                     const int bs,
                     const size_t num_rows,
                     const size_t *row_indices,
                     const size_t *col_indices,
                     const REAL* values,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  long time;

#pragma acc data copyin(row_indices[0:num_rows+1],	\
			col_indices[0:nnz],		\
			values[0:nnz],			\
			x[0:num_rows])			\
  copyout(y[0:num_rows])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      #pragma acc parallel loop vector_length(bs)
      for (size_t i = 0; i < num_rows; i++) {
        size_t row_start = row_indices[i];
        size_t row_end = row_indices[i+1];

        REAL temp = 0;
        for(size_t j = row_start; j < row_end; j++){
          temp += values[j] * x[col_indices[j]];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}

// Reference
// https://github.com/ROCm/rocm-blogs/blob/release/blogs/high-performance-computing/spmv/part-1/examples/vector_csr.cpp
size_t prevPowerOf2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}

// sparse matrix vector multiply using the CSR format
long vector_mv_csr_parallel(const int repeat,
                            const int bs,
                            const size_t num_rows,
                            const size_t *row_indices,
                            const size_t *col_indices,
                            const REAL* values,
                            const REAL* x,
                            const size_t nnz,
                            REAL* matrix,
                            REAL* y)
{
  long time;

  int nnz_per_row = nnz / num_rows;
  int threads_per_row = prevPowerOf2(nnz_per_row);
  int warpSize = 32;
  // limit the number of threads per row to be no larger than the wavefront (warp) size
  threads_per_row = threads_per_row > warpSize ? warpSize : threads_per_row;
  int rows_per_block = bs / threads_per_row;
  if (rows_per_block == 0) rows_per_block = 1;
  int num_blocks = (num_rows + rows_per_block - 1) / rows_per_block;

#pragma acc data copyin(row_indices[0:num_rows+1],	\
			col_indices[0:nnz],		\
			values[0:nnz],			\
			x[0:num_rows])			\
  copyout(y[0:num_rows])
  {
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
#pragma acc parallel loop num_gangs(num_blocks * rows_per_block) vector_length(threads_per_row)
      for (size_t i = 0; i < num_rows; i++) {
        size_t row_start = row_indices[i];
        size_t row_end = row_indices[i+1];

        REAL temp = 0;
#pragma acc loop vector reduction(+:temp)
        for(size_t j = row_start; j < row_end; j++){
          temp += values[j] * x[col_indices[j]];
        }
        y[i] = temp;
      }
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return time;
}
