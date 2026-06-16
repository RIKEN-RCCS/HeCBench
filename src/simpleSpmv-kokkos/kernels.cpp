#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "../simpleSpmv-cuda/mv.h"

// Dense matrix-vector multiply
long mv_dense_parallel(const int repeat,
                       const int bs,
                       const size_t num_rows,
                       const REAL* x,
                       REAL* matrix,
                       REAL* y)
{
  Kokkos::View<REAL*> d_matrix("matrix", num_rows * num_rows);
  Kokkos::View<REAL*> d_x("x",           num_rows);
  Kokkos::View<REAL*> d_y("y",           num_rows);

  {
    auto h_matrix = Kokkos::View<REAL*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(matrix, num_rows * num_rows);
    auto h_x      = Kokkos::View<REAL*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<REAL*>(x), num_rows);
    Kokkos::deep_copy(d_matrix, h_matrix);
    Kokkos::deep_copy(d_x, h_x);
  }

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("mv_dense",
      Kokkos::RangePolicy<>(0, (int)num_rows),
      KOKKOS_LAMBDA(int i) {
        REAL temp = 0;
        for (size_t j = 0; j < num_rows; j++) {
          if (d_matrix(i * num_rows + j) != (REAL)0)
            temp += d_matrix(i * num_rows + j) * d_x(j);
        }
        d_y(i) = temp;
      });
  }
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  long time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  {
    auto h_y = Kokkos::View<REAL*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(y, num_rows);
    Kokkos::deep_copy(h_y, d_y);
  }
  return time;
}

// Sparse CSR matrix-vector multiply
long mv_csr_parallel(const int repeat,
                     const int bs,
                     const size_t num_rows,
                     const size_t* row_indices,
                     const size_t* col_indices,
                     const REAL* values,
                     const REAL* x,
                     const size_t nnz,
                     REAL* matrix,
                     REAL* y)
{
  Kokkos::View<size_t*> d_row("row_idx",  num_rows + 1);
  Kokkos::View<size_t*> d_col("col_idx",  nnz);
  Kokkos::View<REAL*>   d_val("values",   nnz);
  Kokkos::View<REAL*>   d_x("x",          num_rows);
  Kokkos::View<REAL*>   d_y("y",          num_rows);

  {
    auto h_row = Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<size_t*>(row_indices), num_rows + 1);
    auto h_col = Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<size_t*>(col_indices), nnz);
    auto h_val = Kokkos::View<REAL*,   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<REAL*>(values), nnz);
    auto h_x   = Kokkos::View<REAL*,   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<REAL*>(x), num_rows);
    Kokkos::deep_copy(d_row, h_row);
    Kokkos::deep_copy(d_col, h_col);
    Kokkos::deep_copy(d_val, h_val);
    Kokkos::deep_copy(d_x,   h_x);
  }

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("mv_csr",
      Kokkos::RangePolicy<>(0, (int)num_rows),
      KOKKOS_LAMBDA(int i) {
        size_t row_start = d_row(i);
        size_t row_end   = d_row(i + 1);
        REAL temp = 0;
        for (size_t j = row_start; j < row_end; j++)
          temp += d_val(j) * d_x(d_col(j));
        d_y(i) = temp;
      });
  }
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  long time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  {
    auto h_y = Kokkos::View<REAL*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(y, num_rows);
    Kokkos::deep_copy(h_y, d_y);
  }
  return time;
}

// Vector CSR (warp-based) sparse matrix-vector multiply
static size_t prevPowerOf2(size_t v) {
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  v++;
  return v >> 1;
}

long vector_mv_csr_parallel(const int repeat,
                            const int bs,
                            const size_t num_rows,
                            const size_t* row_indices,
                            const size_t* col_indices,
                            const REAL* values,
                            const REAL* x,
                            const size_t nnz,
                            REAL* matrix,
                            REAL* y)
{
  Kokkos::View<size_t*> d_row("row_idx",  num_rows + 1);
  Kokkos::View<size_t*> d_col("col_idx",  nnz);
  Kokkos::View<REAL*>   d_val("values",   nnz);
  Kokkos::View<REAL*>   d_x("x",          num_rows);
  Kokkos::View<REAL*>   d_y("y",          num_rows);

  {
    auto h_row = Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<size_t*>(row_indices), num_rows + 1);
    auto h_col = Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<size_t*>(col_indices), nnz);
    auto h_val = Kokkos::View<REAL*,   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<REAL*>(values), nnz);
    auto h_x   = Kokkos::View<REAL*,   Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(const_cast<REAL*>(x), num_rows);
    Kokkos::deep_copy(d_row, h_row);
    Kokkos::deep_copy(d_col, h_col);
    Kokkos::deep_copy(d_val, h_val);
    Kokkos::deep_copy(d_x,   h_x);
  }

  int nnz_per_row    = (int)(nnz / num_rows);
  int threads_per_row= (int)prevPowerOf2(nnz_per_row);
  int warpSize       = 32;
  threads_per_row    = threads_per_row > warpSize ? warpSize : threads_per_row;
  int rows_per_block = bs / threads_per_row;
  if (rows_per_block == 0) rows_per_block = 1;
  int num_blocks     = ((int)num_rows + rows_per_block - 1) / rows_per_block;

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    Kokkos::parallel_for("vector_mv_csr",
      Kokkos::TeamPolicy<>((int)num_rows, threads_per_row),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        int i = team.league_rank();
        size_t row_start = d_row(i);
        size_t row_end   = d_row(i + 1);
        REAL temp = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, (int)(row_end - row_start)),
          [=](int jj, REAL& lsum) {
            size_t j = row_start + jj;
            lsum += d_val(j) * d_x(d_col(j));
          }, temp);
        if (team.team_rank() == 0) d_y(i) = temp;
      });
  }
  Kokkos::fence();

  auto end  = std::chrono::steady_clock::now();
  long time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  {
    auto h_y = Kokkos::View<REAL*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(y, num_rows);
    Kokkos::deep_copy(h_y, d_y);
  }
  return time;
}
