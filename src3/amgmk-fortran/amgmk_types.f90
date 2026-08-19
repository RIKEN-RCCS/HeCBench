module amgmk_types
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  integer, parameter :: nx = 50, ny = 50, nz = 50
  integer, parameter :: nrows = nx * ny * nz
  integer, parameter :: max_nnz = 7 * nrows
  integer, parameter :: test_iter = 500, block_size = 256
  type :: csr_matrix
    integer :: num_rows = 0, num_cols = 0, num_nonzeros = 0
    integer, allocatable :: row_ptr(:), col_ind(:)
    real(real64), allocatable :: data(:)
  end type csr_matrix
  type :: seq_vector
    integer :: size = 0
    real(real64), allocatable :: data(:)
  end type seq_vector
end module amgmk_types
