program simpleSpmv
  use iso_c_binding, only: c_long
  use iso_fortran_env, only: int64, real32, real64
  use mv_mod
  implicit none

  integer :: argc, repeat, bs, i
  integer(int64) :: nnz, num_rows, num_elems
  integer(int64), allocatable :: row_indices(:), col_indices(:)
  real(real32), allocatable :: values(:), x(:), y0(:), y1(:), y2(:), y3(:), matrix(:)
  integer(int64) :: elapsed(0:2)
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage ./main <number of non-zero elements> <number of rows in a square matrix> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) nnz
  call get_command_argument(2, arg); read(arg, *) num_rows
  call get_command_argument(3, arg); read(arg, *) repeat
  num_elems = num_rows * num_rows
  if (nnz <= 0 .or. num_rows <= 0 .or. nnz > num_elems) stop 1

  allocate(row_indices(0:num_rows), col_indices(0:nnz-1), values(0:nnz-1), x(0:num_rows-1), &
           y0(0:num_rows-1), y1(0:num_rows-1), y2(0:num_rows-1), y3(0:num_rows-1), matrix(0:num_elems-1))
  call c_srand48(int(ishft(1, 12), c_long))
  call init_matrix(matrix, num_rows, nnz)
  call init_vector(x, num_rows)
  call init_csr(row_indices, values, col_indices, matrix, num_rows, nnz)
  call mv_csr_serial(num_rows, row_indices, col_indices, values, x, y0)

  print '(a,i0)', 'Number of non-zero elements: ', nnz
  print '(a,i0)', 'Number of rows in a square matrix: ', num_rows
  print '(a,f12.6,a)', 'Sparsity: ', real(num_elems - nnz, real64) / real(num_elems, real64) * 100.0_real64, '%'

  bs = 32
  do while (bs <= 1024)
    print '(a)', ''
    print '(a,i0)', 'Thread block size: ', bs
    elapsed(0) = mv_dense_parallel(repeat, bs, num_rows, x, matrix, y1)
    elapsed(1) = mv_csr_parallel(repeat, bs, num_rows, row_indices, col_indices, values, x, nnz, matrix, y2)
    elapsed(2) = vector_mv_csr_parallel(repeat, bs, num_rows, row_indices, col_indices, values, x, nnz, matrix, y3)
    write(*,'(a,3(1x,f12.6))') 'Average dense, sparse, and vector sparse kernel execution time (ms):', &
      (real(elapsed(i), real64) * 1.0e-6_real64 / repeat, i = 0, 2)
    write(*,'(a,3(1x,f10.6))') 'Error rate:', check(y0, y1, num_rows), check(y0, y2, num_rows), check(y0, y3, num_rows)
    bs = bs * 2
  end do
end program simpleSpmv
