module mv_mod
  use iso_c_binding, only: c_int, c_long
  use iso_fortran_env, only: int64, real32, real64
  use omp_lib
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
    subroutine c_srand48(seed) bind(C, name="srand48")
      import :: c_long
      integer(c_long), value :: seed
    end subroutine c_srand48
    function c_drand48() bind(C, name="drand48") result(r)
      import :: real64
      real(real64) :: r
    end function c_drand48
  end interface

contains

  subroutine init_vector(vector, m)
    real(real32), intent(out) :: vector(0:)
    integer(int64), intent(in) :: m
    integer(int64) :: i
    do i = 0, m - 1
      vector(i) = real(c_drand48(), real32)
    end do
  end subroutine init_vector

  subroutine init_matrix(matrix, num_rows, nnz)
    real(real32), intent(out) :: matrix(0:)
    integer(int64), intent(in) :: num_rows, nnz
    integer(int64) :: n, i, j, a, b
    integer(int64), allocatable :: d(:)
    integer(int64) :: tmp

    n = num_rows * num_rows
    allocate(d(0:n-1))
    call c_srand(123_c_int)
    do i = 0, n - 1
      d(i) = i
    end do
    do i = n, 1, -1
      a = i - 1
      b = mod(int(c_rand(), int64), i)
      if (a /= b) then
        tmp = d(a)
        d(a) = d(b)
        d(b) = tmp
      end if
    end do
    do i = 0, num_rows - 1
      do j = 0, num_rows - 1
        if (d(i * num_rows + j) >= nnz) then
          matrix(i * num_rows + j) = 0.0_real32
        else
          matrix(i * num_rows + j) = real(c_drand48() + 1.0_real64, real32)
        end if
      end do
    end do
  end subroutine init_matrix

  subroutine init_csr(row_indices, values, col_indices, matrix, num_rows, nnz)
    integer(int64), intent(out) :: row_indices(0:), col_indices(0:)
    real(real32), intent(out) :: values(0:)
    real(real32), intent(in) :: matrix(0:)
    integer(int64), intent(in) :: num_rows, nnz
    integer(int64), allocatable :: non_zero_elements(:)
    integer(int64) :: i, j, tmp, nnz_per_row

    allocate(non_zero_elements(0:num_rows-1))
    row_indices(num_rows) = nnz
    row_indices(0) = 0
    tmp = 0
    do i = 0, num_rows - 1
      nnz_per_row = 0
      do j = 0, num_rows - 1
        if (matrix(i * num_rows + j) /= 0.0_real32) then
          values(tmp) = matrix(i * num_rows + j)
          col_indices(tmp) = j
          tmp = tmp + 1
          nnz_per_row = nnz_per_row + 1
        end if
      end do
      non_zero_elements(i) = nnz_per_row
    end do
    do i = 1, num_rows - 1
      row_indices(i) = row_indices(i - 1) + non_zero_elements(i - 1)
    end do
  end subroutine init_csr

  subroutine mv_csr_serial(num_rows, row_indices, col_indices, values, x, y)
    integer(int64), intent(in) :: num_rows, row_indices(0:), col_indices(0:)
    real(real32), intent(in) :: values(0:), x(0:)
    real(real32), intent(out) :: y(0:)
    integer(int64) :: row, i
    real(real32) :: dot
    do row = 0, num_rows - 1
      dot = 0.0_real32
      do i = row_indices(row), row_indices(row + 1) - 1
        dot = dot + values(i) * x(col_indices(i))
      end do
      y(row) = dot
    end do
  end subroutine mv_csr_serial

  real(real32) function check(a, b, n) result(rate)
    real(real32), intent(in) :: a(0:), b(0:)
    integer(int64), intent(in) :: n
    integer(int64) :: i
    real(real64) :: diffsum, sumv
    diffsum = 0.0_real64
    sumv = 0.0_real64
    do i = 0, n - 1
      diffsum = diffsum + abs(real(a(i) - b(i), real64))
      sumv = sumv + abs(real(b(i), real64))
    end do
    rate = real(diffsum / sumv, real32)
  end function check

  integer function prev_power_of_2(v_in) result(r)
    integer, intent(in) :: v_in
    integer :: v
    v = v_in - 1
    v = ior(v, ishft(v, -1))
    v = ior(v, ishft(v, -2))
    v = ior(v, ishft(v, -4))
    v = ior(v, ishft(v, -8))
    v = ior(v, ishft(v, -16))
    v = v + 1
    r = ishft(v, -1)
  end function prev_power_of_2

  integer(int64) function mv_dense_parallel(repeat, bs, num_rows, x, matrix, y) result(elapsed_ns)
    integer, intent(in) :: repeat, bs
    integer(int64), intent(in) :: num_rows
    real(real32), intent(in) :: x(0:), matrix(0:)
    real(real32), intent(out) :: y(0:)
    integer :: n
    integer(int64) :: i, j
    real(real32) :: temp
    real(real64) :: start_time, end_time
    !$omp target data map(to: matrix(0:num_rows*num_rows-1), x(0:num_rows-1)) map(from: y(0:num_rows-1))
      start_time = omp_get_wtime()
      do n = 0, repeat - 1
        !$omp target teams distribute parallel do num_threads(bs) private(j,temp)
        do i = 0, num_rows - 1
          temp = 0.0_real32
          do j = 0, num_rows - 1
            if (matrix(i * num_rows + j) /= 0.0_real32) temp = temp + matrix(i * num_rows + j) * x(j)
          end do
          y(i) = temp
        end do
        !$omp end target teams distribute parallel do
      end do
      end_time = omp_get_wtime()
      elapsed_ns = int((end_time - start_time) * 1.0e9_real64, int64)
    !$omp end target data
  end function mv_dense_parallel

  integer(int64) function mv_csr_parallel(repeat, bs, num_rows, row_indices, col_indices, values, x, nnz, matrix, y) result(elapsed_ns)
    integer, intent(in) :: repeat, bs
    integer(int64), intent(in) :: num_rows, nnz, row_indices(0:), col_indices(0:)
    real(real32), intent(in) :: values(0:), x(0:), matrix(0:)
    real(real32), intent(out) :: y(0:)
    integer :: n
    integer(int64) :: i, j, row_start, row_end
    real(real32) :: temp
    real(real64) :: start_time, end_time
    !$omp target data map(to: row_indices(0:num_rows), col_indices(0:nnz-1), values(0:nnz-1), x(0:num_rows-1)) map(from: y(0:num_rows-1))
      start_time = omp_get_wtime()
      do n = 0, repeat - 1
        !$omp target teams distribute parallel do num_threads(bs) private(j,row_start,row_end,temp)
        do i = 0, num_rows - 1
          row_start = row_indices(i); row_end = row_indices(i + 1)
          temp = 0.0_real32
          do j = row_start, row_end - 1
            temp = temp + values(j) * x(col_indices(j))
          end do
          y(i) = temp
        end do
        !$omp end target teams distribute parallel do
      end do
      end_time = omp_get_wtime()
      elapsed_ns = int((end_time - start_time) * 1.0e9_real64, int64)
    !$omp end target data
  end function mv_csr_parallel

  integer(int64) function vector_mv_csr_parallel(repeat, bs, num_rows, row_indices, col_indices, values, x, nnz, matrix, y) result(elapsed_ns)
    integer, intent(in) :: repeat, bs
    integer(int64), intent(in) :: num_rows, nnz, row_indices(0:), col_indices(0:)
    real(real32), intent(in) :: values(0:), x(0:), matrix(0:)
    real(real32), intent(out) :: y(0:)
    integer :: n, nnz_per_row, threads_per_row, warp_size, rows_per_block, num_blocks
    integer(int64) :: i, j, row_start, row_end
    real(real32) :: temp
    real(real64) :: start_time, end_time

    nnz_per_row = int(nnz / num_rows)
    threads_per_row = prev_power_of_2(nnz_per_row)
    warp_size = 32
    if (threads_per_row > warp_size) threads_per_row = warp_size
    rows_per_block = bs / threads_per_row
    if (rows_per_block == 0) rows_per_block = 1
    num_blocks = int((num_rows + rows_per_block - 1) / rows_per_block)
    !$omp target data map(to: row_indices(0:num_rows), col_indices(0:nnz-1), values(0:nnz-1), x(0:num_rows-1)) map(from: y(0:num_rows-1))
      start_time = omp_get_wtime()
      do n = 0, repeat - 1
        !$omp target teams distribute num_teams(num_blocks * rows_per_block)
        do i = 0, num_rows - 1
          row_start = row_indices(i); row_end = row_indices(i + 1)
          temp = 0.0_real32
          !$omp parallel do num_threads(threads_per_row) reduction(+:temp)
          do j = row_start, row_end - 1
            temp = temp + values(j) * x(col_indices(j))
          end do
          !$omp end parallel do
          y(i) = temp
        end do
        !$omp end target teams distribute
      end do
      end_time = omp_get_wtime()
      elapsed_ns = int((end_time - start_time) * 1.0e9_real64, int64)
    !$omp end target data
  end function vector_mv_csr_parallel

end module mv_mod
