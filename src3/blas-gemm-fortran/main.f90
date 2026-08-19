! Faithful OpenMP GPU Fortran port of blas-gemm-omp/main.cpp.
!
! The OMP source dispatches MKL hgemm/sgemm/dgemm from inside one target-data
! region for each precision.  NVFORTRAN has no compatible MKL dispatch API, so
! each dispatch is represented by its equivalent row-major GEMM computation on
! the target device.  Matrix storage and update remain C = 2*A*B + 0.5*C.
program blas_gemm
  use, intrinsic :: iso_fortran_env, only : real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  integer :: m, k, n, repeat
  character(len=64) :: argument
  integer :: ios
  ! NVFORTRAN selects its IEEE binary16 kind here.  Compilers without native
  ! binary16 (such as the local gfortran syntax checker) select real32 instead.
  integer, parameter :: real16 = selected_real_kind(3, 5)

  if (command_argument_count() /= 4) then
    write(*, '(A)') 'Usage: ./main <m> <k> <n> <repeat>'
    stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *, iostat=ios) m
  if (ios /= 0) stop 1
  call get_command_argument(2, argument)
  read(argument, *, iostat=ios) k
  if (ios /= 0) stop 1
  call get_command_argument(3, argument)
  read(argument, *, iostat=ios) n
  if (ios /= 0) stop 1
  call get_command_argument(4, argument)
  read(argument, *, iostat=ios) repeat
  if (ios /= 0) stop 1

  if (m <= 0 .or. k <= 0 .or. n <= 0 .or. repeat <= 0) then
    write(*, '(A)') 'All dimensions and repeat must be positive.'
    stop 1
  end if

  write(*, '(A)') achar(9)//'Running with half precision data type:'
  call run_gemm_r2(m, k, n, repeat)

  write(*, '(A)') achar(9)//'Running with single precision data type:'
  call run_gemm_r4(m, k, n, repeat)

  write(*, '(A)') achar(9)//'Running with double precision data type:'
  call run_gemm_r8(m, k, n, repeat)

contains

  subroutine fill_r2(matrix, rows, columns)
    integer, intent(in) :: rows, columns
    real(real16), intent(out) :: matrix(0:)
    integer :: i, j

    do i = 0, rows - 1
      do j = 0, columns - 1
        matrix(i * columns + j) = real(mod(c_rand(), 2_c_int), kind=real16)
      end do
    end do
  end subroutine fill_r2

  subroutine fill_r4(matrix, rows, columns)
    integer, intent(in) :: rows, columns
    real(real32), intent(out) :: matrix(0:)
    integer :: i, j

    do i = 0, rows - 1
      do j = 0, columns - 1
        matrix(i * columns + j) = real(mod(c_rand(), 2_c_int), kind=real32)
      end do
    end do
  end subroutine fill_r4

  subroutine fill_r8(matrix, rows, columns)
    integer, intent(in) :: rows, columns
    real(real64), intent(out) :: matrix(0:)
    integer :: i, j

    do i = 0, rows - 1
      do j = 0, columns - 1
        matrix(i * columns + j) = real(mod(c_rand(), 2_c_int), kind=real64)
      end do
    end do
  end subroutine fill_r8

  subroutine run_gemm_r2(m, k, n, repeat)
    integer, intent(in) :: m, k, n, repeat
    real(real16), allocatable :: a(:), b(:), c(:)
    real(real16) :: alpha, beta, sum
    integer :: i, j, l, iteration, count_start, count_stop, count_rate
    real(real64) :: elapsed_us

    alpha = real(2.0, kind=real16)
    beta = real(0.5, kind=real16)
    allocate(a(0:m * k - 1), b(0:k * n - 1), c(0:m * n - 1))
    call c_srand(2_c_int)
    call fill_r2(a, m, k)
    call fill_r2(b, k, n)
    call fill_r2(c, m, n)

    !$omp target data map(to:a, b) map(tofrom:c)
      call system_clock(count_start, count_rate)
      do iteration = 1, repeat
        !$omp target teams distribute parallel do collapse(2) private(sum, l)
        do i = 0, m - 1
          do j = 0, n - 1
            sum = real(0.0, kind=real16)
            do l = 0, k - 1
              sum = sum + a(i * k + l) * b(l * n + j)
            end do
            c(i * n + j) = alpha * sum + beta * c(i * n + j)
          end do
        end do
        !$omp end target teams distribute parallel do
      end do
      call system_clock(count_stop)
    !$omp end target data

    elapsed_us = real(count_stop - count_start, real64) * 1.0e6_real64 / real(count_rate, real64) / real(repeat, real64)
    write(*, '(A,F0.6,A)') 'Average GEMM execution time: ', elapsed_us, ' (us)'
    call print_c_r2(c, n)
    deallocate(a, b, c)
  end subroutine run_gemm_r2

  subroutine run_gemm_r4(m, k, n, repeat)
    integer, intent(in) :: m, k, n, repeat
    real(real32), allocatable :: a(:), b(:), c(:)
    real(real32) :: alpha, beta, sum
    integer :: i, j, l, iteration, count_start, count_stop, count_rate
    real(real64) :: elapsed_us

    alpha = 2.0_real32
    beta = 0.5_real32
    allocate(a(0:m * k - 1), b(0:k * n - 1), c(0:m * n - 1))
    call c_srand(2_c_int)
    call fill_r4(a, m, k)
    call fill_r4(b, k, n)
    call fill_r4(c, m, n)

    !$omp target data map(to:a, b) map(tofrom:c)
      call system_clock(count_start, count_rate)
      do iteration = 1, repeat
        !$omp target teams distribute parallel do collapse(2) private(sum, l)
        do i = 0, m - 1
          do j = 0, n - 1
            sum = 0.0_real32
            do l = 0, k - 1
              sum = sum + a(i * k + l) * b(l * n + j)
            end do
            c(i * n + j) = alpha * sum + beta * c(i * n + j)
          end do
        end do
        !$omp end target teams distribute parallel do
      end do
      call system_clock(count_stop)
    !$omp end target data

    elapsed_us = real(count_stop - count_start, real64) * 1.0e6_real64 / real(count_rate, real64) / real(repeat, real64)
    write(*, '(A,F0.6,A)') 'Average GEMM execution time: ', elapsed_us, ' (us)'
    call print_c_r4(c, n)
    deallocate(a, b, c)
  end subroutine run_gemm_r4

  subroutine run_gemm_r8(m, k, n, repeat)
    integer, intent(in) :: m, k, n, repeat
    real(real64), allocatable :: a(:), b(:), c(:)
    real(real64) :: alpha, beta, sum
    integer :: i, j, l, iteration, count_start, count_stop, count_rate
    real(real64) :: elapsed_us

    alpha = 2.0_real64
    beta = 0.5_real64
    allocate(a(0:m * k - 1), b(0:k * n - 1), c(0:m * n - 1))
    call c_srand(2_c_int)
    call fill_r8(a, m, k)
    call fill_r8(b, k, n)
    call fill_r8(c, m, n)

    !$omp target data map(to:a, b) map(tofrom:c)
      call system_clock(count_start, count_rate)
      do iteration = 1, repeat
        !$omp target teams distribute parallel do collapse(2) private(sum, l)
        do i = 0, m - 1
          do j = 0, n - 1
            sum = 0.0_real64
            do l = 0, k - 1
              sum = sum + a(i * k + l) * b(l * n + j)
            end do
            c(i * n + j) = alpha * sum + beta * c(i * n + j)
          end do
        end do
        !$omp end target teams distribute parallel do
      end do
      call system_clock(count_stop)
    !$omp end target data

    elapsed_us = real(count_stop - count_start, real64) * 1.0e6_real64 / real(count_rate, real64) / real(repeat, real64)
    write(*, '(A,F0.6,A)') 'Average GEMM execution time: ', elapsed_us, ' (us)'
    call print_c_r8(c, n)
    deallocate(a, b, c)
  end subroutine run_gemm_r8

  subroutine print_c_r2(c, n)
    real(real16), intent(in) :: c(0:)
    integer, intent(in) :: n
    write(*, '(A)') ''
    write(*, '(A,F0.6,A,F0.6,A)') achar(9)//achar(9)//achar(9)//'C = [ ', real(c(0), real32), ', ', real(c(n), real32), ', ...'
    write(*, '(A,F0.6,A,F0.6,A)') achar(9)//achar(9)//achar(9)//'    [ ', real(c(1), real32), ', ', real(c(n + 1), real32), ', ...'
    write(*, '(A)') achar(9)//achar(9)//achar(9)//'    [ ...'
    write(*, '(A)') ''
  end subroutine print_c_r2

  subroutine print_c_r4(c, n)
    real(real32), intent(in) :: c(0:)
    integer, intent(in) :: n
    write(*, '(A)') ''
    write(*, '(A,F0.6,A,F0.6,A)') achar(9)//achar(9)//achar(9)//'C = [ ', c(0), ', ', c(n), ', ...'
    write(*, '(A,F0.6,A,F0.6,A)') achar(9)//achar(9)//achar(9)//'    [ ', c(1), ', ', c(n + 1), ', ...'
    write(*, '(A)') achar(9)//achar(9)//achar(9)//'    [ ...'
    write(*, '(A)') ''
  end subroutine print_c_r4

  subroutine print_c_r8(c, n)
    real(real64), intent(in) :: c(0:)
    integer, intent(in) :: n
    write(*, '(A)') ''
    write(*, '(A,F0.6,A,F0.6,A)') achar(9)//achar(9)//achar(9)//'C = [ ', c(0), ', ', c(n), ', ...'
    write(*, '(A,F0.6,A,F0.6,A)') achar(9)//achar(9)//achar(9)//'    [ ', c(1), ', ', c(n + 1), ', ...'
    write(*, '(A)') achar(9)//achar(9)//achar(9)//'    [ ...'
    write(*, '(A)') ''
  end subroutine print_c_r8

end program blas_gemm
