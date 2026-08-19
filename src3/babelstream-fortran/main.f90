! BabelStream OpenMP target Fortran translation of babelstream-omp/main.cpp.
program babelstream
  use, intrinsic :: iso_fortran_env, only : real32, real64
  use omp_lib
  implicit none

  integer, parameter :: tbsize = 256
  integer :: array_size = 33554432
  integer :: num_times = 100

  call parse_arguments(array_size, num_times)
  call run_sp(array_size, num_times)
  call run_dp(array_size, num_times)

contains

  subroutine parse_arguments(n, ntimes)
    integer, intent(inout) :: n, ntimes
    integer :: i, ios, value
    character(len=256) :: arg, next

    i = 1
    do while (i <= command_argument_count())
      call get_command_argument(i, arg)
      select case (trim(arg))
      case ('-s', '--arraysize')
        if (i == command_argument_count()) error stop 'Invalid array size.'
        i = i + 1
        call get_command_argument(i, next)
        read(next, *, iostat=ios) value
        if (ios /= 0 .or. value <= 0) error stop 'Invalid array size.'
        n = value
      case ('-n', '--numtimes')
        if (i == command_argument_count()) error stop 'Invalid number of times.'
        i = i + 1
        call get_command_argument(i, next)
        read(next, *, iostat=ios) value
        if (ios /= 0 .or. value < 2) error stop 'Number of times must be 2 or more'
        ntimes = value
      case ('-h', '--help')
        print '(a)', ''
        print '(a)', 'Usage: ./main [OPTIONS]'
        print '(a)', ''
        print '(a)', 'Options:'
        print '(a)', '  -h  --help               Print the message'
        print '(a)', '  -s  --arraysize  SIZE    Use SIZE elements in the array'
        print '(a)', '  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)'
        stop
      case default
        write(*, '(3a)') 'Unrecognized argument ''', trim(arg), ''' (try ''--help'')'
        error stop
      end select
      i = i + 1
    end do
  end subroutine parse_arguments

  subroutine run_sp(n, ntimes)
    integer, intent(in) :: n, ntimes
    real(real32), allocatable :: a(:), b(:), c(:)
    real(real64) :: timings(6, ntimes)
    real(real64) :: t1, t2
    real(real32) :: sumdot
    integer :: i, k

    call print_header(n, ntimes, storage_size(0.0_real32)/8, 'float')
    if (mod(n, tbsize) /= 0) error stop 'Array size must be a multiple of 256'
    allocate(a(0:n-1), b(0:n-1), c(0:n-1))

!$omp target data map(alloc:a(0:n-1), b(0:n-1), c(0:n-1))
!$omp target teams distribute parallel do simd thread_limit(tbsize)
    do i = 0, n - 1
      a(i) = 0.1_real32
      b(i) = 0.2_real32
      c(i) = 0.0_real32
    end do
!$omp end target teams distribute parallel do simd

    do k = 1, ntimes
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        c(i) = a(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(1, k) = t2 - t1

      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        b(i) = 0.4_real32 * c(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(2, k) = t2 - t1

      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        c(i) = a(i) + b(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(3, k) = t2 - t1

      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        a(i) = b(i) + 0.4_real32 * c(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(4, k) = t2 - t1

      sumdot = 0.0_real32
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize) reduction(+:sumdot)
      do i = 0, n - 1
        sumdot = sumdot + a(i) * b(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(5, k) = t2 - t1

      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        a(i) = a(i) + b(i) + 0.4_real32 * c(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(6, k) = t2 - t1
    end do
!$omp end target data

    call print_results(timings, n, ntimes, storage_size(0.0_real32)/8)
    deallocate(a, b, c)
  end subroutine run_sp

  subroutine run_dp(n, ntimes)
    integer, intent(in) :: n, ntimes
    real(real64), allocatable :: a(:), b(:), c(:)
    real(real64) :: timings(6, ntimes)
    real(real64) :: t1, t2, sumdot
    integer :: i, k

    call print_header(n, ntimes, storage_size(0.0_real64)/8, 'double')
    if (mod(n, tbsize) /= 0) error stop 'Array size must be a multiple of 256'
    allocate(a(0:n-1), b(0:n-1), c(0:n-1))

!$omp target data map(alloc:a(0:n-1), b(0:n-1), c(0:n-1))
!$omp target teams distribute parallel do simd thread_limit(tbsize)
    do i = 0, n - 1
      a(i) = 0.1_real64; b(i) = 0.2_real64; c(i) = 0.0_real64
    end do
!$omp end target teams distribute parallel do simd
    do k = 1, ntimes
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        c(i) = a(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(1, k) = t2 - t1
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        b(i) = 0.4_real64 * c(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(2, k) = t2 - t1
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        c(i) = a(i) + b(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(3, k) = t2 - t1
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        a(i) = b(i) + 0.4_real64 * c(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(4, k) = t2 - t1
      sumdot = 0.0_real64
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize) reduction(+:sumdot)
      do i = 0, n - 1
        sumdot = sumdot + a(i) * b(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(5, k) = t2 - t1
      t1 = omp_get_wtime()
!$omp target teams distribute parallel do simd thread_limit(tbsize)
      do i = 0, n - 1
        a(i) = a(i) + b(i) + 0.4_real64 * c(i)
      end do
!$omp end target teams distribute parallel do simd
      t2 = omp_get_wtime(); timings(6, k) = t2 - t1
    end do
!$omp end target data

    call print_results(timings, n, ntimes, storage_size(0.0_real64)/8)
    deallocate(a, b, c)
  end subroutine run_dp

  subroutine print_header(n, ntimes, bytes, precision_name)
    integer, intent(in) :: n, ntimes, bytes
    character(*), intent(in) :: precision_name
    print '(a,i0,a)', 'Running kernels ', ntimes, ' times'
    print '(a,a)', 'Precision: ', trim(precision_name)
    write(*, '(a,f9.1,a,f9.1,a)') 'Array size: ', real(n*bytes, real64)*1.0e-6_real64, ' MB (= ', &
      real(n*bytes, real64)*1.0e-9_real64, ' GB)'
    write(*, '(a,f9.1,a,f9.1,a)') 'Total size: ', real(3*n*bytes, real64)*1.0e-6_real64, ' MB (= ', &
      real(3*n*bytes, real64)*1.0e-9_real64, ' GB)'
  end subroutine print_header

  subroutine print_results(timings, n, ntimes, bytes)
    real(real64), intent(in) :: timings(:, :)
    integer, intent(in) :: n, ntimes, bytes
    character(len=12), parameter :: labels(6) = [character(len=12) :: 'Copy', 'Mul', 'Add', 'Triad', 'Dot', 'Nstream']
    integer, parameter :: factors(6) = [2, 2, 3, 3, 2, 4]
    integer :: j
    real(real64) :: minimum, maximum, average, bandwidth

    print '(a12,a12,a12,a12,a12)', 'Function', 'MBytes/sec', 'Min (sec)', 'Max', 'Average'
    do j = 1, 6
      minimum = minval(timings(j, 2:ntimes))
      maximum = maxval(timings(j, 2:ntimes))
      average = sum(timings(j, 2:ntimes)) / real(ntimes - 1, real64)
      bandwidth = 1.0e-6_real64 * real(factors(j) * bytes * n, real64) / minimum
      write(*, '(a12,1x,f11.3,1x,f11.5,1x,f11.5,1x,f11.5)') trim(labels(j)), bandwidth, minimum, maximum, average
    end do
    print '(a)', ''
  end subroutine print_results
end program babelstream
