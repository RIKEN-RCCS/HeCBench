module popcount_mod
  use iso_fortran_env, only: int32, int64, real64
  implicit none
  integer(int64), parameter :: m1 = int(z'5555555555555555', int64)
  integer(int64), parameter :: m2 = int(z'3333333333333333', int64)
  integer(int64), parameter :: m4 = int(z'0f0f0f0f0f0f0f0f', int64)
  integer(int64), parameter :: h01 = int(z'0101010101010101', int64)
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  integer function popcount_ref(x0) result(count)
    integer(int64), intent(in) :: x0
    integer(int64) :: x
    count = 0; x = x0
    do while (x /= 0_int64)
      count = count + 1
      x = iand(x, x - 1_int64)
    end do
  end function

  subroutine check_results(d, r, length)
    integer, intent(in) :: length
    integer(int64), intent(in) :: d(0:length-1)
    integer(int32), intent(in) :: r(0:length-1)
    integer :: i
    logical :: error
    error = .false.
    do i = 0, length-1
      if (popcount_ref(d(i)) /= r(i)) then
        error = .true.; exit
      end if
    end do
    print '(a)', merge('Fail   ', 'Success', error)
  end subroutine
end module

program main
  use popcount_mod
  implicit none
  integer :: length, repeat, i, n, c, ios, byte
  character(len=64) :: arg
  integer(int64), allocatable :: data(:)
  integer(int32), allocatable :: result(:)
  integer(int64) :: x
  integer, parameter :: block_size = 256
  real(real64) :: t0, t1
  integer, parameter :: lut(0:255) = [ &
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, &
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, &
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, &
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7, &
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, &
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7, &
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7, &
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8 ]

  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: main <length> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) length
  call get_command_argument(2,arg); read(arg,*,iostat=ios) repeat
  allocate(data(0:length-1), result(0:length-1))
  call c_srand(2_int32)
  do i = 0, length-1
    data(i) = ior(ishft(int(c_rand(),int64), 32), int(c_rand(),int64))
  end do

  !$omp target data map(to:data(0:length-1)) map(alloc:result(0:length-1))
  t0 = seconds()
  do n = 1, repeat
    !$omp target teams distribute parallel do private(x) thread_limit(block_size)
    do i = 0, length-1
      x = data(i)
      x = x - iand(ishft(x,-1), m1)
      x = iand(x, m2) + iand(ishft(x,-2), m2)
      x = iand(x + ishft(x,-4), m4)
      x = x + ishft(x,-8); x = x + ishft(x,-16); x = x + ishft(x,-32)
      result(i) = int(iand(x, 127_int64), int32)
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (pc1): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp target update from(result(0:length-1))
  call check_results(data, result, length)

  t0 = seconds()
  do n = 1, repeat
    !$omp target teams distribute parallel do private(x) thread_limit(block_size)
    do i = 0, length-1
      x = data(i)
      x = x - iand(ishft(x,-1), m1)
      x = iand(x, m2) + iand(ishft(x,-2), m2)
      x = iand(x + ishft(x,-4), m4)
      result(i) = int(ishft(x * h01, -56), int32)
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (pc2): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp target update from(result(0:length-1))
  call check_results(data, result, length)

  t0 = seconds()
  do n = 1, repeat
    !$omp target teams distribute parallel do private(x,c) thread_limit(block_size)
    do i = 0, length-1
      c = 0; x = data(i)
      do while (x /= 0_int64)
        c = c + 1; x = iand(x, x - 1_int64)
      end do
      result(i) = c
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (pc3): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp target update from(result(0:length-1))
  call check_results(data, result, length)

  t0 = seconds()
  do n = 1, repeat
    !$omp target teams distribute parallel do private(x,c,byte) thread_limit(block_size)
    do i = 0, length-1
      c = 0; x = data(i)
      do byte = 0, 63
        c = c + int(iand(x, 1_int64), int32)
        x = ishft(x, -1)
      end do
      result(i) = c
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (pc4): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp target update from(result(0:length-1))
  call check_results(data, result, length)

  t0 = seconds()
  do n = 1, repeat
    !$omp target teams distribute parallel do private(x,c) thread_limit(block_size)
    do i = 0, length-1
      x = data(i)
      c = lut(int(iand(x,255_int64))) + lut(int(iand(ishft(x,-8),255_int64))) + &
          lut(int(iand(ishft(x,-16),255_int64))) + lut(int(iand(ishft(x,-24),255_int64))) + &
          lut(int(iand(ishft(x,-32),255_int64))) + lut(int(iand(ishft(x,-40),255_int64))) + &
          lut(int(iand(ishft(x,-48),255_int64))) + lut(int(iand(ishft(x,-56),255_int64)))
      result(i) = c
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (pc5): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp target update from(result(0:length-1))
  call check_results(data, result, length)

  t0 = seconds()
  do n = 1, repeat
    !$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, length-1
      result(i) = popcnt(data(i))
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (pc6): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp target update from(result(0:length-1))
  call check_results(data, result, length)
  !$omp end target data
end program
