program main
  use iso_fortran_env, only: real32, real64, int64
  use svd3x3_kernels
  implicit none
  integer :: argc, repeat, tests_size, i, j, count, unit, stat
  character(len=256) :: filename
  character(len=128) :: arg
  real(real32), allocatable :: input(:), result(:), result_h(:), tmp(:)
  real(real64) :: t0, t1
  logical :: ok
  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <path to file> <repeat>'
    stop 1
  end if
  call get_command_argument(1, filename)
  call get_command_argument(2, arg); read(arg, *, iostat=stat) repeat
  open(newunit=unit, file=trim(filename), status='old', action='read', iostat=stat)
  if (stat /= 0) then
    print '(a,a)', 'ERROR: failed to open ', trim(filename)
    stop 1
  end if
  read(unit, *) tests_size
  print '(a,i0)', 'dataset size: ', tests_size
  if (tests_size <= 0) then
    print '(a)', 'ERROR: invalid dataset size'
    stop 1
  end if
  allocate(input(0:9*tests_size-1), result(0:21*tests_size-1), result_h(0:21*tests_size-1), tmp(0:20))
  count = 0
  do i = 0, tests_size - 1
    do j = 0, 8
      read(unit, *) input(count)
      count = count + 1
    end do
  end do
  close(unit)
!$omp target data map(to:input) map(from:result)
  t0 = wall_seconds()
  call run_svd_device(input, result, tests_size, repeat)
  t1 = wall_seconds()
  print '(a,f12.6,a)', 'Average kernel execution time: ', (t1 - t0) * 1.0e6_real64 / real(repeat, real64), ' (us)'
!$omp end target data
  do i = 0, tests_size - 1
    call svd(input(i+0*tests_size), input(i+1*tests_size), input(i+2*tests_size), &
             input(i+3*tests_size), input(i+4*tests_size), input(i+5*tests_size), &
             input(i+6*tests_size), input(i+7*tests_size), input(i+8*tests_size), tmp)
    do j = 0, 20
      result_h(i + j * tests_size) = tmp(j)
    end do
  end do
  ok = .true.
  do i = 0, tests_size - 1
    if (abs(result(i) - result_h(i)) > 1.0e-3_real32) then
      print '(2f12.6)', result(i), result_h(i)
      ok = .false.
      exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count_clock, rate
    call system_clock(count_clock, rate)
    t = real(count_clock, real64) / real(rate, real64)
  end function wall_seconds
end program main
