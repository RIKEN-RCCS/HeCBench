module zeropoint_mod
  use iso_fortran_env, only: real32, real64, int32, int64
  implicit none
contains
  function wall_time() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_time

  subroutine zero_point(x_min, x_max, qmin, qmax, size, preserve_sparsity, scale, zp)
    real(real32), intent(in) :: x_min(0:), x_max(0:)
    integer(int32), intent(in) :: qmin, qmax
    integer, intent(in) :: size
    logical, intent(in) :: preserve_sparsity
    real(real32), intent(out) :: scale(0:)
    integer(int32), intent(out) :: zp(0:)
    integer :: i
    integer(int32) :: symmetric_qmin, symmetric_qmax, nudged_zero_point
    real(real32) :: min_val, max_val
    real(real64) :: max_scale, zpf_min, zpf_max, err_min, err_max, initial_zp
!$omp target teams distribute parallel do thread_limit(256) &
!$omp& map(to:x_min,x_max) map(from:scale,zp) private(i,min_val,max_val,symmetric_qmin,symmetric_qmax,max_scale,zpf_min,zpf_max,err_min,err_max,initial_zp,nudged_zero_point)
    do i = 0, size - 1
      min_val = x_min(i)
      max_val = x_max(i)
      if (min_val < 0.0_real32 .and. max_val > 0.0_real32 .and. preserve_sparsity) then
        symmetric_qmin = -((qmax - qmin) / 2 + 1)
        symmetric_qmax = (qmax - qmin) / 2
        max_scale = max(abs(real(min_val, real64) / real(symmetric_qmin, real64)), &
                        abs(real(max_val, real64) / real(symmetric_qmax, real64)))
        min_val = real(max_scale * real(symmetric_qmin, real64), real32)
        max_val = real(max_scale * real(symmetric_qmax, real64), real32)
      end if
      min_val = min(min_val, 0.0_real32)
      max_val = max(max_val, 0.0_real32)
      scale(i) = real((real(max_val, real64) - real(min_val, real64)) / real(qmax - qmin, real64), real32)
      if (scale(i) == 0.0_real32 .or. abs(1.0_real32 / scale(i)) == huge(1.0_real32)) scale(i) = 0.1_real32
      zpf_min = real(qmin, real64) - real(min_val, real64) / real(scale(i), real64)
      zpf_max = real(qmax, real64) - real(max_val, real64) / real(scale(i), real64)
      err_min = abs(real(qmin, real64)) + abs(real(min_val, real64) / real(scale(i), real64))
      err_max = abs(real(qmax, real64)) + abs(real(max_val, real64) / real(scale(i), real64))
      if (err_min < err_max) then
        initial_zp = zpf_min
      else
        initial_zp = zpf_max
      end if
      if (min_val < 0.0_real32 .and. max_val > 0.0_real32 .and. preserve_sparsity) then
        initial_zp = real(qmin + qmax, real64) / 2.0_real64
      end if
      if (initial_zp < real(qmin, real64)) then
        nudged_zero_point = qmin
      else if (initial_zp > real(qmax, real64)) then
        nudged_zero_point = qmax
      else
        nudged_zero_point = int(anint(initial_zp), int32)
      end if
      zp(i) = nudged_zero_point
    end do
!$omp end target teams distribute parallel do
  end subroutine zero_point
end module zeropoint_mod

program main
  use iso_fortran_env, only: real32, real64, int32
  use zeropoint_mod
  implicit none
  integer :: argc, size, repeat, i, rep, stat
  integer(int32) :: qmin, qmax
  character(len=128) :: arg
  logical :: preserve_sparsity, ok
  real(real32), allocatable :: scale(:), scale_ref(:), x_min(:), x_max(:)
  integer(int32), allocatable :: zp(:), zp_ref(:)
  real(real64) :: t0, t1

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <number of min/max values> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) size
  call get_command_argument(2, arg); read(arg, *, iostat=stat) repeat
  qmin = -127_int32
  qmax = 127_int32
  preserve_sparsity = .true.
  allocate(scale(0:size-1), scale_ref(0:size-1), zp(0:size-1), zp_ref(0:size-1), x_min(0:size-1), x_max(0:size-1))
  call random_seed()
  do i = 0, size - 1
    call random_number(x_min(i)); call random_number(x_max(i))
    x_min(i) = -1.0_real32 + 2.0_real32 * x_min(i)
    x_max(i) = -1.0_real32 + 2.0_real32 * x_max(i)
  end do
  call zero_point(x_min, x_max, qmin, qmax, size, preserve_sparsity, scale_ref, zp_ref)
!$omp target data map(to:x_min,x_max) map(from:scale,zp)
  t0 = wall_time()
  do rep = 1, repeat
    call zero_point(x_min, x_max, qmin, qmax, size, preserve_sparsity, scale, zp)
  end do
  t1 = wall_time()
  print '(a,f12.6,a)', 'Average execution time of zero-point kernel: ', (t1 - t0) * 1.0e6_real64 / real(repeat, real64), ' (us)'
!$omp end target data
  ok = .true.
  do i = 0, size - 1
    if (zp(i) /= zp_ref(i) .or. scale(i) - scale_ref(i) > 1.0e-3_real32) then
      ok = .false.
      exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
end program main
