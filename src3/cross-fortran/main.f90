program cross_benchmark
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  implicit none
  integer :: nrows, repeat
  character(len=64) :: arg

  if (command_argument_count() /= 2) then
    write(*,'(a)') 'Usage: ./main <number of rows in a 2D tensor> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) nrows
  call get_command_argument(2, arg); read(arg, *) repeat
  if (nrows <= 0 .or. repeat <= 0) error stop 'arguments must be positive'

  write(*,'(a)') '=========== Data type is FP32 =========='
  call eval_sp(nrows, repeat)
  write(*,'(a)') '=========== Data type is FP64 =========='
  call eval_dp(nrows, repeat)

contains

  subroutine eval_sp(nrows, repeat)
    integer, intent(in) :: nrows, repeat
    integer(int64) :: num_elems, index, start_count, end_count, clock_rate, state
    integer :: iteration
    real(real64) :: elapsed_us
    logical :: ok
    real(real32), allocatable :: a(:), b(:), o(:), o2(:), o3(:)
    num_elems = int(nrows, int64) * 3_int64
    allocate(a(0:num_elems-1), b(0:num_elems-1), o(0:num_elems-1), o2(0:num_elems-1), o3(0:num_elems-1))
    state = 123_int64
    do index = 0, num_elems - 1
      a(index) = random_sp(state)
      b(index) = random_sp(state)
    end do

!$omp target data map(to:a(0:num_elems-1),b(0:num_elems-1)) map(from:o(0:num_elems-1),o2(0:num_elems-1),o3(0:num_elems-1))
    call system_clock(start_count, clock_rate)
    do iteration = 1, repeat
      call cross1_sp(nrows, o, a, b, 1, 1, 1)
    end do
    call system_clock(end_count)
    elapsed_us = real(end_count-start_count,real64) * 1.0e6_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average execution time of cross1 kernel: ', elapsed_us, ' (us)'

    call system_clock(start_count, clock_rate)
    do iteration = 1, repeat
      call cross2_sp(nrows, o2, a, b, 1, 1, 1)
    end do
    call system_clock(end_count)
    elapsed_us = real(end_count-start_count,real64) * 1.0e6_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average execution time of cross2 kernel: ', elapsed_us, ' (us)'

    call system_clock(start_count, clock_rate)
    do iteration = 1, repeat
      call cross3_sp(nrows, o3, a, b)
    end do
    call system_clock(end_count)
    elapsed_us = real(end_count-start_count,real64) * 1.0e6_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average execution time of cross3 kernel: ', elapsed_us, ' (us)'
!$omp end target data
    ok = all(abs(o-o2) <= 1.0e-3_real32) .and. all(abs(o-o3) <= 1.0e-3_real32)
    if (ok) then; write(*,'(a)') 'PASS'; else; write(*,'(a)') 'FAIL'; end if
    deallocate(a,b,o,o2,o3)
  end subroutine eval_sp

  subroutine eval_dp(nrows, repeat)
    integer, intent(in) :: nrows, repeat
    integer(int64) :: num_elems, index, start_count, end_count, clock_rate, state
    integer :: iteration
    real(real64) :: elapsed_us
    logical :: ok
    real(real64), allocatable :: a(:), b(:), o(:), o2(:), o3(:)
    num_elems = int(nrows, int64) * 3_int64
    allocate(a(0:num_elems-1), b(0:num_elems-1), o(0:num_elems-1), o2(0:num_elems-1), o3(0:num_elems-1))
    state = 123_int64
    do index = 0, num_elems - 1
      a(index) = random_dp(state)
      b(index) = random_dp(state)
    end do

!$omp target data map(to:a(0:num_elems-1),b(0:num_elems-1)) map(from:o(0:num_elems-1),o2(0:num_elems-1),o3(0:num_elems-1))
    call system_clock(start_count, clock_rate)
    do iteration = 1, repeat
      call cross1_dp(nrows, o, a, b, 1, 1, 1)
    end do
    call system_clock(end_count)
    elapsed_us = real(end_count-start_count,real64) * 1.0e6_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average execution time of cross1 kernel: ', elapsed_us, ' (us)'

    call system_clock(start_count, clock_rate)
    do iteration = 1, repeat
      call cross2_dp(nrows, o2, a, b, 1, 1, 1)
    end do
    call system_clock(end_count)
    elapsed_us = real(end_count-start_count,real64) * 1.0e6_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average execution time of cross2 kernel: ', elapsed_us, ' (us)'

    call system_clock(start_count, clock_rate)
    do iteration = 1, repeat
      call cross3_dp(nrows, o3, a, b)
    end do
    call system_clock(end_count)
    elapsed_us = real(end_count-start_count,real64) * 1.0e6_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average execution time of cross3 kernel: ', elapsed_us, ' (us)'
!$omp end target data
    ok = all(abs(o-o2) <= 1.0e-3_real64) .and. all(abs(o-o3) <= 1.0e-3_real64)
    if (ok) then; write(*,'(a)') 'PASS'; else; write(*,'(a)') 'FAIL'; end if
    deallocate(a,b,o,o2,o3)
  end subroutine eval_dp

  subroutine cross1_sp(n, out, x1, x2, ostride, x1stride, x2stride)
    integer, intent(in) :: n, ostride, x1stride, x2stride
    real(real32), intent(out) :: out(0:)
    real(real32), intent(in) :: x1(0:), x2(0:)
    integer :: i, base
!$omp target teams distribute parallel do thread_limit(256) private(base)
    do i=0,n-1
      base=3*i
      out(base)=x1(base+x1stride)*x2(base+2*x2stride)-x1(base+2*x1stride)*x2(base+x2stride)
      out(base+ostride)=x1(base+2*x1stride)*x2(base)-x1(base)*x2(base+2*x2stride)
      out(base+2*ostride)=x1(base)*x2(base+x2stride)-x1(base+x1stride)*x2(base)
    end do
!$omp end target teams distribute parallel do
  end subroutine cross1_sp

  subroutine cross2_sp(n, out, x1, x2, ostride, x1stride, x2stride)
    integer, intent(in) :: n, ostride, x1stride, x2stride
    real(real32), intent(out) :: out(0:)
    real(real32), intent(in) :: x1(0:), x2(0:)
    integer :: i, base
    real(real32) :: a0,a1,a2,b0,b1,b2
!$omp target teams distribute parallel do thread_limit(256) private(base,a0,a1,a2,b0,b1,b2)
    do i=0,n-1
      base=3*i; a0=x1(base); a1=x1(base+x1stride); a2=x1(base+2*x1stride); b0=x2(base); b1=x2(base+x2stride); b2=x2(base+2*x2stride)
      out(base)=a1*b2-a2*b1; out(base+ostride)=a2*b0-a0*b2; out(base+2*ostride)=a0*b1-a1*b0
    end do
!$omp end target teams distribute parallel do
  end subroutine cross2_sp

  subroutine cross3_sp(n, out, x1, x2)
    integer, intent(in) :: n
    real(real32), intent(out) :: out(0:)
    real(real32), intent(in) :: x1(0:), x2(0:)
    integer :: i, base
    real(real32) :: a0,a1,a2,b0,b1,b2
!$omp target teams distribute parallel do thread_limit(256) private(base,a0,a1,a2,b0,b1,b2)
    do i=0,n-1
      base=3*i; a0=x1(base); a1=x1(base+1); a2=x1(base+2); b0=x2(base); b1=x2(base+1); b2=x2(base+2)
      out(base)=a1*b2-a2*b1; out(base+1)=a2*b0-a0*b2; out(base+2)=a0*b1-a1*b0
    end do
!$omp end target teams distribute parallel do
  end subroutine cross3_sp

  subroutine cross1_dp(n, out, x1, x2, ostride, x1stride, x2stride)
    integer, intent(in) :: n, ostride, x1stride, x2stride
    real(real64), intent(out) :: out(0:)
    real(real64), intent(in) :: x1(0:), x2(0:)
    integer :: i, base
!$omp target teams distribute parallel do thread_limit(256) private(base)
    do i=0,n-1
      base=3*i
      out(base)=x1(base+x1stride)*x2(base+2*x2stride)-x1(base+2*x1stride)*x2(base+x2stride)
      out(base+ostride)=x1(base+2*x1stride)*x2(base)-x1(base)*x2(base+2*x2stride)
      out(base+2*ostride)=x1(base)*x2(base+x2stride)-x1(base+x1stride)*x2(base)
    end do
!$omp end target teams distribute parallel do
  end subroutine cross1_dp

  subroutine cross2_dp(n, out, x1, x2, ostride, x1stride, x2stride)
    integer, intent(in) :: n, ostride, x1stride, x2stride
    real(real64), intent(out) :: out(0:)
    real(real64), intent(in) :: x1(0:), x2(0:)
    integer :: i, base
    real(real64) :: a0,a1,a2,b0,b1,b2
!$omp target teams distribute parallel do thread_limit(256) private(base,a0,a1,a2,b0,b1,b2)
    do i=0,n-1
      base=3*i; a0=x1(base); a1=x1(base+x1stride); a2=x1(base+2*x1stride); b0=x2(base); b1=x2(base+x2stride); b2=x2(base+2*x2stride)
      out(base)=a1*b2-a2*b1; out(base+ostride)=a2*b0-a0*b2; out(base+2*ostride)=a0*b1-a1*b0
    end do
!$omp end target teams distribute parallel do
  end subroutine cross2_dp

  subroutine cross3_dp(n, out, x1, x2)
    integer, intent(in) :: n
    real(real64), intent(out) :: out(0:)
    real(real64), intent(in) :: x1(0:), x2(0:)
    integer :: i, base
    real(real64) :: a0,a1,a2,b0,b1,b2
!$omp target teams distribute parallel do thread_limit(256) private(base,a0,a1,a2,b0,b1,b2)
    do i=0,n-1
      base=3*i; a0=x1(base); a1=x1(base+1); a2=x1(base+2); b0=x2(base); b1=x2(base+1); b2=x2(base+2)
      out(base)=a1*b2-a2*b1; out(base+1)=a2*b0-a0*b2; out(base+2)=a0*b1-a1*b0
    end do
!$omp end target teams distribute parallel do
  end subroutine cross3_dp

  real(real32) function random_sp(state)
    integer(int64), intent(inout) :: state
    state = modulo(16807_int64*state, 2147483647_int64)
    random_sp = real(-2.0_real64 + 4.0_real64 * real(state,real64) / 2147483647.0_real64,real32)
  end function random_sp

  real(real64) function random_dp(state)
    integer(int64), intent(inout) :: state
    state = modulo(16807_int64*state, 2147483647_int64)
    random_dp = -2.0_real64 + 4.0_real64 * real(state,real64) / 2147483647.0_real64
  end function random_dp

end program cross_benchmark
