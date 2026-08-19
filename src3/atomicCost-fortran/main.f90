module atomic_cost_kinds
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  implicit none
  integer, parameter :: block_size = 256
contains

  subroutine start_clock(tick, rate)
    integer(int64), intent(out) :: tick, rate
    call system_clock(tick, rate)
  end subroutine start_clock

  real(real64) function elapsed_us(start_tick, rate) result(value)
    integer(int64), intent(in) :: start_tick, rate
    integer(int64) :: stop_tick
    call system_clock(stop_tick)
    value = real(stop_tick - start_tick, real64) * 1.0e6_real64 / real(rate, real64)
  end function elapsed_us

  subroutine atomic_cost_r8(length, size, repeat)
    integer, intent(in) :: length, size, repeat
    integer :: num_threads, tid, i, rep
    integer(int64) :: tick, rate
    real(real64) :: time_us
    real(real64), allocatable :: result_wi(:), result_wo(:)

    print '(A)', ''
    print '(A)', ''
    print '(A,I0)', 'Each thread sums up ', size, ' elements'
    num_threads = length / size
    if (mod(length, size) /= 0) error stop 'length % size assertion failed'
    if (mod(num_threads, block_size) /= 0) error stop 'num_threads % BLOCK_SIZE assertion failed'
    allocate(result_wi(0:num_threads-1), result_wo(0:num_threads-1))
    result_wi = 0.0_real64
    result_wo = 0.0_real64

!$omp target data map(alloc:result_wi(0:num_threads-1),result_wo(0:num_threads-1))
    call start_clock(tick, rate)
    do rep = 0, repeat - 1
!$omp target teams distribute parallel do thread_limit(block_size)
      do tid = 0, num_threads - 1
        do i = tid * size, (tid + 1) * size - 1
!$omp atomic update
          result_wi(tid) = result_wi(tid) + real(mod(i, 2), real64)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    time_us = elapsed_us(tick, rate) / real(repeat, real64)
    print '(A,F0.6,A)', 'Average execution time of WithAtomicOnGlobalMem: ', time_us, ' (us)'
!$omp target update from(result_wi(0:num_threads-1))

    call start_clock(tick, rate)
    do rep = 0, repeat - 1
!$omp target teams distribute parallel do thread_limit(block_size)
      do tid = 0, num_threads - 1
        do i = tid * size, (tid + 1) * size - 1
          result_wo(tid) = result_wo(tid) + real(mod(i, 2), real64)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    time_us = elapsed_us(tick, rate) / real(repeat, real64)
    print '(A,F0.6,A)', 'Average execution time of WithoutAtomicOnGlobalMem: ', time_us, ' (us)'
!$omp target update from(result_wo(0:num_threads-1))
    if (all(result_wi == result_wo)) then
      print '(A)', 'PASS'
    else
      print '(A)', 'FAIL'
    end if
!$omp end target data
    deallocate(result_wi, result_wo)
  end subroutine atomic_cost_r8

  subroutine atomic_cost_i4(length, size, repeat)
    integer, intent(in) :: length, size, repeat
    integer :: num_threads, tid, i, rep
    integer(int64) :: tick, rate
    real(real64) :: time_us
    integer(int32), allocatable :: result_wi(:), result_wo(:)

    print '(A)', ''
    print '(A)', ''
    print '(A,I0)', 'Each thread sums up ', size, ' elements'
    num_threads = length / size
    if (mod(length, size) /= 0) error stop 'length % size assertion failed'
    if (mod(num_threads, block_size) /= 0) error stop 'num_threads % BLOCK_SIZE assertion failed'
    allocate(result_wi(0:num_threads-1), result_wo(0:num_threads-1))
    result_wi = 0_int32
    result_wo = 0_int32

!$omp target data map(alloc:result_wi(0:num_threads-1),result_wo(0:num_threads-1))
    call start_clock(tick, rate)
    do rep = 0, repeat - 1
!$omp target teams distribute parallel do thread_limit(block_size)
      do tid = 0, num_threads - 1
        do i = tid * size, (tid + 1) * size - 1
!$omp atomic update
          result_wi(tid) = result_wi(tid) + int(mod(i, 2), int32)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    time_us = elapsed_us(tick, rate) / real(repeat, real64)
    print '(A,F0.6,A)', 'Average execution time of WithAtomicOnGlobalMem: ', time_us, ' (us)'
!$omp target update from(result_wi(0:num_threads-1))

    call start_clock(tick, rate)
    do rep = 0, repeat - 1
!$omp target teams distribute parallel do thread_limit(block_size)
      do tid = 0, num_threads - 1
        do i = tid * size, (tid + 1) * size - 1
          result_wo(tid) = result_wo(tid) + int(mod(i, 2), int32)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    time_us = elapsed_us(tick, rate) / real(repeat, real64)
    print '(A,F0.6,A)', 'Average execution time of WithoutAtomicOnGlobalMem: ', time_us, ' (us)'
!$omp target update from(result_wo(0:num_threads-1))
    if (all(result_wi == result_wo)) then
      print '(A)', 'PASS'
    else
      print '(A)', 'FAIL'
    end if
!$omp end target data
    deallocate(result_wi, result_wo)
  end subroutine atomic_cost_i4

  subroutine atomic_cost_r4(length, size, repeat)
    integer, intent(in) :: length, size, repeat
    integer :: num_threads, tid, i, rep
    integer(int64) :: tick, rate
    real(real64) :: time_us
    real(real32), allocatable :: result_wi(:), result_wo(:)

    print '(A)', ''
    print '(A)', ''
    print '(A,I0)', 'Each thread sums up ', size, ' elements'
    num_threads = length / size
    if (mod(length, size) /= 0) error stop 'length % size assertion failed'
    if (mod(num_threads, block_size) /= 0) error stop 'num_threads % BLOCK_SIZE assertion failed'
    allocate(result_wi(0:num_threads-1), result_wo(0:num_threads-1))
    result_wi = 0.0_real32
    result_wo = 0.0_real32

!$omp target data map(alloc:result_wi(0:num_threads-1),result_wo(0:num_threads-1))
    call start_clock(tick, rate)
    do rep = 0, repeat - 1
!$omp target teams distribute parallel do thread_limit(block_size)
      do tid = 0, num_threads - 1
        do i = tid * size, (tid + 1) * size - 1
!$omp atomic update
          result_wi(tid) = result_wi(tid) + real(mod(i, 2), real32)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    time_us = elapsed_us(tick, rate) / real(repeat, real64)
    print '(A,F0.6,A)', 'Average execution time of WithAtomicOnGlobalMem: ', time_us, ' (us)'
!$omp target update from(result_wi(0:num_threads-1))

    call start_clock(tick, rate)
    do rep = 0, repeat - 1
!$omp target teams distribute parallel do thread_limit(block_size)
      do tid = 0, num_threads - 1
        do i = tid * size, (tid + 1) * size - 1
          result_wo(tid) = result_wo(tid) + real(mod(i, 2), real32)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    time_us = elapsed_us(tick, rate) / real(repeat, real64)
    print '(A,F0.6,A)', 'Average execution time of WithoutAtomicOnGlobalMem: ', time_us, ' (us)'
!$omp target update from(result_wo(0:num_threads-1))
    if (all(result_wi == result_wo)) then
      print '(A)', 'PASS'
    else
      print '(A)', 'FAIL'
    end if
!$omp end target data
    deallocate(result_wi, result_wo)
  end subroutine atomic_cost_r4
end module atomic_cost_kinds

program atomic_cost
  use atomic_cost_kinds
  implicit none
  character(len=64) :: arg
  integer :: nelems, repeat, ios
  integer, parameter :: length = 922521600

  if (command_argument_count() /= 2) then
    print '(A)', 'Usage: ./main <N> <repeat>'
    print '(A)', 'N: the number of elements to sum per thread (1 - 16)'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *, iostat=ios) nelems
  if (ios /= 0) then
    print '(A)', 'Usage: ./main <N> <repeat>'
    print '(A)', 'N: the number of elements to sum per thread (1 - 16)'
    stop 1
  end if
  call get_command_argument(2, arg)
  read(arg, *, iostat=ios) repeat
  if (ios /= 0) then
    print '(A)', 'Usage: ./main <N> <repeat>'
    print '(A)', 'N: the number of elements to sum per thread (1 - 16)'
    stop 1
  end if
  if (mod(length, block_size) /= 0) error stop 'length % BLOCK_SIZE assertion failed'

  print '(A)', ''
  print '(A)', 'FP64 atomic add'
  call atomic_cost_r8(length, nelems, repeat)
  print '(A)', ''
  print '(A)', 'INT32 atomic add'
  call atomic_cost_i4(length, nelems, repeat)
  print '(A)', ''
  print '(A)', 'FP32 atomic add'
  call atomic_cost_r4(length, nelems, repeat)
end program atomic_cost
