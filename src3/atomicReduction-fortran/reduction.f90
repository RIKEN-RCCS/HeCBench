program atomic_reduction
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int64, real64
  implicit none

  interface
    function c_rand() bind(C, name="rand") result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand

    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
  end interface

  integer(c_int), parameter :: default_array_length = 52428800_c_int
  integer(c_int), parameter :: default_repetitions = 100_c_int
  integer(c_int) :: array_length, repetitions, i, n, k, threads, blocks
  integer(c_int) :: sum_value, checksum
  integer(c_int), dimension(4) :: block_sizes
  integer(c_int), allocatable :: array(:)
  integer(int64) :: tick_start, tick_stop, tick_rate
  real(real64) :: elapsed, gigabytes
  character(len=64) :: argument

  array_length = default_array_length
  repetitions = default_repetitions
  block_sizes = [128_c_int, 256_c_int, 512_c_int, 1024_c_int]

  if (command_argument_count() == 2) then
    call get_command_argument(1, argument)
    read(argument, *) array_length
    call get_command_argument(2, argument)
    read(argument, *) repetitions
  end if

  print '(a,f0.6,a)', 'Array size: ', real(array_length, real64) * 4.0_real64 / &
    1024.0_real64 / 1024.0_real64, ' MB'
  print '(a,i0,a)', 'Repeat the kernel execution: ', repetitions, ' times'

  allocate(array(0:array_length - 1_c_int))
  ! C/C++ rand() uses seed 1 when srand() has not been called.
  call c_srand(1_c_int)
  checksum = 0_c_int
  do i = 0_c_int, array_length - 1_c_int
    array(i) = modulo(c_rand(), 2_c_int)
    checksum = checksum + array(i)
  end do

  ! The C source casts arrayLength to float before this multiplication.
  gigabytes = real(array_length, real64) * 4.0_real64 * real(repetitions, real64)

  !$omp target data map(to: array(0:array_length - 1_c_int)) map(alloc: sum_value)
    ! Same warm-up: N contiguous reduction launches with a fixed 2048 x 256 geometry.
    do n = 1_c_int, repetitions
      sum_value = 0_c_int
      !$omp target update to(sum_value)
      !$omp target teams distribute parallel do reduction(+:sum_value) &
      !$omp& num_teams(2048) thread_limit(256)
      do i = 0_c_int, array_length - 1_c_int
        sum_value = sum_value + array(i)
      end do
      !$omp end target teams distribute parallel do
    end do

    do k = 1_c_int, size(block_sizes, kind=c_int)
      threads = block_sizes(k)
      blocks = min((array_length + threads - 1_c_int) / threads, 2048_c_int)

      call system_clock(tick_start, tick_rate)
      do n = 1_c_int, repetitions
        sum_value = 0_c_int
        !$omp target update to(sum_value)
        !$omp target teams distribute parallel do reduction(+:sum_value) &
        !$omp& num_teams(blocks) thread_limit(threads)
        do i = 0_c_int, array_length - 1_c_int
          sum_value = sum_value + array(i)
        end do
        !$omp end target teams distribute parallel do
      end do
      !$omp target update from(sum_value)
      call system_clock(tick_stop)
      elapsed = real(tick_stop - tick_start, real64) / real(tick_rate, real64)
      call print_result(threads, gigabytes, elapsed, sum_value, checksum, .true.)

      call system_clock(tick_start)
      do n = 1_c_int, repetitions
        sum_value = 0_c_int
        !$omp target update to(sum_value)
        !$omp target teams distribute parallel do reduction(+:sum_value) &
        !$omp& num_teams(blocks / 2_c_int) thread_limit(threads)
        do i = 0_c_int, array_length / 2_c_int - 1_c_int
          sum_value = sum_value + array(i * 2_c_int) + array(i * 2_c_int + 1_c_int)
        end do
        !$omp end target teams distribute parallel do
      end do
      !$omp target update from(sum_value)
      call system_clock(tick_stop)
      elapsed = real(tick_stop - tick_start, real64) / real(tick_rate, real64)
      call print_result(threads, gigabytes, elapsed, sum_value, checksum, .false.)

      call system_clock(tick_start)
      do n = 1_c_int, repetitions
        sum_value = 0_c_int
        !$omp target update to(sum_value)
        !$omp target teams distribute parallel do reduction(+:sum_value) &
        !$omp& num_teams(blocks / 4_c_int) thread_limit(threads)
        do i = 0_c_int, array_length / 4_c_int - 1_c_int
          sum_value = sum_value + array(i * 4_c_int) + array(i * 4_c_int + 1_c_int) + &
            array(i * 4_c_int + 2_c_int) + array(i * 4_c_int + 3_c_int)
        end do
        !$omp end target teams distribute parallel do
      end do
      !$omp target update from(sum_value)
      call system_clock(tick_stop)
      elapsed = real(tick_stop - tick_start, real64) / real(tick_rate, real64)
      call print_result(threads, gigabytes, elapsed, sum_value, checksum, .false.)

      call system_clock(tick_start)
      do n = 1_c_int, repetitions
        sum_value = 0_c_int
        !$omp target update to(sum_value)
        !$omp target teams distribute parallel do reduction(+:sum_value) &
        !$omp& num_teams(blocks / 8_c_int) thread_limit(threads)
        do i = 0_c_int, array_length / 8_c_int - 1_c_int
          sum_value = sum_value + array(i * 8_c_int) + array(i * 8_c_int + 1_c_int) + &
            array(i * 8_c_int + 2_c_int) + array(i * 8_c_int + 3_c_int) + &
            array(i * 8_c_int + 4_c_int) + array(i * 8_c_int + 5_c_int) + &
            array(i * 8_c_int + 6_c_int) + array(i * 8_c_int + 7_c_int)
        end do
        !$omp end target teams distribute parallel do
      end do
      !$omp target update from(sum_value)
      call system_clock(tick_stop)
      elapsed = real(tick_stop - tick_start, real64) / real(tick_rate, real64)
      call print_result(threads, gigabytes, elapsed, sum_value, checksum, .false.)

      call system_clock(tick_start)
      do n = 1_c_int, repetitions
        sum_value = 0_c_int
        !$omp target update to(sum_value)
        !$omp target teams distribute parallel do reduction(+:sum_value) &
        !$omp& num_teams(blocks / 16_c_int) thread_limit(threads)
        do i = 0_c_int, array_length / 16_c_int - 1_c_int
          sum_value = sum_value + array(i * 16_c_int) + array(i * 16_c_int + 1_c_int) + &
            array(i * 16_c_int + 2_c_int) + array(i * 16_c_int + 3_c_int) + &
            array(i * 16_c_int + 4_c_int) + array(i * 16_c_int + 5_c_int) + &
            array(i * 16_c_int + 6_c_int) + array(i * 16_c_int + 7_c_int) + &
            array(i * 16_c_int + 8_c_int) + array(i * 16_c_int + 9_c_int) + &
            array(i * 16_c_int + 10_c_int) + array(i * 16_c_int + 11_c_int) + &
            array(i * 16_c_int + 12_c_int) + array(i * 16_c_int + 13_c_int) + &
            array(i * 16_c_int + 14_c_int) + array(i * 16_c_int + 15_c_int)
        end do
        !$omp end target teams distribute parallel do
      end do
      !$omp target update from(sum_value)
      call system_clock(tick_stop)
      elapsed = real(tick_stop - tick_start, real64) / real(tick_rate, real64)
      call print_result(threads, gigabytes, elapsed, sum_value, checksum, .false.)
    end do
  !$omp end target data

  deallocate(array)

contains

  subroutine print_result(thread_count, bytes_processed, seconds, device_sum, host_checksum, print_numbers)
    integer(c_int), intent(in) :: thread_count, device_sum, host_checksum
    real(real64), intent(in) :: bytes_processed, seconds
    logical, intent(in) :: print_numbers

    print '(a,i0,a)', 'Thread block size: ', thread_count, ', '
    print '(a,es16.8,a)', 'The average performance of reduction is ', &
      1.0e-9_real64 * bytes_processed / seconds, ' GBytes/sec'
    if (print_numbers) print '(i0,1x,i0)', device_sum, host_checksum
    if (device_sum == host_checksum) then
      print '(a)', 'VERIFICATION: PASS'
    else
      print '(a)', 'VERIFICATION: FAIL!!'
    end if
    print '(a)'
  end subroutine print_result

end program atomic_reduction
