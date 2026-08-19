program atomic_intrinsics
  use, intrinsic :: iso_fortran_env, only : int32, int64, output_unit
  use omp_lib
  implicit none

  integer :: argc, num, repeat, ios
  character(len=64) :: argument

  argc = command_argument_count()
  if (argc /= 2) then
    write(output_unit, '(a)') 'Usage: ./main <number of atomic operations> <repeat>'
    error stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *, iostat=ios) num
  if (ios /= 0 .or. num < 0 .or. num > 62) error stop 'invalid number of atomic operations'
  call get_command_argument(2, argument)
  read(argument, *, iostat=ios) repeat
  if (ios /= 0 .or. repeat < 1) error stop 'invalid repeat count'

  call testcase_signed(num, repeat)
  call testcase_unsigned(num, repeat)

contains

  subroutine testcase_signed(num, repeat)
    integer, intent(in) :: num, repeat
    integer(int64) :: len, i
    integer(int32) :: gpu_data(0:6), data(0:6), max_value, min_value
    integer :: n
    real(8) :: start_time, end_time

    len = shiftl(1_int64, num)
    data = [0_int32, 0_int32, -256_int32, 256_int32, 255_int32, 0_int32, 255_int32]

    !$omp target data map(alloc: gpu_data, max_value, min_value)
    do n = 1, repeat
      gpu_data = data
      max_value = data(2)
      min_value = data(3)
      !$omp target update to(gpu_data, max_value, min_value)
      !$omp target teams distribute parallel do thread_limit(256)
      do i = 0_int64, len - 1_int64
        !$omp atomic update
        gpu_data(0) = gpu_data(0) + 10_int32
        !$omp atomic update
        gpu_data(1) = gpu_data(1) - 10_int32
        !$omp atomic update
        gpu_data(4) = iand(gpu_data(4), int(2_int64 * i + 7_int64, int32))
        !$omp atomic update
        gpu_data(5) = ior(gpu_data(5), shiftl(1_int32, int(iand(i, 31_int64))))
        !$omp atomic update
        gpu_data(6) = ieor(gpu_data(6), int(i, int32))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(max:max_value)
      do i = 0_int64, len - 1_int64
        max_value = max(max_value, int(i, int32))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(min:min_value)
      do i = 0_int64, len - 1_int64
        min_value = min(min_value, int(i, int32))
      end do
    end do

    !$omp target update from(gpu_data, max_value, min_value)
    gpu_data(2) = max_value
    gpu_data(3) = min_value
    call compute_gold_signed(gpu_data, len)

    start_time = omp_get_wtime()
    do n = 1, repeat
      !$omp target teams distribute parallel do thread_limit(256)
      do i = 0_int64, len - 1_int64
        !$omp atomic update
        gpu_data(0) = gpu_data(0) + 10_int32
        !$omp atomic update
        gpu_data(1) = gpu_data(1) - 10_int32
        !$omp atomic update
        gpu_data(4) = iand(gpu_data(4), int(2_int64 * i + 7_int64, int32))
        !$omp atomic update
        gpu_data(5) = ior(gpu_data(5), shiftl(1_int32, int(iand(i, 31_int64))))
        !$omp atomic update
        gpu_data(6) = ieor(gpu_data(6), int(i, int32))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(max:max_value)
      do i = 0_int64, len - 1_int64
        max_value = max(max_value, int(i, int32))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(min:min_value)
      do i = 0_int64, len - 1_int64
        min_value = min(min_value, int(i, int32))
      end do
    end do
    end_time = omp_get_wtime()
    !$omp end target data

    write(output_unit, '(a,f0.6,a)') 'Average kernel execution time: ', &
      ((end_time - start_time) * 1.0e6_8) / real(repeat, 8), ' (us)'
  end subroutine testcase_signed

  subroutine testcase_unsigned(num, repeat)
    integer, intent(in) :: num, repeat
    integer(int32), parameter :: sign_bit = int(z'80000000', int32)
    integer(int64) :: len, i
    integer(int32) :: gpu_data(0:6), data(0:6), max_key, min_value
    integer :: n
    real(8) :: start_time, end_time

    len = shiftl(1_int64, num)
    ! max_key holds ieor(unsigned_raw, sign_bit), an order-preserving signed key.
    data = [0_int32, 0_int32, ieor(-256_int32, sign_bit), 256_int32, 255_int32, 0_int32, 255_int32]

    !$omp target data map(alloc: gpu_data, max_key, min_value)
    do n = 1, repeat
      gpu_data = data
      max_key = data(2)
      min_value = data(3)
      !$omp target update to(gpu_data, max_key, min_value)
      !$omp target teams distribute parallel do thread_limit(256)
      do i = 0_int64, len - 1_int64
        !$omp atomic update
        gpu_data(0) = gpu_data(0) + 10_int32
        !$omp atomic update
        gpu_data(1) = gpu_data(1) - 10_int32
        !$omp atomic update
        gpu_data(4) = iand(gpu_data(4), int(2_int64 * i + 7_int64, int32))
        !$omp atomic update
        gpu_data(5) = ior(gpu_data(5), shiftl(1_int32, int(iand(i, 31_int64))))
        !$omp atomic update
        gpu_data(6) = ieor(gpu_data(6), int(i, int32))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(max:max_key)
      do i = 0_int64, len - 1_int64
        max_key = max(max_key, ieor(int(i, int32), sign_bit))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(min:min_value)
      do i = 0_int64, len - 1_int64
        min_value = min(min_value, int(i, int32))
      end do
    end do

    !$omp target update from(gpu_data, max_key, min_value)
    gpu_data(2) = max_key
    gpu_data(3) = min_value
    call compute_gold_unsigned(gpu_data, len)

    start_time = omp_get_wtime()
    do n = 1, repeat
      !$omp target teams distribute parallel do thread_limit(256)
      do i = 0_int64, len - 1_int64
        !$omp atomic update
        gpu_data(0) = gpu_data(0) + 10_int32
        !$omp atomic update
        gpu_data(1) = gpu_data(1) - 10_int32
        !$omp atomic update
        gpu_data(4) = iand(gpu_data(4), int(2_int64 * i + 7_int64, int32))
        !$omp atomic update
        gpu_data(5) = ior(gpu_data(5), shiftl(1_int32, int(iand(i, 31_int64))))
        !$omp atomic update
        gpu_data(6) = ieor(gpu_data(6), int(i, int32))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(max:max_key)
      do i = 0_int64, len - 1_int64
        max_key = max(max_key, ieor(int(i, int32), sign_bit))
      end do
      !$omp target teams distribute parallel do thread_limit(256) reduction(min:min_value)
      do i = 0_int64, len - 1_int64
        min_value = min(min_value, int(i, int32))
      end do
    end do
    end_time = omp_get_wtime()
    !$omp end target data

    write(output_unit, '(a,f0.6,a)') 'Average kernel execution time: ', &
      ((end_time - start_time) * 1.0e6_8) / real(repeat, 8), ' (us)'
  end subroutine testcase_unsigned

  subroutine compute_gold_signed(gpu_data, len)
    integer(int32), intent(in) :: gpu_data(0:6)
    integer(int64), intent(in) :: len
    integer(int32) :: value
    integer(int64) :: i
    logical :: ok

    ok = .true.
    value = 0_int32
    do i = 0_int64, len - 1_int64
      value = value + 10_int32
    end do
    call check_value('Add', value, gpu_data(0), ok)
    value = 0_int32
    do i = 0_int64, len - 1_int64
      value = value - 10_int32
    end do
    call check_value('Sub', value, gpu_data(1), ok)
    value = -256_int32
    do i = 0_int64, len - 1_int64
      value = max(value, int(i, int32))
    end do
    call check_value('Max', value, gpu_data(2), ok)
    value = 256_int32
    do i = 0_int64, len - 1_int64
      value = min(value, int(i, int32))
    end do
    call check_value('Min', value, gpu_data(3), ok)
    call check_bitwise_values(gpu_data, len, ok)
    if (ok) then
      write(output_unit, '(a)') 'PASS'
    else
      write(output_unit, '(a)') 'FAIL'
    end if
  end subroutine compute_gold_signed

  subroutine compute_gold_unsigned(gpu_data, len)
    integer(int32), intent(in) :: gpu_data(0:6)
    integer(int64), intent(in) :: len
    integer(int32), parameter :: sign_bit = int(z'80000000', int32)
    integer(int32) :: value, key
    integer(int64) :: i
    logical :: ok

    ok = .true.
    value = 0_int32
    do i = 0_int64, len - 1_int64
      value = value + 10_int32
    end do
    call check_value('Add', value, gpu_data(0), ok)
    value = 0_int32
    do i = 0_int64, len - 1_int64
      value = value - 10_int32
    end do
    call check_value('Sub', value, gpu_data(1), ok)
    key = ieor(-256_int32, sign_bit)
    do i = 0_int64, len - 1_int64
      key = max(key, ieor(int(i, int32), sign_bit))
    end do
    call check_value('Max', key, gpu_data(2), ok)
    value = 256_int32
    do i = 0_int64, len - 1_int64
      value = min(value, int(i, int32))
    end do
    call check_value('Min', value, gpu_data(3), ok)
    call check_bitwise_values(gpu_data, len, ok)
    if (ok) then
      write(output_unit, '(a)') 'PASS'
    else
      write(output_unit, '(a)') 'FAIL'
    end if
  end subroutine compute_gold_unsigned

  subroutine check_bitwise_values(gpu_data, len, ok)
    integer(int32), intent(in) :: gpu_data(0:6)
    integer(int64), intent(in) :: len
    logical, intent(inout) :: ok
    integer(int32) :: value
    integer(int64) :: i

    value = 255_int32
    do i = 0_int64, len - 1_int64
      value = iand(value, int(2_int64 * i + 7_int64, int32))
    end do
    call check_value('And', value, gpu_data(4), ok)
    value = 0_int32
    do i = 0_int64, len - 1_int64
      value = ior(value, shiftl(1_int32, int(modulo(i, 32_int64))))
    end do
    call check_value('Or', value, gpu_data(5), ok)
    value = 255_int32
    do i = 0_int64, len - 1_int64
      value = ieor(value, int(i, int32))
    end do
    call check_value('Xor', value, gpu_data(6), ok)
  end subroutine check_bitwise_values

  subroutine check_value(name, expected, actual, ok)
    character(len=*), intent(in) :: name
    integer(int32), intent(in) :: expected, actual
    logical, intent(inout) :: ok

    if (expected /= actual) then
      write(output_unit, '(a,a,i0,a,i0)') trim(name), ' failed: ', expected, ' != ', actual
      ok = .false.
    end if
  end subroutine check_value

end program atomic_intrinsics
