module cbsfil_kernels
  use iso_fortran_env, only : int32, real32
  implicit none

  real(real32), parameter :: pole = sqrt(3.0_real32) - 2.0_real32

!$omp declare target (initial_causal_coefficient, initial_anti_causal_coefficient, &
!$omp& convert_to_interpolation_coefficients)

contains

  integer function pow_two_divider(n) result(divider)
    integer, intent(in) :: n

    if (n == 0) then
      divider = 0
      return
    end if

    divider = 1
    do while (iand(n, divider) == 0)
      divider = shiftl(divider, 1)
    end do
  end function pow_two_divider

  real(real32) function initial_causal_coefficient(coefficients, base, data_length, step) &
      result(sum_value)
    real(real32), intent(in) :: coefficients(0:*)
    integer, intent(in) :: base, data_length, step
    integer :: n, horizon
    real(real32) :: zn

    horizon = min(12, data_length)
    zn = pole
    sum_value = coefficients(base)
    do n = 0, horizon - 1
      sum_value = sum_value + zn * coefficients(base + n * step)
      zn = zn * pole
    end do
  end function initial_causal_coefficient

  real(real32) function initial_anti_causal_coefficient(coefficients, position) result(value)
    real(real32), intent(in) :: coefficients(0:*)
    integer, intent(in) :: position

    value = (pole / (pole - 1.0_real32)) * coefficients(position)
  end function initial_anti_causal_coefficient

  subroutine convert_to_interpolation_coefficients(coefficients, base, data_length, step)
    real(real32), intent(inout) :: coefficients(0:*)
    integer, intent(in) :: base, data_length, step
    integer :: n, position
    real(real32) :: lambda, previous_coefficient

    lambda = (1.0_real32 - pole) * (1.0_real32 - 1.0_real32 / pole)

    position = base
    coefficients(position) = lambda * initial_causal_coefficient(coefficients, position, data_length, step)
    previous_coefficient = coefficients(position)

    do n = 1, data_length - 1
      position = position + step
      coefficients(position) = lambda * coefficients(position) + pole * previous_coefficient
      previous_coefficient = coefficients(position)
    end do

    coefficients(position) = initial_anti_causal_coefficient(coefficients, position)
    previous_coefficient = coefficients(position)

    do n = data_length - 2, 0, -1
      position = position - step
      coefficients(position) = pole * (previous_coefficient - coefficients(position))
      previous_coefficient = coefficients(position)
    end do
  end subroutine convert_to_interpolation_coefficients

end module cbsfil_kernels

program cbsfil
  use iso_c_binding, only : c_int
  use iso_fortran_env, only : int32, int64, real32, real64
  use cbsfil_kernels, only : pow_two_divider, convert_to_interpolation_coefficients
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name = 'srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    integer(c_int) function c_rand() bind(C, name = 'rand')
      import :: c_int
    end function c_rand
  end interface

  character(len = 64) :: argument
  integer :: width, height, repeat_count, image_pitch, num_pix
  integer :: num_threads_x, num_threads_y, i, x, y, read_status
  integer(c_int) :: random_x, random_y, random_z, random_w
  integer(int32) :: bits
  integer(int64) :: count_start, count_end, count_rate
  real(real32), allocatable :: image(:)
  real(real32) :: checksum
  real(real64) :: total_seconds

  if (command_argument_count() /= 3) then
    print '(A)', 'Usage: ./main <width> <height> <repeat>'
    stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *, iostat = read_status) width
  if (read_status /= 0) stop 1
  call get_command_argument(2, argument)
  read(argument, *, iostat = read_status) height
  if (read_status /= 0) stop 1
  call get_command_argument(3, argument)
  read(argument, *, iostat = read_status) repeat_count
  if (read_status /= 0) stop 1

  image_pitch = width * storage_size(0.0_real32) / 8
  num_pix = width * height
  allocate(image(0:num_pix - 1))

  ! Preserve the reference libc sequence and its packed RGBA bit pattern.
  call c_srand(123_c_int)
  do i = 0, num_pix - 1
    random_x = modulo(c_rand(), 256_c_int)
    random_y = modulo(c_rand(), 256_c_int)
    random_z = modulo(c_rand(), 256_c_int)
    random_w = modulo(c_rand(), 256_c_int)
    bits = ior(ior(int(random_x, int32), shiftl(int(random_y, int32), 8)), &
        ior(shiftl(int(random_z, int32), 16), shiftl(int(random_w, int32), 24)))
    image(i) = transfer(bits, image(i))
  end do

  total_seconds = 0.0_real64

  ! The reference allocates the device image once, uploads it before each pair
  ! of kernels, and copies the final coefficients only when the region exits.
!$omp target data map(from : image(0:num_pix - 1))
  num_threads_x = min(pow_two_divider(height), 64)
  num_threads_y = min(pow_two_divider(width), 64)

  do i = 1, repeat_count
!$omp target update to(image(0:num_pix - 1))
    call system_clock(count_start, count_rate)

!$omp target teams distribute parallel do thread_limit(num_threads_x)
    do y = 0, height - 1
      call convert_to_interpolation_coefficients(image, y * width, width, 1)
    end do
!$omp end target teams distribute parallel do

!$omp target teams distribute parallel do thread_limit(num_threads_y)
    do x = 0, width - 1
      call convert_to_interpolation_coefficients(image, x, height, width)
    end do
!$omp end target teams distribute parallel do

    call system_clock(count_end)
    total_seconds = total_seconds + real(count_end - count_start, real64) / real(count_rate, real64)
  end do

  print '(A)', 'Average kernel execution time ' // &
      trim(fixed_six(total_seconds / real(repeat_count, real64))) // ' (s)'
!$omp end target data

  checksum = 0.0_real32
  do i = 0, num_pix - 1
    bits = transfer(image(i), bits)
    checksum = checksum + real((iand(bits, 255_int32) + &
        iand(ishft(bits, -8), 255_int32) + iand(ishft(bits, -16), 255_int32) + &
        iand(ishft(bits, -24), 255_int32)) / 4, real32)
  end do
  print '(A)', 'Checksum: ' // trim(fixed_six(real(checksum, real64) / real(num_pix, real64)))

  deallocate(image)

contains

  function fixed_six(value) result(text)
    real(real64), intent(in) :: value
    character(len = 64) :: text, raw

    write(raw, '(F0.6)') value
    raw = adjustl(raw)
    if (raw(1:1) == '.') then
      text = '0' // trim(raw)
    else if (raw(1:2) == '-.') then
      text = '-0' // trim(raw(2:))
    else
      text = trim(raw)
    end if
  end function fixed_six
end program cbsfil
