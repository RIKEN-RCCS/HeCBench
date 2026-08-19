module atan2_kernels
  use, intrinsic :: iso_fortran_env, only : real32, real64, int16, int32
  implicit none

!$omp declare target (approx_atan2f, unsafe_atan2f, safe_atan2f, approx_atan2i, unsafe_atan2i, approx_atan2s, unsafe_atan2s)
contains

  pure function approx_atan2f(degree, x) result(value)
    integer, intent(in) :: degree
    real(real32), intent(in) :: x
    real(real32) :: value, z

    z = x * x
    select case (degree)
    case (3)
      value = x * (-0.972394109_real32 + x * x * 0.191947937_real32)
    case (5)
      value = x * (-0.995357990_real32 + z * (0.288690388_real32 + z * (-0.0793391615_real32)))
    case (7)
      value = x * (-0.999213815_real32 + z * (0.321174979_real32 + z * (-0.146264464_real32 + z * 0.0389865041_real32)))
    case (9)
      value = x * (-0.999866366_real32 + z * (0.330305159_real32 + z * (-0.180160433_real32 + &
              z * (0.0851577073_real32 + z * (-0.0208456703_real32)))))
    case (11)
      value = x * (-0.999977231_real32 + z * (0.332623005_real32 + z * (-0.193541199_real32 + &
              z * (0.116428122_real32 + z * (-0.0526488200_real32 + z * 0.0117196217_real32)))))
    case (13)
      value = x * (-0.999996066_real32 + z * (0.333172798_real32 + z * (-0.198072404_real32 + &
              z * (0.132316172_real32 + z * (-0.0795975327_real32 + &
              z * (0.0335847437_real32 + z * (-0.00680612121_real32)))))))
    case (15)
      ! The OMP source deliberately casts this whole inner expression to float:
      ! float(0x5.552f9p-4 + z * (...)).  Keep its double addition before the
      ! single-precision cast instead of rewriting it as the usual Horner form.
      value = x * (-0.999999285_real32 + z * real(0.33329731225967407_real64 + &
              real(z * (-0.199454457_real32 + z * (0.139040351_real32 + &
              z * (-0.0963211507_real32 + z * (0.0557907447_real32 + &
              z * (-0.0217869878_real32 + z * 0.00403534528_real32))))), real64), real32))
    end select
  end function approx_atan2f

  pure function unsafe_atan2f(y, x, degree) result(value)
    real(real32), intent(in) :: y, x
    integer, intent(in) :: degree
    real(real32) :: value, r, angle

    r = (abs(x) - abs(y)) / (abs(x) + abs(y))
    if (x < 0.0_real32) r = -r
    if (x >= 0.0_real32) then
      angle = 0.785398185_real32
    else
      angle = 2.35619450_real32
    end if
    angle = angle + approx_atan2f(degree, r)
    if (y < 0.0_real32) angle = -angle
    value = angle
  end function unsafe_atan2f

  pure function safe_atan2f(y, x, degree) result(value)
    real(real32), intent(in) :: y, x
    integer, intent(in) :: degree
    real(real32) :: value, safe_x

    safe_x = x
    if (y == 0.0_real32 .and. x == 0.0_real32) safe_x = 0.2_real32
    value = unsafe_atan2f(y, safe_x, degree)
  end function safe_atan2f

  pure function approx_atan2i(degree, x) result(value)
    integer, intent(in) :: degree
    real(real32), intent(in) :: x
    real(real32) :: value, z

    z = x * x
    select case (degree)
    case (3)
      value = x * (-664694912.0_real32 + z * 131209024.0_real32)
    case (5)
      value = x * (-680392064.0_real32 + z * (197338400.0_real32 + z * (-54233256.0_real32)))
    case (7)
      value = x * (-683027840.0_real32 + z * (219543904.0_real32 + z * (-99981040.0_real32 + z * 26649684.0_real32)))
    case (9)
      value = x * (-683473920.0_real32 + z * (225785056.0_real32 + z * (-123151184.0_real32 + &
              z * (58210592.0_real32 + z * (-14249276.0_real32)))))
    case (11)
      value = x * (-683549696.0_real32 + z * (227369312.0_real32 + z * (-132297008.0_real32 + &
              z * (79584144.0_real32 + z * (-35987016.0_real32 + z * 8010488.0_real32)))))
    case (13, 15)
      value = x * (-683562624.0_real32 + z * (227746080.0_real32 + z * (-135400128.0_real32 + &
              z * (90460848.0_real32 + z * (-54431464.0_real32 + z * (22973256.0_real32 + &
              z * (-4657049.0_real32)))))))
    end select
  end function approx_atan2i

  pure function unsafe_atan2i(y, x, degree) result(value)
    real(real32), intent(in) :: y, x
    integer, intent(in) :: degree
    integer(int32) :: value, angle
    real(real32) :: r

    r = (abs(x) - abs(y)) / (abs(x) + abs(y))
    if (x < 0.0_real32) r = -r
    if (x >= 0.0_real32) then
      angle = 536870912_int32
    else
      angle = 1610612736_int32
    end if
    angle = angle + int(approx_atan2i(degree, r), int32)
    if (y < 0.0_real32) angle = -angle
    value = angle
  end function unsafe_atan2i

  pure function approx_atan2s(degree, x) result(value)
    integer, intent(in) :: degree
    real(real32), intent(in) :: x
    real(real32) :: value, z

    z = x * x
    select case (degree)
    case (3)
      value = x * (-10142.439453125_real32 + z * 2002.0908203125_real32)
    case (5)
      value = x * (-10381.9609375_real32 + z * (3011.1513671875_real32 + z * (-827.538330078125_real32)))
    case (7)
      value = x * (-10422.177734375_real32 + z * (3349.97412109375_real32 + z * &
              (-1525.589599609375_real32 + z * 406.64190673828125_real32)))
    case (9)
      value = x * (-10428.984375_real32 + z * (3445.20654296875_real32 + z * (-1879.137939453125_real32 + &
              z * (888.22314453125_real32 + z * (-217.42669677734375_real32)))))
    end select
  end function approx_atan2s

  pure function unsafe_atan2s(y, x, degree) result(value)
    real(real32), intent(in) :: y, x
    integer, intent(in) :: degree
    integer(int16) :: value, angle
    real(real32) :: r

    r = (abs(x) - abs(y)) / (abs(x) + abs(y))
    if (x < 0.0_real32) r = -r
    if (x >= 0.0_real32) then
      angle = 8192_int16
    else
      angle = 24576_int16
    end if
    angle = transfer(int(angle, int32) + int(approx_atan2s(degree, r), int32), angle)
    if (y < 0.0_real32) angle = -angle
    value = angle
  end function unsafe_atan2s

  subroutine compute_f(n, x, y, r)
    integer(int32), value, intent(in) :: n
    real(real32), intent(in) :: x(0:n-1), y(0:n-1)
    real(real32), intent(out) :: r(0:n-1)
    integer(int32) :: i

!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, n - 1
      r(i) = safe_atan2f(y(i), x(i), 3) + safe_atan2f(y(i), x(i), 5) + &
             safe_atan2f(y(i), x(i), 7) + safe_atan2f(y(i), x(i), 9) + &
             safe_atan2f(y(i), x(i), 11) + safe_atan2f(y(i), x(i), 13) + &
             safe_atan2f(y(i), x(i), 15)
    end do
!$omp end target teams distribute parallel do
  end subroutine compute_f

  subroutine compute_i(n, x, y, r)
    integer(int32), value, intent(in) :: n
    real(real32), intent(in) :: x(0:n-1), y(0:n-1)
    integer(int32), intent(out) :: r(0:n-1)
    integer(int32) :: i

!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, n - 1
      r(i) = unsafe_atan2i(y(i), x(i), 3) + unsafe_atan2i(y(i), x(i), 5) + &
             unsafe_atan2i(y(i), x(i), 7) + unsafe_atan2i(y(i), x(i), 9) + &
             unsafe_atan2i(y(i), x(i), 11) + unsafe_atan2i(y(i), x(i), 13) + &
             unsafe_atan2i(y(i), x(i), 15)
    end do
!$omp end target teams distribute parallel do
  end subroutine compute_i

  subroutine compute_s(n, x, y, r)
    integer(int32), value, intent(in) :: n
    real(real32), intent(in) :: x(0:n-1), y(0:n-1)
    integer(int16), intent(out) :: r(0:n-1)
    integer(int32) :: i

!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, n - 1
      r(i) = transfer(int(unsafe_atan2s(y(i), x(i), 3), int32) + int(unsafe_atan2s(y(i), x(i), 5), int32) + &
                      int(unsafe_atan2s(y(i), x(i), 7), int32) + int(unsafe_atan2s(y(i), x(i), 9), int32), r(i))
    end do
!$omp end target teams distribute parallel do
  end subroutine compute_s

  subroutine reference_f(n, x, y, r)
    integer(int32), value, intent(in) :: n
    real(real32), intent(in) :: x(0:n-1), y(0:n-1)
    real(real32), intent(out) :: r(0:n-1)
    integer(int32) :: i
    do i = 0, n - 1
      r(i) = safe_atan2f(y(i), x(i), 3) + safe_atan2f(y(i), x(i), 5) + &
             safe_atan2f(y(i), x(i), 7) + safe_atan2f(y(i), x(i), 9) + &
             safe_atan2f(y(i), x(i), 11) + safe_atan2f(y(i), x(i), 13) + &
             safe_atan2f(y(i), x(i), 15)
    end do
  end subroutine reference_f

  subroutine reference_i(n, x, y, r)
    integer(int32), value, intent(in) :: n
    real(real32), intent(in) :: x(0:n-1), y(0:n-1)
    integer(int32), intent(out) :: r(0:n-1)
    integer(int32) :: i
    do i = 0, n - 1
      r(i) = unsafe_atan2i(y(i), x(i), 3) + unsafe_atan2i(y(i), x(i), 5) + &
             unsafe_atan2i(y(i), x(i), 7) + unsafe_atan2i(y(i), x(i), 9) + &
             unsafe_atan2i(y(i), x(i), 11) + unsafe_atan2i(y(i), x(i), 13) + &
             unsafe_atan2i(y(i), x(i), 15)
    end do
  end subroutine reference_i

  subroutine reference_s(n, x, y, r)
    integer(int32), value, intent(in) :: n
    real(real32), intent(in) :: x(0:n-1), y(0:n-1)
    integer(int16), intent(out) :: r(0:n-1)
    integer(int32) :: i
    do i = 0, n - 1
      r(i) = transfer(int(unsafe_atan2s(y(i), x(i), 3), int32) + int(unsafe_atan2s(y(i), x(i), 5), int32) + &
                      int(unsafe_atan2s(y(i), x(i), 7), int32) + int(unsafe_atan2s(y(i), x(i), 9), int32), r(i))
    end do
  end subroutine reference_s

  function fixed6(value) result(text)
    real(real64), intent(in) :: value
    character(len=64) :: text, raw

    write(raw, '(f60.6)') value
    text = adjustl(raw)
  end function fixed6
end module atan2_kernels

program main
  use, intrinsic :: iso_fortran_env, only : real32, real64, int16, int32, int64
  use, intrinsic :: iso_c_binding, only : c_int
  use atan2_kernels
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  integer(int32) :: n, repeat, i
  integer(int64) :: start_count, end_count, count_rate
  integer :: argument_status
  character(len=64) :: argument
  real(real64) :: elapsed_us
  real(real32) :: error
  real(real32), allocatable :: x(:), y(:), hf(:), rf(:)
  integer(int32), allocatable :: hi(:), ri(:)
  integer(int16), allocatable :: hs(:), rs(:)

  if (command_argument_count() /= 2) then
    write(*, '(a)') 'Usage: ./main <number of coordinates> <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument, status=argument_status)
  read(argument, *) n
  call get_command_argument(2, argument, status=argument_status)
  read(argument, *) repeat

  allocate(x(0:n-1), y(0:n-1), hf(0:n-1), hi(0:n-1), hs(0:n-1))
  allocate(rf(0:n-1), ri(0:n-1), rs(0:n-1))

  call c_srand(123_c_int)
  do i = 0, n - 1
    x(i) = real(c_rand(), real32) / 2147483647.0_real32 + 1.57_real32
    y(i) = real(c_rand(), real32) / 2147483647.0_real32 + 1.57_real32
  end do

!$omp target data map(to: x(0:n-1), y(0:n-1)) map(alloc: hf(0:n-1), hi(0:n-1), hs(0:n-1))
  write(*, '(a)') ''
  write(*, '(a)') '======== output type is f32 ========'
  call system_clock(start_count, count_rate)
  do i = 1, repeat
    call compute_f(n, y, x, hf)
  end do
  call system_clock(end_count)
  elapsed_us = real(end_count - start_count, real64) * 1.0e6_real64 / real(count_rate, real64) / real(repeat, real64)
  write(*, '(a)') 'Average execution time: ' // trim(fixed6(elapsed_us)) // ' (us)'
!$omp target update from(hf(0:n-1))
  call reference_f(n, y, x, rf)
  error = 0.0_real32
  do i = 0, n - 1
    if (abs(rf(i) - hf(i)) > 1.0e-3_real32) then
      error = error + real((ri(i) - hi(i)) * (ri(i) - hi(i)), real32)
    end if
  end do
  write(*, '(a)') 'RMSE: ' // trim(fixed6(real(sqrt(error / real(n, real32)), real64)))

  write(*, '(a)') ''
  write(*, '(a)') '======== output type is i32 ========'
  call system_clock(start_count, count_rate)
  do i = 1, repeat
    call compute_i(n, y, x, hi)
  end do
  call system_clock(end_count)
  elapsed_us = real(end_count - start_count, real64) * 1.0e6_real64 / real(count_rate, real64) / real(repeat, real64)
  write(*, '(a)') 'Average execution time: ' // trim(fixed6(elapsed_us)) // ' (us)'
!$omp target update from(hi(0:n-1))
  call reference_i(n, y, x, ri)
  error = 0.0_real32
  do i = 0, n - 1
    if (abs(ri(i) - hi(i)) > 0_int32) then
      error = error + real((ri(i) - hi(i)) * (ri(i) - hi(i)), real32)
    end if
  end do
  write(*, '(a)') 'RMSE: ' // trim(fixed6(real(sqrt(error / real(n, real32)), real64)))

  write(*, '(a)') ''
  write(*, '(a)') '======== output type is i16 ========'
  call system_clock(start_count, count_rate)
  do i = 1, repeat
    call compute_s(n, y, x, hs)
  end do
  call system_clock(end_count)
  elapsed_us = real(end_count - start_count, real64) * 1.0e6_real64 / real(count_rate, real64) / real(repeat, real64)
  write(*, '(a)') 'Average execution time: ' // trim(fixed6(elapsed_us)) // ' (us)'
!$omp target update from(hs(0:n-1))
  call reference_s(n, y, x, rs)
  error = 0.0_real32
  do i = 0, n - 1
    if (abs(int(rs(i), int32) - int(hs(i), int32)) > 0_int32) then
      error = error + real((int(rs(i), int32) - int(hs(i), int32)) * &
                           (int(rs(i), int32) - int(hs(i), int32)), real32)
    end if
  end do
  write(*, '(a)') 'RMSE: ' // trim(fixed6(real(sqrt(error / real(n, real32)), real64)))
!$omp end target data

  deallocate(x, y, hf, hi, hs, rf, ri, rs)
end program main
