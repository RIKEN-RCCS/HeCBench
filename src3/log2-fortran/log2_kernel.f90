module log2_kernel
  use iso_fortran_env, only: real32, int32, int64
  implicit none
!$omp declare target (binary_log)
contains
  pure real(real32) function binary_log(input, precision)
    real(real32), intent(in) :: input
    integer, intent(in) :: precision
    integer(int32) :: bits
    integer :: exponent, m, sum_m
    integer(int64) :: one, denom, prev_denom
    real(real32) :: result, y
    logical :: max_condition_met

    bits = transfer(input, bits)
    exponent = int(shiftr(iand(bits, int(z'7F800000', int32)), 23)) - 127
    m = 0
    sum_m = 0
    result = 0.0_real32
    y = input / real(ishft(1, exponent), real32)
    max_condition_met = .false.
    one = 1_int64
    denom = 0_int64
    prev_denom = 0_int64

    do while ((sum_m < precision + 1 .and. y /= 1.0_real32) .or. max_condition_met)
      m = 0
      do while (y < 2.0_real32 .and. sum_m + m < precision + 1)
        y = y * y
        m = m + 1
      end do
      sum_m = sum_m + m
      prev_denom = denom
      denom = shiftl(one, sum_m)
      if (sum_m >= precision) exit
      if (prev_denom > denom) then
        max_condition_met = .true.
        exit
      end if
      result = result + 1.0_real32 / real(denom, real32)
      y = y / 2.0_real32
    end do
    binary_log = real(exponent, real32) + result
  end function binary_log
end module log2_kernel
