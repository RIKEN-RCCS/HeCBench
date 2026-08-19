module rfs_mod
  use iso_fortran_env, only: real32
  implicit none
  real(real32), parameter :: flt_epsilon = epsilon(1.0_real32)
!$omp declare target (create_rounding_factor, truncate_with_rounding_factor)

contains

  real(real32) function create_rounding_factor(max_value, n) result(factor)
    real(real32), intent(in) :: max_value
    integer, intent(in) :: n
    real(real32) :: delta
    integer :: exponent_value

    delta = (max_value * real(n, real32)) / (1.0_real32 - 2.0_real32 * real(n, real32) * flt_epsilon)
    ! Fortran EXPONENT has the same result as the exponent returned by
    ! frexpf() in the C++ source (including exact powers of two).
    exponent_value = exponent(delta)
    factor = scale(1.0_real32, exponent_value)
  end function create_rounding_factor

  real(real32) function truncate_with_rounding_factor(rounding_factor, x) result(y)
    real(real32), intent(in) :: rounding_factor, x
    y = (rounding_factor + x) - rounding_factor
  end function truncate_with_rounding_factor

  subroutine sum_array(factor, length, x, x_offset, r, result_index)
    real(real32), intent(in) :: factor
    integer, intent(in) :: length, x_offset, result_index
    real(real32), intent(in) :: x(0:)
    real(real32), intent(inout) :: r(0:)
    integer :: i
    real(real32) :: q

    !$omp target teams distribute parallel do num_teams(256) thread_limit(256) private(q)
    do i = 0, length - 1
      q = truncate_with_rounding_factor(factor, x(x_offset + i))
      !$omp atomic update
      r(result_index) = r(result_index) + q
    end do
    !$omp end target teams distribute parallel do
  end subroutine sum_array

  subroutine sum_arrays(n_arrays, length, x, r, max_val)
    integer, intent(in) :: n_arrays, length
    real(real32), intent(in) :: x(0:), max_val(0:)
    real(real32), intent(out) :: r(0:)
    integer :: i, n
    real(real32) :: factor, s

    !$omp target teams distribute parallel do num_teams(256) thread_limit(256) private(factor,s,n)
    do i = 0, n_arrays - 1
      factor = create_rounding_factor(max_val(i), length)
      s = 0.0_real32
      do n = length - 1, 0, -1
        s = s + truncate_with_rounding_factor(factor, x(i * length + n))
      end do
      r(i) = s
    end do
    !$omp end target teams distribute parallel do
  end subroutine sum_arrays

end module rfs_mod
