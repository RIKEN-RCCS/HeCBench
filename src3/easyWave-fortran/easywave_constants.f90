module easywave_constants
  use iso_fortran_env, only: real32, real64
  implicit none
  integer, parameter :: sp = real32, dp = real64
  integer, parameter :: max_vars_per_node = 12
  integer, parameter :: id_depth = 1, id_height = 2, id_hmax = 3
  integer, parameter :: id_m = 4, id_n = 5, id_r1 = 6, id_r2 = 7
  integer, parameter :: id_r3 = 8, id_r4 = 9, id_r5 = 10, id_time = 11, id_topo = 12
  real(dp), parameter :: rearth = 6384.0e3_dp, gravity = 9.81_dp, omega = 7.29e-5_dp
  real(dp), parameter :: pi = 3.14159265358979323846_dp
contains
  pure real(dp) function deg2rad(x)
    real(dp), intent(in) :: x
    deg2rad = x * pi / 180.0_dp
  end function deg2rad

  pure real(dp) function sindeg(x)
    real(dp), intent(in) :: x
    sindeg = sin(deg2rad(x))
  end function sindeg

  pure real(dp) function cosdeg(x)
    real(dp), intent(in) :: x
    cosdeg = cos(deg2rad(x))
  end function cosdeg
end module easywave_constants
