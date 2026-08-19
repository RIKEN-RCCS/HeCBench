module particle_mod
  use iso_fortran_env, only: int64, real32, real64
  implicit none
  type :: particle
    real(real32) :: pos(0:2)
    real(real32) :: vel(0:2)
    real(real32) :: acc(0:2)
    real(real32) :: mass
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function
end module
