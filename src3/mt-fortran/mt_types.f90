module mt_types
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  integer, parameter :: mt_rng_count = 4096
  integer, parameter :: mt_mm = 9, mt_nn = 19
  integer(int32), parameter :: mt_wmask = int(z'FFFFFFFF', int32)
  integer(int32), parameter :: mt_umask = int(z'FFFFFFFE', int32)
  integer(int32), parameter :: mt_lmask = int(z'00000001', int32)
  integer, parameter :: mt_shift0 = 12, mt_shiftb = 7, mt_shiftc = 15, mt_shift1 = 18
  real(real32), parameter :: pi = 3.14159265358979_real32
  type :: mt_struct_stripped
    integer(int32) :: matrix_a
    integer(int32) :: mask_b
    integer(int32) :: mask_c
    integer(int32) :: seed
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function
end module
