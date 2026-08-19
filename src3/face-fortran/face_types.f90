module face_types
  use iso_fortran_env, only: int8, int32, real32
  implicit none
  type :: image_t
    integer(int32) :: width = 0, height = 0, maxgrey = 255
    integer(int32), allocatable :: data(:)
  end type image_t
  type :: rect_t
    integer(int32) :: x=0, y=0, width=0, height=0
  end type rect_t
contains
  integer(int32) function iround(x)
    real(real32), intent(in) :: x
    if (x >= 0.0_real32) then; iround = int(x + .5_real32, int32)
    else; iround = int(x - .5_real32, int32); end if
  end function iround
end module face_types
