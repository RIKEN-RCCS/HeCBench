module su3_types
  use iso_fortran_env, only: real64, int64
  implicit none
  integer, parameter :: even = int(z'02'), odd = int(z'01')
  type :: complex_t
    real(real64) :: real = 0.0_real64
    real(real64) :: imag = 0.0_real64
  end type complex_t
  type :: su3_matrix
    type(complex_t) :: e(0:2,0:2)
  end type su3_matrix
  type :: site
    type(su3_matrix) :: link(0:3)
    integer :: x = 0, y = 0, z = 0, t = 0
    integer :: index = 0
    integer :: parity = 0
    integer :: pad(0:9) = 0
  end type site
contains
  pure subroutine cmulsum(a, b, c)
    type(complex_t), intent(in) :: a, b
    type(complex_t), intent(inout) :: c
    c%real = c%real + a%real * b%real - a%imag * b%imag
    c%imag = c%imag + a%real * b%imag + a%imag * b%real
  end subroutine cmulsum
end module su3_types
