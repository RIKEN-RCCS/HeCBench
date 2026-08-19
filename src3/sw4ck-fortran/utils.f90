module sw4ck_utils
  use iso_fortran_env, only: real64, int64
  implicit none
  type :: sarray_t
    character(len=32) :: name = ''
    integer :: g = 0, nc = 0, ni = 0, nj = 0, nk = 0
    integer :: ib = 0, ie = 0, jb = 0, je = 0, kb = 0, ke = 0
    integer(int64) :: base = 0, offi = 0, offj = 0, offk = 0, offc = 0, npts = 0
    real(real64), allocatable :: data(:)
  end type sarray_t
contains
  subroutine init_sarray(s)
    type(sarray_t), intent(inout) :: s
    integer :: i, j, k, c, idx
    real(real64) :: dx, x, y, z
    dx = 0.001_real64
    allocate(s%data(0:s%nc*s%ni*s%nj*s%nk-1))
    do i = 0, s%ni - 1
      do j = 0, s%nj - 1
        do k = 0, s%nk - 1
          do c = 0, s%nc - 1
            idx = c + i*s%nc + j*s%nc*s%ni + k*s%nc*s%ni*s%nj
            x = i * dx; y = j * dx; z = k * dx
            s%data(idx) = sin(x) * sin(y) * sin(z)
          end do
        end do
      end do
    end do
  end subroutine init_sarray

  real(real64) function norm_sarray(s)
    type(sarray_t), intent(in) :: s
    norm_sarray = sum(s%data * s%data)
  end function norm_sarray

  pure integer function idx4(c, i, j, k, nc, ni, nj) result(idx)
    integer, intent(in) :: c, i, j, k, nc, ni, nj
    idx = c + i*nc + j*nc*ni + k*nc*ni*nj
  end function idx4

  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds
end module sw4ck_utils
