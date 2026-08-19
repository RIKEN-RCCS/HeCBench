module lattice_mod
  use iso_fortran_env, only: real64
  use su3_types
  implicit none
contains
  subroutine init_link(s, val)
    type(su3_matrix), intent(out) :: s(0:)
    type(complex_t), intent(in) :: val
    integer :: j, k, l
    do j = 0, 3
      do k = 0, 2
        do l = 0, 2
          s(j)%e(k,l) = val
        end do
      end do
    end do
  end subroutine init_link

  subroutine make_lattice(s, n, val)
    type(site), intent(out) :: s(0:)
    integer, intent(in) :: n
    type(complex_t), intent(in) :: val
    integer :: nx, ny, nz, nt, x, y, z, t, idx
    nx = n; ny = n; nz = n; nt = n
    idx = 0
    do t = 0, nt - 1
      do z = 0, nz - 1
        do y = 0, ny - 1
          do x = 0, nx - 1
            s(idx)%x = x
            s(idx)%y = y
            s(idx)%z = z
            s(idx)%t = t
            s(idx)%index = x + nx * (y + ny * (z + nz * t))
            if (mod(x + y + z + t, 2) == 0) then
              s(idx)%parity = even
            else
              s(idx)%parity = odd
            end if
            call init_link(s(idx)%link, val)
            idx = idx + 1
          end do
        end do
      end do
    end do
  end subroutine make_lattice
end module lattice_mod
