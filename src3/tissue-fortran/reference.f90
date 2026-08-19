module tissue_reference
  use iso_fortran_env, only: real32
  implicit none
contains
  subroutine reference(tisspoints, gtt, gbartt, ct, ctprev, qt, nnt, nntDev, step, isp)
    integer, intent(in) :: tisspoints(0:)
    real(real32), intent(in) :: gtt(0:), gbartt(0:), ctprev(0:), qt(0:)
    real(real32), intent(inout) :: ct(0:)
    integer, intent(in) :: nnt, nntDev, step, isp
    integer :: i, jtp, ixyz, ix, iy, iz, jx, jy, jz, istep, nnt2, itp, itp1
    real(real32) :: p
    nnt2 = 2 * nnt
    do i = 0, step * nnt - 1
      p = 0.0_real32
      itp = i / step
      itp1 = mod(i, step)
      if (itp < nnt) then
        ix = tisspoints(itp)
        iy = tisspoints(itp + nnt)
        iz = tisspoints(itp + nnt2)
        do jtp = itp1, nnt - 1, step
          jx = tisspoints(jtp)
          jy = tisspoints(jtp + nnt)
          jz = tisspoints(jtp + nnt2)
          ixyz = abs(jx - ix) + abs(jy - iy) + abs(jz - iz) + (isp - 1) * nntDev
          p = p + gtt(ixyz) * ctprev(jtp) + gbartt(ixyz) * qt(jtp)
        end do
        if (itp1 == 0) ct(itp) = p
      end if
      do istep = 1, step - 1
        if (itp1 == istep .and. itp < nnt) ct(itp) = ct(itp) + p
      end do
    end do
  end subroutine reference
end module tissue_reference
