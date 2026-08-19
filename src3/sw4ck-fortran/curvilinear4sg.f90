module curvilinear4sg_mod
  use iso_fortran_env, only: real64
  use kernel1_mod
  use kernel2_mod
  use kernel3_mod
  use kernel4_mod
  use kernel5_mod
  implicit none
contains
  subroutine curvilinear4sg_ci(ifirst, ilast, jfirst, jlast, kfirst, klast, u, mu, lambda, met, jac, lu, onesided, cof, str, nk, op, nc, ni, nj, nkk)
    integer, intent(in) :: ifirst, ilast, jfirst, jlast, kfirst, klast, nk, nc, ni, nj, nkk
    integer, intent(in) :: onesided(0:13)
    character(len=1), intent(in) :: op
    real(real64), intent(in) :: u(0:), mu(0:), lambda(0:), met(0:), jac(0:), cof(0:), str(0:)
    real(real64), intent(inout) :: lu(0:)
    integer :: kstart, kend, stry_offset
    real(real64) :: a1, sgn
    if (op == '=') then
      a1 = 0.0_real64; sgn = 1.0_real64
    else if (op == '+') then
      a1 = 1.0_real64; sgn = 1.0_real64
    else
      a1 = 1.0_real64; sgn = -1.0_real64
    end if
    kstart = kfirst + 2
    kend = klast - 2
    if (onesided(5) == 1) kend = nk - 6
    stry_offset = ilast - ifirst + 1
    if (onesided(4) == 1) then
      kstart = 7
      call kernel1(ifirst+2, ilast, jfirst+2, jlast, 1, 7, u, mu, lambda, met, jac, lu, str, str(stry_offset:), nc, ni, nj, nkk, a1, sgn)
    end if
    call kernel2(ifirst+2, ilast, jfirst+2, jlast, kstart, kend+1, u, mu, lambda, met, jac, lu, str, str(stry_offset:), nc, ni, nj, nkk, a1, sgn)
    call kernel3(ifirst+2, ilast, jfirst+2, jlast, kstart, kend+1, u, mu, lambda, met, jac, lu, str, str(stry_offset:), nc, ni, nj, nkk, a1, sgn)
    call kernel4(ifirst+2, ilast, jfirst+2, jlast, kstart, kend+1, u, mu, lambda, met, jac, lu, str, str(stry_offset:), nc, ni, nj, nkk, a1, sgn)
    if (onesided(5) == 1) then
      call kernel5(ifirst+2, ilast, jfirst+2, jlast, nk-5, nk+1, u, mu, lambda, met, jac, lu, str, str(stry_offset:), nc, ni, nj, nkk, a1, sgn)
    end if
  end subroutine curvilinear4sg_ci
end module curvilinear4sg_mod
