module easywave_okada
  use easywave_constants, only: dp, pi
  implicit none
contains
  subroutine okada_vertical(l, w, d, sind, cosd, u1, u2, x, y, uz)
    real(dp), intent(in) :: l, w, d, sind, cosd, u1, u2, x, y
    real(dp), intent(out) :: uz
    real(dp) :: p, q, u1z, u2z
    p = y*cosd + d*sind
    q = y*sind - d*cosd
    u1z = 0.0_dp
    u2z = 0.0_dp
    if (u1 /= 0.0_dp) u1z = -u1/(2.0_dp*pi) * chinnery_ss_uz(x, p, q, l, w, sind, cosd)
    if (u2 /= 0.0_dp) u2z = -u2/(2.0_dp*pi) * chinnery_ds_uz(x, p, q, l, w, sind, cosd)
    if (abs(u1z) > 1000.0_dp) u1z = 0.0_dp
    if (abs(u2z) > 1000.0_dp) u2z = 0.0_dp
    uz = u1z + u2z
  end subroutine okada_vertical

  pure real(dp) function chinnery_ss_uz(x, p, q, l, w, sd, cd)
    real(dp), intent(in) :: x, p, q, l, w, sd, cd
    chinnery_ss_uz = ss_uz(x,p,q,sd,cd) - ss_uz(x,p-w,q,sd,cd) - &
                      ss_uz(x-l,p,q,sd,cd) + ss_uz(x-l,p-w,q,sd,cd)
  end function chinnery_ss_uz

  pure real(dp) function chinnery_ds_uz(x, p, q, l, w, sd, cd)
    real(dp), intent(in) :: x, p, q, l, w, sd, cd
    chinnery_ds_uz = ds_uz(x,p,q,sd,cd) - ds_uz(x,p-w,q,sd,cd) - &
                      ds_uz(x-l,p,q,sd,cd) + ds_uz(x-l,p-w,q,sd,cd)
  end function chinnery_ds_uz

  pure real(dp) function radius(ksi, eta, q)
    real(dp), intent(in) :: ksi, eta, q
    radius = sqrt(ksi*ksi + eta*eta + q*q)
  end function radius

  pure real(dp) function dpval(eta, q, sd, cd)
    real(dp), intent(in) :: eta, q, sd, cd
    dpval = eta*sd - q*cd
  end function dpval

  pure real(dp) function ypval(eta, q, sd, cd)
    real(dp), intent(in) :: eta, q, sd, cd
    ypval = eta*cd + q*sd
  end function ypval

  pure real(dp) function i4(ksi, eta, q, sd, cd)
    real(dp), intent(in) :: ksi, eta, q, sd, cd
    real(dp) :: r, dpp
    r = radius(ksi,eta,q); dpp = dpval(eta,q,sd,cd)
    if (abs(cd) > 1.0e-14_dp) then
      i4 = 0.5_dp/cd * (log(r+dpp) - sd*log(r+eta))
    else
      i4 = -0.5_dp*q/(r+dpp)
    end if
  end function i4

  pure real(dp) function i5(ksi, eta, q, sd, cd)
    real(dp), intent(in) :: ksi, eta, q, sd, cd
    real(dp) :: r, xx, dpp
    if (ksi == 0.0_dp) then
      i5 = 0.0_dp
      return
    end if
    r = radius(ksi,eta,q); xx = sqrt(ksi*ksi+q*q); dpp = dpval(eta,q,sd,cd)
    if (abs(cd) > 1.0e-14_dp) then
      i5 = 1.0_dp/cd * atan((eta*(xx+q*cd)+xx*(r+xx)*sd)/(ksi*(r+xx)*cd))
    else
      i5 = -0.5_dp*ksi*sd/(r+dpp)
    end if
  end function i5

  pure real(dp) function ss_uz(ksi, eta, q, sd, cd)
    real(dp), intent(in) :: ksi, eta, q, sd, cd
    real(dp) :: r, dpp
    r = radius(ksi,eta,q); dpp = dpval(eta,q,sd,cd)
    ss_uz = dpp*q/r/(r+eta) + q*sd/(r+eta) + i4(ksi,eta,q,sd,cd)*sd
  end function ss_uz

  pure real(dp) function atan_term(ksi, eta, q, r)
    real(dp), intent(in) :: ksi, eta, q, r
    if (q*r == 0.0_dp) then
      if (ksi*eta == 0.0_dp) then
        atan_term = 0.0_dp
      else if (ksi*eta*q*r > 0.0_dp) then
        atan_term = pi
      else
        atan_term = -pi
      end if
    else
      atan_term = atan(ksi*eta/q/r)
    end if
  end function atan_term

  pure real(dp) function ds_uz(ksi, eta, q, sd, cd)
    real(dp), intent(in) :: ksi, eta, q, sd, cd
    real(dp) :: r, dpp
    r = radius(ksi,eta,q); dpp = dpval(eta,q,sd,cd)
    ds_uz = dpp*q/r/(r+ksi) + sd*atan_term(ksi,eta,q,r) - i5(ksi,eta,q,sd,cd)*sd*cd
  end function ds_uz
end module easywave_okada
