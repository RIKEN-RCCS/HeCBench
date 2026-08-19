module wsm5_kernel_mod
  use iso_fortran_env, only: real32
  use omp_lib
  use wsm5_spt
  implicit none
  real(real32), parameter :: r_v=0.461600006103516e3_real32, rv=r_v
  real(real32), parameter :: cice=0.210600000000000e4_real32, cliq=0.419000000000000e4_real32
  real(real32), parameter :: denr=0.100000000000000e4_real32, den0=0.127999997138977e1_real32
  real(real32), parameter :: xlf0=0.350000000000000e6_real32, xlv0=0.250000000000000e7_real32
  real(real32), parameter :: xls=0.285000000000000e7_real32, t0c=0.273149993896484e3_real32
  real(real32), parameter :: qmin=0.100000000362749e-14_real32, ep2=0.621750414371490e0_real32
  real(real32), parameter :: psat=0.610780029296875e3_real32, alpha=0.120000000000000e0_real32
  real(real32), parameter :: n0smax=0.100000000000000e12_real32, n0s=0.200000000000000e7_real32
  real(real32), parameter :: n0r=0.800000000000000e7_real32, qcrmin=0.100000000000000e-08_real32
  real(real32), parameter :: avtr=0.841900000000000e3_real32, bvtr=0.800000000000000e0_real32
  real(real32), parameter :: g4pbr=0.178173289058329e2_real32, avts=0.117200000000000e2_real32
  real(real32), parameter :: bvts=0.410000000000000e0_real32, g4pbs=0.102654190601850e2_real32
  real(real32), parameter :: pi=0.314159265358979e1_real32, dicon=0.119000000000000e2_real32
  real(real32), parameter :: dimax=0.500000000000000e-03_real32, pfrz1=0.100000000000000e3_real32
  real(real32), parameter :: pfrz2=0.660000000000000e0_real32, dens=100.0_real32
  real(real32), parameter :: cpv=4.0_real32*r_v, cp=7.0_real32*287.0_real32/2.0_real32, cv=cp-287.0_real32, cpd=cp
  real(real32), parameter :: pvtr=avtr*g4pbr/6.0_real32, pvts=avts*g4pbs/6.0_real32, xlv1=cliq-cv
  real(real32), parameter :: lamdarmax=0.800000000000000e5_real32, rslopermax=1.0_real32/lamdarmax
  real(real32), parameter :: rslopesmax=0.10000000000000001e-04_real32
  real(real32), parameter :: rsloperbmax=0.11954406247375457e-03_real32, rslopesbmax=0.89125093813374589e-02_real32
  real(real32), parameter :: rsloper2max=rslopermax*rslopermax, rslopes2max=rslopesmax*rslopesmax
  real(real32), parameter :: rsloper3max=rsloper2max*rslopermax, rslopes3max=rslopes2max*rslopesmax
  real(real32), parameter :: pidn0r=pi*denr*n0r, pidn0s=pi*dens*n0s
  real(real32), parameter :: precs1=4.0_real32*n0s*0.65_real32, precs2=4.0_real32*n0s*0.44_real32*sqrt(avts)*1.550308_real32
  real(real32), parameter :: qc0=4.0_real32/3.0_real32*pi*denr*(0.800000000000000e-05_real32**3)*0.300000000000000e09_real32/den0
!$omp declare target (cpmcal, xlcal, wsm_column)
contains
  pure real(real32) function cpmcal(x)
    real(real32), intent(in) :: x
    cpmcal = cpd * (1.0_real32 - max(x, qmin)) + max(x, qmin) * cpv
  end function cpmcal

  pure real(real32) function xlcal(x)
    real(real32), intent(in) :: x
    xlcal = xlv0 - xlv1 * (x - t0c)
  end function xlcal

  subroutine wsm(th, pii, q, qc, qi, qr, qs, den, p, delz, rain, rainncv, sr, snow, snowncv, delt, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, teamX, teamY)
    real(real32), intent(inout) :: th(0:), q(0:), qc(0:), qi(0:), qr(0:), qs(0:), rain(0:), rainncv(0:), sr(0:), snow(0:), snowncv(0:)
    real(real32), intent(in) :: pii(0:), den(0:), p(0:), delz(0:), delt
    integer, intent(in) :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, teamX, teamY
!$omp target teams num_teams(teamX*teamY) thread_limit(xxx*yyy) &
!$omp& map(to:pii,den,p,delz) map(tofrom:th,q,qc,qi,qr,qs,rain,rainncv,sr,snow,snowncv)
!$omp parallel
    call wsm_column(th, pii, q, qc, qi, qr, qs, den, p, delz, rain, rainncv, sr, snow, snowncv, delt, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, teamX, teamY)
!$omp end parallel
!$omp end target teams
  end subroutine wsm

  subroutine wsm_column(th, pii, q, qc, qi, qr, qs, den, p, delz, rain, rainncv, sr, snow, snowncv, delt, &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, teamX, teamY)
    real(real32), intent(inout) :: th(0:), q(0:), qc(0:), qi(0:), qr(0:), qs(0:), rain(0:), rainncv(0:), sr(0:), snow(0:), snowncv(0:)
    real(real32), intent(in) :: pii(0:), den(0:), p(0:), delz(0:), delt
    integer, intent(in) :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, teamX, teamY
    integer :: bi, bj, ti, tj, bx, by, ig, jg, k, loop, loops, mstep, nstep, idx, idx2
    real(real32) :: t(0:mkx-1), cpm(0:mkx-1), xl(0:mkx-1), qs1(0:mkx-1), qs2(0:mkx-1), rh1(0:mkx-1), rh2(0:mkx-1)
    real(real32) :: rsloper(0:mkx-1), rslopes(0:mkx-1), rslopebr(0:mkx-1), rslopebs(0:mkx-1)
    real(real32) :: rslope2r(0:mkx-1), rslope2s(0:mkx-1), rslope3r(0:mkx-1), rslope3s(0:mkx-1), denfac(0:mkx-1), n0sfac(0:mkx-1)
    real(real32) :: w1(0:mkx-1), w2(0:mkx-1), dtcldcr, dtcld, ttp, xa, xb, xai, xbi, tr, ltr, qq, pp, tt
    real(real32) :: supcol, xmi, diameter, rmstep, falk1, falk2, fallsum, fallsum_qsi, psmlt
    bi = omp_get_team_num() - (omp_get_team_num() / teamX) * teamX
    bj = omp_get_team_num() / teamX
    ti = omp_get_thread_num() - (omp_get_thread_num() / xxx) * xxx
    tj = omp_get_thread_num() / xxx
    bx = xxx; by = yyy
    ig = ti + bi * bx + ips - ims
    jg = tj + bj * by + jps - jms
    if (ig >= ide - ids + 1 .or. jg >= jde - jds + 1) return
    do k = kps - 1, kpe - 1
      idx = p3(ti,k,tj,bi,bj,bx,by,ips,ims,jps,jms,ime,kms,kme)
      t(k) = th(idx) * pii(idx)
      qc(idx) = max(qc(idx), 0.0_real32)
      qi(idx) = max(qi(idx), 0.0_real32)
      qr(idx) = max(qr(idx), 0.0_real32)
      qs(idx) = max(qs(idx), 0.0_real32)
      cpm(k) = cpmcal(q(idx))
      xl(k) = xlcal(t(k))
    end do
    dtcldcr = 120.0_real32
    loops = max(int(delt / dtcldcr + 0.5_real32), 1)
    dtcld = delt / real(loops, real32)
    if (delt <= dtcldcr) dtcld = delt
    do loop = 1, loops
      ttp = t0c + 0.01_real32
      xa = -(cpv - cliq) / rv
      xb = xa + xlv0 / (rv * ttp)
      xai = -(cpv - cice) / rv
      xbi = xai + xls / (rv * ttp)
      do k = kps - 1, kpe - 1
        idx = p3(ti,k,tj,bi,bj,bx,by,ips,ims,jps,jms,ime,kms,kme)
        pp = p(idx); tt = t(k); tr = ttp / tt; ltr = log(tr)
        qq = psat * exp(ltr * xa + xb * (1.0_real32 - tr))
        qq = ep2 * qq / (pp - qq)
        qs1(k) = max(qq, qmin)
        rh1(k) = max(q(idx) / qs1(k), qmin)
        if (tt < ttp) then
          qq = psat * exp(ltr * xai + xbi * (1.0_real32 - tr))
        else
          qq = psat * exp(ltr * xa + xb * (1.0_real32 - tr))
        end if
        qq = ep2 * qq / (pp - qq)
        qs2(k) = max(qq, qmin)
        rh2(k) = max(q(idx) / qs2(k), qmin)
        supcol = t0c - t(k)
        n0sfac(k) = max(min(exp(alpha * supcol), n0smax / n0s), 1.0_real32)
        if (qr(idx) <= qcrmin) then
          rsloper(k)=rslopermax; rslopebr(k)=rsloperbmax; rslope2r(k)=rsloper2max; rslope3r(k)=rsloper3max
        else
          rsloper(k)=1.0_real32/sqrt(sqrt(pidn0r/(qr(idx)*den(idx))))
          rslopebr(k)=exp(log(rsloper(k))*bvtr); rslope2r(k)=rsloper(k)*rsloper(k); rslope3r(k)=rslope2r(k)*rsloper(k)
        end if
        if (qs(idx) <= qcrmin) then
          rslopes(k)=rslopesmax; rslopebs(k)=rslopesbmax; rslope2s(k)=rslopes2max; rslope3s(k)=rslopes3max
        else
          rslopes(k)=1.0_real32/sqrt(sqrt(pidn0s*n0sfac(k)/(qs(idx)*den(idx))))
          rslopebs(k)=exp(log(rslopes(k))*bvts); rslope2s(k)=rslopes(k)*rslopes(k); rslope3s(k)=rslope2s(k)*rslopes(k)
        end if
        denfac(k) = sqrt(den0 / den(idx))
        w1(k) = pvtr * rslopebr(k) * denfac(k) / delz(idx)
        w2(k) = pvts * rslopebs(k) * denfac(k) / delz(idx)
      end do
      mstep = 1
      rmstep = 1.0_real32 / real(mstep, real32)
      fallsum = 0.0_real32
      fallsum_qsi = 0.0_real32
      do nstep = 1, mstep
        do k = kpe - 1, kps - 1, -1
          idx = p3(ti,k,tj,bi,bj,bx,by,ips,ims,jps,jms,ime,kms,kme)
          falk1 = den(idx) * qr(idx) * w1(k) * rmstep
          falk2 = den(idx) * qs(idx) * w2(k) * rmstep
          qr(idx) = max(qr(idx) - falk1 * dtcld / den(idx), 0.0_real32)
          qs(idx) = max(qs(idx) - falk2 * dtcld / den(idx), 0.0_real32)
          fallsum = falk1 + falk2
          fallsum_qsi = falk2
          if (t(k) > t0c .and. qs(idx) > 0.0_real32) then
            psmlt = min(max((t0c - t(k)) * rslope2s(k) * dtcld * rmstep, -qs(idx) * rmstep), 0.0_real32)
            qs(idx) = qs(idx) + psmlt
            qr(idx) = qr(idx) - psmlt
            t(k) = t(k) + xlf0 / cpmcal(q(idx)) * psmlt
          end if
        end do
      end do
      idx2 = p2(ti,tj,bi,bj,bx,by,ips,ims,jps,jms,ime)
      rainncv(idx2) = 0.0_real32
      if (fallsum > 0.0_real32) then
        rainncv(idx2) = fallsum * delz(p3(ti,1,tj,bi,bj,bx,by,ips,ims,jps,jms,ime,kms,kme)) / denr * dtcld * 1000.0_real32
        rain(idx2) = rain(idx2) + rainncv(idx2)
      end if
      snowncv(idx2) = 0.0_real32
      if (fallsum_qsi > 0.0_real32) then
        snowncv(idx2) = fallsum_qsi * delz(p3(ti,0,tj,bi,bj,bx,by,ips,ims,jps,jms,ime,kms,kme)) / denr * dtcld * 1000.0_real32
        snow(idx2) = snow(idx2) + snowncv(idx2)
      end if
      sr(idx2) = 0.0_real32
      if (fallsum > 0.0_real32) sr(idx2) = snowncv(idx2) / (rainncv(idx2) + 1.0e-12_real32)
      do k = kps - 1, kpe - 1
        idx = p3(ti,k,tj,bi,bj,bx,by,ips,ims,jps,jms,ime,kms,kme)
        if (t(k) <= t0c) then
          qc(idx) = max(qc(idx) - min(qc(idx), qc0) * dtcld, 0.0_real32)
          qi(idx) = max(qi(idx), 0.0_real32)
        else
          qr(idx) = max(qr(idx) + min(qc(idx), qc0) * dtcld, 0.0_real32)
          qc(idx) = max(qc(idx) - min(qc(idx), qc0) * dtcld, 0.0_real32)
        end if
        th(idx) = t(k) / pii(idx)
      end do
    end do
  end subroutine wsm_column
end module wsm5_kernel_mod
