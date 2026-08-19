module svd3x3_kernels
  use iso_fortran_env, only: real32
  implicit none
contains
  subroutine svd(a11,a12,a13,a21,a22,a23,a31,a32,a33, out)
    real(real32), intent(in) :: a11,a12,a13,a21,a22,a23,a31,a32,a33
    real(real32), intent(out) :: out(0:20)
    real(real32) :: a(0:2,0:2), ata(0:2,0:2), v(0:2,0:2), u(0:2,0:2), s(0:2)
    integer :: i, j, k
    a = reshape([a11,a21,a31,a12,a22,a32,a13,a23,a33], [3,3])
    ata = 0.0_real32
    do i = 0, 2
      do j = 0, 2
        do k = 0, 2
          ata(i,j) = ata(i,j) + a(k,i) * a(k,j)
        end do
      end do
    end do
    call jacobi_sym3(ata, v)
    do i = 0, 2
      s(i) = sqrt(max(ata(i,i), 0.0_real32))
    end do
    call sort_desc(s, v)
    u = 0.0_real32
    do j = 0, 2
      if (s(j) > 1.0e-12_real32) then
        do i = 0, 2
          do k = 0, 2
            u(i,j) = u(i,j) + a(i,k) * v(k,j)
          end do
          u(i,j) = u(i,j) / s(j)
        end do
      end if
    end do
    call normalize_columns(u)
    out(0)=u(0,0); out(1)=u(1,0); out(2)=u(2,0)
    out(3)=u(0,1); out(4)=u(1,1); out(5)=u(2,1)
    out(6)=u(0,2); out(7)=u(1,2); out(8)=u(2,2)
    out(9)=s(0); out(10)=s(1); out(11)=s(2)
    out(12)=v(0,0); out(13)=v(1,0); out(14)=v(2,0)
    out(15)=v(0,1); out(16)=v(1,1); out(17)=v(2,1)
    out(18)=v(0,2); out(19)=v(1,2); out(20)=v(2,2)
  end subroutine svd

  subroutine jacobi_sym3(a, v)
    real(real32), intent(inout) :: a(0:2,0:2)
    real(real32), intent(out) :: v(0:2,0:2)
    integer :: sweep
    v = 0.0_real32
    v(0,0)=1.0_real32; v(1,1)=1.0_real32; v(2,2)=1.0_real32
    do sweep = 1, 8
      call rotate(a, v, 0, 1)
      call rotate(a, v, 0, 2)
      call rotate(a, v, 1, 2)
    end do
  end subroutine jacobi_sym3

  subroutine rotate(a, v, p, q)
    real(real32), intent(inout) :: a(0:2,0:2), v(0:2,0:2)
    integer, intent(in) :: p, q
    real(real32) :: app, aqq, apq, tau, t, c, s, aip, aiq, vip, viq
    integer :: i
    apq = a(p,q)
    if (abs(apq) <= 1.0e-20_real32) return
    app = a(p,p); aqq = a(q,q)
    tau = (aqq - app) / (2.0_real32 * apq)
    if (tau >= 0.0_real32) then
      t = 1.0_real32 / (tau + sqrt(1.0_real32 + tau * tau))
    else
      t = -1.0_real32 / (-tau + sqrt(1.0_real32 + tau * tau))
    end if
    c = 1.0_real32 / sqrt(1.0_real32 + t * t)
    s = t * c
    do i = 0, 2
      if (i /= p .and. i /= q) then
        aip = a(i,p); aiq = a(i,q)
        a(i,p) = c * aip - s * aiq
        a(p,i) = a(i,p)
        a(i,q) = s * aip + c * aiq
        a(q,i) = a(i,q)
      end if
    end do
    a(p,p) = c*c*app - 2.0_real32*s*c*apq + s*s*aqq
    a(q,q) = s*s*app + 2.0_real32*s*c*apq + c*c*aqq
    a(p,q) = 0.0_real32; a(q,p) = 0.0_real32
    do i = 0, 2
      vip = v(i,p); viq = v(i,q)
      v(i,p) = c * vip - s * viq
      v(i,q) = s * vip + c * viq
    end do
  end subroutine rotate

  subroutine sort_desc(s, v)
    real(real32), intent(inout) :: s(0:2), v(0:2,0:2)
    integer :: i, j
    real(real32) :: ts, tv(0:2)
    do i = 0, 1
      do j = i + 1, 2
        if (s(j) > s(i)) then
          ts = s(i); s(i) = s(j); s(j) = ts
          tv = v(:,i); v(:,i) = v(:,j); v(:,j) = tv
        end if
      end do
    end do
  end subroutine sort_desc

  subroutine normalize_columns(u)
    real(real32), intent(inout) :: u(0:2,0:2)
    integer :: j
    real(real32) :: nrm
    do j = 0, 2
      nrm = sqrt(sum(u(:,j) * u(:,j)))
      if (nrm > 1.0e-12_real32) u(:,j) = u(:,j) / nrm
    end do
  end subroutine normalize_columns

  subroutine run_svd_device(input, output, testsize, repeat)
    real(real32), intent(in) :: input(0:)
    real(real32), intent(out) :: output(0:)
    integer, intent(in) :: testsize, repeat
    integer :: rep, tid
    real(real32) :: tmp(0:20)
    do rep = 1, repeat
!$omp target teams distribute parallel do thread_limit(256) map(to:input) map(from:output) private(tid,tmp)
      do tid = 0, testsize - 1
        call svd(input(tid+0*testsize), input(tid+1*testsize), input(tid+2*testsize), &
                 input(tid+3*testsize), input(tid+4*testsize), input(tid+5*testsize), &
                 input(tid+6*testsize), input(tid+7*testsize), input(tid+8*testsize), tmp)
        output(tid+0*testsize)=tmp(0); output(tid+1*testsize)=tmp(1); output(tid+2*testsize)=tmp(2)
        output(tid+3*testsize)=tmp(3); output(tid+4*testsize)=tmp(4); output(tid+5*testsize)=tmp(5)
        output(tid+6*testsize)=tmp(6); output(tid+7*testsize)=tmp(7); output(tid+8*testsize)=tmp(8)
        output(tid+9*testsize)=tmp(9); output(tid+10*testsize)=tmp(10); output(tid+11*testsize)=tmp(11)
        output(tid+12*testsize)=tmp(12); output(tid+13*testsize)=tmp(13); output(tid+14*testsize)=tmp(14)
        output(tid+15*testsize)=tmp(15); output(tid+16*testsize)=tmp(16); output(tid+17*testsize)=tmp(17)
        output(tid+18*testsize)=tmp(18); output(tid+19*testsize)=tmp(19); output(tid+20*testsize)=tmp(20)
      end do
!$omp end target teams distribute parallel do
    end do
  end subroutine run_svd_device
end module svd3x3_kernels
