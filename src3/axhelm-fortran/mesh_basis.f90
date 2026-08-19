module mesh_basis
  use, intrinsic :: iso_fortran_env, only : real32, real64
  implicit none
contains
  real(real32) function mygamma(x) result(v)
    real(real32), intent(in) :: x
    v = real(exp(log_gamma(real(x, real64))), real32)
  end function mygamma

  real(real32) function jacobi_p(a, alpha, beta, n) result(pn)
    real(real32), intent(in) :: a, alpha, beta
    integer, intent(in) :: n
    real(real32) :: p(0:n), gamma0, gamma1, aold, anew, bnew, h1
    integer :: i
    gamma0 = 2.0_real32 ** (alpha + beta + 1.0_real32) / (alpha + beta + 1.0_real32) * &
      mygamma(1.0_real32 + alpha) * mygamma(1.0_real32 + beta) / mygamma(1.0_real32 + alpha + beta)
    p(0) = 1.0_real32 / sqrt(gamma0)
    if (n == 0) then; pn = p(0); return; end if
    gamma1 = (alpha + 1.0_real32) * (beta + 1.0_real32) / (alpha + beta + 3.0_real32) * gamma0
    p(1) = ((alpha + beta + 2.0_real32) * a / 2.0_real32 + (alpha - beta) / 2.0_real32) / sqrt(gamma1)
    if (n == 1) then; pn = p(1); return; end if
    aold = 2.0_real32 / (2.0_real32 + alpha + beta) * sqrt((alpha + 1.0_real32) * (beta + 1.0_real32) / (alpha + beta + 3.0_real32))
    do i = 1, n - 1
      h1 = 2.0_real32 * real(i, real32) + alpha + beta
      anew = 2.0_real32 / (h1 + 2.0_real32) * sqrt((real(i + 1, real32)) * (real(i + 1, real32) + alpha + beta) * &
        (real(i + 1, real32) + alpha) * (real(i + 1, real32) + beta) / (h1 + 1.0_real32) / (h1 + 3.0_real32))
      bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2.0_real32)
      p(i + 1) = (-aold * p(i - 1) + (a - bnew) * p(i)) / anew
      aold = anew
    end do
    pn = p(n)
  end function jacobi_p

  real(real32) function grad_jacobi_p(a, alpha, beta, n) result(v)
    real(real32), intent(in) :: a, alpha, beta
    integer, intent(in) :: n
    v = 0.0_real32
    if (n > 0) v = sqrt(real(n, real32) * (real(n, real32) + alpha + beta + 1.0_real32)) * &
      jacobi_p(a, alpha + 1.0_real32, beta + 1.0_real32, n - 1)
  end function grad_jacobi_p

  subroutine dgeev_interface(jobvl, jobvr, n, a, wr, wi, vl, vr, work, info)
    character(len=1), intent(in) :: jobvl, jobvr
    integer, intent(in) :: n
    real(real64), intent(inout) :: a(n,n), wr(n), wi(n), vl(n,n), vr(n,n), work(8*n)
    integer, intent(out) :: info
    interface
      subroutine dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
        import real64
        character(len=1) :: jobvl, jobvr
        integer :: n, lda, ldvl, ldvr, lwork, info
        real(real64) :: a(lda,*), wr(*), wi(*), vl(ldvl,*), vr(ldvr,*), work(*)
      end subroutine dgeev
    end interface
    call dgeev(jobvl, jobvr, n, a, n, wr, wi, vl, n, vr, n, work, 8*n, info)
  end subroutine dgeev_interface

  subroutine dgesv_interface(n, nrhs, a, b, info)
    integer, intent(in) :: n, nrhs
    real(real64), intent(inout) :: a(n,n), b(n,nrhs)
    integer, intent(out) :: info
    integer :: ipiv(n)
    interface
      subroutine dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
        import real64
        integer :: n, nrhs, lda, ldb, ipiv(*), info
        real(real64) :: a(lda,*), b(ldb,*)
      end subroutine dgesv
    end interface
    call dgesv(n, nrhs, a, n, ipiv, b, n, info)
  end subroutine dgesv_interface

  subroutine mesh_jacobi_gq(n, x, w)
    integer, intent(in) :: n
    real(real32), intent(out) :: x(0:n), w(0:n)
    real(real32) :: jmat(0:n,0:n), h1, off, eps
    real(real64) :: a64(n+1,n+1), wr(n+1), wi(n+1), vl(n+1,n+1), vr(n+1,n+1), work(8*(n+1))
    real(real32) :: tx, tw
    integer :: i, m, info
    jmat = 0.0_real32
    do i = 0, n
      h1 = 2.0_real32 * real(i, real32)
      if (i < n) then
        off = 2.0_real32 / (h1 + 2.0_real32) * sqrt(real(i + 1, real32)**4 / ((h1 + 1.0_real32) * (h1 + 3.0_real32)))
        jmat(i,i+1) = off; jmat(i+1,i) = off
      end if
    end do
    eps = 1.0_real32
    do while (1.0_real32 + eps > 1.0_real32); eps = eps / 2.0_real32; end do
    jmat(0,0) = 0.0_real32
    a64 = real(jmat, real64)
    call dgeev_interface('N','V',n+1,a64,wr,wi,vl,vr,work,info)
    if (info /= 0) error stop 'dgeev failed'
    do i=0,n
      x(i)=real(wr(i+1),real32)
      w(i)=real(2.0_real64*vr(1,i+1)*vr(1,i+1),real32)
    end do
    do i=0,n
      do m=i+1,n
        if(x(i)>x(m)) then
          tx=x(m); tw=w(m); x(m)=x(i); w(m)=w(i); x(i)=tx; w(i)=tw
        end if
      end do
    end do
  end subroutine mesh_jacobi_gq

  subroutine mesh_dmatrix_1d(n, r, d)
    integer, intent(in) :: n
    real(real32), intent(in) :: r(0:n)
    real(real32), intent(out) :: d(0:n,0:n)
    real(real32) :: v(0:n,0:n), vr(0:n,0:n)
    real(real64) :: a64(n+1,n+1), b64(n+1,n+1)
    integer :: row, col, info
    do row=0,n
      do col=0,n
        v(row,col)=jacobi_p(r(row),0.0_real32,0.0_real32,col)
        vr(row,col)=grad_jacobi_p(r(row),0.0_real32,0.0_real32,col)
      end do
    end do
    a64=transpose(real(v,real64)); b64=transpose(real(vr,real64))
    call dgesv_interface(n+1,n+1,a64,b64,info)
    if(info/=0) error stop 'dgesv failed'
    d=real(transpose(b64),real32)
  end subroutine mesh_dmatrix_1d
end module mesh_basis
