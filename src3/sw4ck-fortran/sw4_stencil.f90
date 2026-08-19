module sw4_stencil_mod
  use iso_fortran_env, only: real64
  use sw4ck_utils
  implicit none
contains
  subroutine sw4_stencil(start0,n0,start1,n1,start2,n2,u,mu,lambda,met,jac,lu,strx,stry,nc,ni,nj,nk,a1,sgn,scale)
    integer, intent(in) :: start0,n0,start1,n1,start2,n2,nc,ni,nj,nk
    real(real64), intent(in) :: u(0:), mu(0:), lambda(0:), met(0:), jac(0:), strx(0:), stry(0:), a1, sgn, scale
    real(real64), intent(inout) :: lu(0:)
    integer :: i,j,k,c,idx,idxm,idxp
    real(real64) :: r, ijac
!$omp target teams distribute parallel do collapse(3) thread_limit(256) map(to:u,mu,lambda,met,jac,strx,stry) map(tofrom:lu) private(i,j,k,c,idx,idxm,idxp,r,ijac)
    do k = start2, n2 - 1
      do j = start1, n1 - 1
        do i = start0, n0 - 1
          if (i > 1 .and. i < ni - 2 .and. j > 1 .and. j < nj - 2 .and. k > 1 .and. k < nk - 2) then
            do c = 0, min(nc,3) - 1
              idx = idx4(c,i,j,k,nc,ni,nj)
              idxm = idx4(c,i-1,j,k,nc,ni,nj)
              idxp = idx4(c,i+1,j,k,nc,ni,nj)
              ijac = strx(i-start0) * stry(j-start1) / max(jac(min(idx,size(jac)-1)), 1.0e-30_real64)
              r = (2.0_real64*mu(min(idx,size(mu)-1)) + lambda(min(idx,size(lambda)-1))) * &
                  (u(idxm) - 2.0_real64*u(idx) + u(idxp))
              lu(idx) = a1 * lu(idx) + sgn * scale * ijac * r
            end do
          end if
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine sw4_stencil
end module sw4_stencil_mod
