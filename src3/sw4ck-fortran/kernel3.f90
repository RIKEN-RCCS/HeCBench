module kernel3_mod
  use iso_fortran_env, only: real64
  use sw4ck_utils
  use sw4_stencil_mod
  implicit none
contains
  subroutine kernel3(start0,n0,start1,n1,start2,n2,u,mu,lambda,met,jac,lu,strx,stry,nc,ni,nj,nk,a1,sgn)
    integer, intent(in) :: start0,n0,start1,n1,start2,n2,nc,ni,nj,nk
    real(real64), intent(in) :: u(0:), mu(0:), lambda(0:), met(0:), jac(0:), strx(0:), stry(0:), a1, sgn
    real(real64), intent(inout) :: lu(0:)
    call sw4_stencil(start0,n0,start1,n1,start2,n2,u,mu,lambda,met,jac,lu,strx,stry,nc,ni,nj,nk,a1,sgn,3.0_real64)
  end subroutine kernel3
end module kernel3_mod
