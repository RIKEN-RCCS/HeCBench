module force_kernel_mod
  use iso_fortran_env, only: real64
  implicit none
  real(real64), parameter :: PI = 3.1415926535897932384626433832795_real64
  type :: d_atom
    real(real64) :: pos(0:2)=0.0_real64
    real(real64) :: eps=0.0_real64
    real(real64) :: sig=0.0_real64
    real(real64) :: charge=0.0_real64
    real(real64) :: f(0:2)=0.0_real64
    integer :: molid=0
    integer :: frozen=0
    real(real64) :: u(0:2)=0.0_real64
    real(real64) :: polar=0.0_real64
  end type d_atom
contains
  subroutine calculate_force_kernel(atom_list, n_atoms, cutoffd, basis, reciprocal_basis, pformd, ewald_alpha, kmax, kspace, polar_damp)
    type(d_atom), intent(inout) :: atom_list(0:)
    integer, intent(in) :: n_atoms, pformd, kmax, kspace
    real(real64), intent(in) :: cutoffd, basis(0:8), reciprocal_basis(0:8), ewald_alpha, polar_damp
    integer :: i,j,n,q,l0,l1,l2,p
    real(real64) :: rimg, rsq, sqrtpi, d(0:2), di(0:2), img(0:2), dimg(0:2), r, r2, ri, ri2
    real(real64) :: sig, eps, r6, s6, af(0:2), holder, chargeprod, invv, kvec(0:2), k_sq, fourpi
    real(real64) :: common_factor, rinv, r2inv, r3, r3inv, r5inv, r7inv, x2,y2,z2,x,y,z,udotu,ujdotr,uidotr
    real(real64) :: cc2inv, t1,t2,t3,p1,p2,p3,p4,p5,u_i(0:2),u_j(0:2)
!$omp target teams distribute parallel do thread_limit(256) private(j,n,q,l0,l1,l2,p,rimg,rsq,sqrtpi,d,di,img,dimg,r,r2,ri,ri2,sig,eps,r6,s6,af,holder,chargeprod,invv,kvec,k_sq,fourpi,common_factor,rinv,r2inv,r3,r3inv,r5inv,r7inv,x2,y2,z2,x,y,z,udotu,ujdotr,uidotr,cc2inv,t1,t2,t3,p1,p2,p3,p4,p5,u_i,u_j)
    do i=0,n_atoms-1
      sqrtpi = sqrt(PI)
      af = 0.0_real64
      if (pformd == 0 .or. pformd == 1 .or. pformd == 2) then
        do j=i+1,n_atoms-1
          if (atom_list(i)%molid == atom_list(j)%molid) cycle
          if (atom_list(i)%frozen /= 0 .and. atom_list(j)%frozen /= 0) cycle
          sig = atom_list(i)%sig; if (sig /= atom_list(j)%sig) sig = 0.5_real64*(sig+atom_list(j)%sig)
          eps = atom_list(i)%eps; if (eps /= atom_list(j)%eps) eps = sqrt(eps*atom_list(j)%eps)
          if (sig == 0.0_real64 .or. eps == 0.0_real64) cycle
          call nearest_image(atom_list(i), atom_list(j), basis, reciprocal_basis, d, dimg, rimg, rsq)
          if (rimg <= cutoffd) then
            r6 = rsq*rsq*rsq
            s6 = sig*sig; s6 = s6*s6*s6
            do n=0,2
              holder = 24.0_real64*dimg(n)*eps*(2.0_real64*(s6*s6)/(r6*r6*rsq)-s6/(r6*rsq))
!$omp atomic update
              atom_list(j)%f(n) = atom_list(j)%f(n) - holder
              af(n) = af(n) + holder
            end do
          end if
        end do
        do n=0,2
!$omp atomic update
          atom_list(i)%f(n) = atom_list(i)%f(n) + af(n)
        end do
      end if

      if (pformd == 1 .or. pformd == 2) then
        af = 0.0_real64
        invv = basis(0)*(basis(4)*basis(8)-basis(7)*basis(5)) + basis(3)*(basis(7)*basis(2)-basis(1)*basis(8)) + basis(6)*(basis(1)*basis(5)-basis(5)*basis(2))
        invv = 1.0_real64 / invv
        fourpi = 4.0_real64 * PI
        do j=0,n_atoms-1
          if (atom_list(i)%frozen /= 0 .and. atom_list(j)%frozen /= 0) cycle
          if (atom_list(i)%charge == 0.0_real64 .or. atom_list(j)%charge == 0.0_real64) cycle
          if (i == j) cycle
          call nearest_image(atom_list(i), atom_list(j), basis, reciprocal_basis, d, dimg, rimg, rsq)
          if (rimg <= cutoffd .and. atom_list(i)%molid < atom_list(j)%molid) then
            chargeprod = atom_list(i)%charge * atom_list(j)%charge
            do n=0,2
              holder = -((-2.0_real64*chargeprod*ewald_alpha*exp(-ewald_alpha*ewald_alpha*rsq))/(sqrtpi*rimg) - (chargeprod*erfc(ewald_alpha*rimg)/rsq)) * dimg(n)/rimg
              af(n) = af(n) + holder
!$omp atomic update
              atom_list(j)%f(n) = atom_list(j)%f(n) - holder
            end do
          end if
          if (kspace /= 0 .and. atom_list(i)%molid < atom_list(j)%molid) then
            chargeprod = atom_list(i)%charge * atom_list(j)%charge
            do n=0,2; do l0=0,kmax; do l1=merge(0,-kmax,l0==0),kmax; do l2=merge(1,-kmax,l0==0 .and. l1==0),kmax
              if (l0*l0+l1*l1+l2*l2 > kmax*kmax) cycle
              do p=0,2
                kvec(p)=2.0_real64*PI*(reciprocal_basis(0*3+p)*l0+reciprocal_basis(1*3+p)*l1+reciprocal_basis(2*3+p)*l2)
              end do
              k_sq = kvec(0)*kvec(0)+kvec(1)*kvec(1)+kvec(2)*kvec(2)
              holder = chargeprod*invv*fourpi*kvec(n)*exp(-k_sq/(4.0_real64*ewald_alpha*ewald_alpha))*sin(kvec(0)*dimg(0)+kvec(1)*dimg(1)+kvec(2)*dimg(2))/k_sq*2.0_real64
              af(n)=af(n)+holder
!$omp atomic update
              atom_list(j)%f(n)=atom_list(j)%f(n)-holder
            end do; end do; end do; end do
          end if
        end do
        do n=0,2
!$omp atomic update
          atom_list(i)%f(n)=atom_list(i)%f(n)+af(n)
        end do
      end if

      if (pformd == 2) then
        cc2inv = 1.0_real64/(cutoffd*cutoffd)
        u_i = atom_list(i)%u
        do j=i+1,n_atoms-1
          af = 0.0_real64
          if (atom_list(i)%molid == atom_list(j)%molid) cycle
          call nearest_image(atom_list(i), atom_list(j), basis, reciprocal_basis, d, dimg, rimg, rsq)
          if (rimg > cutoffd) cycle
          r=rimg; x=dimg(0); y=dimg(1); z=dimg(2); x2=x*x; y2=y*y; z2=z*z; r2=r*r
          rinv=1.0_real64/r; r2inv=rinv*rinv; r3=r2*r; r3inv=r2inv*rinv; u_j=atom_list(j)%u
          if (atom_list(j)%charge /= 0.0_real64 .and. atom_list(i)%polar /= 0.0_real64) then
            common_factor=atom_list(j)%charge*r3inv
            af(0)=af(0)+common_factor*(u_i(0)*(r2inv*(-2*x2+y2+z2)-cc2inv*(y2+z2))+u_i(1)*(r2inv*(-3*x*y)+cc2inv*x*y)+u_i(2)*(r2inv*(-3*x*z)+cc2inv*x*z))
            af(1)=af(1)+common_factor*(u_i(0)*(r2inv*(-3*x*y)+cc2inv*x*y)+u_i(1)*(r2inv*(-2*y2+x2+z2)-cc2inv*(x2+z2))+u_i(2)*(r2inv*(-3*y*z)+cc2inv*y*z))
            af(2)=af(2)+common_factor*(u_i(0)*(r2inv*(-3*x*z)+cc2inv*x*z)+u_i(1)*(r2inv*(-3*y*z)+cc2inv*y*z)+u_i(2)*(r2inv*(-2*z2+x2+y2)-cc2inv*(x2+y2)))
          end if
          if (atom_list(i)%charge /= 0.0_real64 .and. atom_list(j)%polar /= 0.0_real64) then
            common_factor=atom_list(i)%charge*r3inv
            af(0)=af(0)-common_factor*(u_j(0)*(r2inv*(-2*x2+y2+z2)-cc2inv*(y2+z2))+u_j(1)*(r2inv*(-3*x*y)+cc2inv*x*y)+u_j(2)*(r2inv*(-3*x*z)+cc2inv*x*z))
            af(1)=af(1)-common_factor*(u_j(0)*(r2inv*(-3*x*y)+cc2inv*x*y)+u_j(1)*(r2inv*(-2*y2+x2+z2)-cc2inv*(x2+z2))+u_j(2)*(r2inv*(-3*y*z)+cc2inv*y*z))
            af(2)=af(2)-common_factor*(u_j(0)*(r2inv*(-3*x*z)+cc2inv*x*z)+u_j(1)*(r2inv*(-3*y*z)+cc2inv*y*z)+u_j(2)*(r2inv*(-2*z2+x2+y2)-cc2inv*(x2+y2)))
          end if
          if (atom_list(i)%polar /= 0.0_real64 .and. atom_list(j)%polar /= 0.0_real64) then
            r5inv=r2inv*r3inv; r7inv=r5inv*r2inv; udotu=sum(u_i*u_j); uidotr=sum(u_i*dimg); ujdotr=sum(u_j*dimg)
            t1=exp(-polar_damp*r); t2=1.0_real64+polar_damp*r+0.5_real64*polar_damp*polar_damp*r2; t3=t2+polar_damp**3*r3/6.0_real64
            p1=3*r5inv*udotu*(1-t1*t2)-r7inv*15*uidotr*ujdotr*(1-t1*t3)
            p2=3*r5inv*ujdotr*(1-t1*t3); p3=3*r5inv*uidotr*(1-t1*t3)
            p4=-udotu*r3inv*(-t1*(polar_damp*rinv+polar_damp*polar_damp)+rinv*t1*polar_damp*t2)
            p5=3*r5inv*uidotr*ujdotr*(-t1*(rinv*polar_damp+polar_damp*polar_damp+0.5_real64*r*polar_damp**3)+rinv*t1*polar_damp*t3)
            af = af + p1*dimg + p2*u_i + p3*u_j + p4*dimg + p5*dimg
          end if
          do n=0,2
!$omp atomic update
            atom_list(i)%f(n)=atom_list(i)%f(n)+af(n)
!$omp atomic update
            atom_list(j)%f(n)=atom_list(j)%f(n)-af(n)
          end do
        end do
      end if
    end do
!$omp end target teams distribute parallel do
  end subroutine calculate_force_kernel

  subroutine nearest_image(ai, aj, basis, reciprocal_basis, d, dimg, rimg, rsq)
    type(d_atom),intent(in)::ai,aj
    real(real64),intent(in)::basis(0:8),reciprocal_basis(0:8)
    real(real64),intent(out)::d(0:2),dimg(0:2),rimg,rsq
    real(real64)::img(0:2),di(0:2),r2,ri2,r,ri
    integer::n,q
    d=ai%pos-aj%pos
    do n=0,2
      img(n)=0.0_real64
      do q=0,2
        img(n)=img(n)+reciprocal_basis(n*3+q)*d(q)
      end do
      img(n)=anint(img(n))
    end do
    do n=0,2
      di(n)=0.0_real64
      do q=0,2
        di(n)=di(n)+basis(n*3+q)*img(q)
      end do
      di(n)=d(n)-di(n)
    end do
    r2=sum(d*d); ri2=sum(di*di); r=sqrt(r2); ri=sqrt(ri2)
    if (ri /= ri) then
      rimg=r; rsq=r2; dimg=d
    else
      rimg=ri; rsq=ri2; dimg=di
    end if
  end subroutine nearest_image

  subroutine force_kernel(total_atoms, block_size, pform, cutoff, ewald_alpha, ewald_kmax, kspace_option, polar_damp, h_basis, h_rbasis, h_atom_list)
    integer,intent(in)::total_atoms,block_size,pform,ewald_kmax,kspace_option
    real(real64),intent(in)::cutoff,ewald_alpha,polar_damp,h_basis(0:8),h_rbasis(0:8)
    type(d_atom),intent(inout)::h_atom_list(0:)
!$omp target data map(to:h_basis(0:8),h_rbasis(0:8)) map(tofrom:h_atom_list(0:total_atoms-1))
    call calculate_force_kernel(h_atom_list,total_atoms,cutoff,h_basis,h_rbasis,pform,ewald_alpha,ewald_kmax,kspace_option,polar_damp)
!$omp end target data
  end subroutine force_kernel
end module force_kernel_mod
