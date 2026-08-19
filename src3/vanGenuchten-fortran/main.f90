module vangenuchten_kernel
  use iso_fortran_env,only:real64
  implicit none
  real(real64),parameter::alpha=0.02_real64,theta_s=0.45_real64,theta_r=0.1_real64,exponent_n=1.8_real64
contains
  subroutine vg_gpu(ksat,psi,capacity,theta,k,size)
    integer,intent(in)::size
    real(real64),intent(in)::ksat(0:size-1),psi(0:size-1)
    real(real64),intent(out)::capacity(0:size-1),theta(0:size-1),k(0:size-1)
    integer::i
    real(real64)::se,thetai,psii,lambda,m,t
!$omp target teams distribute parallel do thread_limit(256) private(se,thetai,psii,lambda,m,t)
    do i=0,size-1
      lambda=exponent_n-1.0_real64;m=lambda/exponent_n;psii=psi(i)*100.0_real64
      if(psii<0.0_real64)then;thetai=(theta_s-theta_r)/(1.0_real64+(alpha*(-psii))**exponent_n)**m+theta_r;else;thetai=theta_s;end if
      theta(i)=thetai;se=(thetai-theta_r)/(theta_s-theta_r);t=1.0_real64-(1.0_real64-se**(1.0_real64/m))**m;k(i)=ksat(i)*sqrt(se)*t*t
      if(psii<0.0_real64)then;capacity(i)=100.0_real64*alpha*exponent_n*(1.0_real64/exponent_n-1.0_real64)*(alpha*abs(psii))**(exponent_n-1.0_real64)*(theta_r-theta_s)*((alpha*abs(psii))**exponent_n+1.0_real64)**(1.0_real64/exponent_n-2.0_real64);else;capacity(i)=0.0_real64;end if
    end do
!$omp end target teams distribute parallel do
  end subroutine
  subroutine vg_reference(ksat,psi,capacity,theta,k,size)
    integer,intent(in)::size
    real(real64),intent(in)::ksat(0:size-1),psi(0:size-1)
    real(real64),intent(out)::capacity(0:size-1),theta(0:size-1),k(0:size-1)
    integer::i
    real(real64)::se,thetai,psii,lambda,m,t
    do i=0,size-1
      lambda=exponent_n-1.0_real64;m=lambda/exponent_n;psii=psi(i)*100.0_real64
      if(psii<0.0_real64)then;thetai=(theta_s-theta_r)/(1.0_real64+(alpha*(-psii))**exponent_n)**m+theta_r;else;thetai=theta_s;end if
      theta(i)=thetai;se=(thetai-theta_r)/(theta_s-theta_r);t=1.0_real64-(1.0_real64-se**(1.0_real64/m))**m;k(i)=ksat(i)*sqrt(se)*t*t
      if(psii<0.0_real64)then;capacity(i)=100.0_real64*alpha*exponent_n*(1.0_real64/exponent_n-1.0_real64)*(alpha*abs(psii))**(exponent_n-1.0_real64)*(theta_r-theta_s)*((alpha*abs(psii))**exponent_n+1.0_real64)**(1.0_real64/exponent_n-2.0_real64);else;capacity(i)=0.0_real64;end if
    end do
  end subroutine
end module
program vangenuchten
  use iso_fortran_env,only:int64,real64
  use vangenuchten_kernel
  implicit none
  integer::argc,dx,dy,dz,repeats,size,i,j
  integer(int64)::t0,t1,rate
  real(real64),allocatable::ksat(:),psi(:),capacity(:),theta(:),k(:),rc(:),rt(:),rk(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=4)then;print '(a)','Usage: ./main <dimX> <dimY> <dimZ> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)dx;call get_command_argument(2,arg);read(arg,*)dy;call get_command_argument(3,arg);read(arg,*)dz;call get_command_argument(4,arg);read(arg,*)repeats
  size=dx*dy*dz;allocate(ksat(0:size-1),psi(0:size-1),capacity(0:size-1),theta(0:size-1),k(0:size-1),rc(0:size-1),rt(0:size-1),rk(0:size-1))
  do i=0,size-1;ksat(i)=1.0e-6_real64+(1.0_real64-1.0e-6_real64)*i/real(size,real64);psi(i)=-100.0_real64+101.0_real64*i/real(size,real64);end do
  call vg_reference(ksat,psi,rc,rt,rk,size)
!$omp target data map(to:ksat(0:size-1),psi(0:size-1)) map(from:capacity(0:size-1),theta(0:size-1),k(0:size-1))
  call system_clock(t0,rate);do j=1,repeats;call vg_gpu(ksat,psi,capacity,theta,k,size);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average kernel execution time: ',real(t1-t0,real64)/real(rate,real64)/repeats,' (s)'
!$omp end target data
  if(all(abs(capacity-rc)<=1.0e-3_real64).and.all(abs(theta-rt)<=1.0e-3_real64).and.all(abs(k-rk)<=1.0e-3_real64))then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
