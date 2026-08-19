module fluid_constants
  use iso_fortran_env, only: int32, real64
  implicit none
  integer, parameter :: nx=256, ny=256, ncell=nx*ny
  integer, parameter :: dx(0:8)=[0,1,0,-1,0,1,-1,-1,1]
  integer, parameter :: dy(0:8)=[0,0,1,0,-1,1,1,-1,-1]
  real(real64), parameter :: omega=1.2_real64
!$omp declare target (dx, dy, omega)
end module fluid_constants

program fluid_sim
  use iso_fortran_env, only:int32,int64,real64
  use iso_c_binding, only:c_int
  use fluid_constants
  implicit none
  integer::iterations,i,x,y,pos,k,iter
  integer(int64)::startc,endc,rate
  real(real64)::w(0:8),u0(0:1),elapsed,maxerr
  integer(int32),allocatable::ctype(:)
  real(real64),allocatable::a0(:),a14(:),a58(:),b0(:),b14(:),b58(:),r0(:),r14(:),r58(:),q0(:),q14(:),q58(:)
  character(64)::arg
  interface
    subroutine c_srand(seed) bind(C,name='srand');import c_int;integer(c_int),value::seed;end subroutine
    function c_rand() bind(C,name='rand') result(v);import c_int;integer(c_int)::v;end function
  end interface
!$omp declare target (feq, cell)
  if(command_argument_count()/=1)then;write(*,'(a)')'Usage ./main <iterations>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)iterations;if(iterations<=0)error stop 'iterations must be positive'
  w=[4d0/9d0,1d0/9d0,1d0/9d0,1d0/9d0,1d0/9d0,1d0/36d0,1d0/36d0,1d0/36d0,1d0/36d0];u0=[.01d0,.01d0]
  allocate(ctype(0:ncell-1),a0(0:ncell-1),b0(0:ncell-1),r0(0:ncell-1),q0(0:ncell-1),a14(0:4*ncell-1),a58(0:4*ncell-1),b14(0:4*ncell-1),b58(0:4*ncell-1),r14(0:4*ncell-1),r58(0:4*ncell-1),q14(0:4*ncell-1),q58(0:4*ncell-1))
  call c_srand(123_c_int)
  do y=0,ny-1;do x=0,nx-1
    pos=x+y*nx;call initialize_cell(a0,a14,a58,pos,real(modulo(c_rand(),10_c_int)+1,real64),w,u0)
    ctype(pos)=merge(1_int32,0_int32,x==0 .or. x==nx-1 .or. y==0 .or. y==ny-1)
  end do;end do
  b0=a0;b14=a14;b58=a58;r0=a0;r14=a14;r58=a58;q0=a0;q14=a14;q58=a58
  do iter=1,iterations
    if(mod(iter,2)==1)then;call lbm_host(r0,r14,r58,q0,q14,q58,ctype,w);else;call lbm_host(q0,q14,q58,r0,r14,r58,ctype,w);end if
  end do
!$omp target data map(to:a0,a14,a58,b0,b14,b58,ctype,w)
  call system_clock(startc,rate)
  do iter=1,iterations
    if(mod(iter,2)==1)then;call lbm_device(a0,a14,a58,b0,b14,b58,ctype,w);else;call lbm_device(b0,b14,b58,a0,a14,a58,ctype,w);end if
  end do
  call system_clock(endc);elapsed=real(endc-startc,real64)/real(rate,real64)/real(iterations,real64);write(*,'(a,f0.6,a)')'Average kernel execution time ',elapsed,' (s)'
!$omp target update from(a0,a14,a58,b0,b14,b58)
!$omp end target data
  if(mod(iterations,2)==1)then
    maxerr=max(maxval(abs(b0-q0)),max(maxval(abs(b14-q14)),maxval(abs(b58-q58))))
  else
    maxerr=max(maxval(abs(a0-r0)),max(maxval(abs(a14-r14)),maxval(abs(a58-r58))))
  end if
  if(maxerr<=1d-3)then;write(*,'(a)')'PASS';else;write(*,'(a,f0.6)')'FAIL max error ',maxerr;end if
contains
  subroutine initialize_cell(f0,f14,f58,p,den,weight,vel)
    real(real64),intent(inout)::f0(0:),f14(0:),f58(0:);integer,intent(in)::p;real(real64),intent(in)::den,weight(0:),vel(0:)
    integer::j;f0(p)=feq(den,weight(0),0d0,0d0,vel(0),vel(1));do j=1,4;f14(4*p+j-1)=feq(den,weight(j),real(dx(j),real64),real(dy(j),real64),vel(0),vel(1));f58(4*p+j-1)=feq(den,weight(j+4),real(dx(j+4),real64),real(dy(j+4),real64),vel(0),vel(1));end do
  end subroutine
  real(real64) function feq(rho,weight,dx,dy,ux,uy)
    real(real64),intent(in)::rho,weight,dx,dy,ux,uy;real(real64)::u2,eu
    u2=ux**2+uy**2;eu=dx*ux+dy*uy;feq=rho*weight*(1d0+3d0*eu+4.5d0*eu*eu-1.5d0*u2)
  end function
  subroutine lbm_host(i0,i14,i58,o0,o14,o58,t,weight)
    real(real64),intent(in)::i0(0:ncell-1),i14(0:4*ncell-1),i58(0:4*ncell-1),weight(0:8);real(real64),intent(inout)::o0(0:ncell-1),o14(0:4*ncell-1),o58(0:4*ncell-1);integer(int32),intent(in)::t(0:ncell-1)
    integer::xx,yy,p;do yy=0,ny-1;do xx=0,nx-1;p=xx+yy*nx;call cell(xx,yy,p,i0,i14,i58,o0,o14,o58,t,weight);end do;end do
  end subroutine
  subroutine lbm_device(i0,i14,i58,o0,o14,o58,t,weight)
    real(real64),intent(in)::i0(0:ncell-1),i14(0:4*ncell-1),i58(0:4*ncell-1),weight(0:8);real(real64),intent(inout)::o0(0:ncell-1),o14(0:4*ncell-1),o58(0:4*ncell-1);integer(int32),intent(in)::t(0:ncell-1);integer::xx,yy,p
!$omp target teams distribute parallel do collapse(2) thread_limit(256)
    do yy=0,ny-1;do xx=0,nx-1;p=xx+yy*nx;call cell(xx,yy,p,i0,i14,i58,o0,o14,o58,t,weight);end do;end do
!$omp end target teams distribute parallel do
  end subroutine
  subroutine cell(xx,yy,p,i0,i14,i58,o0,o14,o58,t,weight)
    integer,intent(in)::xx,yy,p;real(real64),intent(in)::i0(0:ncell-1),i14(0:4*ncell-1),i58(0:4*ncell-1),weight(0:8);real(real64),intent(inout)::o0(0:ncell-1),o14(0:4*ncell-1),o58(0:4*ncell-1);integer(int32),intent(in)::t(0:ncell-1)
    real(real64)::f(0:8),g(0:8),rho,ux,uy;integer::j,np
    f(0)=i0(p);do j=1,4;f(j)=i14(4*p+j-1);f(j+4)=i58(4*p+j-1);end do
    if(t(p)/=0)then;g(1:4)=[f(3),f(4),f(1),f(2)];g(5:8)=[f(7),f(8),f(5),f(6)]
    else
      rho=sum(f);ux=(f(1)-f(3)+f(5)-f(6)-f(7)+f(8))/rho;uy=(f(2)-f(4)+f(5)+f(6)-f(7)-f(8))/rho
      g(0)=(1d0-omega)*f(0)+omega*feq(rho,weight(0),0d0,0d0,ux,uy)
      do j=1,8;g(j)=(1d0-omega)*f(j)+omega*feq(rho,weight(j),real(dx(j),real64),real(dy(j),real64),ux,uy);end do
    end if
    if(xx>0.and.xx<nx-1.and.yy>0.and.yy<ny-1)then
      o0(p)=g(0);do j=1,8;np=(xx+dx(j))+(yy+dy(j))*nx;if(j<=4)then;o14(4*np+j-1)=g(j);else;o58(4*np+j-5)=g(j);end if;end do
    end if
  end subroutine
end program fluid_sim
