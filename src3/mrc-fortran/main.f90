module mrc_kernels
  use iso_fortran_env, only: real32
  implicit none
contains
  subroutine gradient(n,y,x1,x2,o,margin,dx1,dx2)
    integer,intent(in)::n
    integer,intent(in)::y(0:n-1)
    real(real32),intent(in)::x1(0:n-1),x2(0:n-1),o(0:n-1),margin
    real(real32),intent(out)::dx1(0:n-1),dx2(0:n-1)
    integer::i
    real(real32)::dist
!$omp target teams distribute parallel do thread_limit(256) private(dist)
    do i=0,n-1
      dist=-real(y(i),real32)*(x1(i)-x2(i))+margin
      if(dist<0.0_real32)then;dx1(i)=0.0_real32;dx2(i)=0.0_real32
      else;dx1(i)=-real(y(i),real32)*o(i);dx2(i)=real(y(i),real32)*o(i);end if
    end do
!$omp end target teams distribute parallel do
  end subroutine
  subroutine gradient2(n,y,x1,x2,o,margin,dx1,dx2)
    integer,intent(in)::n
    integer,intent(in)::y(0:n-1)
    real(real32),intent(in)::x1(0:n-1),x2(0:n-1),o(0:n-1),margin
    real(real32),intent(out)::dx1(0:n-1),dx2(0:n-1)
    integer::i
    real(real32)::dist,yy
!$omp target teams distribute parallel do thread_limit(256) private(dist,yy)
    do i=0,n-1
      yy=real(y(i),real32);dist=-yy*(x1(i)-x2(i))+margin
      if(dist<0.0_real32)then;dx1(i)=0.0_real32;dx2(i)=0.0_real32
      else;dx1(i)=-yy*o(i);dx2(i)=yy*o(i);end if
    end do
!$omp end target teams distribute parallel do
  end subroutine
end module
program mrc
  use iso_fortran_env,only:int64,real32,real64
  use mrc_kernels
  implicit none
  integer(int64),parameter::mul=16807_int64,modulus=2147483647_int64
  integer::argc,n,repeats,i,k
  integer(int64)::state,t0,t1,rate
  real(real32),parameter::margin=0.01_real32
  real(real32),allocatable::x1(:),x2(:),o(:),dx1(:),dx2(:),ref1(:),ref2(:)
  integer,allocatable::y(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <number of elements> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)n;call get_command_argument(2,arg);read(arg,*)repeats
  allocate(x1(0:n-1),x2(0:n-1),o(0:n-1),dx1(0:n-1),dx2(0:n-1),ref1(0:n-1),ref2(0:n-1),y(0:n-1))
  state=123_int64
  do i=0,n-1
    state=modulo(mul*state,modulus);x1(i)=-2.0_real32+4.0_real32*real(state,real32)/modulus
    state=modulo(mul*state,modulus);x2(i)=-2.0_real32+4.0_real32*real(state,real32)/modulus
    state=modulo(mul*state,modulus);o(i)=-2.0_real32+4.0_real32*real(state,real32)/modulus
    state=modulo(mul*state,modulus);y(i)=merge(-1,1,(-2.0_real32+4.0_real32*real(state,real32)/modulus)<0.0_real32)
  end do
!$omp target data map(to:x1(0:n-1),x2(0:n-1),o(0:n-1),y(0:n-1)) map(from:dx1(0:n-1),dx2(0:n-1))
  do k=1,repeats;call gradient(n,y,x1,x2,o,margin,dx1,dx2);call gradient2(n,y,x1,x2,o,margin,dx1,dx2);end do
  call system_clock(t0,rate);do k=1,repeats;call gradient(n,y,x1,x2,o,margin,dx1,dx2);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average execution time of MRC kernel: ',real(t1-t0,real64)*1e6_real64/real(rate,real64)/repeats,' (us)'
  call system_clock(t0,rate);do k=1,repeats;call gradient2(n,y,x1,x2,o,margin,dx1,dx2);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average execution time of MRC2 kernel: ',real(t1-t0,real64)*1e6_real64/real(rate,real64)/repeats,' (us)'
!$omp end target data
  do i=0,n-1
    if(-real(y(i),real32)*(x1(i)-x2(i))+margin<0.0_real32)then;ref1(i)=0.0_real32;ref2(i)=0.0_real32
    else;ref1(i)=-real(y(i),real32)*o(i);ref2(i)=real(y(i),real32)*o(i);end if
  end do
  if(all(abs(dx1-ref1)<=1.0e-3_real32).and.all(abs(dx2-ref2)<=1.0e-3_real32))then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
