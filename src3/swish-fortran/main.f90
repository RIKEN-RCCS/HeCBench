module swish_kernels
  use iso_fortran_env,only:real32
  implicit none
contains
  subroutine swish_kernel(n,x,y)
    integer,intent(in)::n
    real(real32),intent(in)::x(0:n-1)
    real(real32),intent(out)::y(0:n-1)
    integer::i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,n-1;y(i)=x(i)/(1.0_real32+exp(-x(i)));end do
!$omp end target teams distribute parallel do
  end subroutine swish_kernel
  subroutine swish_gradient(n,x,y,dy,dx)
    integer,intent(in)::n
    real(real32),intent(in)::x(0:n-1),y(0:n-1),dy(0:n-1)
    real(real32),intent(out)::dx(0:n-1)
    integer::i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,n-1;dx(i)=dy(i)*(y(i)+(1.0_real32-y(i))/(1.0_real32+exp(-x(i))));end do
!$omp end target teams distribute parallel do
  end subroutine
end module
program swish
  use iso_fortran_env,only:int64,real32,real64
  use swish_kernels
  implicit none
  integer(int64),parameter::mul=16807_int64,modulus=2147483647_int64
  integer::argc,n,repeats,i,k
  integer(int64)::state,t0,t1,rate
  real(real32),allocatable::x(:),y(:),dy(:),dx(:),ry(:),rdx(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <number of elements> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)n;call get_command_argument(2,arg);read(arg,*)repeats
  allocate(x(0:n-1),y(0:n-1),dy(0:n-1),dx(0:n-1),ry(0:n-1),rdx(0:n-1));state=123_int64
  do i=0,n-1;state=modulo(mul*state,modulus);x(i)=-2.0_real32+4.0_real32*real(state,real32)/modulus;state=modulo(mul*state,modulus);dy(i)=-2.0_real32+4.0_real32*real(state,real32)/modulus;end do
!$omp target data map(to:x(0:n-1),dy(0:n-1)) map(from:y(0:n-1),dx(0:n-1))
  call system_clock(t0,rate);do k=1,repeats;call swish_kernel(n,x,y);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average execution time of Swish kernel: ',real(t1-t0,real64)*1.0e6_real64/real(rate,real64)/repeats,' (us)'
  call system_clock(t0,rate);do k=1,repeats;call swish_gradient(n,x,y,dy,dx);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average execution time of SwishGradient kernel: ',real(t1-t0,real64)*1.0e6_real64/real(rate,real64)/repeats,' (us)'
!$omp end target data
  do i=0,n-1;ry(i)=x(i)/(1.0_real32+exp(-x(i)));rdx(i)=dy(i)*(ry(i)+(1.0_real32-ry(i))/(1.0_real32+exp(-x(i))));end do
  if(all(abs(y-ry)<=1.0e-3_real32).and.all(abs(dx-rdx)<=1.0e-3_real32))then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
