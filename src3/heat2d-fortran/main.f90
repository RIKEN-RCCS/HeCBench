program heat2d
  use, intrinsic :: iso_fortran_env, only : int32, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib, only : omp_get_wtime
  implicit none
  integer(int32)::lx,ly,niter,i,j,x,y,ios
  real(real32),parameter::sigma=0.01_real32,xdelta=sigma/(1.0_real32+4.0_real32*sigma),xnorm=1.0_real32/(1.0_real32+4.0_real32*sigma)
  real(real32),allocatable::buffer(:),host_a(:),host_b(:),device_a(:),device_b(:)
  real(real64)::start,elapsed
  logical::ok
  character(len=64)::arg
  interface
    subroutine c_srand(seed) bind(C,name='srand');import c_int;integer(c_int),value::seed;end subroutine
    function c_rand() bind(C,name='rand') result(value);import c_int;integer(c_int)::value;end function
  end interface
  if(command_argument_count()/=3)then;write(*,'(a)')' Usage: ./main LX LY NITER';stop 1;end if
  call get_command_argument(1,arg);read(arg,*,iostat=ios)lx;if(ios/=0.or.lx<=0.or.mod(lx,16)/=0)error stop 'LX must be positive multiple of 16'
  call get_command_argument(2,arg);read(arg,*,iostat=ios)ly;if(ios/=0.or.ly<=0.or.mod(ly,16)/=0)error stop 'LY must be positive multiple of 16'
  call get_command_argument(3,arg);read(arg,*,iostat=ios)niter;if(ios/=0.or.niter<1)error stop 'NITER must be positive'
  write(*,'(a,i0,a,i0)')' Ly,Lx = ',ly,',',lx;write(*,'(a,i0)')' niter = ',niter
  allocate(buffer(0:lx*ly-1),host_a(0:lx*ly-1),host_b(0:lx*ly-1),device_a(0:lx*ly-1),device_b(0:lx*ly-1))
  call c_srand(123_c_int)
  do i=0,lx-1,16;x=mod(c_rand(),lx);do j=0,ly-1;buffer(x+j*lx)=1.0_real32;end do;end do
  do i=0,ly-1,16;y=mod(c_rand(),ly);do j=0,lx-1;buffer(j+y*lx)=1.0_real32;end do;end do
  host_a=buffer
  do i=1,niter
    if(mod(i,2)==1)then;call laplace(host_b,host_a,xdelta,xnorm,lx,ly);else;call laplace(host_a,host_b,xdelta,xnorm,lx,ly);end if
  end do
  device_a=buffer
!$omp target data map(tofrom:device_a(0:lx*ly-1)) map(alloc:device_b(0:lx*ly-1))
  start=omp_get_wtime()
  do i=1,niter
    if(mod(i,2)==1)then;call device_laplace(device_b,device_a,xdelta,xnorm,lx,ly);else;call device_laplace(device_a,device_b,xdelta,xnorm,lx,ly);end if
  end do
  elapsed=(omp_get_wtime()-start)*1.0e6_real64/real(niter,real64)
!$omp end target data
  write(*,'(a,i0,a,i0,a,i0,a,f8.1,a,f6.3,a,f6.3,a)')'Device: iters = ',niter,', (Lx,Ly) = ',lx,', ',ly,', t = ',elapsed,' usec/iter, BW = ',real(lx*ly,real64)*8.0_real64/(elapsed*1.0e3_real64), ' GB/s, P = ',real(lx*ly,real64)*6.0_real64/(elapsed*1.0e3_real64),' Gflop/s'
  ok=.true.
  do i=0,lx*ly-1
    if(mod(niter,2)==0)then
      if(abs(host_a(i)-device_a(i))>1.0e-2_real32)then;ok=.false.;write(*,'(a,i0,a,f0.6,a,f0.6)')'Mismatch at ',i,' cpu=',host_a(i),' gpu=',device_a(i);exit;end if
    else
      ! The source maps only the original d_in pointer; odd iteration counts leave the swapped final allocation unmapped.
      if(abs(host_b(i)-device_a(i))>1.0e-2_real32)then;ok=.false.;write(*,'(a,i0,a,f0.6,a,f0.6)')'Mismatch at ',i,' cpu=',host_b(i),' gpu=',device_a(i);exit;end if
    end if
  end do
  if(ok)then;write(*,'(a)')'PASS';else;write(*,'(a)')'FAIL';end if
  deallocate(buffer,host_a,host_b,device_a,device_b)
contains
  subroutine laplace(output,input,delta,norm,lx,ly)
    integer(int32),intent(in)::lx,ly;real(real32),intent(in)::input(0:),delta,norm;real(real32),intent(out)::output(0:);integer(int32)::x,y,v00,v0p,v0m,vp0,vm0
!$omp parallel do collapse(2)
    do y=0,ly-1;do x=0,lx-1;v00=y*lx+x;v0p=y*lx+mod(x+1,lx);v0m=y*lx+mod(lx+x-1,lx);vp0=mod(y+1,ly)*lx+x;vm0=mod(ly+y-1,ly)*lx+x;output(v00)=norm*input(v00)+delta*(input(v0p)+input(v0m)+input(vp0)+input(vm0));end do;end do
!$omp end parallel do
  end subroutine laplace
  subroutine device_laplace(output,input,delta,norm,lx,ly)
    integer(int32),intent(in)::lx,ly;real(real32),intent(in)::input(0:),delta,norm;real(real32),intent(out)::output(0:);integer(int32)::x,y,v00,v0p,v0m,vp0,vm0
!$omp target teams distribute parallel do collapse(2) thread_limit(256)
    do y=0,ly-1;do x=0,lx-1;v00=y*lx+x;v0p=y*lx+mod(x+1,lx);v0m=y*lx+mod(lx+x-1,lx);vp0=mod(y+1,ly)*lx+x;vm0=mod(ly+y-1,ly)*lx+x;output(v00)=norm*input(v00)+delta*(input(v0p)+input(v0m)+input(vp0)+input(vm0));end do;end do
!$omp end target teams distribute parallel do
  end subroutine device_laplace
end program heat2d
