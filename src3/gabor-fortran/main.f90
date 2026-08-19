program gabor
  use iso_fortran_env, only:int64,real64
  implicit none
  integer::height,width,repeat
  character(64)::arg
  if(command_argument_count()/=3)then;write(*,'(a)')'Usage: ./main <height> <width> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)height
  call get_command_argument(2,arg);read(arg,*)width
  call get_command_argument(3,arg);read(arg,*)repeat
  if(height<=0.or.width<=0.or.repeat<=0)error stop 'arguments must be positive'
  call run_gabor(height,width,repeat)
contains
  subroutine run_gabor(h,wid,rep)
    integer,intent(in)::h,wid,rep
    integer(int64)::n,i,startc,endc,rate
    integer::x,y,it
    real(real64)::sx,sy,sx2,sy2,fx,ctheta,stheta,cy,cx,scale,u,v,time_us
    real(real64),allocatable::host_filter(:),device_filter(:)
    n=int(h,int64)*wid;allocate(host_filter(0:n-1),device_filter(0:n-1))
    sx=13d0/(2d0*sqrt(2d0*log(2d0)));sy=2.65d0*sx;sx2=sx*sx;sy2=sy*sy;fx=1d0/13d0
    ctheta=cos(45d0);stheta=sin(45d0);cy=real(h,real64)/2d0;cx=real(wid,real64)/2d0;scale=1d0/(2d0*acos(-1d0)*sx*sy)
!$omp target data map(from:device_filter(0:n-1))
    call system_clock(startc,rate)
    do it=1,rep
!$omp target teams distribute parallel do collapse(2) thread_limit(256) private(u,v)
      do y=0,h-1
        do x=0,wid-1
          u=ctheta*(real(x,real64)-cx)-stheta*(real(y,real64)-cy)
          v=ctheta*(real(y,real64)-cy)+stheta*(real(x,real64)-cx)
          device_filter(y*wid+x)=scale*exp(-.5d0*(u*u/sx2+v*v/sy2))*cos(2d0*acos(-1d0)*fx*u)
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(endc);time_us=real(endc-startc,real64)*1d6/real(rate,real64)/real(rep,real64)
    write(*,'(a,f0.6,a)')'Average kernel execution time: ',time_us,' (us)'
!$omp end target data
    do y=0,h-1;do x=0,wid-1
      u=ctheta*(real(x,real64)-cx)-stheta*(real(y,real64)-cy);v=ctheta*(real(y,real64)-cy)+stheta*(real(x,real64)-cx)
      host_filter(y*wid+x)=scale*exp(-.5d0*(u*u/sx2+v*v/sy2))*cos(2d0*acos(-1d0)*fx*u)
    end do;end do
    if(all(abs(host_filter-device_filter)<=1d-3))then;write(*,'(a)')'PASS';else;write(*,'(a)')'FAIL';end if
    deallocate(host_filter,device_filter)
  end subroutine
end program gabor
