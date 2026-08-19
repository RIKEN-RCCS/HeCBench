module colorwheel
  use iso_fortran_env, only: real32,real64,int8,int32
  implicit none
  integer,parameter::ry=15,yg=6,gc=4,cb=11,bm=13,mr=6,maxcols=ry+yg+gc+cb+bm+mr
  real(real32),parameter::pi=3.1415927410125732421875_real32
  !$omp declare target (compute_color)
contains
  subroutine compute_color(fx,fy,pix,offset)
    real(real32),intent(in)::fx,fy
    integer(int8),intent(inout)::pix(0:)
    integer,intent(in)::offset
    integer(int32)::cw(0:maxcols-1,0:2),i,k,b,value
    integer::k0,k1
    real(real32)::rad,a,f,col0,col1,col
    k=0
    do i=0,ry-1;cw(k,0)=255;cw(k,1)=255*i/ry;cw(k,2)=0;k=k+1;end do
    do i=0,yg-1;cw(k,0)=255-255*i/yg;cw(k,1)=255;cw(k,2)=0;k=k+1;end do
    do i=0,gc-1;cw(k,0)=0;cw(k,1)=255;cw(k,2)=255*i/gc;k=k+1;end do
    do i=0,cb-1;cw(k,0)=0;cw(k,1)=255-255*i/cb;cw(k,2)=255;k=k+1;end do
    do i=0,bm-1;cw(k,0)=255*i/bm;cw(k,1)=0;cw(k,2)=255;k=k+1;end do
    do i=0,mr-1;cw(k,0)=255;cw(k,1)=0;cw(k,2)=255-255*i/mr;k=k+1;end do
    rad=sqrt(fx*fx+fy*fy);a=atan2(-fy,-fx)/pi;f=(a+1.0_real32)/2.0_real32*real(maxcols-1,real32);k0=int(f);k1=mod(k0+1,maxcols);f=f-real(k0,real32)
    do b=0,2
      col0=real(cw(k0,b),real32)/255.0_real32;col1=real(cw(k1,b),real32)/255.0_real32;col=(1.0_real32-f)*col0+f*col1
      if(rad<=1.0_real32)then;col=1.0_real32-rad*(1.0_real32-col);else;col=col*.75_real32;end if
      value=int(255.0_real32*col,int32);pix(offset+2-b)=transfer(value,pix(offset+2-b))
    end do
  end subroutine compute_color
end module colorwheel
program main
  use iso_fortran_env,only:real32,real64,int8,int32
  use omp_lib,only:omp_get_wtime
  use colorwheel
  implicit none
  integer::argc,size,rep,x,y,half,ios,idx,i,maxerr,value
  real(real32)::truerange,range,fx,fy
  real(real64)::a,z
  integer(int8),allocatable::pix(:),d_pix(:),res(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=3)then;call get_command_argument(0,arg);print '(a,a,a)','Usage: ',trim(arg),' <range> <size> <repeat>';error stop 1;end if
  call get_command_argument(1,arg);read(arg,*,iostat=ios)truerange;if(ios/=0)error stop 1
  call get_command_argument(2,arg);read(arg,*,iostat=ios)size;if(ios/=0)error stop 1
  call get_command_argument(3,arg);read(arg,*,iostat=ios)rep;if(ios/=0)error stop 1
  range=1.04_real32*truerange;half=size/2;allocate(pix(0:size*size*3-1),d_pix(0:size*size*3-1),res(0:size*size*3-1));pix=0_int8
  do y=0,size-1;do x=0,size-1
    fx=real(x,real32)/real(half,real32)*range-range;fy=real(y,real32)/real(half,real32)*range-range
    if(x/=half .and. y/=half)then;idx=(y*size+x)*3;call compute_color(fx/truerange,fy/truerange,pix,idx);end if
  end do;end do
  print '(a)','Start execution on a device';d_pix=0_int8
  !$omp target data map(tofrom:d_pix(0:size*size*3-1))
  a=omp_get_wtime()
  do i=0,rep-1
    !$omp target teams distribute parallel do collapse(2) private(fx,fy,idx)
    do y=0,size-1;do x=0,size-1
      fx=real(x,real32)/real(half,real32)*range-range;fy=real(y,real32)/real(half,real32)*range-range
      if(x/=half .and. y/=half)then;idx=(y*size+x)*3;call compute_color(fx/truerange,fy/truerange,d_pix,idx);end if
    end do;end do
    !$omp end target teams distribute parallel do
  end do
  z=omp_get_wtime();print '(a,f0.6,a)','Average kernel execution time : ',(z-a)*1.0e3_real64/real(rep,real64),' (ms)'
  !$omp end target data
  if(any(pix/=d_pix))then
    maxerr=0;do i=0,size*size*3-1;value=abs(iand(int(d_pix(i),int32),255)-iand(int(pix(i),int32),255));if(value>maxerr)maxerr=value;end do
    print '(a,i0)','Maximum error between host and device results: ',maxerr
  else
    print '(a)','PASS'
  end if
  deallocate(d_pix,pix,res)
end program main
