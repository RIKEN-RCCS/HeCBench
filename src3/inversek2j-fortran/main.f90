module inverse_kinematics
  use iso_fortran_env,only:real32
  implicit none
contains
  subroutine solve_cpu(x_target,y_target,angles,size)
    integer,intent(in)::size
    real(real32),intent(in)::x_target(0:size-1),y_target(0:size-1)
    real(real32),intent(out)::angles(0:3*size-1)
    integer::idx,i,loop,iter
    real(real32)::aout(0:2),xdata(0:3),ydata(0:3),angle,pex,pey,pcx,pcy,dx,dy,tx,ty,la,lb,ax,ay,bx,by,dot,direction
    do idx=0,size-1
      aout=0.0_real32;do i=0,3;xdata(i)=real(i,real32);ydata(i)=0.0_real32;end do
      do loop=1,25
        do iter=3,1,-1
          pex=xdata(3);pey=ydata(3);pcx=xdata(iter-1);pcy=ydata(iter-1);dx=pex-pcx;dy=pey-pcy;tx=x_target(idx)-pcx;ty=y_target(idx)-pcy
          la=sqrt(dx*dx+dy*dy);lb=sqrt(tx*tx+ty*ty);ax=dx/la;ay=dy/la;bx=tx/lb;by=ty/lb;dot=max(-1.0_real32,min(1.0_real32,ax*bx+ay*by))
          angle=acos(dot)*(180.0_real32/3.14159265358979_real32);direction=ax*by-ay*bx;if(direction<0.0_real32)angle=-angle;angle=max(-30.0_real32,min(30.0_real32,angle));aout(iter-1)=angle
          do i=0,1;aout(i+1)=aout(i+1)+aout(i);end do
        end do
      end do
      angles(3*idx)=aout(0);angles(3*idx+1)=aout(1);angles(3*idx+2)=aout(2)
    end do
  end subroutine
  subroutine solve_gpu(x_target,y_target,angles,size)
    integer,intent(in)::size
    real(real32),intent(in)::x_target(0:size-1),y_target(0:size-1)
    real(real32),intent(out)::angles(0:3*size-1)
    integer::idx,i,loop,iter
    real(real32)::aout(0:2),xdata(0:3),ydata(0:3),angle,pex,pey,pcx,pcy,dx,dy,tx,ty,la,lb,ax,ay,bx,by,dot,direction
!$omp target teams distribute parallel do simd thread_limit(128) private(aout,xdata,ydata,i,loop,iter,angle,pex,pey,pcx,pcy,dx,dy,tx,ty,la,lb,ax,ay,bx,by,dot,direction)
    do idx=0,size-1
      aout=0.0_real32;do i=0,3;xdata(i)=real(i,real32);ydata(i)=0.0_real32;end do
      do loop=1,25
        do iter=3,1,-1
          pex=xdata(3);pey=ydata(3);pcx=xdata(iter-1);pcy=ydata(iter-1);dx=pex-pcx;dy=pey-pcy;tx=x_target(idx)-pcx;ty=y_target(idx)-pcy
          la=sqrt(dx*dx+dy*dy);lb=sqrt(tx*tx+ty*ty);ax=dx/la;ay=dy/la;bx=tx/lb;by=ty/lb;dot=max(-1.0_real32,min(1.0_real32,ax*bx+ay*by))
          angle=acos(dot)*(180.0_real32/3.14159265358979_real32);direction=ax*by-ay*bx;if(direction<0.0_real32)angle=-angle;angle=max(-30.0_real32,min(30.0_real32,angle));aout(iter-1)=angle
          do i=0,1;aout(i+1)=aout(i+1)+aout(i);end do
        end do
      end do
      angles(3*idx)=aout(0);angles(3*idx+1)=aout(1);angles(3*idx+2)=aout(2)
    end do
!$omp end target teams distribute parallel do simd
  end subroutine
end module
program inversek2j
  use iso_fortran_env,only:int64,real32,real64
  use inverse_kinematics
  implicit none
  integer::argc,unit,ios,size,i,repeats,n
  integer(int64)::t0,t1,rate
  real(real32),allocatable::x(:),y(:),angles(:),cpu(:)
  character(len=1024)::filename;character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <input file coefficients> <iterations>';stop 1;end if
  call get_command_argument(1,filename);call get_command_argument(2,arg);read(arg,*)repeats
  open(newunit=unit,file=trim(filename),status='old',action='read',iostat=ios);if(ios/=0)stop 1;read(unit,*,iostat=ios)size;if(ios/=0)stop 1;print '(a,i0)','# Data Size = ',size
  allocate(x(0:size-1),y(0:size-1),angles(0:3*size-1),cpu(0:3*size-1))
  do i=0,size-1;read(unit,*,iostat=ios)x(i),y(i);if(ios/=0)stop 1;end do;close(unit);print '(a)','# Coordinates are read from file...'
!$omp target data map(to:x(0:size-1),y(0:size-1)) map(from:angles(0:3*size-1))
  call system_clock(t0,rate);do n=1,repeats;call solve_gpu(x,y,angles,size);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average kernel execution time ',real(t1-t0,real64)*1.0e6_real64/real(rate,real64)/repeats,' (us)'
!$omp end target data
  call solve_cpu(x,y,cpu,size)
  if(all(abs(angles-cpu)<=1.0e-3_real32))then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
