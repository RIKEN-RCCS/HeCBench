program affine
  use iso_fortran_env, only: int16, int32, real32, real64
  use omp_lib
  implicit none
  integer,parameter::x_size=512,y_size=512
  integer::x,y,iterations,unit,iostat,iteration,ixp,n
  character(len=256)::input_name,output_name,arg
  integer(int16),allocatable::input_image(:),output_image(:)
  real(real32)::lx_rot,ly_rot,lx_expan,ly_expan,det,xnew,ynew,xfrac,yfrac,gray
  real(real32)::a00,a01,a10,a11,ia00,ia01,ia10,ia11,pi
  real(real64)::start_time,elapsed
  if(command_argument_count()/=3)then;print'(a)','Usage: ./main <input image> <output image> <iterations>';stop 1;end if
  call get_command_argument(1,input_name);call get_command_argument(2,output_name);call get_command_argument(3,arg);read(arg,*)iterations
  allocate(input_image(0:x_size*y_size-1),output_image(0:x_size*y_size-1))
  print'(a)','Reading input image...';open(newunit=unit,file=trim(input_name),access='stream',form='unformatted',status='old',iostat=iostat)
  if(iostat/=0)then;print'(a,a)','Error: Unable to open input image ',trim(input_name);stop 1;end if
  read(unit)input_image;close(unit);print'(a,i0)','   Bytes read = ',size(input_image)*storage_size(input_image)/8
  pi=acos(-1._real32);lx_rot=30._real32;ly_rot=0._real32;lx_expan=.5_real32;ly_expan=.5_real32
  a00=lx_expan*cos(lx_rot*pi/180._real32);a01=ly_expan*sin(ly_rot*pi/180._real32);a10=lx_expan*sin(lx_rot*pi/180._real32);a11=ly_expan*cos(ly_rot*pi/180._real32)
  det=a00*a11-a01*a10;if(det==0._real32)then;ia00=1.;ia01=0.;ia10=0.;ia11=1.;else;ia00=a11/det;ia01=-a01/det;ia10=-a10/det;ia11=a00/det;end if
  start_time=omp_get_wtime()
  do iteration=1,iterations
!$omp target teams distribute parallel do collapse(2) map(to:input_image) map(from:output_image) private(xnew,ynew,xfrac,yfrac,gray,n,ixp)
    do y=0,y_size-1;do x=0,x_size-1
      xnew=ia00*(x-x_size/2._real32)+ia01*(y-y_size/2._real32)+x_size/2._real32
      ynew=ia10*(x-x_size/2._real32)+ia11*(y-y_size/2._real32)+y_size/2._real32;n=int(floor(ynew));ixp=int(floor(xnew));xfrac=xnew-ixp;yfrac=ynew-n
      if(ixp>=0 .and. ixp+1<x_size .and. n>=0 .and. n+1<y_size)then
        gray=(1-yfrac)*((1-xfrac)*real(input_image(n*x_size+ixp),real32)+xfrac*real(input_image(n*x_size+ixp+1),real32))+yfrac*((1-xfrac)*real(input_image((n+1)*x_size+ixp),real32)+xfrac*real(input_image((n+1)*x_size+ixp+1),real32));output_image(y*x_size+x)=transfer(iand(int(gray,int32),65535_int32),output_image(y*x_size+x))
      else if((ixp+1==x_size .and. n>=0 .and. n<y_size) .or. (n+1==y_size .and. ixp>=0 .and. ixp<x_size))then;output_image(y*x_size+x)=input_image(n*x_size+ixp)
      else;output_image(y*x_size+x)=1_int16;end if
    end do;end do
!$omp end target teams distribute parallel do
  end do
  elapsed=omp_get_wtime()-start_time;print'(a,f12.6,a)','   Average kernel execution time ',elapsed/iterations,' (s)'
  open(newunit=unit,file=trim(output_name),access='stream',form='unformatted',status='replace');write(unit)output_image;close(unit);print'(a,i0)','   Bytes written = ',size(output_image)*storage_size(output_image)/8
end program affine
