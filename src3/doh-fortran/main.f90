program doh
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use omp_lib, only : omp_get_wtime
  implicit none
!$omp declare target (integ, clip)
  integer(int32) :: h,w,repeat,img_size,i,j,ios
  real(real32), allocatable :: input_img(:), integral_img(:), output_img(:)
  real(real64) :: begin_time, elapsed_us, checksum
  integer(int64) :: seed
  character(len=64) :: argument

  if(command_argument_count()/=3) then
    write(*,'(a)') 'Usage: ./main <height> <width> <repeat>'; stop 1
  end if
  call get_command_argument(1,argument);read(argument,*,iostat=ios)h
  if(ios/=0 .or. h<=0) error stop 'height must be positive'
  call get_command_argument(2,argument);read(argument,*,iostat=ios)w
  if(ios/=0 .or. w<=0) error stop 'width must be positive'
  call get_command_argument(3,argument);read(argument,*,iostat=ios)repeat
  if(ios/=0 .or. repeat<=0) error stop 'repeat must be positive'
  img_size=h*w;allocate(input_img(0:img_size-1),integral_img(0:img_size-1),output_img(0:img_size-1))
  seed=123_int64
  do i=0,img_size-1; input_img(i)=normal_sample(seed); end do
  write(*,'(a)') 'Integrating the input image may take a while...'
  do i=0,h-1
    do j=0,w-1
      integral_img(i*w+j)=sum_integral(input_img,h,w,i,j)
    end do
  end do
!$omp target data map(to:integral_img(0:img_size-1)) map(from:output_img(0:img_size-1))
  begin_time=omp_get_wtime()
  do i=1,repeat
    call hessian_matrix_det(integral_img,h,w,4.0_real32,output_img)
  end do
  elapsed_us=(omp_get_wtime()-begin_time)*1.0e6_real64/real(repeat,real64)
!$omp end target data
  checksum=0.0_real64
  do i=0,img_size-1;checksum=checksum+real(output_img(i),real64);end do
  write(*,'(a,f0.6,a)') 'Average kernel execution time : ',elapsed_us,' (us)'
  write(*,'(a,f0.6)') 'Kernel checksum: ',checksum
  deallocate(input_img,integral_img,output_img)
contains
  function normal_sample(state) result(value)
    integer(int64),intent(inout)::state
    real(real32)::value
    real(real64)::u1,u2
    state=mod(16807_int64*state,2147483647_int64);u1=real(state,real64)/2147483647.0_real64
    state=mod(16807_int64*state,2147483647_int64);u2=real(state,real64)/2147483647.0_real64
    value=real(sqrt(-2.0_real64*log(u1))*cos(2.0_real64*acos(-1.0_real64)*u2),real32)
  end function normal_sample
  function sum_integral(image,rows,columns,row,column) result(value)
    real(real32),intent(in)::image(0:);integer(int32),intent(in)::rows,columns,row,column
    real(real32)::value;integer(int32)::x,y
    value=0.0_real32
    do y=0,row;do x=0,column;value=value+image(y*columns+x);end do;end do
  end function sum_integral
  function clip(value,low,high) result(result_value)
    integer(int32),intent(in)::value,low,high;integer(int32)::result_value
    result_value=max(low,min(value,high))
  end function clip
  function integ(image,rows,columns,row,column,row_length,column_length) result(answer)
    real(real32),intent(in)::image(*);integer(int32),intent(in)::rows,columns,row,column,row_length,column_length
    real(real32)::answer;integer(int32)::r,c,r2,c2
    r=clip(row,0_int32,rows-1);c=clip(column,0_int32,columns-1)
    r2=clip(r+row_length,0_int32,rows-1);c2=clip(c+column_length,0_int32,columns-1)
    answer=image(r*columns+c+1)+image(r2*columns+c2+1)-image(r*columns+c2+1)-image(r2*columns+c+1)
    answer=max(0.0_real32,answer)
  end function integ
  subroutine hessian_matrix_det(image,rows,columns,sigma,output)
    real(real32),intent(in)::image(0:),sigma;integer(int32),intent(in)::rows,columns
    real(real32),intent(out)::output(0:)
    integer(int32)::tid,r,c,size,b,l,wid
    real(real32)::wi,tl,br,bl,tr,dxy,mid,side,dxx,dyy
!$omp target teams distribute parallel do thread_limit(256) private(r,c,size,b,l,wid,wi,tl,br,bl,tr,dxy,mid,side,dxx,dyy)
    do tid=0,rows*columns-1
      r=tid/columns;c=mod(tid,columns);size=int(3.0_real32*sigma,int32)
      b=(size-1)/2+1;l=size/3;wid=size;wi=1.0_real32/real(size*size,real32)
      tl=integ(image,rows,columns,r-l,c-l,l,l);br=integ(image,rows,columns,r+1,c+1,l,l)
      bl=integ(image,rows,columns,r-l,c+1,l,l);tr=integ(image,rows,columns,r+1,c-l,l,l)
      dxy=-(bl+tr-tl-br)*wi
      mid=integ(image,rows,columns,r-l+1,c-l,2*l-1,wid);side=integ(image,rows,columns,r-l+1,c-l/2,2*l-1,l)
      dxx=-(mid-3.0_real32*side)*wi
      mid=integ(image,rows,columns,r-l,c-b+1,wid,2*b-1);side=integ(image,rows,columns,r-b/2,c-b+1,b,2*b-1)
      dyy=-(mid-3.0_real32*side)*wi
      output(tid)=dxx*dyy-0.81_real32*dxy*dxy
    end do
!$omp end target teams distribute parallel do
  end subroutine hessian_matrix_det
end program doh
