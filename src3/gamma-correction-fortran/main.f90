module gamma_module
  use, intrinsic :: iso_fortran_env, only : real32, real64, int32
  use, intrinsic :: iso_c_binding, only : c_int8_t
  implicit none
  type, bind(C) :: img_pixel
    integer(c_int8_t) :: b, g, r, a
  end type img_pixel
!$omp declare target (unsigned_byte, set_pixel, gamma_pixel)
contains
  integer(int32) function unsigned_byte(value) result(result_value)
    integer(c_int8_t), intent(in) :: value
    result_value = iand(int(value,int32),255_int32)
  end function unsigned_byte

  subroutine set_pixel(pixel, blue, green, red, alpha)
    type(img_pixel), intent(inout) :: pixel
    integer(int32), intent(in) :: blue, green, red, alpha
    pixel%b = transfer(iand(blue,255_int32),pixel%b); pixel%g = transfer(iand(green,255_int32),pixel%g)
    pixel%r = transfer(iand(red,255_int32),pixel%r); pixel%a = transfer(iand(alpha,255_int32),pixel%a)
  end subroutine set_pixel

  real(real64) function fractal_value(x, y, width, height) result(result_value)
    integer, intent(in) :: x, y, width, height
    real(real64) :: fx, fy, res, nx, ny, val
    integer :: iteration
    fx = (real(x,real64)-real(width,real64)/2.0_real64)*(1.0_real64/2000000.0_real64)-0.7436_real64
    fy = (real(y,real64)-real(height,real64)/2.0_real64)*(1.0_real64/2000000.0_real64)+0.1319_real64
    res=0.0_real64; nx=0.0_real64; ny=0.0_real64
    do iteration=0,999
      if(nx*nx+ny*ny>4.0_real64) exit
      val=nx*nx-ny*ny+fx; ny=2.0_real64*nx*ny+fy; nx=val
      res=res+exp(-sqrt(nx*nx+ny*ny))
    end do
    result_value=res
  end function fractal_value

  subroutine gamma_pixel(pixel)
    type(img_pixel), intent(inout) :: pixel
    real(real32) :: v
    integer(int32) :: gamma_value
    v=(0.3_real32*real(unsigned_byte(pixel%r),real32)+0.59_real32*real(unsigned_byte(pixel%g),real32)+ &
      0.11_real32*real(unsigned_byte(pixel%b),real32))/255.0_real32
    gamma_value=int(255.0_real32*v*v,int32)
    if(gamma_value>255_int32)gamma_value=255_int32
    call set_pixel(pixel,gamma_value,gamma_value,gamma_value,gamma_value)
  end subroutine gamma_pixel
end module gamma_module

program gamma_correction
  use, intrinsic :: iso_fortran_env, only : real64, int32
  use omp_lib
  use gamma_module
  implicit none
  integer :: argc,width,height,block_size,repeat,image_size,index,x,y,i,iteration
  type(img_pixel), allocatable :: image(:), image2(:)
  real(real64) :: fractal_pixel,start_time,end_time
  real(real32) :: total_nanoseconds
  character(len=64) :: argument
  argc=command_argument_count()
  if(argc/=4)then
    print '(a)','Usage: ./main <image width> <image height> <block size> <repeat>'; stop 1
  end if
  call get_command_argument(1,argument);read(argument,*)width
  call get_command_argument(2,argument);read(argument,*)height
  call get_command_argument(3,argument);read(argument,*)block_size
  call get_command_argument(4,argument);read(argument,*)repeat
  image_size=width*height; allocate(image(0:image_size-1),image2(0:image_size-1))
  index=0
  do i=0,image_size-1
    x=modulo(index,width); y=index/width; fractal_pixel=fractal_value(x,y,width,height)
    if(fractal_pixel<0.0_real64)fractal_pixel=0.0_real64
    if(fractal_pixel>255.0_real64)fractal_pixel=255.0_real64
    call set_pixel(image(i),int(fractal_pixel,int32),int(fractal_pixel,int32),int(fractal_pixel,int32),int(fractal_pixel,int32))
    index=index+1
  end do
  image2=image
  do i=0,image_size-1; call gamma_pixel(image(i)); end do
!$omp target data map(from:image2)
  total_nanoseconds=0.0
  do iteration=1,repeat
!$omp target update to(image2)
    start_time=omp_get_wtime()
!$omp target teams distribute parallel do thread_limit(block_size)
    do i=0,image_size-1
      call gamma_pixel(image2(i))
    end do
!$omp end target teams distribute parallel do
    end_time=omp_get_wtime()
    total_nanoseconds=total_nanoseconds+real((end_time-start_time)*1.0e9_real64,kind(total_nanoseconds))
  end do
  print '(a,f0.6,a)','Average kernel execution time ',real(total_nanoseconds*1.0e-9/real(repeat,kind(total_nanoseconds)),real64),' (s)'
!$omp end target data
  do i=0,image_size-1
    if(image(i)%b/=image2(i)%b .or. image(i)%g/=image2(i)%g .or. image(i)%r/=image2(i)%r .or. image(i)%a/=image2(i)%a)then
      print '(a)','FAIL'; deallocate(image,image2); stop 0
    end if
  end do
  print '(a)','PASS'
  deallocate(image,image2)
end program gamma_correction
