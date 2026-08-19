module distort_device
  use, intrinsic :: iso_fortran_env, only : int32, real32
  use, intrinsic :: iso_c_binding, only : c_int, c_int8_t
  implicit none
  type, bind(C) :: uchar3
    integer(c_int8_t) :: x, y, z, padding
  end type uchar3
  type, bind(C) :: properties
    real(real32) :: k, center_x, center_y
    integer(c_int) :: width, height
    real(real32) :: thresh, xscale, yscale, xshift, yshift
  end type properties
!$omp declare target (radial_x_dev, radial_y_dev, sample_image_dev, pack_u8_dev, unpack_u8_dev)
contains
  real(real32) function radial_x_dev(x, y, prop)
    real(real32), intent(in) :: x, y
    type(properties), intent(in) :: prop
    real(real32) :: scaled_x, scaled_y
    scaled_x=x*prop%xscale+prop%xshift; scaled_y=y*prop%yscale+prop%yshift
    radial_x_dev=scaled_x+(scaled_x-prop%center_x)*prop%k*((scaled_x-prop%center_x)**2+(scaled_y-prop%center_y)**2)
  end function radial_x_dev
  real(real32) function radial_y_dev(x, y, prop)
    real(real32), intent(in) :: x, y
    type(properties), intent(in) :: prop
    real(real32) :: scaled_x, scaled_y
    scaled_x=x*prop%xscale+prop%xshift; scaled_y=y*prop%yscale+prop%yshift
    radial_y_dev=scaled_y+(scaled_y-prop%center_y)*prop%k*((scaled_x-prop%center_x)**2+(scaled_y-prop%center_y)**2)
  end function radial_y_dev
  integer(c_int8_t) function pack_u8_dev(value)
    integer(int32), intent(in) :: value
    pack_u8_dev=transfer(iand(value,255_int32),pack_u8_dev)
  end function pack_u8_dev
  integer(int32) function unpack_u8_dev(value)
    integer(c_int8_t), intent(in) :: value
    unpack_u8_dev=iand(int(value,int32),255_int32)
  end function unpack_u8_dev
  subroutine sample_image_dev(source, index0, index1, pixel, prop)
    type(uchar3), intent(in) :: source(*)
    real(real32), intent(in) :: index0,index1
    type(uchar3), intent(out) :: pixel
    type(properties), intent(in) :: prop
    integer :: rf,rc,cf,cc
    real(real32) :: x,y,value
    type(uchar3) :: s1,s2,s3,s4
    if(index0<0.0_real32 .or. index1<0.0_real32 .or. index0>real(prop%height-1,real32) .or. index1>real(prop%width-1,real32))then
      pixel%x=0_c_int8_t;pixel%y=0_c_int8_t;pixel%z=0_c_int8_t;pixel%padding=0_c_int8_t;return
    end if
    rf=floor(index0);rc=ceiling(index0);cf=floor(index1);cc=ceiling(index1)
    s1=source(rf*prop%width+cf+1);s2=source(rf*prop%width+cc+1);s3=source(rc*prop%width+cc+1);s4=source(rc*prop%width+cf+1)
    x=index0-real(rf,real32);y=index1-real(cf,real32)
    value=real(unpack_u8_dev(s1%x),real32)*(1.0_real32-x)*(1.0_real32-y)+real(unpack_u8_dev(s2%x),real32)*(1.0_real32-x)*y+real(unpack_u8_dev(s3%x),real32)*x*y+real(unpack_u8_dev(s4%x),real32)*x*(1.0_real32-y)
    pixel%x=pack_u8_dev(int(value,int32))
    value=real(unpack_u8_dev(s1%y),real32)*(1.0_real32-x)*(1.0_real32-y)+real(unpack_u8_dev(s2%y),real32)*(1.0_real32-x)*y+real(unpack_u8_dev(s3%y),real32)*x*y+real(unpack_u8_dev(s4%y),real32)*x*(1.0_real32-y)
    pixel%y=pack_u8_dev(int(value,int32))
    value=real(unpack_u8_dev(s1%z),real32)*(1.0_real32-x)*(1.0_real32-y)+real(unpack_u8_dev(s2%z),real32)*(1.0_real32-x)*y+real(unpack_u8_dev(s3%z),real32)*x*y+real(unpack_u8_dev(s4%z),real32)*x*(1.0_real32-y)
    pixel%z=pack_u8_dev(int(value,int32));pixel%padding=0_c_int8_t
  end subroutine sample_image_dev
end module distort_device

program distort_benchmark
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int, c_int8_t
  use distort_device
  implicit none

  integer :: width, height, repeat
  real(real32) :: coefficient
  character(len=64) :: arg

  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  if (command_argument_count() /= 4) then
    write(*,'(a)') 'Usage: ./main <input image width> <input image height> <coefficient of distortion> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) width
  call get_command_argument(2, arg); read(arg, *) height
  call get_command_argument(3, arg); read(arg, *) coefficient
  call get_command_argument(4, arg); read(arg, *) repeat
  call run_distort(width, height, coefficient, repeat)

contains

  subroutine run_distort(width, height, coefficient, repeat)
    integer, intent(in) :: width, height, repeat
    real(real32), intent(in) :: coefficient
    type(properties) :: prop
    integer(int64) :: image_size, index, start_count, end_count, clock_rate
    integer(c_int) :: value
    integer :: ex, ey, ez
    real(real64) :: elapsed_ms
    real(real32) :: new_center_x, new_center_y, xshift_2, yshift_2
    type(uchar3), allocatable :: source(:), destination(:), reference_image(:)

    if (width <= 0 .or. height <= 0 .or. repeat <= 0) error stop 'invalid dimensions or repeat'
    prop%k = coefficient
    prop%center_x = real(width / 2, real32)
    prop%center_y = real(height / 2, real32)
    prop%width = width
    prop%height = height
    prop%thresh = 1.0_real32
    prop%xshift = calc_shift(0.0_real32, prop%center_x - 1.0_real32, prop%center_x, prop%k, prop%thresh)
    new_center_x = real(prop%width, real32) - prop%center_x
    xshift_2 = calc_shift(0.0_real32, new_center_x - 1.0_real32, new_center_x, prop%k, prop%thresh)
    prop%yshift = calc_shift(0.0_real32, prop%center_y - 1.0_real32, prop%center_y, prop%k, prop%thresh)
    new_center_y = real(prop%height, real32) - prop%center_y
    yshift_2 = calc_shift(0.0_real32, new_center_y - 1.0_real32, new_center_y, prop%k, prop%thresh)
    prop%xscale = (real(prop%width, real32) - prop%xshift - xshift_2) / real(prop%width, real32)
    prop%yscale = (real(prop%height, real32) - prop%yshift - yshift_2) / real(prop%height, real32)

    image_size = int(width, int64) * height
    allocate(source(0:image_size-1), destination(0:image_size-1), reference_image(0:image_size-1))
    call c_srand(123_c_int)
    do index = 0, image_size - 1
      value = c_rand(); source(index)%x = pack_u8(modulo(value, 256_c_int))
      value = c_rand(); source(index)%y = pack_u8(modulo(value, 256_c_int))
      value = c_rand(); source(index)%z = pack_u8(modulo(value, 256_c_int))
      source(index)%padding = 0_c_int8_t
    end do

!$omp target data map(to:source(0:image_size-1),prop) map(from:destination(0:image_size-1))
    call system_clock(start_count, clock_rate)
    do index = 1, repeat
      call barrel_distort(source, destination, prop)
    end do
    call system_clock(end_count)
    elapsed_ms = real(end_count-start_count,real64) * 1.0e3_real64 / real(clock_rate,real64) / real(repeat,real64)
    write(*,'(a,f0.6,a)') 'Average kernel execution time: ', elapsed_ms, ' (ms)'
!$omp end target data

    call reference(source, reference_image, prop)
    ex = 0; ey = 0; ez = 0
    do index = 0, image_size - 1
      ex = max(ex, abs(unpack_u8(destination(index)%x) - unpack_u8(reference_image(index)%x)))
      ey = max(ey, abs(unpack_u8(destination(index)%y) - unpack_u8(reference_image(index)%y)))
      ez = max(ez, abs(unpack_u8(destination(index)%z) - unpack_u8(reference_image(index)%z)))
    end do
    write(*,'(a,i0,a,i0,a,i0)') 'Max error of each channel: ', ex, ' ', ey, ' ', ez
    deallocate(source, destination, reference_image)
  end subroutine run_distort

  recursive real(real32) function calc_shift(x1, x2, center, coefficient, threshold) result(shift)
    real(real32), intent(in) :: x1, x2, center, coefficient, threshold
    real(real32) :: x3, result1, result3
    x3 = x1 + (x2-x1) * 0.5_real32
    result1 = x1 + ((x1-center) * coefficient * ((x1-center) * (x1-center)))
    result3 = x3 + ((x3-center) * coefficient * ((x3-center) * (x3-center)))
    if (result1 > -threshold .and. result1 < threshold) then
      shift = x1
    else if (result3 < 0.0_real32) then
      shift = calc_shift(x3, x2, center, coefficient, threshold)
    else
      shift = calc_shift(x1, x3, center, coefficient, threshold)
    end if
  end function calc_shift

  real(real32) function radial_x(x, y, prop)
    real(real32), intent(in) :: x, y
    type(properties), intent(in) :: prop
    real(real32) :: scaled_x, scaled_y
    scaled_x = x * prop%xscale + prop%xshift
    scaled_y = y * prop%yscale + prop%yshift
    radial_x = scaled_x + ((scaled_x-prop%center_x)*prop%k*((scaled_x-prop%center_x)*(scaled_x-prop%center_x) + &
      (scaled_y-prop%center_y)*(scaled_y-prop%center_y)))
  end function radial_x

  real(real32) function radial_y(x, y, prop)
    real(real32), intent(in) :: x, y
    type(properties), intent(in) :: prop
    real(real32) :: scaled_x, scaled_y
    scaled_x = x * prop%xscale + prop%xshift
    scaled_y = y * prop%yscale + prop%yshift
    radial_y = scaled_y + ((scaled_y-prop%center_y)*prop%k*((scaled_x-prop%center_x)*(scaled_x-prop%center_x) + &
      (scaled_y-prop%center_y)*(scaled_y-prop%center_y)))
  end function radial_y

  subroutine sample_image(source, index0, index1, pixel, prop)
    type(uchar3), intent(in) :: source(0:)
    real(real32), intent(in) :: index0, index1
    type(uchar3), intent(out) :: pixel
    type(properties), intent(in) :: prop
    integer :: row_floor, row_ceil, column_floor, column_ceil
    real(real32) :: x, y, value
    type(uchar3) :: s1, s2, s3, s4
    if (index0 < 0.0_real32 .or. index1 < 0.0_real32 .or. index0 > real(prop%height-1,real32) .or. &
        index1 > real(prop%width-1,real32)) then
      pixel%x = 0_c_int8_t; pixel%y = 0_c_int8_t; pixel%z = 0_c_int8_t; pixel%padding = 0_c_int8_t
      return
    end if
    row_floor = floor(index0); row_ceil = ceiling(index0)
    column_floor = floor(index1); column_ceil = ceiling(index1)
    s1 = source(row_floor*prop%width + column_floor)
    s2 = source(row_floor*prop%width + column_ceil)
    s3 = source(row_ceil*prop%width + column_ceil)
    s4 = source(row_ceil*prop%width + column_floor)
    x = index0 - real(row_floor,real32); y = index1 - real(column_floor,real32)
    value = real(unpack_u8(s1%x),real32)*(1.0_real32-x)*(1.0_real32-y) + real(unpack_u8(s2%x),real32)*(1.0_real32-x)*y + &
      real(unpack_u8(s3%x),real32)*x*y + real(unpack_u8(s4%x),real32)*x*(1.0_real32-y)
    pixel%x = pack_u8(int(value,int32))
    value = real(unpack_u8(s1%y),real32)*(1.0_real32-x)*(1.0_real32-y) + real(unpack_u8(s2%y),real32)*(1.0_real32-x)*y + &
      real(unpack_u8(s3%y),real32)*x*y + real(unpack_u8(s4%y),real32)*x*(1.0_real32-y)
    pixel%y = pack_u8(int(value,int32))
    value = real(unpack_u8(s1%z),real32)*(1.0_real32-x)*(1.0_real32-y) + real(unpack_u8(s2%z),real32)*(1.0_real32-x)*y + &
      real(unpack_u8(s3%z),real32)*x*y + real(unpack_u8(s4%z),real32)*x*(1.0_real32-y)
    pixel%z = pack_u8(int(value,int32)); pixel%padding = 0_c_int8_t
  end subroutine sample_image
  subroutine barrel_distort(source, destination, prop)
    type(uchar3), intent(in) :: source(0:)
    type(uchar3), intent(out) :: destination(0:)
    type(properties), intent(in) :: prop
    integer :: row, column
    real(real32) :: x, y
    type(uchar3) :: pixel
!$omp target teams distribute parallel do collapse(2) thread_limit(256) private(x,y,pixel)
    do row = 0, prop%height - 1
      do column = 0, prop%width - 1
        x = radial_x_dev(real(column,real32), real(row,real32), prop)
        y = radial_y_dev(real(column,real32), real(row,real32), prop)
        call sample_image_dev(source, y, x, pixel, prop)
        destination(row*prop%width + column) = pixel
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine barrel_distort

  subroutine reference(source, destination, prop)
    type(uchar3), intent(in) :: source(0:)
    type(uchar3), intent(out) :: destination(0:)
    type(properties), intent(in) :: prop
    integer :: row, column
    real(real32) :: x, y
    type(uchar3) :: pixel
    do row = 0, prop%height - 1
      do column = 0, prop%width - 1
        x = radial_x(real(column,real32), real(row,real32), prop)
        y = radial_y(real(column,real32), real(row,real32), prop)
        call sample_image(source, y, x, pixel, prop)
        destination(row*prop%width + column) = pixel
      end do
    end do
  end subroutine reference

  integer(c_int8_t) function pack_u8(value)
    integer(int32), intent(in) :: value
    pack_u8 = transfer(iand(value,255_int32),pack_u8)
  end function pack_u8

  integer(int32) function unpack_u8(value)
    integer(c_int8_t), intent(in) :: value
    unpack_u8 = iand(int(value,int32),255_int32)
  end function unpack_u8

end program distort_benchmark
