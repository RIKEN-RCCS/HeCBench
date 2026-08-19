program main
  use iso_fortran_env, only: real32, real64, int64
  use urng_kernel_mod
  implicit none
  integer :: argc, iterations, height, width, size, image_size, stat, i, rep
  character(len=256) :: file_path
  character(len=128) :: arg
  integer, allocatable :: input_image(:), output_image(:)
  real(real64) :: t0, t1
  real(real32) :: mean
  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <path to file> <repeat>'
    stop 1
  end if
  call get_command_argument(1, file_path)
  call get_command_argument(2, arg); read(arg, *, iostat=stat) iterations
  call load_bitmap_rgba(trim(file_path), input_image, width, height)
  size = height * width
  image_size = size * 4
  allocate(output_image(0:image_size-1))
  output_image = 0
  print '(a,a,a,i0,a,i0)', 'Image ', trim(file_path), ' height: ', height, ' width: ', width
  print '(a,i0,a)', 'Executing kernel for ', iterations, ' iterations'
  print '(a)', '-------------------------------------------'
!$omp target data map(to:input_image) map(from:output_image)
  t0 = wall_seconds()
  do rep = 1, iterations
    call urng_kernel(input_image, output_image, size)
  end do
  t1 = wall_seconds()
!$omp end target data
  print '(a,f12.6,a)', 'Average kernel execution time: ', (t1 - t0) * 1.0e6_real64 / real(iterations, real64), ' (us)'
  mean = 0.0_real32
  do i = 0, image_size - 1
    mean = mean + real(output_image(i) - input_image(i), real32)
  end do
  mean = mean / real(image_size * factor, real32)
  print '(a,f12.6)', 'The averaged mean of the image: ', mean
  print '(a)', merge('PASS', 'FAIL', abs(mean) < 1.0_real32)
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine load_bitmap_rgba(path, pixels, width, height)
    character(len=*), intent(in) :: path
    integer, allocatable, intent(out) :: pixels(:)
    integer, intent(out) :: width, height
    integer :: unit, stat, file_size, offbits, dib_size, bit_count, row_stride, x, y, c
    character(len=1), allocatable :: bytes(:)
    open(newunit=unit, file=path, access='stream', form='unformatted', status='old', action='read', iostat=stat)
    if (stat /= 0) then
      print '(a)', 'Failed to load input image!'
      stop 1
    end if
    inquire(unit, size=file_size)
    allocate(bytes(file_size))
    read(unit) bytes
    close(unit)
    offbits = le32(bytes, 11)
    dib_size = le32(bytes, 15)
    width = le32(bytes, 19)
    height = le32(bytes, 23)
    bit_count = le16(bytes, 29)
    if (dib_size < 40 .or. bit_count /= 24) then
      print '(a)', 'Failed to read pixel Data!'
      stop 1
    end if
    allocate(pixels(0:width*height*4-1))
    row_stride = ((width * 3 + 3) / 4) * 4
    do y = 0, height - 1
      do x = 0, width - 1
        c = offbits + (height - 1 - y) * row_stride + x * 3
        pixels((y*width+x)*4+0) = ichar(bytes(c+3))
        pixels((y*width+x)*4+1) = ichar(bytes(c+2))
        pixels((y*width+x)*4+2) = ichar(bytes(c+1))
        pixels((y*width+x)*4+3) = 255
      end do
    end do
  end subroutine load_bitmap_rgba

  pure integer function le16(bytes, one_based)
    character(len=1), intent(in) :: bytes(:)
    integer, intent(in) :: one_based
    le16 = ichar(bytes(one_based)) + ishft(ichar(bytes(one_based+1)), 8)
  end function le16

  pure integer function le32(bytes, one_based)
    character(len=1), intent(in) :: bytes(:)
    integer, intent(in) :: one_based
    le32 = ichar(bytes(one_based)) + ishft(ichar(bytes(one_based+1)), 8) + &
           ishft(ichar(bytes(one_based+2)), 16) + ishft(ichar(bytes(one_based+3)), 24)
  end function le32
end program main
