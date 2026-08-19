program sobel
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use sobel_mod
  use bmp_mod
  implicit none
  integer :: argc, iterations, width, height, image_size, i
  type(uchar4), allocatable :: input_image(:), output_image(:), verification_output(:)
  real(real32), allocatable :: output_device(:), output_reference(:)
  real(real64) :: start_time, end_time
  character(len=512) :: file_path, arg

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <path to file> <repeat>'
    stop 1
  end if
  call get_command_argument(1, file_path)
  call get_command_argument(2, arg); read(arg, *) iterations
  call read_bmp24(trim(file_path), input_image, width, height)
  image_size = width * height * 4
  print '(a,i0,a,i0)', 'Image height = ', height, ' and width = ', width
  allocate(output_image(0:width*height-1), verification_output(0:width*height-1))
  output_image = uchar4(0_int8,0_int8,0_int8,0_int8)
  verification_output = uchar4(0_int8,0_int8,0_int8,0_int8)
  print '(a,i0,a)', 'Executing kernel for ', iterations, ' iterations'
  print '(a)', '-------------------------------------------'
  !$omp target data map(to: input_image(0:width*height-1)) map(tofrom: output_image(0:width*height-1))
    start_time = omp_get_wtime()
    call sobel_kernel(input_image, output_image, width, height, iterations)
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time: ', ((end_time-start_time)*1.0e6_real64)/iterations, ' (us)'
  !$omp end target data
  call reference(verification_output, input_image, width, height)
  allocate(output_device(0:image_size-1), output_reference(0:image_size-1))
  do i = 0, width*height - 1
    output_device(i*4+0) = real(u8(output_image(i)%x), real32)
    output_device(i*4+1) = real(u8(output_image(i)%y), real32)
    output_device(i*4+2) = real(u8(output_image(i)%z), real32)
    output_device(i*4+3) = real(u8(output_image(i)%w), real32)
    output_reference(i*4+0) = real(u8(verification_output(i)%x), real32)
    output_reference(i*4+1) = real(u8(verification_output(i)%y), real32)
    output_reference(i*4+2) = real(u8(verification_output(i)%z), real32)
    output_reference(i*4+3) = real(u8(verification_output(i)%w), real32)
  end do
  if (compare(output_reference, output_device, image_size, 1.0e-6_real32)) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
end program sobel
