program sad
  use iso_fortran_env, only: int8, real64
  use omp_lib
  use bmp_mod
  use sad_mod
  implicit none

  integer :: argc, repeat, main_width, main_height, main_size, template_width, template_height, template_size
  integer :: height_difference, width_difference, sad_array_size, h_num_occurrences, h_min_mse, i
  integer, allocatable :: h_sad_array(:)
  integer(int8), allocatable :: main_rgb(:), template_rgb(:)
  type(bmp_image) :: main_image, template_image
  real(real64) :: kernel_time, begin_time, end_time, elapsed_time
  character(len=512) :: image_path, template_path, arg

  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage: ./main <image> <template image> <repeat>'
    stop 1
  end if
  call get_command_argument(1, image_path)
  call get_command_argument(2, template_path)
  call get_command_argument(3, arg); read(arg, *) repeat

  call read_bmp24(trim(image_path), main_image)
  call read_bmp24(trim(template_path), template_image)
  main_width = main_image%width
  main_height = main_image%height
  main_size = main_width * main_height
  template_width = template_image%width
  template_height = template_image%height
  template_size = template_width * template_height
  call move_alloc(main_image%rgb, main_rgb)
  call move_alloc(template_image%rgb, template_rgb)
  height_difference = main_height - template_height
  width_difference = main_width - template_width
  sad_array_size = (height_difference + 1) * (width_difference + 1)
  allocate(h_sad_array(0:sad_array_size-1))

  !$omp target data map(to: main_rgb(0:3*main_size-1), template_rgb(0:3*template_size-1)) &
  !$omp& map(alloc: h_sad_array(0:sad_array_size-1))
    kernel_time = 0.0_real64
    begin_time = omp_get_wtime()
    do i = 0, repeat - 1
      call compute_sad_array(h_sad_array, main_rgb, template_rgb, sad_array_size, &
                             h_min_mse, h_num_occurrences, main_width, main_height, &
                             template_width, template_height, template_size, kernel_time)
    end do
    end_time = omp_get_wtime()
    elapsed_time = (end_time - begin_time) * 1000.0_real64

    print '(a)', 'Parallel Computation Results: '
    print '(a,f12.6)', 'Kernel time in msec: ', kernel_time
    print '(a,f12.6)', 'Elapsed time in msec = ', elapsed_time
    print '(a,i0,a,i0)', 'Main Image Dimensions: ', main_width, '*', main_height
    print '(a,i0,a,i0)', 'Template Image Dimensions: ', template_width, '*', template_height
    print '(a,i0)', 'Found Minimum:  ', h_min_mse
    print '(a,i0)', 'Number of Occurances: ', h_num_occurrences
  !$omp end target data
end program sad
