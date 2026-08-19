program s8n
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real64
  use omp_lib
  use s8n_mod
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
  end interface

  integer :: argc, b, n, repeat, input_size, output_size, radius, i, error_count
  integer, allocatable :: h_xyz(:), h_out(:), h_out2(:), h_out4(:), r_out(:), r_out2(:), r_out4(:)
  real(real64) :: start_time, end_time
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage: ./main <number of batches> <number of points> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) b
  call get_command_argument(2, arg); read(arg, *) n
  call get_command_argument(3, arg); read(arg, *) repeat

  input_size = b * n * 3
  output_size = b * n * 8
  radius = 512
  allocate(h_xyz(0:input_size-1), h_out(0:output_size-1), r_out(0:output_size-1), &
           h_out2(0:2*output_size-1), r_out2(0:2*output_size-1), &
           h_out4(0:4*output_size-1), r_out4(0:4*output_size-1))

  call c_srand(123_c_int)
  do i = 0, input_size - 1
    h_xyz(i) = mod(c_rand(), 512_c_int) - 256
  end do

  !$omp target data map(to: h_xyz(0:input_size-1)) map(alloc: h_out(0:output_size-1), h_out2(0:2*output_size-1), h_out4(0:4*output_size-1))
    start_time = omp_get_wtime()
    do i = 0, repeat - 1
      call k_cube_select(b, n, radius, h_xyz, h_out)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average execution time of select kernel: ', ((end_time-start_time)*1.0e6_real64)/repeat, ' (us)'
    !$omp target update from(h_out(0:output_size-1))
    call cube_select(b, n, radius, h_xyz, r_out)
    error_count = count(h_out /= r_out)

    start_time = omp_get_wtime()
    do i = 0, repeat - 1
      call k_cube_select_two(b, n, radius, h_xyz, h_out2)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average execution time of select2 kernel: ', ((end_time-start_time)*1.0e6_real64)/repeat, ' (us)'
    !$omp target update from(h_out2(0:2*output_size-1))
    call cube_select_two(b, n, radius, h_xyz, r_out2)
    error_count = error_count + count(h_out2 /= r_out2)

    start_time = omp_get_wtime()
    do i = 0, repeat - 1
      call k_cube_select_four(b, n, radius, h_xyz, h_out4)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average execution time of select4 kernel: ', ((end_time-start_time)*1.0e6_real64)/repeat, ' (us)'
    !$omp target update from(h_out4(0:4*output_size-1))
    call cube_select_four(b, n, radius, h_xyz, r_out4)
    error_count = error_count + count(h_out4 /= r_out4)

    if (error_count == 0) then
      print '(a)', 'PASS'
    else
      print '(a)', 'FAIL'
    end if
  !$omp end target data
end program s8n
