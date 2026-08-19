program resize
  use iso_fortran_env, only: int8, int16, int32, real32, real64
  use omp_lib
  use resize_mod
  implicit none

  integer :: argc, in_width, in_height, out_width, out_height, num_channels, repeat
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 6) then
    print '(a)', 'Usage: ./main <input image width> <input image height>'
    print '(a)', '          <output image width> <output image height>'
    print '(a)', '          <image channels> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) in_width
  call get_command_argument(2, arg); read(arg, *) in_height
  call get_command_argument(3, arg); read(arg, *) out_width
  call get_command_argument(4, arg); read(arg, *) out_height
  call get_command_argument(5, arg); read(arg, *) num_channels
  call get_command_argument(6, arg); read(arg, *) repeat

  print '(a,i0,a,i0,a,i0,a,i0,a,i0,a)', 'Resize ', num_channels, ' images from (', in_width, ' x ', in_height, ') to (', out_width, ' x ', out_height, ')'

  print '(a)', ''
  print '(a)', 'The size of each pixel is 1 byte'
  call resize_image_i8(in_width, in_height, out_width, out_height, num_channels, repeat, .false.)
  print '(a)', ''
  print '(a)', 'Bilinear resizing'
  call resize_image_i8(in_width, in_height, out_width, out_height, num_channels, repeat, .true.)

  print '(a)', ''
  print '(a)', 'The size of each pixel is 2 bytes'
  call resize_image_i16(in_width, in_height, out_width, out_height, num_channels, repeat, .false.)
  print '(a)', ''
  print '(a)', 'Bilinear resizing'
  call resize_image_i16(in_width, in_height, out_width, out_height, num_channels, repeat, .true.)

  print '(a)', ''
  print '(a)', 'The size of each pixel is 4 bytes'
  call resize_image_i32(in_width, in_height, out_width, out_height, num_channels, repeat, .false.)
  print '(a)', ''
  print '(a)', 'Bilinear resizing'
  call resize_image_i32(in_width, in_height, out_width, out_height, num_channels, repeat, .true.)

contains

  subroutine resize_image_i8(iw, ih, ow, oh, nc, reps, bilinear)
    integer, intent(in) :: iw, ih, ow, oh, nc, reps
    logical, intent(in) :: bilinear
    integer :: i, in_size, out_size, in_bytes, out_bytes
    real(real32) :: fx, fy
    real(real64) :: start_time, end_time, elapsed_ns
    integer(int8), allocatable :: input(:), output(:)
    in_size = nc * ih * iw
    out_size = nc * oh * ow
    in_bytes = in_size
    out_bytes = out_size
    allocate(input(0:in_size-1), output(0:out_size-1))
    do i = 0, in_size - 1
      input(i) = int(mod(i + 1, 13), int8)
    end do
    ! Preserve the integer division performed by the C++ source before the
    ! quotient is assigned to float.
    fx = real(iw / ow, real32)
    fy = real(ih / oh, real32)
    !$omp target data map(to: input(0:in_size-1)) map(from: output(0:out_size-1))
      start_time = omp_get_wtime()
      do i = 0, reps - 1
        if (bilinear) then
          call resize_bilinear_i8(output, out_size, oh, ow, input, ih, iw, fy, fx, .true.)
        else
          call resize_i8(output, out_size, oh, ow, input, ih, iw, fy, fx, .true., .true.)
        end if
      end do
      end_time = omp_get_wtime()
      elapsed_ns = (end_time - start_time) * 1.0e9_real64
      print '(a,f12.6,a,f12.6,a)', 'Average kernel execution time: ', elapsed_ns * 1.0e-3_real64 / reps, ' (us)    Perf: ', &
        real(in_bytes + out_bytes, real64) * reps / elapsed_ns, ' (GB/s)'
    !$omp end target data
  end subroutine resize_image_i8

  subroutine resize_image_i16(iw, ih, ow, oh, nc, reps, bilinear)
    integer, intent(in) :: iw, ih, ow, oh, nc, reps
    logical, intent(in) :: bilinear
    integer :: i, in_size, out_size, in_bytes, out_bytes
    real(real32) :: fx, fy
    real(real64) :: start_time, end_time, elapsed_ns
    integer(int16), allocatable :: input(:), output(:)
    in_size = nc * ih * iw; out_size = nc * oh * ow
    in_bytes = 2 * in_size; out_bytes = 2 * out_size
    allocate(input(0:in_size-1), output(0:out_size-1))
    do i = 0, in_size - 1
      input(i) = int(mod(i + 1, 13), int16)
    end do
    fx = real(iw / ow, real32); fy = real(ih / oh, real32)
    !$omp target data map(to: input(0:in_size-1)) map(from: output(0:out_size-1))
      start_time = omp_get_wtime()
      do i = 0, reps - 1
        if (bilinear) then
          call resize_bilinear_i16(output, out_size, oh, ow, input, ih, iw, fy, fx, .true.)
        else
          call resize_i16(output, out_size, oh, ow, input, ih, iw, fy, fx, .true., .true.)
        end if
      end do
      end_time = omp_get_wtime()
      elapsed_ns = (end_time - start_time) * 1.0e9_real64
      print '(a,f12.6,a,f12.6,a)', 'Average kernel execution time: ', elapsed_ns * 1.0e-3_real64 / reps, ' (us)    Perf: ', &
        real(in_bytes + out_bytes, real64) * reps / elapsed_ns, ' (GB/s)'
    !$omp end target data
  end subroutine resize_image_i16

  subroutine resize_image_i32(iw, ih, ow, oh, nc, reps, bilinear)
    integer, intent(in) :: iw, ih, ow, oh, nc, reps
    logical, intent(in) :: bilinear
    integer :: i, in_size, out_size, in_bytes, out_bytes
    real(real32) :: fx, fy
    real(real64) :: start_time, end_time, elapsed_ns
    integer(int32), allocatable :: input(:), output(:)
    in_size = nc * ih * iw; out_size = nc * oh * ow
    in_bytes = 4 * in_size; out_bytes = 4 * out_size
    allocate(input(0:in_size-1), output(0:out_size-1))
    do i = 0, in_size - 1
      input(i) = int(mod(i + 1, 13), int32)
    end do
    fx = real(iw / ow, real32); fy = real(ih / oh, real32)
    !$omp target data map(to: input(0:in_size-1)) map(from: output(0:out_size-1))
      start_time = omp_get_wtime()
      do i = 0, reps - 1
        if (bilinear) then
          call resize_bilinear_i32(output, out_size, oh, ow, input, ih, iw, fy, fx, .true.)
        else
          call resize_i32(output, out_size, oh, ow, input, ih, iw, fy, fx, .true., .true.)
        end if
      end do
      end_time = omp_get_wtime()
      elapsed_ns = (end_time - start_time) * 1.0e9_real64
      print '(a,f12.6,a,f12.6,a)', 'Average kernel execution time: ', elapsed_ns * 1.0e-3_real64 / reps, ' (us)    Perf: ', &
        real(in_bytes + out_bytes, real64) * reps / elapsed_ns, ' (GB/s)'
    !$omp end target data
  end subroutine resize_image_i32

end program resize
