program convolution_separable
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  implicit none

  integer :: image_w, image_h, num_iterations
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

  if (command_argument_count() /= 3) then
    write(*,'(a)') 'Usage: ./main <image width> <image height> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) image_w
  call get_command_argument(2, arg); read(arg, *) image_h
  call get_command_argument(3, arg); read(arg, *) num_iterations
  call run_benchmark(image_w, image_h, num_iterations)

contains

  subroutine run_benchmark(image_w, image_h, num_iterations)
    integer, intent(in) :: image_w, image_h, num_iterations
    integer(int64) :: image_size, index
    integer :: iteration
    integer(int64) :: start_count, end_count, clock_rate
    integer(c_int) :: random_value
    real(real64) :: elapsed_seconds, delta, sum, l2norm
    logical :: ok
    real(real32), allocatable :: kernel(:), input(:), buffer(:), output_cpu(:), output_gpu(:)

    if (num_iterations <= 0) error stop 'repeat must be positive'
    if (modulo(image_w, 128) /= 0 .or. modulo(image_h, 4) /= 0 .or. modulo(image_w, 16) /= 0 .or. &
        modulo(image_h, 64) /= 0) error stop 'image dimensions violate convolution kernel assertions'

    image_size = int(image_w, int64) * image_h
    allocate(kernel(0:16), input(0:image_size-1), buffer(0:image_size-1), output_cpu(0:image_size-1), output_gpu(0:image_size-1))
    call c_srand(2009_c_int)
    do index = 0, 16
      random_value = c_rand()
      kernel(index) = real(modulo(random_value, 16_c_int), real32)
    end do
    do index = 0, image_size - 1
      random_value = c_rand()
      input(index) = real(modulo(random_value, 16_c_int), real32)
    end do

!$omp target data map(to:kernel(0:16), input(0:image_size-1)) map(alloc:buffer(0:image_size-1)) map(from:output_gpu(0:image_size-1))
    call convolution_rows(buffer, input, kernel, image_w, image_h, image_w)
    call convolution_columns(output_gpu, buffer, kernel, image_w, image_h, image_w)

    call system_clock(start_count, clock_rate)
    do iteration = 1, num_iterations
      call convolution_rows(buffer, input, kernel, image_w, image_h, image_w)
      call convolution_columns(output_gpu, buffer, kernel, image_w, image_h, image_w)
    end do
    call system_clock(end_count)
    elapsed_seconds = real(end_count - start_count, real64) / real(clock_rate, real64) / real(num_iterations, real64)
    write(*,'(a,f0.6,a)') 'Average kernel execution time ', elapsed_seconds, ' (s)'
!$omp end target data

    write(*,'(a)') 'Comparing against Host/C++ computation...'
    call convolution_row_host(buffer, input, kernel, image_w, image_h, 8)
    call convolution_column_host(output_cpu, buffer, kernel, image_w, image_h, 8)
    sum = 0.0_real64
    delta = 0.0_real64
    do index = 0, image_size - 1
      delta = delta + real(output_cpu(index) - output_gpu(index), real64) * real(output_cpu(index) - output_gpu(index), real64)
      sum = sum + real(output_cpu(index), real64) * real(output_cpu(index), real64)
    end do
    l2norm = sqrt(delta / sum)
    write(*,'(a,es10.3,/)') 'Relative L2 norm: ', l2norm
    ok = l2norm < 1.0e-6_real64
    if (ok) then
      write(*,'(a)') 'PASS'
    else
      write(*,'(a)') 'FAIL'
    end if
    deallocate(kernel, input, buffer, output_cpu, output_gpu)
  end subroutine run_benchmark

  subroutine convolution_rows(dst, src, kernel, image_w, image_h, pitch)
    use omp_lib
    integer, parameter :: block_x = 16, block_y = 4, result_steps = 8, halo_steps = 1
    real(real32), intent(out) :: dst(0:)
    real(real32), intent(in) :: src(0:), kernel(0:)
    integer, intent(in) :: image_w, image_h, pitch
    integer :: team_x, team_y, num_teams, gid_x, gid_y, lid_x, lid_y, base_x, base_y, i, j
    real(real32) :: l_data(0:(result_steps + 2 * halo_steps) * block_x - 1, 0:block_y-1)
    real(real32) :: value

    team_x = (image_w / result_steps) / block_x
    team_y = image_h / block_y
    num_teams = team_x * team_y
!$omp target teams num_teams(num_teams) thread_limit(block_y*block_x) private(l_data)
!$omp parallel private(gid_x,gid_y,lid_x,lid_y,base_x,base_y,i,j,value)
    gid_x = modulo(omp_get_team_num(), team_x)
    gid_y = omp_get_team_num() / team_x
    lid_x = modulo(omp_get_thread_num(), block_x)
    lid_y = omp_get_thread_num() / block_x
    base_x = (gid_x * result_steps - halo_steps) * block_x + lid_x
    base_y = gid_y * block_y + lid_y

    do i = halo_steps, halo_steps + result_steps - 1
      l_data(lid_x + i * block_x, lid_y) = src((base_y * pitch) + base_x + i * block_x)
    end do
    do i = 0, halo_steps - 1
      if (base_x + i * block_x >= 0) then
        l_data(lid_x + i * block_x, lid_y) = src((base_y * pitch) + base_x + i * block_x)
      else
        l_data(lid_x + i * block_x, lid_y) = 0.0_real32
      end if
    end do
    do i = halo_steps + result_steps, halo_steps + result_steps + halo_steps - 1
      if (base_x + i * block_x < image_w) then
        l_data(lid_x + i * block_x, lid_y) = src((base_y * pitch) + base_x + i * block_x)
      else
        l_data(lid_x + i * block_x, lid_y) = 0.0_real32
      end if
    end do
!$omp barrier
    do i = halo_steps, halo_steps + result_steps - 1
      value = 0.0_real32
      do j = -8, 8
        value = value + kernel(8-j) * l_data(lid_x + i * block_x + j, lid_y)
      end do
      dst((base_y * pitch) + base_x + i * block_x) = value
    end do
!$omp end parallel
!$omp end target teams
  end subroutine convolution_rows

  subroutine convolution_columns(dst, src, kernel, image_w, image_h, pitch)
    use omp_lib
    integer, parameter :: block_x = 16, block_y = 8, result_steps = 8, halo_steps = 1
    real(real32), intent(out) :: dst(0:)
    real(real32), intent(in) :: src(0:), kernel(0:)
    integer, intent(in) :: image_w, image_h, pitch
    integer :: team_x, team_y, num_teams, gid_x, gid_y, lid_x, lid_y, base_x, base_y, i, j
    real(real32) :: l_data(0:(result_steps + 2 * halo_steps) * block_y, 0:block_x-1)
    real(real32) :: value

    team_x = image_w / block_x
    team_y = image_h / result_steps / block_y
    num_teams = team_x * team_y
!$omp target teams num_teams(num_teams) thread_limit(block_y*block_x) private(l_data)
!$omp parallel private(gid_x,gid_y,lid_x,lid_y,base_x,base_y,i,j,value)
    gid_x = modulo(omp_get_team_num(), team_x)
    gid_y = omp_get_team_num() / team_x
    lid_x = modulo(omp_get_thread_num(), block_x)
    lid_y = omp_get_thread_num() / block_x
    base_x = gid_x * block_x + lid_x
    base_y = (gid_y * result_steps - halo_steps) * block_y + lid_y

    do i = halo_steps, halo_steps + result_steps - 1
      l_data(lid_y + i * block_y, lid_x) = src((base_y + i * block_y) * pitch + base_x)
    end do
    do i = 0, halo_steps - 1
      if (base_y + i * block_y >= 0) then
        l_data(lid_y + i * block_y, lid_x) = src((base_y + i * block_y) * pitch + base_x)
      else
        l_data(lid_y + i * block_y, lid_x) = 0.0_real32
      end if
    end do
    do i = halo_steps + result_steps, halo_steps + result_steps + halo_steps - 1
      if (base_y + i * block_y < image_h) then
        l_data(lid_y + i * block_y, lid_x) = src((base_y + i * block_y) * pitch + base_x)
      else
        l_data(lid_y + i * block_y, lid_x) = 0.0_real32
      end if
    end do
!$omp barrier
    do i = halo_steps, halo_steps + result_steps - 1
      value = 0.0_real32
      do j = -8, 8
        value = value + kernel(8-j) * l_data(lid_y + i * block_y + j, lid_x)
      end do
      dst((base_y + i * block_y) * pitch + base_x) = value
    end do
!$omp end parallel
!$omp end target teams
  end subroutine convolution_columns

  subroutine convolution_row_host(dst, src, kernel, image_w, image_h, radius)
    real(real32), intent(out) :: dst(0:)
    real(real32), intent(in) :: src(0:), kernel(0:)
    integer, intent(in) :: image_w, image_h, radius
    integer :: x, y, offset
    real(real64) :: value
    do y = 0, image_h - 1
      do x = 0, image_w - 1
        value = 0.0_real64
        do offset = -radius, radius
          if (x + offset >= 0 .and. x + offset < image_w) value = value + real(src(y * image_w + x + offset), real64) * real(kernel(radius-offset), real64)
        end do
        dst(y * image_w + x) = real(value, real32)
      end do
    end do
  end subroutine convolution_row_host

  subroutine convolution_column_host(dst, src, kernel, image_w, image_h, radius)
    real(real32), intent(out) :: dst(0:)
    real(real32), intent(in) :: src(0:), kernel(0:)
    integer, intent(in) :: image_w, image_h, radius
    integer :: x, y, offset
    real(real64) :: value
    do y = 0, image_h - 1
      do x = 0, image_w - 1
        value = 0.0_real64
        do offset = -radius, radius
          if (y + offset >= 0 .and. y + offset < image_h) value = value + real(src((y + offset) * image_w + x), real64) * real(kernel(radius-offset), real64)
        end do
        dst(y * image_w + x) = real(value, real32)
      end do
    end do
  end subroutine convolution_column_host

end program convolution_separable
