module tonemapping_mod
  use iso_fortran_env, only: real32, real64, int64
  implicit none
contains
  pure real(real32) function luminance(r, g, b)
    real(real32), intent(in) :: r, g, b
    luminance = 0.2126_real32 * r + 0.7152_real32 * g + 0.0722_real32 * b
  end function luminance

  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  pure integer function pidx(width, num_channels, x, y, c) result(pos)
    integer, intent(in) :: width, num_channels, x, y, c
    pos = width * num_channels * y + x * num_channels + c
  end function pidx

  subroutine pattanaik(input, output, average_luminance, gamma, c, delta, width, num_channels, height)
    real(real32), intent(in) :: input(0:)
    real(real32), intent(out) :: output(0:)
    real(real32), intent(in) :: average_luminance, gamma, c, delta
    integer, intent(in) :: width, num_channels, height
    integer :: x, y
    real(real32) :: r, g, b, r1, g1, b1, y_luminance, gc_pattanaik
    real(real32) :: left_up, up, right_up, left, right, left_down, down, right_down
    real(real32) :: c_l_pattanaik, y_l_pattanaik, y_d_pattanaik
!$omp target teams distribute parallel do collapse(2) thread_limit(256) &
!$omp& map(to:input) map(from:output) private(x,y,r,g,b,r1,g1,b1,y_luminance,gc_pattanaik,left_up,up,right_up,left,right,left_down,down,right_down,c_l_pattanaik,y_l_pattanaik,y_d_pattanaik)
    do y = 0, height - 1
      do x = 0, width - 1
        r1 = input(pidx(width, num_channels, x, y, 0))
        g1 = input(pidx(width, num_channels, x, y, 1))
        b1 = input(pidx(width, num_channels, x, y, 2))
        y_luminance = luminance(r1, g1, b1)
        gc_pattanaik = c * average_luminance
        if (x /= 0 .and. y /= 0 .and. x /= width - 1 .and. y /= height - 1) then
          r = input(pidx(width, num_channels, x - 1, y - 1, 0)); g = input(pidx(width, num_channels, x - 1, y - 1, 1)); b = input(pidx(width, num_channels, x - 1, y - 1, 2))
          left_up = luminance(r, g, b)
          r = input(pidx(width, num_channels, x, y - 1, 0)); g = input(pidx(width, num_channels, x, y - 1, 1)); b = input(pidx(width, num_channels, x, y - 1, 2))
          up = luminance(r, g, b)
          r = input(pidx(width, num_channels, x + 1, y - 1, 0)); g = input(pidx(width, num_channels, x + 1, y - 1, 1)); b = input(pidx(width, num_channels, x + 1, y - 1, 2))
          right_up = luminance(r, g, b)
          r = input(pidx(width, num_channels, x - 1, y, 0)); g = input(pidx(width, num_channels, x - 1, y, 1)); b = input(pidx(width, num_channels, x - 1, y, 2))
          left = luminance(r, g, b)
          r = input(pidx(width, num_channels, x + 1, y, 0)); g = input(pidx(width, num_channels, x + 1, y, 1)); b = input(pidx(width, num_channels, x + 1, y, 2))
          right = luminance(r, g, b)
          r = input(pidx(width, num_channels, x - 1, y + 1, 0)); g = input(pidx(width, num_channels, x - 1, y + 1, 1)); b = input(pidx(width, num_channels, x - 1, y + 1, 2))
          left_down = luminance(r, g, b)
          r = input(pidx(width, num_channels, x, y + 1, 0)); g = input(pidx(width, num_channels, x, y + 1, 1)); b = input(pidx(width, num_channels, x, y + 1, 2))
          down = luminance(r, g, b)
          r = input(pidx(width, num_channels, x + 1, y + 1, 0)); g = input(pidx(width, num_channels, x + 1, y + 1, 1)); b = input(pidx(width, num_channels, x + 1, y + 1, 2))
          right_down = luminance(r, g, b)
          y_l_pattanaik = (left_up + up + right_up + left + right + left_down + down + right_down) / 8.0_real32
        else
          y_l_pattanaik = y_luminance
        end if
        c_l_pattanaik = y_l_pattanaik * log(delta + y_l_pattanaik / y_luminance) + gc_pattanaik
        y_d_pattanaik = y_luminance / (y_luminance + c_l_pattanaik)
        output(pidx(width, num_channels, x, y, 0)) = (r1 / y_luminance) ** gamma * y_d_pattanaik
        output(pidx(width, num_channels, x, y, 1)) = (g1 / y_luminance) ** gamma * y_d_pattanaik
        output(pidx(width, num_channels, x, y, 2)) = (b1 / y_luminance) ** gamma * y_d_pattanaik
        output(pidx(width, num_channels, x, y, 3)) = input(pidx(width, num_channels, x, y, 3))
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine pattanaik
end module tonemapping_mod

program main
  use iso_fortran_env, only: real32, real64
  use tonemapping_mod
  implicit none
  integer, parameter :: num_channels = 4
  integer :: argc, iterations, width, height, x, y, i, unit, stat
  character(len=256) :: input_name
  character(len=128) :: arg
  real(real32), parameter :: c_pattanaik = 0.25_real32, gamma_pattanaik = 0.4_real32, delta_pattanaik = 0.000002_real32
  real(real32), allocatable :: input(:), output(:), reference_output(:)
  real(real32) :: average_luminance, error
  real(real64) :: time_sum, t0, t1

  argc = command_argument_count()
  if (argc < 2) stop 1
  call get_command_argument(1, input_name)
  call get_command_argument(2, arg); read(arg, *, iostat=stat) iterations
  print '(a,a)', 'Input file name ', trim(input_name)
  open(newunit=unit, file=trim(input_name), status='old', action='read', iostat=stat)
  if (stat /= 0) then
    print '(a,a)', 'not able to open the file  ', trim(input_name)
    stop 1
  end if
  read(unit, *) width
  read(unit, *) height
  allocate(input(0:height*width*num_channels-1), output(0:height*width*num_channels-1), reference_output(0:height*width*num_channels-1))
  do y = 0, height - 1
    do x = 0, width - 1
      read(unit, *) input(pidx(width, num_channels, x, y, 0))
      read(unit, *) input(pidx(width, num_channels, x, y, 1))
      read(unit, *) input(pidx(width, num_channels, x, y, 2))
      read(unit, *) input(pidx(width, num_channels, x, y, 3))
    end do
  end do
  close(unit)
  print '(a,i0)', 'Width of the image ', width
  print '(a,i0)', 'Height of the image ', height
  average_luminance = 0.0_real32
  do y = 0, height - 1
    do x = 0, width - 1
      average_luminance = average_luminance + luminance(input(pidx(width,num_channels,x,y,0)), &
        input(pidx(width,num_channels,x,y,1)), input(pidx(width,num_channels,x,y,2)))
    end do
  end do
  average_luminance = average_luminance / real(width * height, real32)
  print '(a,f12.6)', 'Average luminance value in the image ', average_luminance
!$omp target data map(alloc:input,output)
  do i = 1, merge(2, 0, iterations /= 1)
!$omp target update to(input)
    call pattanaik(input, output, average_luminance, gamma_pattanaik, c_pattanaik, delta_pattanaik, width, num_channels, height)
!$omp target update from(output)
  end do
  print '(a,i0,a)', 'Executing kernel for ', iterations, ' iterations'
  print '(a)', '-------------------------------------------'
  time_sum = 0.0_real64
  do i = 1, iterations
!$omp target update to(input)
    t0 = wall_seconds()
    call pattanaik(input, output, average_luminance, gamma_pattanaik, c_pattanaik, delta_pattanaik, width, num_channels, height)
    t1 = wall_seconds()
!$omp target update from(output)
    time_sum = time_sum + (t1 - t0)
  end do
  print '(a,f12.6,a)', 'Average kernel execution time: ', time_sum * 1.0e6_real64 / real(iterations, real64), ' (us)'
!$omp end target data
  call pattanaik(input, reference_output, average_luminance, gamma_pattanaik, c_pattanaik, delta_pattanaik, width, num_channels, height)
  error = 0.0_real32
  do i = 0, height * width * num_channels - 1
    error = error + reference_output(i) - output(i)
  end do
  error = error / real(height * width, real32)
  if (error > 0.000001_real32) then
    print '(a,f12.6)', 'FAIL with normalized error: ', error
    stop 1
  else
    print '(a)', 'PASS'
  end if
end program main
