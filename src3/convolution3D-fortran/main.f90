program convolution3d
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  implicit none

  integer :: n, c, m, win, hin, k, repeat
  integer :: argc
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

  argc = command_argument_count()
  if (argc /= 7) then
    write(*,'(a)',advance='no') 'Usage: ./main <batch size:N> <input channels:C> <output feature maps:M>'
    write(*,'(a)') ' <input width:Win> <input height:Hin> <kernel size:K> <repeat>'
    stop 1
  end if

  call get_command_argument(1, arg); read(arg, *) n
  call get_command_argument(2, arg); read(arg, *) c
  call get_command_argument(3, arg); read(arg, *) m
  call get_command_argument(4, arg); read(arg, *) win
  call get_command_argument(5, arg); read(arg, *) hin
  call get_command_argument(6, arg); read(arg, *) k
  call get_command_argument(7, arg); read(arg, *) repeat

  write(*,'(a)') '3D convolution (FP32)'
  write(*,'(/,a)') '========== Warmup start =========='
  call conv3d(n, c, m, win, hin, k, 1000)
  write(*,'(/,a)') '========== Warmup done =========='
  call conv3d(n, c, m, win, hin, k, repeat)

contains

  subroutine conv3d(n, channels, maps, width_in, height_in, kernel, repetitions)
    integer, intent(in) :: n, channels, maps, width_in, height_in, kernel, repetitions
    integer :: height_out, width_out
    integer(int64) :: x_size, w_size, y_size, idx
    integer :: batch, fmap, row, col, channel, p, q, iteration
    integer(int64) :: clock_rate, start_count, end_count
    integer(c_int) :: random_value
    real(real64) :: elapsed_us
    real(real32) :: sum
    logical :: ok
    real(real32), allocatable :: x(:), w(:), y(:), y_ref(:)

    height_out = height_in - kernel + 1
    width_out = width_in - kernel + 1
    x_size = int(n, int64) * channels * height_in * width_in
    w_size = int(maps, int64) * channels * kernel * kernel
    y_size = int(n, int64) * maps * height_out * width_out
    if (height_out <= 0 .or. width_out <= 0 .or. repetitions <= 0) error stop 'invalid convolution dimensions'

    allocate(x(0:x_size-1), w(0:w_size-1), y(0:y_size-1), y_ref(0:y_size-1))
    call c_srand(123_c_int)
    do idx = 0, w_size - 1
      random_value = c_rand()
      w(idx) = real(modulo(random_value, 31_c_int), real32)
    end do
    do idx = 0, x_size - 1
      random_value = c_rand()
      x(idx) = real(modulo(random_value, 13_c_int), real32)
    end do
    y = -1.0_real32
    y_ref = -1.0_real32

    write(*,'(a,i0,a,i0,a,i0)') 'input dimensions: C=', channels, ' Win=', width_in, ' Hin=', height_in
    write(*,'(a,i0,a,i0,a,i0)') 'output dimensions: M=', maps, ' Wout=', width_out, ' Hout=', height_out

    call system_clock(start_count, clock_rate)
!$omp target data map(to:x(0:x_size-1), w(0:w_size-1)) map(tofrom:y(0:y_size-1))
    do iteration = 1, repetitions
!$omp target teams distribute parallel do collapse(4) thread_limit(256) private(sum,channel,p,q)
      do batch = 0, n - 1
        do fmap = 0, maps - 1
          do row = 0, height_out - 1
            do col = 0, width_out - 1
              sum = 0.0_real32
              do channel = 0, channels - 1
                do p = 0, kernel - 1
                  do q = 0, kernel - 1
                    sum = sum + x(input_index(batch, channel, row + p, col + q, channels, height_in, width_in)) * &
                                w(weight_index(fmap, channel, p, q, channels, kernel))
                  end do
                end do
              end do
              y(output_index(batch, fmap, row, col, maps, height_out, width_out)) = sum
            end do
          end do
        end do
      end do
!$omp end target teams distribute parallel do
    end do
!$omp end target data
    call system_clock(end_count)
    elapsed_us = real(end_count - start_count, real64) * 1.0e6_real64 / real(clock_rate, real64) / real(repetitions, real64)
    write(*,'(a,f0.6,a)') 'Average kernel execution time of conv3d kernel: ', elapsed_us, ' (us)'

    call reference(x, w, y_ref, n, maps, channels, kernel, height_in, width_in, height_out, width_out)
    ok = .true.
    do idx = 0, y_size - 1
      if (abs(y(idx) - y_ref(idx)) > 1.0e-3_real32) then
        write(*,'(f0.6,a,f0.6)') y(idx), ' (device) != ', y_ref(idx)
        ok = .false.
        exit
      end if
    end do
    if (ok) then
      write(*,'(a)') 'PASS'
    else
      write(*,'(a)') 'FAIL'
    end if
    deallocate(x, w, y, y_ref)
  end subroutine conv3d

  subroutine reference(x, w, y, n, maps, channels, kernel, height_in, width_in, height_out, width_out)
    integer, intent(in) :: n, maps, channels, kernel, height_in, width_in, height_out, width_out
    real(real32), intent(in) :: x(0:), w(0:)
    real(real32), intent(out) :: y(0:)
    integer :: batch, fmap, row, col, channel, p, q
    do batch = 0, n - 1
      do fmap = 0, maps - 1
        do row = 0, height_out - 1
          do col = 0, width_out - 1
            y(output_index(batch, fmap, row, col, maps, height_out, width_out)) = 0.0_real32
            do channel = 0, channels - 1
              do p = 0, kernel - 1
                do q = 0, kernel - 1
                  y(output_index(batch, fmap, row, col, maps, height_out, width_out)) = &
                    y(output_index(batch, fmap, row, col, maps, height_out, width_out)) + &
                    x(input_index(batch, channel, row + p, col + q, channels, height_in, width_in)) * &
                    w(weight_index(fmap, channel, p, q, channels, kernel))
                end do
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine reference

  pure integer(int64) function input_index(batch, channel, row, col, channels, height, width)
    integer, intent(in) :: batch, channel, row, col, channels, height, width
    input_index = int(batch, int64) * channels * height * width + int(channel, int64) * height * width + int(row, int64) * width + col
  end function input_index

  pure integer(int64) function weight_index(fmap, channel, row, col, channels, kernel)
    integer, intent(in) :: fmap, channel, row, col, channels, kernel
    weight_index = int(fmap, int64) * channels * kernel * kernel + int(channel, int64) * kernel * kernel + int(row, int64) * kernel + col
  end function weight_index

  pure integer(int64) function output_index(batch, fmap, row, col, maps, height, width)
    integer, intent(in) :: batch, fmap, row, col, maps, height, width
    output_index = int(batch, int64) * maps * height * width + int(fmap, int64) * height * width + int(row, int64) * width + col
  end function output_index

end program convolution3d
