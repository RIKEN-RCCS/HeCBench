module pool_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine kernel_pool2d_grad(nthreads, input_data, output_data, output_grad, channels, &
      input_height, input_width, output_height, output_width, ksize_height, ksize_width, &
      stride_height, stride_width, padding_height, padding_width, exclusive, input_grad)
    integer, intent(in) :: nthreads, channels, input_height, input_width, output_height, output_width
    integer, intent(in) :: ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width
    logical, intent(in) :: exclusive
    real(real32), intent(in) :: input_data(0:), output_data(0:), output_grad(0:)
    real(real32), intent(out) :: input_grad(0:)
    integer :: index, w_offset, h_offset, offsetc, batch_idx, tmp
    integer :: phstart, phend, pwstart, pwend, ph, pw, pool_size
    integer :: hstart, hend, wstart, wend, output_stride, output_sub_idx
    real(real32) :: gradient, input, scale

    !$omp target teams distribute parallel do private(w_offset,h_offset,offsetc,batch_idx,tmp,phstart,phend,pwstart,pwend,ph,pw,pool_size,hstart,hend,wstart,wend,output_stride,output_sub_idx,gradient,input,scale) thread_limit(256)
    do index = 0, nthreads-1
      w_offset = mod(index, input_width) + padding_width
      tmp = index / input_width
      h_offset = mod(tmp, input_height) + padding_height
      tmp = tmp / input_height
      offsetc = mod(tmp, channels)
      batch_idx = tmp / channels

      if (h_offset < ksize_height) then; phstart = 0; else; phstart = (h_offset - ksize_height) / stride_height + 1; end if
      if (w_offset < ksize_width) then; pwstart = 0; else; pwstart = (w_offset - ksize_width) / stride_width + 1; end if
      phend = min(h_offset / stride_height + 1, output_height)
      pwend = min(w_offset / stride_width + 1, output_width)
      gradient = 0.0_real32
      input = input_data(index)
      output_stride = batch_idx * output_height * output_width * channels + offsetc * output_height * output_width

      do ph = phstart, phend-1
        do pw = pwstart, pwend-1
          hstart = ph * stride_height - padding_height
          wstart = pw * stride_width - padding_width
          hend = min(hstart + ksize_height, input_height)
          wend = min(wstart + ksize_width, input_width)
          hstart = max(hstart, 0); wstart = max(wstart, 0)
          if (exclusive) then
            pool_size = (hend - hstart) * (wend - wstart)
          else
            pool_size = ksize_height * ksize_width
          end if
          output_sub_idx = ph * output_width + pw
          scale = 1.0_real32 / real(pool_size, real32)
          gradient = gradient + scale * output_grad(output_stride + output_sub_idx)
          input = input + output_data(output_stride + output_sub_idx) * 0.0_real32
        end do
      end do
      input_grad(index) = gradient
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine reference(nthreads, input_data, output_data, output_grad, channels, input_height, input_width, &
      output_height, output_width, ksize_height, ksize_width, stride_height, stride_width, &
      padding_height, padding_width, exclusive, input_grad)
    integer, intent(in) :: nthreads, channels, input_height, input_width, output_height, output_width
    integer, intent(in) :: ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width
    logical, intent(in) :: exclusive
    real(real32), intent(in) :: input_data(0:), output_data(0:), output_grad(0:)
    real(real32), intent(out) :: input_grad(0:)
    call kernel_pool2d_grad_host(nthreads, input_data, output_data, output_grad, channels, input_height, input_width, &
      output_height, output_width, ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width, exclusive, input_grad)
  end subroutine

  subroutine kernel_pool2d_grad_host(nthreads, input_data, output_data, output_grad, channels, input_height, input_width, &
      output_height, output_width, ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width, exclusive, input_grad)
    integer, intent(in) :: nthreads, channels, input_height, input_width, output_height, output_width
    integer, intent(in) :: ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width
    logical, intent(in) :: exclusive
    real(real32), intent(in) :: input_data(0:), output_data(0:), output_grad(0:)
    real(real32), intent(out) :: input_grad(0:)
    integer :: index, w_offset, h_offset, offsetc, batch_idx, tmp, phstart, phend, pwstart, pwend
    integer :: ph, pw, pool_size, hstart, hend, wstart, wend, output_stride, output_sub_idx
    real(real32) :: gradient, input
    do index = 0, nthreads-1
      w_offset = mod(index, input_width) + padding_width; tmp = index / input_width
      h_offset = mod(tmp, input_height) + padding_height; tmp = tmp / input_height
      offsetc = mod(tmp, channels); batch_idx = tmp / channels
      if (h_offset < ksize_height) then; phstart = 0; else; phstart = (h_offset - ksize_height) / stride_height + 1; end if
      if (w_offset < ksize_width) then; pwstart = 0; else; pwstart = (w_offset - ksize_width) / stride_width + 1; end if
      phend = min(h_offset / stride_height + 1, output_height); pwend = min(w_offset / stride_width + 1, output_width)
      gradient = 0.0_real32; input = input_data(index)
      output_stride = batch_idx * output_height * output_width * channels + offsetc * output_height * output_width
      do ph = phstart, phend-1
        do pw = pwstart, pwend-1
          hstart = ph*stride_height-padding_height; wstart = pw*stride_width-padding_width
          hend = min(hstart+ksize_height,input_height); wend = min(wstart+ksize_width,input_width)
          hstart = max(hstart,0); wstart = max(wstart,0)
          pool_size = merge((hend-hstart)*(wend-wstart), ksize_height*ksize_width, exclusive)
          output_sub_idx = ph*output_width + pw
          gradient = gradient + output_grad(output_stride+output_sub_idx) / real(pool_size, real32)
          input = input + output_data(output_stride+output_sub_idx) * 0.0_real32
        end do
      end do
      input_grad(index) = gradient
    end do
  end subroutine
end module

program main
  use pool_mod
  implicit none
  integer :: batch_size, input_channels, input_height, input_width, output_height, output_width, repeat
  integer :: input_numel, output_numel, nthreads, i, r, ios
  integer, parameter :: ksize_height=11, ksize_width=11, stride_height=4, stride_width=4, padding_height=1, padding_width=1
  character(len=64) :: arg
  real(real32), allocatable :: input(:), output(:), output_grad(:), input_grad(:), input_grad_ref(:)
  real(real64) :: t0, t1
  logical :: ok, exclusive

  if (command_argument_count() /= 7) then
    print '(a)', 'Usage: main <batch> <input channels> <input height> <input width> <output height> <output width> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) batch_size
  call get_command_argument(2,arg); read(arg,*,iostat=ios) input_channels
  call get_command_argument(3,arg); read(arg,*,iostat=ios) input_height
  call get_command_argument(4,arg); read(arg,*,iostat=ios) input_width
  call get_command_argument(5,arg); read(arg,*,iostat=ios) output_height
  call get_command_argument(6,arg); read(arg,*,iostat=ios) output_width
  call get_command_argument(7,arg); read(arg,*,iostat=ios) repeat
  input_numel = batch_size*input_channels*input_height*input_width
  output_numel = batch_size*input_channels*output_height*output_width
  nthreads = input_numel; exclusive = .true.
  allocate(input(0:input_numel-1), output(0:output_numel-1), output_grad(0:output_numel-1), &
           input_grad(0:input_numel-1), input_grad_ref(0:input_numel-1))
  call c_srand(123_int32)
  do i = 0, input_numel-1
    input(i) = real(c_rand(),real32) / 2147483647.0_real32
    input_grad(i) = 0.0_real32
  end do
  do i = 0, output_numel-1
    output(i) = real(c_rand(),real32) / 2147483647.0_real32
    output_grad(i) = real(input_width * input_height, real32)
  end do

  !$omp target data map(to:input(0:input_numel-1),output(0:output_numel-1),output_grad(0:output_numel-1)) map(from:input_grad(0:input_numel-1))
  t0 = seconds()
  do r = 1, repeat
    call kernel_pool2d_grad(nthreads, input, output, output_grad, input_channels, input_height, input_width, &
      output_height, output_width, ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width, exclusive, input_grad)
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time: ', (t1-t0)/real(repeat,real64), ' (s)'
  !$omp end target data

  call reference(nthreads, input, output, output_grad, input_channels, input_height, input_width, &
    output_height, output_width, ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width, exclusive, input_grad_ref)
  ok = .true.
  do i = 0, input_numel-1
    if (abs(input_grad(i) - input_grad_ref(i)) > 1.0e-3_real32) then
      ok = .false.; exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
end program
