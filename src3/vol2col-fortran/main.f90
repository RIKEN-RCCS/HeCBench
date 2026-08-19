module vol2col_mod
  use iso_fortran_env, only: real32, real64, int64
  implicit none
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine vol2col_kernel(data_vol, channels, depth, height, width, ksize_t, ksize_h, ksize_w, &
      pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, &
      depth_col, height_col, width_col, data_col)
    real(real32), intent(in) :: data_vol(0:)
    integer, intent(in) :: channels, depth, height, width, ksize_t, ksize_h, ksize_w
    integer, intent(in) :: pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w
    integer, intent(in) :: depth_col, height_col, width_col
    real(real32), intent(inout) :: data_col(0:)
    integer :: channel_in, t_out, h_out, w_out, channel_out, t_in, h_in, w_in, i, j, k
    integer(int64) :: c, vbase
    integer :: t, h, w
!$omp target teams distribute parallel do collapse(4) num_threads(512) &
!$omp& map(to:data_vol) map(tofrom:data_col) private(channel_in,t_out,h_out,w_out,channel_out,t_in,h_in,w_in,i,j,k,c,vbase,t,h,w)
    do channel_in = 0, channels - 1
      do t_out = 0, depth_col - 1
        do h_out = 0, height_col - 1
          do w_out = 0, width_col - 1
            channel_out = channel_in * ksize_t * ksize_h * ksize_w
            t_in = t_out * stride_t - pad_t
            h_in = h_out * stride_h - pad_h
            w_in = w_out * stride_w - pad_w
            vbase = int(((channel_in * depth + t_in) * height + h_in) * width + w_in, int64)
            c = int(((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out, int64)
            do i = 0, ksize_t - 1
              do j = 0, ksize_h - 1
                do k = 0, ksize_w - 1
                  t = t_in + i * dilation_t
                  h = h_in + j * dilation_h
                  w = w_in + k * dilation_w
                  if (t >= 0 .and. h >= 0 .and. w >= 0 .and. t < depth .and. h < height .and. w < width) then
                    data_col(c) = data_vol(vbase + int(i * dilation_t * height * width + j * dilation_h * width + k * dilation_w, int64))
                  else
                    data_col(c) = 0.0_real32
                  end if
                  c = c + int(depth_col * height_col * width_col, int64)
                end do
              end do
            end do
          end do
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine vol2col_kernel

  subroutine col2vol_kernel(data_col, channels, depth, height, width, kernel_t, kernel_h, kernel_w, &
      pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, &
      depth_col, height_col, width_col, data_vol)
    real(real32), intent(in) :: data_col(0:)
    integer(int64), intent(in) :: channels
    integer, intent(in) :: depth, height, width, kernel_t, kernel_h, kernel_w
    integer, intent(in) :: pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w
    integer, intent(in) :: depth_col, height_col, width_col
    real(real32), intent(inout) :: data_vol(0:)
    integer(int64) :: channel_in, idx_k, data_col_index, outidx, t_k, h_k, w_k
    integer :: t_out, h_out, w_out, w_im, h_im, t_im, c_im
    integer :: kernel_extent_w, kernel_extent_h, kernel_extent_t
    integer :: w_col_start, w_col_end, h_col_start, h_col_end, t_col_start, t_col_end
    integer :: t_col, h_col, w_col
    real(real32) :: val
!$omp target teams distribute parallel do collapse(4) num_threads(512) &
!$omp& map(to:data_col) map(tofrom:data_vol) private(channel_in,t_out,h_out,w_out,val,w_im,h_im,t_im,c_im,kernel_extent_w,kernel_extent_h,kernel_extent_t,w_col_start,w_col_end,h_col_start,h_col_end,t_col_start,t_col_end,t_col,h_col,w_col,t_k,h_k,w_k,idx_k,data_col_index,outidx)
    do channel_in = 0_int64, channels - 1_int64
      do t_out = 0, depth - 1
        do h_out = 0, height - 1
          do w_out = 0, width - 1
            val = 0.0_real32
            w_im = w_out + pad_w
            h_im = h_out + pad_h
            t_im = t_out + pad_t
            c_im = int(channel_in)
            kernel_extent_w = (kernel_w - 1) * dilation_w + 1
            kernel_extent_h = (kernel_h - 1) * dilation_h + 1
            kernel_extent_t = (kernel_t - 1) * dilation_t + 1
            if (w_im < kernel_extent_w) then
              w_col_start = 0
            else
              w_col_start = (w_im - kernel_extent_w) / stride_w + 1
            end if
            w_col_end = min(w_im / stride_w + 1, width_col)
            if (h_im < kernel_extent_h) then
              h_col_start = 0
            else
              h_col_start = (h_im - kernel_extent_h) / stride_h + 1
            end if
            h_col_end = min(h_im / stride_h + 1, height_col)
            if (t_im < kernel_extent_t) then
              t_col_start = 0
            else
              t_col_start = (t_im - kernel_extent_t) / stride_t + 1
            end if
            t_col_end = min(t_im / stride_t + 1, depth_col)
            do t_col = t_col_start, t_col_end - 1
              do h_col = h_col_start, h_col_end - 1
                do w_col = w_col_start, w_col_end - 1
                  t_k = t_im - t_col * stride_t
                  h_k = h_im - h_col * stride_h
                  w_k = w_im - w_col * stride_w
                  if (mod(t_k, dilation_t) == 0 .and. mod(h_k, dilation_h) == 0 .and. mod(w_k, dilation_w) == 0) then
                    t_k = t_k / dilation_t
                    h_k = h_k / dilation_h
                    w_k = w_k / dilation_w
                    idx_k = ((int(c_im, int64) * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k
                    data_col_index = ((idx_k * depth_col + t_col) * height_col + h_col) * width_col + w_col
                    val = val + data_col(data_col_index)
                  end if
                end do
              end do
            end do
            outidx = channel_in * width * height * depth + t_out * width * height + h_out * width + w_out
            data_vol(outidx) = val
          end do
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine col2vol_kernel
end module vol2col_mod

program main
  use iso_fortran_env, only: real32, real64, int64
  use vol2col_mod
  implicit none
  integer :: argc, repeat, stat, k
  character(len=128) :: arg
  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) repeat
  do k = 1, 9, 2
    print '(/a,i0)', 'kernel size: ', k
    call eval(repeat, 4, 3, 255, 255, 3, 255, 255, k, k, k, 1, 1, 1, 2, 2, 2, 2, 2, 2)
  end do
contains
  subroutine eval(repeat, channels, depth, height, width, depth_col, height_col, width_col, &
      ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w)
    integer, intent(in) :: repeat, channels, depth, height, width, depth_col, height_col, width_col
    integer, intent(in) :: ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w
    integer(int64) :: vol_size, col_size, i
    integer :: rep
    real(real32), allocatable :: data_vol(:), data_col(:), data_vol_ref(:), data_col_ref(:)
    real(real64) :: t0, t1
    logical :: ok
    vol_size = int(channels, int64) * (2 * pad_t + depth) * (2 * pad_h + height) * (2 * pad_w + width)
    col_size = (int(channels, int64) * ksize_t * ksize_h * ksize_w + 1_int64) * &
               (depth_col + pad_t) * (height_col + pad_h) * (width_col + pad_w)
    allocate(data_vol(0:vol_size-1), data_col(0:col_size-1), data_vol_ref(0:vol_size-1), data_col_ref(0:col_size-1))
    call random_seed()
    do i = 0, vol_size - 1
      call random_number(data_vol(i))
      data_vol_ref(i) = data_vol(i)
    end do
    data_col = 0.0_real32
    data_col_ref = 0.0_real32
!$omp target data map(to:data_vol) map(to:data_col)
    t0 = wall_seconds()
    do rep = 1, repeat
      call vol2col_kernel(data_vol, channels, depth, height, width, ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, &
        stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, depth_col, height_col, width_col, data_col)
    end do
    t1 = wall_seconds()
    print '(a,f12.6,a)', 'Average execution time of vol2col kernel: ', (t1 - t0) * 1.0e6_real64 / real(repeat, real64), ' (us)'
!$omp target update from(data_col)
    call vol2col_kernel(data_vol_ref, channels, depth, height, width, ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, &
      stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, depth_col, height_col, width_col, data_col_ref)
    ok = all(data_col_ref == data_col)
    print '(a)', merge('PASS', 'FAIL', ok)
    t0 = wall_seconds()
    do rep = 1, repeat
      call col2vol_kernel(data_col, int(channels, int64), depth, height, width, ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, &
        stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, depth_col, height_col, width_col, data_vol)
    end do
    t1 = wall_seconds()
    print '(a,f12.6,a)', 'Average execution time of col2vol kernel: ', (t1 - t0) * 1.0e6_real64 / real(repeat, real64), ' (us)'
!$omp target update from(data_vol)
    call col2vol_kernel(data_col_ref, int(channels, int64), depth, height, width, ksize_t, ksize_h, ksize_w, pad_t, pad_h, pad_w, &
      stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, depth_col, height_col, width_col, data_vol_ref)
    ok = all(abs(data_vol_ref - data_vol) <= 1.0e-3_real32)
    print '(a)', merge('PASS', 'FAIL', ok)
!$omp end target data
  end subroutine eval
end program main
