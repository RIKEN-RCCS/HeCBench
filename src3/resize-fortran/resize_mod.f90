module resize_mod
  use iso_fortran_env, only: int8, int16, int32, real32
  implicit none
  integer, parameter :: channels_per_iter = 8
!$omp declare target (round_nearest)

contains

  integer function round_nearest(x) result(v)
    real(real32), intent(in) :: x
    if (x >= 0.0_real32) then
      v = int(x + 0.5_real32)
    else
      v = int(x - 0.5_real32)
    end if
  end function round_nearest

  subroutine resize_i8(output, output_size, out_height, out_width, input, in_height, in_width, o2i_fy, o2i_fx, do_round, half_pixel_centers)
    integer(int8), intent(out) :: output(0:)
    integer, intent(in) :: output_size, out_height, out_width, in_height, in_width
    integer(int8), intent(in) :: input(0:)
    real(real32), intent(in) :: o2i_fy, o2i_fx
    logical, intent(in) :: do_round, half_pixel_centers
    integer :: iter, iters_required, in_image_size, out_image_size, c_start, y, x, in_y, in_x, in_idx, out_idx, c
    real(real32) :: in_yf, in_xf
    iters_required = output_size / channels_per_iter
    !$omp target teams distribute parallel do num_teams(29184) thread_limit(256) private(in_image_size,out_image_size,c_start,y,x,in_yf,in_xf,in_y,in_x,in_idx,out_idx,c)
    do iter = 0, iters_required - 1
      in_image_size = in_height * in_width
      out_image_size = out_height * out_width
      c_start = (iter / out_image_size) * channels_per_iter
      y = mod(iter, out_image_size) / out_width
      x = mod(iter, out_width)
      if (half_pixel_centers) then
        in_yf = (real(y, real32) + 0.5_real32) * o2i_fy
        in_xf = (real(x, real32) + 0.5_real32) * o2i_fx
      else
        in_yf = real(y, real32) * o2i_fy
        in_xf = real(x, real32) * o2i_fx
      end if
      if (do_round) then
        in_y = round_nearest(in_yf)
        in_x = round_nearest(in_xf)
      else
        in_y = int(in_yf)
        in_x = int(in_xf)
      end if
      in_x = min(in_x, in_width - 1)
      in_y = min(in_y, in_height - 1)
      in_idx = c_start * in_image_size + in_y * in_width + in_x
      out_idx = c_start * out_image_size + y * out_width + x
      do c = 0, channels_per_iter - 1
        output(out_idx) = input(in_idx)
        in_idx = in_idx + in_image_size
        out_idx = out_idx + out_image_size
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine resize_i8

  subroutine resize_i16(output, output_size, out_height, out_width, input, in_height, in_width, o2i_fy, o2i_fx, do_round, half_pixel_centers)
    integer(int16), intent(out) :: output(0:)
    integer, intent(in) :: output_size, out_height, out_width, in_height, in_width
    integer(int16), intent(in) :: input(0:)
    real(real32), intent(in) :: o2i_fy, o2i_fx
    logical, intent(in) :: do_round, half_pixel_centers
    integer :: iter, iters_required, in_image_size, out_image_size, c_start, y, x, in_y, in_x, in_idx, out_idx, c
    real(real32) :: in_yf, in_xf
    iters_required = output_size / channels_per_iter
    !$omp target teams distribute parallel do num_teams(29184) thread_limit(256) private(in_image_size,out_image_size,c_start,y,x,in_yf,in_xf,in_y,in_x,in_idx,out_idx,c)
    do iter = 0, iters_required - 1
      in_image_size = in_height * in_width
      out_image_size = out_height * out_width
      c_start = (iter / out_image_size) * channels_per_iter
      y = mod(iter, out_image_size) / out_width
      x = mod(iter, out_width)
      if (half_pixel_centers) then
        in_yf = (real(y, real32) + 0.5_real32) * o2i_fy
        in_xf = (real(x, real32) + 0.5_real32) * o2i_fx
      else
        in_yf = real(y, real32) * o2i_fy
        in_xf = real(x, real32) * o2i_fx
      end if
      if (do_round) then
        in_y = round_nearest(in_yf)
        in_x = round_nearest(in_xf)
      else
        in_y = int(in_yf)
        in_x = int(in_xf)
      end if
      in_x = min(in_x, in_width - 1)
      in_y = min(in_y, in_height - 1)
      in_idx = c_start * in_image_size + in_y * in_width + in_x
      out_idx = c_start * out_image_size + y * out_width + x
      do c = 0, channels_per_iter - 1
        output(out_idx) = input(in_idx)
        in_idx = in_idx + in_image_size
        out_idx = out_idx + out_image_size
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine resize_i16

  subroutine resize_i32(output, output_size, out_height, out_width, input, in_height, in_width, o2i_fy, o2i_fx, do_round, half_pixel_centers)
    integer(int32), intent(out) :: output(0:)
    integer, intent(in) :: output_size, out_height, out_width, in_height, in_width
    integer(int32), intent(in) :: input(0:)
    real(real32), intent(in) :: o2i_fy, o2i_fx
    logical, intent(in) :: do_round, half_pixel_centers
    integer :: iter, iters_required, in_image_size, out_image_size, c_start, y, x, in_y, in_x, in_idx, out_idx, c
    real(real32) :: in_yf, in_xf
    iters_required = output_size / channels_per_iter
    !$omp target teams distribute parallel do num_teams(29184) thread_limit(256) private(in_image_size,out_image_size,c_start,y,x,in_yf,in_xf,in_y,in_x,in_idx,out_idx,c)
    do iter = 0, iters_required - 1
      in_image_size = in_height * in_width
      out_image_size = out_height * out_width
      c_start = (iter / out_image_size) * channels_per_iter
      y = mod(iter, out_image_size) / out_width
      x = mod(iter, out_width)
      if (half_pixel_centers) then
        in_yf = (real(y, real32) + 0.5_real32) * o2i_fy
        in_xf = (real(x, real32) + 0.5_real32) * o2i_fx
      else
        in_yf = real(y, real32) * o2i_fy
        in_xf = real(x, real32) * o2i_fx
      end if
      if (do_round) then
        in_y = round_nearest(in_yf)
        in_x = round_nearest(in_xf)
      else
        in_y = int(in_yf)
        in_x = int(in_xf)
      end if
      in_x = min(in_x, in_width - 1)
      in_y = min(in_y, in_height - 1)
      in_idx = c_start * in_image_size + in_y * in_width + in_x
      out_idx = c_start * out_image_size + y * out_width + x
      do c = 0, channels_per_iter - 1
        output(out_idx) = input(in_idx)
        in_idx = in_idx + in_image_size
        out_idx = out_idx + out_image_size
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine resize_i32

  subroutine resize_bilinear_i8(output, output_size, out_height, out_width, input, in_height, in_width, o2i_fy, o2i_fx, half_pixel_centers)
    integer(int8), intent(out) :: output(0:)
    integer, intent(in) :: output_size, out_height, out_width, in_height, in_width
    integer(int8), intent(in) :: input(0:)
    real(real32), intent(in) :: o2i_fy, o2i_fx
    logical, intent(in) :: half_pixel_centers
    integer :: iter, iters_required, in_image_size, out_image_size, c_start, c_end, c, y, x, in_x0, in_x1, in_y0, in_y1, in_y2
    integer :: in_offset_r0, in_offset_r1, out_idx
    real(real32) :: in_x, in_y, v00, v01, v10, v11, val
    iters_required = output_size / channels_per_iter
    !$omp target teams distribute parallel do num_teams(29184) thread_limit(256) private(in_image_size,out_image_size,c_start,c_end,c,y,x,in_x0,in_x1,in_y0,in_y1,in_y2,in_offset_r0,in_offset_r1,out_idx,in_x,in_y,v00,v01,v10,v11,val)
    do iter = 0, iters_required - 1
      in_image_size = in_height * in_width
      out_image_size = out_height * out_width
      c_start = (iter / out_image_size) * channels_per_iter
      c_end = c_start + channels_per_iter
      y = mod(iter, out_image_size) / out_width
      x = mod(iter, out_width)
      if (half_pixel_centers) then
        in_x = max((real(x, real32) + 0.5_real32) * o2i_fx - 0.5_real32, 0.0_real32)
        in_y = max((real(y, real32) + 0.5_real32) * o2i_fy - 0.5_real32, 0.0_real32)
      else
        in_x = real(x, real32) * o2i_fx
        in_y = real(y, real32) * o2i_fy
      end if
      in_x0 = int(in_x)
      in_x1 = min(in_x0 + 1, in_width - 1)
      in_y0 = int(in_y)
      in_y1 = min(in_y0, in_height - 1)
      in_y2 = min(in_y0 + 1, in_height - 1)
      in_offset_r0 = c_start * in_image_size + in_y1 * in_width
      in_offset_r1 = c_start * in_image_size + in_y2 * in_width
      out_idx = c_start * out_image_size + y * out_width + x
      do c = c_start, c_end - 1
        v00 = real(input(in_offset_r0 + in_x0), real32)
        v01 = real(input(in_offset_r0 + in_x1), real32)
        v10 = real(input(in_offset_r1 + in_x0), real32)
        v11 = real(input(in_offset_r1 + in_x1), real32)
        val = v00 + (in_y - real(in_y0, real32)) * (v10 - v00) + &
              (in_x - real(in_x0, real32)) * (v01 - v00) + &
              (in_y - real(in_y0, real32)) * (in_x - real(in_x0, real32)) * (v11 - v01 - v10 + v00)
        ! In the C++ template each interpolation fraction is converted to T
        ! before multiplication.  For the unsigned integer pixel types used
        ! here that conversion truncates every fraction in [0,1) to zero.
        output(out_idx) = input(in_offset_r0 + in_x0)
        in_offset_r0 = in_offset_r0 + in_image_size
        in_offset_r1 = in_offset_r1 + in_image_size
        out_idx = out_idx + out_image_size
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine resize_bilinear_i8

  subroutine resize_bilinear_i16(output, output_size, out_height, out_width, input, in_height, in_width, o2i_fy, o2i_fx, half_pixel_centers)
    integer(int16), intent(out) :: output(0:)
    integer, intent(in) :: output_size, out_height, out_width, in_height, in_width
    integer(int16), intent(in) :: input(0:)
    real(real32), intent(in) :: o2i_fy, o2i_fx
    logical, intent(in) :: half_pixel_centers
    integer :: iter, iters_required, in_image_size, out_image_size, c_start, c_end, c, y, x, in_x0, in_x1, in_y0, in_y1, in_y2
    integer :: in_offset_r0, in_offset_r1, out_idx
    real(real32) :: in_x, in_y, v00, v01, v10, v11, val
    iters_required = output_size / channels_per_iter
    !$omp target teams distribute parallel do num_teams(29184) thread_limit(256) private(in_image_size,out_image_size,c_start,c_end,c,y,x,in_x0,in_x1,in_y0,in_y1,in_y2,in_offset_r0,in_offset_r1,out_idx,in_x,in_y,v00,v01,v10,v11,val)
    do iter = 0, iters_required - 1
      in_image_size = in_height * in_width
      out_image_size = out_height * out_width
      c_start = (iter / out_image_size) * channels_per_iter
      c_end = c_start + channels_per_iter
      y = mod(iter, out_image_size) / out_width
      x = mod(iter, out_width)
      in_x = merge(max((real(x, real32) + 0.5_real32) * o2i_fx - 0.5_real32, 0.0_real32), real(x, real32) * o2i_fx, half_pixel_centers)
      in_y = merge(max((real(y, real32) + 0.5_real32) * o2i_fy - 0.5_real32, 0.0_real32), real(y, real32) * o2i_fy, half_pixel_centers)
      in_x0 = int(in_x); in_x1 = min(in_x0 + 1, in_width - 1)
      in_y0 = int(in_y); in_y1 = min(in_y0, in_height - 1); in_y2 = min(in_y0 + 1, in_height - 1)
      in_offset_r0 = c_start * in_image_size + in_y1 * in_width
      in_offset_r1 = c_start * in_image_size + in_y2 * in_width
      out_idx = c_start * out_image_size + y * out_width + x
      do c = c_start, c_end - 1
        v00 = real(input(in_offset_r0 + in_x0), real32); v01 = real(input(in_offset_r0 + in_x1), real32)
        v10 = real(input(in_offset_r1 + in_x0), real32); v11 = real(input(in_offset_r1 + in_x1), real32)
        val = v00 + (in_y-real(in_y0,real32))*(v10-v00) + (in_x-real(in_x0,real32))*(v01-v00) + &
              (in_y-real(in_y0,real32))*(in_x-real(in_x0,real32))*(v11-v01-v10+v00)
        output(out_idx) = input(in_offset_r0 + in_x0)
        in_offset_r0 = in_offset_r0 + in_image_size; in_offset_r1 = in_offset_r1 + in_image_size; out_idx = out_idx + out_image_size
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine resize_bilinear_i16

  subroutine resize_bilinear_i32(output, output_size, out_height, out_width, input, in_height, in_width, o2i_fy, o2i_fx, half_pixel_centers)
    integer(int32), intent(out) :: output(0:)
    integer, intent(in) :: output_size, out_height, out_width, in_height, in_width
    integer(int32), intent(in) :: input(0:)
    real(real32), intent(in) :: o2i_fy, o2i_fx
    logical, intent(in) :: half_pixel_centers
    integer :: iter, iters_required, in_image_size, out_image_size, c_start, c_end, c, y, x, in_x0, in_x1, in_y0, in_y1, in_y2
    integer :: in_offset_r0, in_offset_r1, out_idx
    real(real32) :: in_x, in_y, v00, v01, v10, v11, val
    iters_required = output_size / channels_per_iter
    !$omp target teams distribute parallel do num_teams(29184) thread_limit(256) private(in_image_size,out_image_size,c_start,c_end,c,y,x,in_x0,in_x1,in_y0,in_y1,in_y2,in_offset_r0,in_offset_r1,out_idx,in_x,in_y,v00,v01,v10,v11,val)
    do iter = 0, iters_required - 1
      in_image_size = in_height * in_width
      out_image_size = out_height * out_width
      c_start = (iter / out_image_size) * channels_per_iter
      c_end = c_start + channels_per_iter
      y = mod(iter, out_image_size) / out_width
      x = mod(iter, out_width)
      in_x = merge(max((real(x, real32) + 0.5_real32) * o2i_fx - 0.5_real32, 0.0_real32), real(x, real32) * o2i_fx, half_pixel_centers)
      in_y = merge(max((real(y, real32) + 0.5_real32) * o2i_fy - 0.5_real32, 0.0_real32), real(y, real32) * o2i_fy, half_pixel_centers)
      in_x0 = int(in_x); in_x1 = min(in_x0 + 1, in_width - 1)
      in_y0 = int(in_y); in_y1 = min(in_y0, in_height - 1); in_y2 = min(in_y0 + 1, in_height - 1)
      in_offset_r0 = c_start * in_image_size + in_y1 * in_width
      in_offset_r1 = c_start * in_image_size + in_y2 * in_width
      out_idx = c_start * out_image_size + y * out_width + x
      do c = c_start, c_end - 1
        v00 = real(input(in_offset_r0 + in_x0), real32); v01 = real(input(in_offset_r0 + in_x1), real32)
        v10 = real(input(in_offset_r1 + in_x0), real32); v11 = real(input(in_offset_r1 + in_x1), real32)
        val = v00 + (in_y-real(in_y0,real32))*(v10-v00) + (in_x-real(in_x0,real32))*(v01-v00) + &
              (in_y-real(in_y0,real32))*(in_x-real(in_x0,real32))*(v11-v01-v10+v00)
        output(out_idx) = input(in_offset_r0 + in_x0)
        in_offset_r0 = in_offset_r0 + in_image_size; in_offset_r1 = in_offset_r1 + in_image_size; out_idx = out_idx + out_image_size
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine resize_bilinear_i32

end module resize_mod
