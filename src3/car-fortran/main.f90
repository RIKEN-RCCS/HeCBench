module car_support
  use iso_fortran_env, only: int32, int64, real64
  implicit none

  integer(int64), parameter :: modulus_minus_one = 9223372036854775807_int64
  integer(int64), parameter :: multiplier = 2806196910506780709_int64

contains

  pure integer(int64) function add_mod_2_63(left, right)
    integer(int64), intent(in) :: left, right

    if (left > modulus_minus_one - right) then
      add_mod_2_63 = left - (modulus_minus_one - right) - 1_int64
    else
      add_mod_2_63 = left + right
    end if
  end function add_mod_2_63

  pure integer(int64) function multiply_mod_2_63(left, right)
    integer(int64), intent(in) :: left, right
    integer(int64) :: accumulator, factor
    integer :: bit

    accumulator = 0_int64
    factor = left
    do bit = 0, 62
      if (btest(right, bit)) accumulator = add_mod_2_63(accumulator, factor)
      if (bit < 62) factor = add_mod_2_63(factor, factor)
    end do
    multiply_mod_2_63 = accumulator
  end function multiply_mod_2_63

  subroutine lcg_random_double(seed, value)
    integer(int64), intent(inout) :: seed
    real(real64), intent(out) :: value

    seed = add_mod_2_63(multiply_mod_2_63(multiplier, seed), 1_int64)
    value = real(seed, real64) / 9223372036854775808.0_real64
  end subroutine lcg_random_double
end module car_support

program car
  use iso_fortran_env, only: int32, int64, real32, real64
  use omp_lib, only: omp_get_wtime
  use car_support, only: lcg_random_double
  implicit none

  integer(int32), parameter :: dim_b = 128_int32
  integer(int32), parameter :: dim_c = 3_int32
  integer(int32), parameter :: dim_h = 480_int32
  integer(int32), parameter :: dim_w = 640_int32
  integer(int32), parameter :: kernels_size = 9_int32
  integer(int32), parameter :: img_w = 1024_int32
  integer(int32), parameter :: img_h = 1024_int32
  integer(int32), parameter :: padding = 1_int32

  integer(int32) :: repeat, k_size, width_without_padding, height_without_padding
  integer(int32) :: idb, idc, idy, idx, k_y, k_x
  integer(int32) :: x_left, x_right, y_top, y_bottom, i
  integer(int64) :: image_size, offset_size, kernel_size, output_size, seed
  real(real64) :: start_time, end_time, random_value
  real(real32) :: result, offset_h_value, offset_v_value, p_x, p_y
  real(real32) :: alpha, beta, value, rmse
  real(real32), allocatable :: img(:), offsets_h(:), offsets_v(:)
  real(real32), allocatable :: kernel(:), output(:), output_ref(:)
  character(len=128) :: argument

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument, *) repeat

  k_size = int(sqrt(real(kernels_size, real32)), int32)
  width_without_padding = img_w - 2_int32 * padding
  height_without_padding = img_h - 2_int32 * padding
  image_size = int(dim_b, int64) * dim_c * (img_w + padding) * (img_h + padding)
  offset_size = int(dim_b, int64) * kernels_size * dim_w * dim_h
  kernel_size = offset_size
  output_size = int(dim_b, int64) * dim_c * dim_w * dim_h

  allocate(img(0:image_size-1), offsets_h(0:offset_size-1), offsets_v(0:offset_size-1))
  allocate(kernel(0:kernel_size-1), output(0:output_size-1), output_ref(0:output_size-1))

  seed = 123_int64
  do i = 0, int(image_size, int32) - 1
    call lcg_random_double(seed, random_value)
    img(i) = real(int(256.0_real64 * random_value, int32), real32)
  end do
  do i = 0, int(kernel_size, int32) - 1
    call lcg_random_double(seed, random_value)
    kernel(i) = real(int(256.0_real64 * random_value, int32), real32)
  end do
  do i = 0, int(offset_size, int32) - 1
    call lcg_random_double(seed, random_value)
    offsets_h(i) = real(random_value, real32)
    call lcg_random_double(seed, random_value)
    offsets_v(i) = real(random_value, real32)
  end do

!$omp target data map(to: img(0:image_size-1), offsets_h(0:offset_size-1), &
!$omp& offsets_v(0:offset_size-1), kernel(0:kernel_size-1)) map(from: output(0:output_size-1))
  start_time = omp_get_wtime()
  do i = 0, repeat - 1
!$omp target teams distribute parallel do collapse(4) thread_limit(256) &
!$omp& private(result, offset_h_value, offset_v_value, p_x, p_y, alpha, beta, x_left, x_right, y_top, y_bottom, value, k_y, k_x)
    do idb = 0, dim_b - 1
      do idc = 0, dim_c - 1
        do idy = 0, dim_h - 1
          do idx = 0, dim_w - 1
            result = 0.0_real32
            do k_y = 0, k_size - 1
              do k_x = 0, k_size - 1
                offset_h_value = offsets_h(idb*k_size*k_size*dim_w*dim_h + (k_size*k_y+k_x)*dim_w*dim_h + idy*dim_w + idx)
                offset_v_value = offsets_v(idb*k_size*k_size*dim_w*dim_h + (k_size*k_y+k_x)*dim_w*dim_h + idy*dim_w + idx)
                p_x = (real(idx, real32) + 0.5_real32) / real(dim_w, real32) * real(width_without_padding, real32) + &
                  real(k_x, real32) + offset_h_value - 0.5_real32
                p_y = (real(idy, real32) + 0.5_real32) / real(dim_h, real32) * real(height_without_padding, real32) + &
                  real(k_y, real32) + offset_v_value - 0.5_real32
                alpha = p_x - floor(p_x)
                beta = p_y - floor(p_y)
                x_left = max(min(int(floor(p_x), int32), width_without_padding + 2*padding - 1), 0_int32)
                x_right = max(min(x_left + 1, width_without_padding + 2*padding - 1), 0_int32)
                y_top = max(min(int(floor(p_y), int32), height_without_padding + 2*padding - 1), 0_int32)
                y_bottom = max(min(y_top + 1, height_without_padding + 2*padding - 1), 0_int32)
                value = (1.0_real32-alpha)*(1.0_real32-beta)*img(idb*dim_c*img_w*img_h + idc*img_w*img_h + y_top*img_w + x_left)
                value = value + alpha*(1.0_real32-beta)*img(idb*dim_c*img_w*img_h + idc*img_w*img_h + y_top*img_w + x_right)
                value = value + (1.0_real32-alpha)*beta*img(idb*dim_c*img_w*img_h + idc*img_w*img_h + y_bottom*img_w + x_left)
                value = value + alpha*beta*img(idb*dim_c*img_w*img_h + idc*img_w*img_h + y_bottom*img_w + x_right)
                result = result + value * kernel(idb*k_size*k_size*dim_w*dim_h + (k_size*k_y+k_x)*dim_w*dim_h + idy*dim_w + idx)
              end do
            end do
            output(idb*dim_c*dim_w*dim_h + idc*dim_w*dim_h + idy*dim_w + idx) = result
          end do
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  end_time = omp_get_wtime()
  print '(a,f0.6,a)', 'Average kernel execution time ', &
    real((end_time-start_time)/real(repeat, real64), real32), ' (s)'
!$omp end target data

  call car_reference(img, kernel, offsets_h, offsets_v, output_ref, k_size, width_without_padding, height_without_padding)
  rmse = 0.0_real32
  do i = 0, int(output_size, int32) - 1
    rmse = rmse + (output_ref(i) - output(i)) * (output_ref(i) - output(i))
  end do
  print '(a,f0.6)', 'RMSE: ', sqrt(rmse / real(output_size, real32))

  deallocate(img, offsets_h, offsets_v, kernel, output, output_ref)

contains

  subroutine car_reference(source_img, source_kernel, source_offsets_h, source_offsets_v, destination, root_kernel_size, width, height)
    real(real32), intent(in) :: source_img(0:), source_kernel(0:)
    real(real32), intent(in) :: source_offsets_h(0:), source_offsets_v(0:)
    real(real32), intent(out) :: destination(0:)
    integer(int32), intent(in) :: root_kernel_size, width, height
    integer(int32) :: b, c, row, column, ky, kx, xl, xr, yt, yb
    real(real32) :: total, oh, ov, px, py, a, bta, sampled

    do b = 0, dim_b - 1
      do c = 0, dim_c - 1
        do row = 0, dim_h - 1
          do column = 0, dim_w - 1
            total = 0.0_real32
            do ky = 0, root_kernel_size - 1
              do kx = 0, root_kernel_size - 1
                oh = source_offsets_h(b*root_kernel_size*root_kernel_size*dim_w*dim_h + (root_kernel_size*ky+kx)*dim_w*dim_h + row*dim_w + column)
                ov = source_offsets_v(b*root_kernel_size*root_kernel_size*dim_w*dim_h + (root_kernel_size*ky+kx)*dim_w*dim_h + row*dim_w + column)
                px = (real(column,real32)+0.5_real32)/real(dim_w,real32)*real(width,real32) + real(kx,real32) + oh - 0.5_real32
                py = (real(row,real32)+0.5_real32)/real(dim_h,real32)*real(height,real32) + real(ky,real32) + ov - 0.5_real32
                a = px - floor(px)
                bta = py - floor(py)
                xl = max(min(int(floor(px),int32), width+2*padding-1), 0_int32)
                xr = max(min(xl+1, width+2*padding-1), 0_int32)
                yt = max(min(int(floor(py),int32), height+2*padding-1), 0_int32)
                yb = max(min(yt+1, height+2*padding-1), 0_int32)
                sampled = (1.0_real32-a)*(1.0_real32-bta)*source_img(b*dim_c*img_w*img_h + c*img_w*img_h + yt*img_w + xl)
                sampled = sampled + a*(1.0_real32-bta)*source_img(b*dim_c*img_w*img_h + c*img_w*img_h + yt*img_w + xr)
                sampled = sampled + (1.0_real32-a)*bta*source_img(b*dim_c*img_w*img_h + c*img_w*img_h + yb*img_w + xl)
                sampled = sampled + a*bta*source_img(b*dim_c*img_w*img_h + c*img_w*img_h + yb*img_w + xr)
                total = total + sampled * source_kernel(b*root_kernel_size*root_kernel_size*dim_w*dim_h + &
                  (root_kernel_size*ky+kx)*dim_w*dim_h + row*dim_w + column)
              end do
            end do
            destination(b*dim_c*dim_w*dim_h + c*dim_w*dim_h + row*dim_w + column) = total
          end do
        end do
      end do
    end do
  end subroutine car_reference
end program car
