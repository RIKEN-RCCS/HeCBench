module winograd_utils
  use iso_fortran_env, only: real32, real64, int64
  implicit none
  integer, parameter :: map_size = 1024
  integer, parameter :: dim_local_work_group_x = 32
  integer, parameter :: dim_local_work_group_y = 8
  real(real32), parameter :: percent_diff_error_threshold = 1.05_real32
  real(real32), parameter :: small_float_val = 1.0e-8_real32
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine filter_transformation(tf)
    real(real32), intent(out) :: tf(0:15)
    real(real32) :: filter(0:2,0:2), tmp(0:3,0:2)
    integer :: i, j
    filter(0,0)=0.2_real32; filter(1,0)=0.5_real32; filter(2,0)=-0.8_real32
    filter(0,1)=-0.3_real32; filter(1,1)=0.6_real32; filter(2,1)=-0.9_real32
    filter(0,2)=0.4_real32; filter(1,2)=0.7_real32; filter(2,2)=0.10_real32
    do j = 0, 2
      tmp(0,j) = filter(0,j)
      tmp(1,j) = 0.5_real32 * filter(0,j) + 0.5_real32 * filter(1,j) + 0.5_real32 * filter(2,j)
      tmp(2,j) = 0.5_real32 * filter(0,j) - 0.5_real32 * filter(1,j) + 0.5_real32 * filter(2,j)
      tmp(3,j) = filter(2,j)
    end do
    do i = 0, 3
      tf(i*4+0) = tmp(i,0)
      tf(i*4+1) = 0.5_real32 * tmp(i,0) + 0.5_real32 * tmp(i,1) + 0.5_real32 * tmp(i,2)
      tf(i*4+2) = 0.5_real32 * tmp(i,0) - 0.5_real32 * tmp(i,1) + 0.5_real32 * tmp(i,2)
      tf(i*4+3) = tmp(i,2)
    end do
  end subroutine filter_transformation

  subroutine winograd_tile(input, output, tf, tile_i, tile_j, offset_i, offset_j)
    real(real32), intent(in) :: input(0:), tf(0:)
    real(real32), intent(inout) :: output(0:)
    integer, intent(in) :: tile_i, tile_j, offset_i, offset_j
    integer :: i, j, x, y, out_map_size
    real(real32) :: input_tile(0:3,0:3), tmp_tile(0:3,0:3), transformed_tile(0:3,0:3)
    real(real32) :: multiplied_tile(0:3,0:3), tmp_tile_1(0:1,0:3), final_tile(0:1,0:1)
    out_map_size = map_size - 2
    do i = 0, 3
      do j = 0, 3
        x = 2 * (tile_i + offset_i) + i
        y = 2 * (tile_j + offset_j) + j
        if (x >= map_size .or. y >= map_size) then
          input_tile(i,j) = 0.0_real32
        else
          input_tile(i,j) = input(x * map_size + y)
        end if
      end do
    end do
    do j = 0, 3
      tmp_tile(0,j) = input_tile(0,j) - input_tile(2,j)
      tmp_tile(1,j) = input_tile(1,j) + input_tile(2,j)
      tmp_tile(2,j) = -input_tile(1,j) + input_tile(2,j)
      tmp_tile(3,j) = input_tile(1,j) - input_tile(3,j)
    end do
    do i = 0, 3
      transformed_tile(i,0) = tmp_tile(i,0) - tmp_tile(i,2)
      transformed_tile(i,1) = tmp_tile(i,1) + tmp_tile(i,2)
      transformed_tile(i,2) = -tmp_tile(i,1) + tmp_tile(i,2)
      transformed_tile(i,3) = tmp_tile(i,1) - tmp_tile(i,3)
    end do
    do i = 0, 3
      do j = 0, 3
        multiplied_tile(i,j) = transformed_tile(i,j) * tf(i*4+j)
      end do
    end do
    do j = 0, 3
      tmp_tile_1(0,j) = multiplied_tile(0,j) + multiplied_tile(1,j) + multiplied_tile(2,j)
      tmp_tile_1(1,j) = multiplied_tile(1,j) - multiplied_tile(2,j) - multiplied_tile(3,j)
    end do
    do i = 0, 1
      final_tile(i,0) = tmp_tile_1(i,0) + tmp_tile_1(i,1) + tmp_tile_1(i,2)
      final_tile(i,1) = tmp_tile_1(i,1) - tmp_tile_1(i,2) - tmp_tile_1(i,3)
    end do
    do i = 0, 1
      do j = 0, 1
        x = 2 * (tile_i + offset_i) + i
        y = 2 * (tile_j + offset_j) + j
        if (x < out_map_size .and. y < out_map_size) output(x * out_map_size + y) = final_tile(i,j)
      end do
    end do
  end subroutine winograd_tile

  subroutine winograd_cpu(input, output, tf, cpu_global_size)
    real(real32), intent(in) :: input(0:), tf(0:)
    real(real32), intent(inout) :: output(0:)
    integer, intent(in) :: cpu_global_size(0:1)
    integer :: tile_i, tile_j, tile_n
    tile_n = (map_size - 2 + 1) / 2
!$omp parallel do private(tile_i,tile_j)
    do tile_i = 0, cpu_global_size(0) - 1
      do tile_j = 0, tile_n - 1
        call winograd_tile(input, output, tf, tile_i, tile_j, 0, 0)
      end do
    end do
!$omp end parallel do
  end subroutine winograd_cpu

  subroutine winograd_reference(input, output, tf)
    real(real32), intent(in) :: input(0:), tf(0:)
    real(real32), intent(out) :: output(0:)
    integer :: tile_i, tile_j, tile_n
    tile_n = (map_size - 2 + 1) / 2
    do tile_i = 0, tile_n - 1
      do tile_j = 0, tile_n - 1
        call winograd_tile(input, output, tf, tile_i, tile_j, 0, 0)
      end do
    end do
  end subroutine winograd_reference

  logical function compare_results(a, b)
    real(real32), intent(in) :: a(0:), b(0:)
    integer :: i
    compare_results = .true.
    do i = 0, (map_size - 2) * (map_size - 2) - 1
      if (percent_diff(a(i), b(i)) > percent_diff_error_threshold) then
        compare_results = .false.
        return
      end if
    end do
  end function compare_results

  pure real(real32) function percent_diff(v1, v2)
    real(real32), intent(in) :: v1, v2
    if (abs(v1) < 0.01_real32 .and. abs(v2) < 0.01_real32) then
      percent_diff = 0.0_real32
    else
      percent_diff = 100.0_real32 * abs(abs(v1 - v2) / abs(v1 + small_float_val))
    end if
  end function percent_diff
end module winograd_utils
