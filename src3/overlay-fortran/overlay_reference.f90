module overlay_reference
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  type :: float3
    real(real32) :: x, y, z
  end type
  type :: float4
    real(real32) :: x, y, z, w
  end type
  type :: box
    integer(int32) :: width, height, left, top
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine detection_overlay_box(input, output, img_width, img_height, x0, y0, box_width, box_height, color)
    type(float3), intent(in) :: input(0:)
    type(float3), intent(inout) :: output(0:)
    integer, intent(in) :: img_width, img_height, x0, y0, box_width, box_height
    type(float4), intent(in) :: color
    integer :: box_x, box_y, x, y
    type(float3) :: px
    real(real32) :: alpha, ialph
    !$omp target teams distribute parallel do collapse(2) private(x,y,px,alpha,ialph) thread_limit(64)
    do box_y = 0, box_height-1
      do box_x = 0, box_width-1
        x = box_x + x0
        y = box_y + y0
        if (x < img_width .and. y < img_height) then
          px = input(y*img_width+x)
          alpha = color%w / 255.0_real32
          ialph = 1.0_real32 - alpha
          px%x = alpha * color%x + ialph * px%x
          px%y = alpha * color%y + ialph * px%y
          px%z = alpha * color%z + ialph * px%z
          output(y*img_width+x) = px
        end if
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine cpu_detection_overlay_box(input, output, img_width, img_height, x0, y0, box_width, box_height, color)
    type(float3), intent(in) :: input(0:)
    type(float3), intent(inout) :: output(0:)
    integer, intent(in) :: img_width, img_height, x0, y0, box_width, box_height
    type(float4), intent(in) :: color
    integer :: box_x, box_y, x, y
    type(float3) :: px
    real(real32) :: alpha, ialph
    do box_y = 0, box_height-1
      do box_x = 0, box_width-1
        x = box_x + x0; y = box_y + y0
        if (x < img_width .and. y < img_height) then
          px = input(y*img_width+x)
          alpha = color%w / 255.0_real32; ialph = 1.0_real32 - alpha
          px%x = alpha*color%x + ialph*px%x
          px%y = alpha*color%y + ialph*px%y
          px%z = alpha*color%z + ialph*px%z
          output(y*img_width+x) = px
        end if
      end do
    end do
  end subroutine
end module
