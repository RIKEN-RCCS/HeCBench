module sobel_mod
  use iso_fortran_env, only: int8, real32, real64
  use omp_lib
  implicit none

  type :: uchar4
    integer(int8) :: x, y, z, w
  end type uchar4

  type :: float4
    real(real32) :: x, y, z, w
  end type float4
  integer(int8), parameter :: byte_value(0:255) = [ &
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, &
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, &
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, &
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63, &
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79, &
    80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95, &
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111, &
    112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127, &
    -128,-127,-126,-125,-124,-123,-122,-121,-120,-119,-118,-117,-116,-115,-114,-113, &
    -112,-111,-110,-109,-108,-107,-106,-105,-104,-103,-102,-101,-100,-99,-98,-97, &
    -96,-95,-94,-93,-92,-91,-90,-89,-88,-87,-86,-85,-84,-83,-82,-81, &
    -80,-79,-78,-77,-76,-75,-74,-73,-72,-71,-70,-69,-68,-67,-66,-65, &
    -64,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49, &
    -48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33, &
    -32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17, &
    -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ]
!$omp declare target (byte_value, u8, to_u8, convert_float4, convert_uchar4, add4, sub4, muls4)

contains

  integer function u8(v) result(r)
    integer(int8), intent(in) :: v
    r = int(v)
    if (r < 0) r = r + 256
  end function u8

  integer(int8) function to_u8(v) result(r)
    real(real32), intent(in) :: v
    integer :: iv
    iv = int(max(0.0_real32, min(255.0_real32, v)))
    r = byte_value(iv)
  end function to_u8

  type(float4) function convert_float4(data) result(r)
    type(uchar4), intent(in) :: data
    r = float4(real(u8(data%x), real32), real(u8(data%y), real32), real(u8(data%z), real32), real(u8(data%w), real32))
  end function convert_float4

  type(uchar4) function convert_uchar4(v) result(r)
    type(float4), intent(in) :: v
    r = uchar4(to_u8(v%x), to_u8(v%y), to_u8(v%z), to_u8(v%w))
  end function convert_uchar4

  type(float4) function add4(a,b) result(r)
    type(float4), intent(in) :: a,b
    r = float4(a%x+b%x, a%y+b%y, a%z+b%z, a%w+b%w)
  end function add4

  type(float4) function sub4(a,b) result(r)
    type(float4), intent(in) :: a,b
    r = float4(a%x-b%x, a%y-b%y, a%z-b%z, a%w-b%w)
  end function sub4

  type(float4) function muls4(s,a) result(r)
    real(real32), intent(in) :: s
    type(float4), intent(in) :: a
    r = float4(s*a%x, s*a%y, s*a%z, s*a%w)
  end function muls4

  subroutine sobel_kernel(input_image, output_image, width, height, iterations)
    type(uchar4), intent(in) :: input_image(0:)
    type(uchar4), intent(inout) :: output_image(0:)
    integer, intent(in) :: width, height, iterations
    integer :: iter, x, y, c
    type(float4) :: i00,i01,i02,i10,i12,i20,i21,i22,gx,gy
    do iter = 0, iterations - 1
      !$omp target teams distribute parallel do collapse(2) thread_limit(256) private(c,i00,i01,i02,i10,i12,i20,i21,i22,gx,gy)
      do y = 1, height - 2
        do x = 1, width - 2
          c = x + y * width
          i00 = convert_float4(input_image(c - 1 - width))
          i01 = convert_float4(input_image(c - width))
          i02 = convert_float4(input_image(c + 1 - width))
          i10 = convert_float4(input_image(c - 1))
          i12 = convert_float4(input_image(c + 1))
          i20 = convert_float4(input_image(c - 1 + width))
          i21 = convert_float4(input_image(c + width))
          i22 = convert_float4(input_image(c + 1 + width))
          gx = sub4(sub4(add4(add4(i00, muls4(2.0_real32, i10)), i20), i02), add4(muls4(2.0_real32, i12), i22))
          gy = sub4(sub4(add4(add4(i00, muls4(2.0_real32, i01)), i02), i20), add4(muls4(2.0_real32, i21), i22))
          output_image(c) = convert_uchar4(float4(sqrt(gx%x*gx%x + gy%x*gy%x)/2.0_real32, &
                                                  sqrt(gx%y*gx%y + gy%y*gy%y)/2.0_real32, &
                                                  sqrt(gx%z*gx%z + gy%z*gy%z)/2.0_real32, &
                                                  sqrt(gx%w*gx%w + gy%w*gy%w)/2.0_real32))
        end do
      end do
      !$omp end target teams distribute parallel do
    end do
  end subroutine sobel_kernel

  subroutine reference(verification_output, input_image, width, height)
    type(uchar4), intent(out) :: verification_output(0:)
    type(uchar4), intent(in) :: input_image(0:)
    integer, intent(in) :: width, height
    integer :: x, y, c
    type(float4) :: i00,i01,i02,i10,i12,i20,i21,i22,gx,gy
    do y = 0, height - 1
      do x = 0, width - 1
        if (x >= 1 .and. x < width - 1 .and. y >= 1 .and. y < height - 1) then
          c = x + y * width
          i00 = convert_float4(input_image(c - 1 - width)); i01 = convert_float4(input_image(c - width)); i02 = convert_float4(input_image(c + 1 - width))
          i10 = convert_float4(input_image(c - 1)); i12 = convert_float4(input_image(c + 1))
          i20 = convert_float4(input_image(c - 1 + width)); i21 = convert_float4(input_image(c + width)); i22 = convert_float4(input_image(c + 1 + width))
          gx = sub4(sub4(add4(add4(i00, muls4(2.0_real32, i10)), i20), i02), add4(muls4(2.0_real32, i12), i22))
          gy = sub4(sub4(add4(add4(i00, muls4(2.0_real32, i01)), i02), i20), add4(muls4(2.0_real32, i21), i22))
          verification_output(c) = convert_uchar4(float4(sqrt(gx%x*gx%x + gy%x*gy%x)/2.0_real32, sqrt(gx%y*gx%y + gy%y*gy%y)/2.0_real32, &
                                                  sqrt(gx%z*gx%z + gy%z*gy%z)/2.0_real32, sqrt(gx%w*gx%w + gy%w*gy%w)/2.0_real32))
        end if
      end do
    end do
  end subroutine reference

  logical function compare(ref_data, data, length, epsilon) result(ok)
    real(real32), intent(in) :: ref_data(0:), data(0:), epsilon
    integer, intent(in) :: length
    integer :: i
    real(real32) :: error, ref, diff, norm_ref, norm_error
    error = 0.0_real32
    ref = 0.0_real32
    do i = 1, length - 1
      diff = ref_data(i) - data(i)
      error = error + diff * diff
      ref = ref + ref_data(i) * ref_data(i)
    end do
    norm_ref = sqrt(ref)
    if (abs(ref) < 1.0e-7_real32) then
      ok = .false.
      return
    end if
    norm_error = sqrt(error)
    ok = (norm_error / norm_ref) < epsilon
  end function compare

end module sobel_mod
