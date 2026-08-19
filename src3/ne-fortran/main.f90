module ne_support
  use iso_fortran_env, only: int32, int64, real32, real64
  use iso_c_binding, only: c_float
  implicit none
  type, bind(C) :: float3
    real(c_float) :: x, y, z, padding
  end type
  type, bind(C) :: float4
    real(c_float) :: x, y, z, w
  end type
!$omp declare target(sub3,dot3,scale3,cross3,len3,normalize3,normal_estimate,ieee_is_nan_real)
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

  pure function sub3(a, b) result(c)
    type(float3), intent(in) :: a, b
    type(float3) :: c
    c%x = a%x - b%x; c%y = a%y - b%y; c%z = a%z - b%z
  end function

  pure real(real32) function dot3(a, b) result(v)
    type(float3), intent(in) :: a, b
    v = a%x*b%x + a%y*b%y + a%z*b%z
  end function

  pure function scale3(a, b) result(c)
    type(float3), intent(in) :: a
    real(real32), intent(in) :: b
    type(float3) :: c
    c%x = a%x*b; c%y = a%y*b; c%z = a%z*b
  end function

  pure function cross3(a, b) result(c)
    type(float3), intent(in) :: a, b
    type(float3) :: c
    c%x = a%y*b%z - a%z*b%y
    c%y = a%z*b%x - a%x*b%z
    c%z = a%x*b%y - a%y*b%x
  end function

  pure real(real32) function len3(a) result(v)
    type(float3), intent(in) :: a
    v = sqrt(dot3(a, a))
  end function

  pure function normalize3(a) result(c)
    type(float3), intent(in) :: a
    type(float3) :: c
    c = scale3(a, 1.0_real32 / sqrt(dot3(a, a)))
  end function

  pure function normal_estimate(points, idx, width, height) result(res)
    type(float3), intent(in) :: points(0:*)
    integer, intent(in) :: idx, width, height
    type(float4) :: res
    type(float3) :: query_pt, horiz, vert, normal, mc
    integer :: xidx, yidx
    logical :: west_valid, east_valid, north_valid, south_valid
    real(real32) :: curvature

    query_pt = points(idx)
    if (ieee_is_nan_real(query_pt%z)) then
      res = float4(0.0_real32, 0.0_real32, 0.0_real32, 0.0_real32)
      return
    end if

    xidx = mod(idx, width)
    yidx = idx / width
    west_valid = .false.
    if (xidx > 1) west_valid = (.not. ieee_is_nan_real(points(idx-1)%z)) .and. &
      abs(points(idx-1)%z - query_pt%z) < 200.0_real32
    east_valid = .false.
    if (xidx < width-1) east_valid = (.not. ieee_is_nan_real(points(idx+1)%z)) .and. &
      abs(points(idx+1)%z - query_pt%z) < 200.0_real32
    north_valid = .false.
    if (yidx > 1) north_valid = (.not. ieee_is_nan_real(points(idx-width)%z)) .and. &
      abs(points(idx-width)%z - query_pt%z) < 200.0_real32
    south_valid = .false.
    if (yidx < height-1) south_valid = (.not. ieee_is_nan_real(points(idx+width)%z)) .and. &
      abs(points(idx+width)%z - query_pt%z) < 200.0_real32

    if (west_valid .and. east_valid) horiz = sub3(points(idx+1), points(idx-1))
    if (west_valid .and. (.not. east_valid)) horiz = sub3(points(idx), points(idx-1))
    if ((.not. west_valid) .and. east_valid) horiz = sub3(points(idx+1), points(idx))
    if ((.not. west_valid) .and. (.not. east_valid)) then
      res = float4(0.0_real32, 0.0_real32, 0.0_real32, 1.0_real32)
      return
    end if

    if (south_valid .and. north_valid) vert = sub3(points(idx-width), points(idx+width))
    if (south_valid .and. (.not. north_valid)) vert = sub3(points(idx), points(idx+width))
    if ((.not. south_valid) .and. north_valid) vert = sub3(points(idx-width), points(idx))
    if ((.not. south_valid) .and. (.not. north_valid)) then
      res = float4(0.0_real32, 0.0_real32, 0.0_real32, 1.0_real32)
      return
    end if

    normal = cross3(horiz, vert)
    curvature = merge(1.0_real32, 0.0_real32, abs(horiz%z) > 0.04_real32 .or. &
      abs(vert%z) > 0.04_real32 .or. (.not. west_valid) .or. (.not. east_valid) .or. &
      (.not. north_valid) .or. (.not. south_valid))
    mc = normalize3(normal)
    if (dot3(query_pt, mc) > 0.0_real32) mc = scale3(mc, -1.0_real32)
    res = float4(mc%x, mc%y, mc%z, curvature)
  end function

  pure logical function ieee_is_nan_real(x) result(v)
    real(real32), intent(in) :: x
    v = x /= x
  end function
end module

program main
  use ne_support
  implicit none
  integer :: width, height, repeat, num_pts, idx, r, ios
  character(len=64) :: arg
  type(float3), allocatable :: points(:)
  type(float4), allocatable :: normal_points(:)
  real(real64) :: t0, t1
  real(real32) :: sx, sy, sz, sw

  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: main <width> <height> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=ios) width
  call get_command_argument(2, arg); read(arg, *, iostat=ios) height
  call get_command_argument(3, arg); read(arg, *, iostat=ios) repeat

  num_pts = width * height
  allocate(points(0:num_pts-1), normal_points(0:num_pts-1))
  call c_srand(123_int32)
  do idx = 0, num_pts-1
    points(idx)%x = real(mod(c_rand(), width), real32)
    points(idx)%y = real(mod(c_rand(), height), real32)
    points(idx)%z = real(mod(c_rand(), 256), real32)
    points(idx)%padding = 0.0_real32
  end do

  !$omp target data map(to:points(0:num_pts-1)) map(from:normal_points(0:num_pts-1))
  t0 = seconds()
  do r = 1, repeat
    !$omp target teams distribute parallel do thread_limit(256) &
    !$omp& map(to:points(0:num_pts-1)) map(from:normal_points(0:num_pts-1))
    do idx = 0, num_pts-1
      normal_points(idx) = normal_estimate(points, idx, width, height)
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time: ', (t1-t0)/real(repeat,real64), ' (s)'
  !$omp end target data

  sx = 0.0_real32; sy = 0.0_real32; sz = 0.0_real32; sw = 0.0_real32
  do idx = 0, num_pts-1
    sx = sx + normal_points(idx)%x
    sy = sy + normal_points(idx)%y
    sz = sz + normal_points(idx)%z
    sw = sw + normal_points(idx)%w
  end do
  print '(a,f0.6,a,f0.6,a,f0.6,a,f0.6)', 'Checksum: x=', sx, ' y=', sy, ' z=', sz, ' w=', sw
end program
