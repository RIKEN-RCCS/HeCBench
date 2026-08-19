program main
  use overlay_reference
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
  integer :: width, height, img_size, num_detections, i, n, ios
  character(len=64) :: arg
  type(float3), allocatable :: input(:), output(:), ref_output(:)
  type(box), allocatable :: detections(:)
  type(float4) :: colors
  real(real64) :: t0, t1
  logical :: ok
  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: main <width> <height>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) width
  call get_command_argument(2,arg); read(arg,*,iostat=ios) height
  img_size = width * height
  allocate(input(0:img_size-1), output(0:img_size-1), ref_output(0:img_size-1))
  call c_srand(123_int32)
  do i = 0, img_size-1
    input(i)%x = real(mod(c_rand(),256),real32); output(i)%x = input(i)%x; ref_output(i)%x = input(i)%x
    input(i)%y = real(mod(c_rand(),256),real32); output(i)%y = input(i)%y; ref_output(i)%y = input(i)%y
    input(i)%z = real(mod(c_rand(),256),real32); output(i)%z = input(i)%z; ref_output(i)%z = input(i)%z
  end do
  colors = float4(255.0_real32, 204.0_real32, 203.0_real32, 1.0_real32)
  num_detections = int(real(img_size,real32) * 0.8_real32)
  allocate(detections(0:num_detections-1))
  do i = 0, num_detections-1
    detections(i)%width = 64 + mod(c_rand(), 128)
    detections(i)%height = 64 + mod(c_rand(), 128)
    detections(i)%left = mod(c_rand(), width - 64)
    detections(i)%top = mod(c_rand(), height - 64)
  end do
  !$omp target data map(to:input(0:img_size-1)) map(tofrom:output(0:img_size-1))
  t0 = seconds()
  do n = 0, num_detections-1
    call detection_overlay_box(input, output, width, height, detections(n)%left, detections(n)%top, &
      detections(n)%width, detections(n)%height, colors)
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Total kernel execution time: ', t1-t0, ' (s)'
  !$omp end target data
  do n = 0, num_detections-1
    call cpu_detection_overlay_box(input, ref_output, width, height, detections(n)%left, detections(n)%top, &
      detections(n)%width, detections(n)%height, colors)
  end do
  ok = .true.
  do i = 0, img_size-1
    if (abs(ref_output(i)%x-output(i)%x) > 1.0e-3_real32 .or. abs(ref_output(i)%y-output(i)%y) > 1.0e-3_real32 .or. &
        abs(ref_output(i)%z-output(i)%z) > 1.0e-3_real32) then
      print '(a,i0)', 'Error at index ', i
      ok = .false.; exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
end program
