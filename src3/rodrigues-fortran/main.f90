program rodrigues
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use rodrigues_mod
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
  end interface

  integer :: argc, n, repeat, i
  real(real32), parameter :: wx = -0.3_real32, wy = -0.6_real32, wz = 0.15_real32
  real(real32) :: norm, angle, a, b, c, den
  type(float3) :: w
  type(float3), allocatable :: h(:)
  type(float4), allocatable :: h2(:)
  real(real64) :: start_time, end_time
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <number of points> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *) n
  call get_command_argument(2, arg)
  read(arg, *) repeat

  norm = 1.0_real32 / sqrt(wx * wx + wy * wy + wz * wz)
  w = float3(wx * norm, wy * norm, wz * norm)
  angle = 0.5_real32
  allocate(h(0:n-1), h2(0:n-1))

  call c_srand(123_c_int)
  do i = 0, n - 1
    a = real(c_rand(), real32)
    b = real(c_rand(), real32)
    c = real(c_rand(), real32)
    den = sqrt(a * a + b * b + c * c)
    h(i) = float3(a / den, b / den, c / den)
    h2(i) = float4(a / den, b / den, c / den, 0.0_real32)
  end do

  !$omp target data map(to: h(0:n-1), h2(0:n-1))
    start_time = omp_get_wtime()
    do i = 0, repeat - 1
      call rotate(n, angle, w, h)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time (float3): ', ((end_time - start_time) * 1.0e6_real64) / repeat, ' (us)'

    start_time = omp_get_wtime()
    do i = 0, repeat - 1
      call rotate2(n, angle, w, h2)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time (float4): ', ((end_time - start_time) * 1.0e6_real64) / repeat, ' (us)'
  !$omp end target data
end program rodrigues
