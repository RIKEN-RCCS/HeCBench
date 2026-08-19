program rtm8
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use rtm8_mod
  implicit none

  integer :: argc, repeat, array_size, x, y, z, idx, t, i
  real(real32), allocatable :: next_s(:), current_s(:), next_r(:), current_r(:), vsq(:), image_gpu(:), image_cpu(:)
  real(real32) :: a(0:4)
  real(real64) :: pts, flops, pt_rate, flop_rate, speedup, memory, t0, t1, dt
  logical :: ok
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) repeat

  array_size = nx * ny * nz
  allocate(next_s(0:array_size-1), current_s(0:array_size-1), next_r(0:array_size-1), current_r(0:array_size-1), &
           vsq(0:array_size-1), image_gpu(0:array_size-1), image_cpu(0:array_size-1))

  memory = real(array_size, real64) * 4.0_real64 * 6.0_real64
  pts = real(repeat, real64) * real(nx - 8, real64) * real(ny - 8, real64) * real(nz - 8, real64)
  flops = 67.0_real64 * pts
  print '(a,f12.6)', 'memory (MB) = ', memory / 1.0e6_real64
  print '(a,f12.6)', 'pts (billions) = ', pts / 1.0e9_real64
  print '(a,f12.6)', 'Tflops = ', flops / 1.0e12_real64

  a = [ -1.0_real32/560.0_real32, 8.0_real32/315.0_real32, -0.2_real32, 1.6_real32, -1435.0_real32/504.0_real32 ]
  do z = 0, nz - 1
    do y = 0, ny - 1
      do x = 0, nx - 1
        idx = index_to_1d(x, y, z)
        vsq(idx) = 1.0_real32
        next_s(idx) = 0.0_real32
        current_s(idx) = 1.0_real32
        next_r(idx) = 0.0_real32
        current_r(idx) = 1.0_real32
        image_gpu(idx) = 0.5_real32
        image_cpu(idx) = 0.5_real32
      end do
    end do
  end do

  !$omp target data map(to: current_s(0:array_size-1), current_r(0:array_size-1), a(0:4), vsq(0:array_size-1)) &
  !$omp& map(alloc: next_r(0:array_size-1), next_s(0:array_size-1)) map(tofrom: image_gpu(0:array_size-1))
    t0 = omp_get_wtime()
    do t = 0, repeat - 1
      call rtm8_kernel(vsq, current_s, current_r, next_s, next_r, image_gpu, a)
    end do
    t1 = omp_get_wtime()
    dt = t1 - t0
  !$omp end target data

  t0 = omp_get_wtime()
  do t = 0, repeat - 1
    call rtm8_cpu(vsq, current_s, next_s, current_r, next_r, image_cpu, a, array_size)
  end do
  t1 = omp_get_wtime()

  ok = .true.
  do i = 0, array_size - 1
    if (abs(image_cpu(i) - image_gpu(i)) > 0.1_real32) then
      print '(a,i0,a,f10.4,a,f10.4)', '@index ', i, ' host: ', image_cpu(i), ' device ', image_gpu(i)
      ok = .false.
      exit
    end if
  end do
  if (ok) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if

  pt_rate = pts / dt
  flop_rate = flops / dt
  speedup = (t1 - t0) / dt
  print '(a,f12.6)', 'dt = ', dt
  print '(a,f12.6)', 'pt_rate (millions/sec) = ', pt_rate / 1.0e6_real64
  print '(a,f12.6)', 'flop_rate (Gflops) = ', flop_rate / 1.0e9_real64
  print '(a,f12.6)', 'speedup over cpu = ', speedup
  print '(a,f12.6,a)', 'average kernel execution time = ', dt / repeat, ' (s)'
end program rtm8
