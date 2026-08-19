program main
  use mt_types
  use genmtrand_mod
  use mt_kernel_mod
  use mt_gold_mod
  implicit none
  integer :: num_iterations, ios, i, j
  integer, parameter :: global_work_size = mt_rng_count, local_work_size = 128
  integer(int32), parameter :: seed = 777_int32
  integer, parameter :: n_per_rng = 5860
  integer, parameter :: n_rand = mt_rng_count * n_per_rng
  character(len=64) :: arg
  character(len=*), parameter :: dat_path = '../../src/mt-omp/data/MersenneTwister.dat'
  type(mt_struct_stripped), allocatable :: h_mt(:)
  real(real32), allocatable :: h_rand_gpu(:), h_rand_cpu(:)
  real(real64) :: t0, t1, gpu_time, sum_delta, sum_ref, r_cpu, r_gpu, delta, l1norm
  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: main <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) num_iterations
  print '(a)', 'Initialization: load MT parameters and init host buffers...'
  allocate(h_mt(0:mt_rng_count-1), h_rand_gpu(0:n_rand-1), h_rand_cpu(0:n_rand-1))
  call load_mt_gpu(dat_path, seed, h_mt)
  print '(a)', 'Allocate memory...'
  !$omp target data map(to:h_mt(0:mt_rng_count-1)) map(alloc:h_rand_gpu(0:n_rand-1))
  print '(a,i0,a)', 'Call Mersenne Twister kernel... (', num_iterations, ' iterations)'
  print *
  t0 = seconds()
  do i = 1, num_iterations
    call mersenne_kernel(h_mt, h_rand_gpu, n_per_rng, global_work_size, local_work_size)
    call box_muller_kernel(h_rand_gpu, n_per_rng, global_work_size, local_work_size)
  end do
  t1 = seconds()
  gpu_time = (t1 - t0) / real(num_iterations, real64)
  print '(a,f8.4,a,f8.5,a,i0,a,i0)', 'MersenneTwister, Throughput = ', &
    real(n_rand,real64)*1.0e-9_real64/gpu_time, ' GNumbers/s, Time = ', gpu_time, &
    ' s, Size = ', n_rand, ' Numbers, Workgroup = ', local_work_size
  print *
  print '(a)', 'Read back results...'
  !$omp target update from(h_rand_gpu(0:n_rand-1))
  !$omp end target data
  print '(a)', 'Compute CPU reference solution...'
  call random_ref(h_mt, h_rand_cpu, n_per_rng, seed)
  print '(a)', 'Compare CPU and GPU results...'
  sum_delta = 0.0_real64; sum_ref = 0.0_real64
  do i = 0, mt_rng_count-1
    do j = 0, n_per_rng-1
      r_cpu = h_rand_cpu(i*n_per_rng+j)
      r_gpu = h_rand_gpu(i+j*mt_rng_count)
      delta = abs(r_cpu - r_gpu)
      sum_delta = sum_delta + delta
      sum_ref = sum_ref + abs(r_cpu)
    end do
  end do
  l1norm = sum_delta / sum_ref
  print '(a,es14.6)', 'L1 norm: ', l1norm
  print *
  print '(a)', merge('PASS', 'FAIL', l1norm < 1.0e-6_real64)
end program
