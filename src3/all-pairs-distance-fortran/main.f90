program all_pairs_distance
  use iso_c_binding, only: c_int, c_long, c_signed_char
  use iso_fortran_env, only: real64
  use omp_lib
  implicit none

  integer, parameter :: instances = 224
  integer, parameter :: attributes = 4096
  integer, parameter :: threads = 128

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    function c_random() bind(C, name="random") result(value)
      import :: c_long
      integer(c_long) :: value
    end function c_random
  end interface

  integer :: iterations, i, j, k, n, status
  integer :: idx, gx, gy, stride
  integer(c_int) :: count
  integer(c_int) :: dist(0:threads - 1)
  integer(c_int), allocatable :: data(:), cpu_distance(:), gpu_distance(:)
  integer(c_signed_char), allocatable :: data_char(:)
  real(real64) :: start_cpu, stop_cpu, start_gpu, stop_gpu, elapsed_time
  character(len=64) :: argument

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <iterations>'
    stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *) iterations

  allocate(data(0:instances * attributes - 1))
  allocate(data_char(0:instances * attributes - 1))
  allocate(cpu_distance(0:instances * instances - 1))
  allocate(gpu_distance(0:instances * instances - 1))

  ! The C/C++ baseline calls srand(2), then random()%3 in this collapsed loop.
  call c_srand(2_c_int)
  !$omp parallel do collapse(2) private(i, j)
  do i = 0, attributes - 1
    do j = 0, instances - 1
      data(i + attributes * j) = int(modulo(c_random(), 3_c_long), c_int)
      data_char(i + attributes * j) = int(data(i + attributes * j), c_signed_char)
    end do
  end do
  !$omp end parallel do

  cpu_distance = 0_c_int
  start_cpu = omp_get_wtime()
  !$omp parallel do collapse(2) private(i, j, k)
  do i = 0, instances - 1
    do j = 0, instances - 1
      do k = 0, attributes - 1
        if (data(i * attributes + k) /= data(j * attributes + k)) then
          cpu_distance(i + instances * j) = cpu_distance(i + instances * j) + 1_c_int
        end if
      end do
    end do
  end do
  !$omp end parallel do
  stop_cpu = omp_get_wtime()
  elapsed_time = (stop_cpu - start_cpu) * 1.0e6_real64
  print '(a,f0.6,a)', 'CPU time: ', elapsed_time, ' (us)'

  !$omp target data map(to: data_char(0:instances * attributes - 1)) &
  !$omp& map(alloc: gpu_distance(0:instances * instances - 1))
    do n = 1, iterations
      ! This host reset and device update are deliberately outside the timed kernel.
      gpu_distance = 0_c_int
      !$omp target update to(gpu_distance(0:instances * instances - 1))

      start_gpu = omp_get_wtime()
      !$omp target teams num_teams(instances * instances) thread_limit(threads)
        !$omp parallel private(idx, gx, gy, i, count)
          idx = omp_get_thread_num()
          gx = modulo(omp_get_team_num(), instances)
          gy = omp_get_team_num() / instances

          do i = 4 * idx, attributes - 1, threads * 4
            count = 0_c_int
            if (data_char(i + attributes * gx) /= data_char(i + attributes * gy)) count = count + 1_c_int
            if (data_char(i + 1 + attributes * gx) /= data_char(i + 1 + attributes * gy)) count = count + 1_c_int
            if (data_char(i + 2 + attributes * gx) /= data_char(i + 2 + attributes * gy)) count = count + 1_c_int
            if (data_char(i + 3 + attributes * gx) /= data_char(i + 3 + attributes * gy)) count = count + 1_c_int
            !$omp atomic update
            gpu_distance(instances * gx + gy) = gpu_distance(instances * gx + gy) + count
          end do
        !$omp end parallel
      !$omp end target teams
      stop_gpu = omp_get_wtime()
      elapsed_time = elapsed_time + (stop_gpu - start_gpu) * 1.0e6_real64
    end do

    !$omp target update from(gpu_distance(0:instances * instances - 1))

    print '(a,f0.6,a)', 'Average kernel execution time (w/o shared memory): ', &
      elapsed_time / real(iterations, real64), ' (us)'
    if (all(cpu_distance == gpu_distance)) then
      print '(a)', 'PASS'
      status = 0
    else
      print '(a)', 'FAIL'
      status = 1
    end if

    elapsed_time = 0.0_real64
    do n = 1, iterations
      ! This host reset and device update are deliberately outside the timed kernel.
      gpu_distance = 0_c_int
      !$omp target update to(gpu_distance(0:instances * instances - 1))

      start_gpu = omp_get_wtime()
      !$omp target teams num_teams(instances * instances) thread_limit(threads) private(dist)

          !$omp parallel private(idx, gx, gy, i, count, stride) shared(dist)
            idx = omp_get_thread_num()
            gx = modulo(omp_get_team_num(), instances)
            gy = omp_get_team_num() / instances

            dist(idx) = 0_c_int
            !$omp barrier

            do i = 4 * idx, attributes - 1, threads * 4
              count = 0_c_int
              if (data_char(i + attributes * gx) /= data_char(i + attributes * gy)) count = count + 1_c_int
              if (data_char(i + 1 + attributes * gx) /= data_char(i + 1 + attributes * gy)) count = count + 1_c_int
              if (data_char(i + 2 + attributes * gx) /= data_char(i + 2 + attributes * gy)) count = count + 1_c_int
              if (data_char(i + 3 + attributes * gx) /= data_char(i + 3 + attributes * gy)) count = count + 1_c_int
              dist(idx) = dist(idx) + count
            end do

            !$omp barrier
            stride = threads / 2
            do while (stride > 0)
              if (idx < stride) dist(idx) = dist(idx) + dist(idx + stride)
              !$omp barrier
              stride = stride / 2
            end do

            if (idx == 0) gpu_distance(instances * gy + gx) = dist(0)
          !$omp end parallel
      !$omp end target teams
      stop_gpu = omp_get_wtime()
      elapsed_time = elapsed_time + (stop_gpu - start_gpu) * 1.0e6_real64
    end do

    !$omp target update from(gpu_distance(0:instances * instances - 1))

    print '(a,f0.6,a)', 'Average kernel execution time (w/ shared memory): ', &
      elapsed_time / real(iterations, real64), ' (us)'
    if (all(cpu_distance == gpu_distance)) then
      print '(a)', 'PASS'
      status = 0
    else
      print '(a)', 'FAIL'
      status = 1
    end if
  !$omp end target data

  deallocate(cpu_distance)
  deallocate(gpu_distance)
  deallocate(data_char)
  deallocate(data)

  if (status /= 0) stop status
end program all_pairs_distance
