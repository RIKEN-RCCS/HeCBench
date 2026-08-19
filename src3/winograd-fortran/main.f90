program main
  use iso_fortran_env, only: real32, real64
  use winograd_utils
  implicit none
  real(real32), allocatable :: a(:), b_host(:), b(:), c(:)
  integer :: i, cpu_offset, tile_n, global_work_size(0:1), local_work_size(0:1)
  integer :: cpu_global_size(0:1), gpu_global_size(0:1), global_offset(0:1)
  integer :: tile_i_size, tile_j_size, offset_i, offset_j, thread_size, tile_i, tile_j
  logical :: pass, cpu_run, gpu_run
  real(real64) :: start_time, end_time, co_start, co_time
  allocate(a(0:map_size*map_size-1), b_host(0:(map_size-2)*(map_size-2)-1), b(0:(map_size-2)*(map_size-2)-1), c(0:15))
  start_time = wall_seconds()
  call random_seed()
  do i = 0, map_size * map_size - 1
    call random_number(a(i))
  end do
  call filter_transformation(c)
  tile_n = (map_size - 2 + 1) / 2
  global_work_size(0) = int(ceiling(real(tile_n, real32) / real(dim_local_work_group_x, real32))) * dim_local_work_group_x
  global_work_size(1) = int(ceiling(real(tile_n, real32) / real(dim_local_work_group_y, real32))) * dim_local_work_group_y
  local_work_size = [dim_local_work_group_x, dim_local_work_group_y]
  pass = .true.
  co_time = 0.0_real64
!$omp target data map(to:a,c) map(alloc:b)
  do cpu_offset = 0, 100
    cpu_global_size(0) = cpu_offset * int(ceiling(real(tile_n, real32) / real(dim_local_work_group_x, real32))) / 100 * dim_local_work_group_x
    cpu_global_size(1) = global_work_size(1)
    gpu_global_size(0) = global_work_size(0) - cpu_global_size(0)
    gpu_global_size(1) = global_work_size(1)
    global_offset(0) = cpu_global_size(0)
    global_offset(1) = 0
    tile_i_size = gpu_global_size(0)
    tile_j_size = gpu_global_size(1)
    offset_i = global_offset(0)
    offset_j = global_offset(1)
    thread_size = local_work_size(1) * local_work_size(0)
    cpu_run = cpu_global_size(0) > 0
    gpu_run = gpu_global_size(0) > 0
    co_start = wall_seconds()
    if (gpu_run) then
!$omp target teams distribute parallel do collapse(2) thread_limit(thread_size) private(tile_i,tile_j)
      do tile_j = 0, tile_j_size - 1
        do tile_i = 0, tile_i_size - 1
          call winograd_tile(a, b, c, tile_i, tile_j, offset_i, offset_j)
        end do
      end do
!$omp end target teams distribute parallel do
    end if
    if (cpu_run) then
      call winograd_cpu(a, b, c, cpu_global_size)
      if (gpu_run) then
!$omp target update to(b(0:offset_i*2*(map_size-2)-1))
      else
!$omp target update to(b)
      end if
    end if
!$omp target update from(b)
    co_time = co_time + wall_seconds() - co_start
    call winograd_reference(a, b_host, c)
    pass = pass .and. compare_results(b_host, b)
  end do
!$omp end target data
  print '(a)', merge('PASS', 'FAIL', pass)
  end_time = wall_seconds()
  print '(a,f12.6,a)', 'Co-execution time: ', co_time, ' s'
  print '(a,f12.6,a)', 'Total time: ', end_time - start_time, ' s'
  print '(a,f8.2,a)', 'Ratio of co-execution time to total time: ', 100.0_real64 * co_time / (end_time - start_time), '%'
end program main
