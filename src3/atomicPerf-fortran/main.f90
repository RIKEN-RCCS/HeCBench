module atomicperf_kernels
  use, intrinsic :: iso_fortran_env, only : int32, real32, real64
  use omp_lib
  implicit none
  integer, parameter :: block_size = 256
contains

  subroutine global_block_r8(data, n)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(mod(i, block_size)) = data(mod(i, block_size)) + 1.0_real64
    end do
!$omp end target teams distribute parallel do
  end subroutine global_block_r8

  subroutine global_warp_r8(data, n)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(iand(i, 31)) = data(iand(i, 31)) + 1.0_real64
    end do
!$omp end target teams distribute parallel do
  end subroutine global_warp_r8

  subroutine global_single_r8(data, offset, n)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: offset, n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(offset) = data(offset) + 1.0_real64
    end do
!$omp end target teams distribute parallel do
  end subroutine global_single_r8

  subroutine shared_block_r8(data, n)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: n
    integer :: team, nteams, nthreads, thread, tid, i
    real(real64) :: smem_data(0:block_size - 1)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num()
      nteams = omp_get_num_teams()
      nthreads = omp_get_num_threads()
      thread = omp_get_thread_num()
      tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(thread) = smem_data(thread) + 1.0_real64
      end do
      if (team == nteams) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_block_r8

  subroutine shared_warp_r8(data, n)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: n
    integer :: team, nteams, nthreads, thread, tid, i
    real(real64) :: smem_data(0:31)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams()
      nthreads = omp_get_num_threads(); thread = omp_get_thread_num()
      tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(iand(i, 31)) = smem_data(iand(i, 31)) + 1.0_real64
      end do
      if (team == nteams .and. thread < 31) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_warp_r8

  subroutine shared_single_r8(data, offset, n)
    real(real64), intent(inout) :: data(0:)
    integer, intent(in) :: offset, n
    integer :: team, nteams, nthreads, thread, tid, i
    real(real64) :: smem_data(0:block_size - 1)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams()
      nthreads = omp_get_num_threads(); thread = omp_get_thread_num()
      tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(offset) = smem_data(offset) + 1.0_real64
      end do
      if (team == nteams .and. thread == 0) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_single_r8

  subroutine global_block_i4(data, n)
    integer(int32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(mod(i, block_size)) = data(mod(i, block_size)) + 1_int32
    end do
!$omp end target teams distribute parallel do
  end subroutine global_block_i4
  subroutine global_warp_i4(data, n)
    integer(int32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(iand(i, 31)) = data(iand(i, 31)) + 1_int32
    end do
!$omp end target teams distribute parallel do
  end subroutine global_warp_i4
  subroutine global_single_i4(data, offset, n)
    integer(int32), intent(inout) :: data(0:); integer, intent(in) :: offset, n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(offset) = data(offset) + 1_int32
    end do
!$omp end target teams distribute parallel do
  end subroutine global_single_i4
  subroutine shared_block_i4(data, n)
    integer(int32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: team, nteams, nthreads, thread, tid, i
    integer(int32) :: smem_data(0:block_size - 1)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); thread = omp_get_thread_num(); tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(thread) = smem_data(thread) + 1_int32
      end do
      if (team == nteams) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_block_i4
  subroutine shared_warp_i4(data, n)
    integer(int32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: team, nteams, nthreads, thread, tid, i
    integer(int32) :: smem_data(0:31)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); thread = omp_get_thread_num(); tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(iand(i, 31)) = smem_data(iand(i, 31)) + 1_int32
      end do
      if (team == nteams .and. thread < 31) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_warp_i4
  subroutine shared_single_i4(data, offset, n)
    integer(int32), intent(inout) :: data(0:); integer, intent(in) :: offset, n
    integer :: team, nteams, nthreads, thread, tid, i
    integer(int32) :: smem_data(0:block_size - 1)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); thread = omp_get_thread_num(); tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(offset) = smem_data(offset) + 1_int32
      end do
      if (team == nteams .and. thread == 0) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_single_i4

  subroutine global_block_r4(data, n)
    real(real32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(mod(i, block_size)) = data(mod(i, block_size)) + 1.0_real32
    end do
!$omp end target teams distribute parallel do
  end subroutine global_block_r4
  subroutine global_warp_r4(data, n)
    real(real32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(iand(i, 31)) = data(iand(i, 31)) + 1.0_real32
    end do
!$omp end target teams distribute parallel do
  end subroutine global_warp_r4
  subroutine global_single_r4(data, offset, n)
    real(real32), intent(inout) :: data(0:); integer, intent(in) :: offset, n
    integer :: i
!$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
!$omp atomic update
      data(offset) = data(offset) + 1.0_real32
    end do
!$omp end target teams distribute parallel do
  end subroutine global_single_r4
  subroutine shared_block_r4(data, n)
    real(real32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: team, nteams, nthreads, thread, tid, i
    real(real32) :: smem_data(0:block_size - 1)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); thread = omp_get_thread_num(); tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(thread) = smem_data(thread) + 1.0_real32
      end do
      if (team == nteams) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_block_r4
  subroutine shared_warp_r4(data, n)
    real(real32), intent(inout) :: data(0:); integer, intent(in) :: n
    integer :: team, nteams, nthreads, thread, tid, i
    real(real32) :: smem_data(0:31)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); thread = omp_get_thread_num(); tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(iand(i, 31)) = smem_data(iand(i, 31)) + 1.0_real32
      end do
      if (team == nteams .and. thread < 31) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_warp_r4
  subroutine shared_single_r4(data, offset, n)
    real(real32), intent(inout) :: data(0:); integer, intent(in) :: offset, n
    integer :: team, nteams, nthreads, thread, tid, i
    real(real32) :: smem_data(0:block_size - 1)
!$omp target teams num_teams(n / block_size) thread_limit(block_size) private(smem_data)
!$omp parallel private(team, nteams, nthreads, thread, tid, i)
      team = omp_get_team_num(); nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); thread = omp_get_thread_num(); tid = team * nthreads + thread
      do i = tid, n - 1, nthreads * nteams
!$omp atomic update
        smem_data(offset) = smem_data(offset) + 1.0_real32
      end do
      if (team == nteams .and. thread == 0) data(thread) = smem_data(thread)
!$omp end parallel
!$omp end target teams
  end subroutine shared_single_r4
end module atomicperf_kernels

program atomicperf
  use, intrinsic :: iso_fortran_env, only : int32, real32, real64
  use atomicperf_kernels
  implicit none
  integer :: argc, repeat, n, len
  character(len=64) :: argument
  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument, *) repeat
  n = 3 * 4 * 7 * 8 * 9 * block_size
  len = 1024
  print '(/a)', 'FP64 atomic add'
  call atomicperf_r8(n, len, repeat)
  print '(/a)', 'INT32 atomic add'
  call atomicperf_i4(n, len, repeat)
  print '(/a)', 'FP32 atomic add'
  call atomicperf_r4(n, len, repeat)
contains
  subroutine clock_start(c); integer, intent(out) :: c; call system_clock(c); end subroutine
  real(real64) function elapsed_us(c, repeat) result(value)
    integer, intent(in) :: c, repeat
    integer :: finish, rate
    call system_clock(finish, rate)
    value = real(finish - c, real64) * 1.0e6_real64 / real(rate * repeat, real64)
  end function elapsed_us
  subroutine atomicperf_r8(n, t, repeat)
    integer, intent(in) :: n, t, repeat
    real(real64), allocatable :: data(:), host(:), reference(:)
    integer :: i, r, start, failure
    allocate(data(0:t-1), host(0:t-1), reference(0:t-1))
    do i=0,t-1; data(i)=real(mod(i,1024)+1,real64); host(i)=data(i); end do
!$omp target data map(alloc:data)
!$omp target update to(data)
    call clock_start(start); do r=0,repeat-1; call global_block_r8(data,n); end do
    print '(a,f0.6,a)', 'Average execution time of BlockRangeAtomicOnGlobalMem: ', elapsed_us(start,repeat), ' (us)'
!$omp target update from(data)
    reference=host; do r=0,repeat-1; do i=0,n-1; reference(mod(i,block_size))=reference(mod(i,block_size))+1.0_real64; end do; end do
    failure=merge(1,0,any(data /= reference)); print '(a)',merge('FAIL','PASS',failure /= 0)
    data=host
!$omp target update to(data)
    call clock_start(start); do r=0,repeat-1; call global_warp_r8(data,n); end do
    print '(a,f0.6,a)', 'Average execution time of WarpRangeAtomicOnGlobalMem: ', elapsed_us(start,repeat), ' (us)'
    reference=host
!$omp target update from(data)
    do r=0,repeat-1; do i=0,n-1; reference(iand(i,31))=reference(iand(i,31))+1.0_real64; end do; end do
    failure=merge(1,0,any(data /= reference)); print '(a)',merge('FAIL','PASS',failure /= 0)
    data=host
!$omp target update to(data)
    call clock_start(start); do r=0,repeat-1; call global_single_r8(data,mod(r,block_size),n); end do
    print '(a,f0.6,a)', 'Average execution time of SingleRangeAtomicOnGlobalMem: ', elapsed_us(start,repeat), ' (us)'
    reference=host
!$omp target update from(data)
    do r=0,repeat-1; do i=0,n-1; reference(mod(r,block_size))=reference(mod(r,block_size))+1.0_real64; end do; end do
    failure=merge(1,0,any(data /= reference)); print '(a)',merge('FAIL','PASS',failure /= 0)
    data=host
!$omp target update to(data)
    call clock_start(start); do r=0,repeat-1; call shared_block_r8(data,n); end do
    print '(a,f0.6,a)', 'Average execution time of BlockRangeAtomicOnSharedMem: ', elapsed_us(start,repeat), ' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data /= host)); print '(a)',merge('FAIL','PASS',failure /= 0)
    data=host
!$omp target update to(data)
    call clock_start(start); do r=0,repeat-1; call shared_warp_r8(data,n); end do
    print '(a,f0.6,a)', 'Average execution time of WarpRangeAtomicOnSharedMem: ', elapsed_us(start,repeat), ' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data /= host)); print '(a)',merge('FAIL','PASS',failure /= 0)
    data=host
!$omp target update to(data)
    call clock_start(start); do r=0,repeat-1; call shared_single_r8(data,mod(r,block_size),n); end do
    print '(a,f0.6,a)', 'Average execution time of SingleRangeAtomicOnSharedMem: ', elapsed_us(start,repeat), ' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data /= host)); print '(a)',merge('FAIL','PASS',failure /= 0)
!$omp end target data
    deallocate(data,host,reference)
  end subroutine atomicperf_r8
  subroutine atomicperf_i4(n,t,repeat)
    integer,intent(in)::n,t,repeat; integer(int32),allocatable::data(:),host(:),reference(:); integer::i,r,start,failure
    allocate(data(0:t-1),host(0:t-1),reference(0:t-1)); do i=0,t-1;data(i)=int(mod(i,1024)+1,int32);host(i)=data(i);end do
!$omp target data map(alloc:data)
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call global_block_i4(data,n);end do;print '(a,f0.6,a)','Average execution time of BlockRangeAtomicOnGlobalMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    reference=host;do r=0,repeat-1;do i=0,n-1;reference(mod(i,block_size))=reference(mod(i,block_size))+1_int32;end do;end do;failure=merge(1,0,any(data/=reference));print '(a)',merge('FAIL','PASS',failure/=0)
    data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call global_warp_i4(data,n);end do;print '(a,f0.6,a)','Average execution time of WarpRangeAtomicOnGlobalMem: ',elapsed_us(start,repeat),' (us)'
    reference=host
!$omp target update from(data)
    do r=0,repeat-1;do i=0,n-1;reference(iand(i,31))=reference(iand(i,31))+1_int32;end do;end do;failure=merge(1,0,any(data/=reference));print '(a)',merge('FAIL','PASS',failure/=0)
    data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call global_single_i4(data,mod(r,block_size),n);end do;print '(a,f0.6,a)','Average execution time of SingleRangeAtomicOnGlobalMem: ',elapsed_us(start,repeat),' (us)'
    reference=host
!$omp target update from(data)
    do r=0,repeat-1;do i=0,n-1;reference(mod(r,block_size))=reference(mod(r,block_size))+1_int32;end do;end do;failure=merge(1,0,any(data/=reference));print '(a)',merge('FAIL','PASS',failure/=0)
    data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call shared_block_i4(data,n);end do;print '(a,f0.6,a)','Average execution time of BlockRangeAtomicOnSharedMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data/=host));print '(a)',merge('FAIL','PASS',failure/=0);data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call shared_warp_i4(data,n);end do;print '(a,f0.6,a)','Average execution time of WarpRangeAtomicOnSharedMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data/=host));print '(a)',merge('FAIL','PASS',failure/=0);data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call shared_single_i4(data,mod(r,block_size),n);end do;print '(a,f0.6,a)','Average execution time of SingleRangeAtomicOnSharedMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data/=host));print '(a)',merge('FAIL','PASS',failure/=0)
!$omp end target data
    deallocate(data,host,reference)
  end subroutine atomicperf_i4
  subroutine atomicperf_r4(n,t,repeat)
    integer,intent(in)::n,t,repeat; real(real32),allocatable::data(:),host(:),reference(:); integer::i,r,start,failure
    allocate(data(0:t-1),host(0:t-1),reference(0:t-1)); do i=0,t-1;data(i)=real(mod(i,1024)+1,real32);host(i)=data(i);end do
!$omp target data map(alloc:data)
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call global_block_r4(data,n);end do;print '(a,f0.6,a)','Average execution time of BlockRangeAtomicOnGlobalMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    reference=host;do r=0,repeat-1;do i=0,n-1;reference(mod(i,block_size))=reference(mod(i,block_size))+1.0_real32;end do;end do;failure=merge(1,0,any(data/=reference));print '(a)',merge('FAIL','PASS',failure/=0)
    data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call global_warp_r4(data,n);end do;print '(a,f0.6,a)','Average execution time of WarpRangeAtomicOnGlobalMem: ',elapsed_us(start,repeat),' (us)'
    reference=host
!$omp target update from(data)
    do r=0,repeat-1;do i=0,n-1;reference(iand(i,31))=reference(iand(i,31))+1.0_real32;end do;end do;failure=merge(1,0,any(data/=reference));print '(a)',merge('FAIL','PASS',failure/=0)
    data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call global_single_r4(data,mod(r,block_size),n);end do;print '(a,f0.6,a)','Average execution time of SingleRangeAtomicOnGlobalMem: ',elapsed_us(start,repeat),' (us)'
    reference=host
!$omp target update from(data)
    do r=0,repeat-1;do i=0,n-1;reference(mod(r,block_size))=reference(mod(r,block_size))+1.0_real32;end do;end do;failure=merge(1,0,any(data/=reference));print '(a)',merge('FAIL','PASS',failure/=0)
    data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call shared_block_r4(data,n);end do;print '(a,f0.6,a)','Average execution time of BlockRangeAtomicOnSharedMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data/=host));print '(a)',merge('FAIL','PASS',failure/=0);data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call shared_warp_r4(data,n);end do;print '(a,f0.6,a)','Average execution time of WarpRangeAtomicOnSharedMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data/=host));print '(a)',merge('FAIL','PASS',failure/=0);data=host
!$omp target update to(data)
    call clock_start(start);do r=0,repeat-1;call shared_single_r4(data,mod(r,block_size),n);end do;print '(a,f0.6,a)','Average execution time of SingleRangeAtomicOnSharedMem: ',elapsed_us(start,repeat),' (us)'
!$omp target update from(data)
    failure=merge(1,0,any(data/=host));print '(a)',merge('FAIL','PASS',failure/=0)
!$omp end target data
    deallocate(data,host,reference)
  end subroutine atomicperf_r4
end program atomicperf
