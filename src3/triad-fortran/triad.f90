module triad_kernel_mod
  use iso_fortran_env, only: real32, real64, int64
  implicit none
  integer, parameter :: n_sizes = 9
  integer, parameter :: block_sizes(0:n_sizes-1) = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
  integer, parameter :: mem_size = 16384
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine run_benchmark(verbose, n_passes)
    logical, intent(in) :: verbose
    integer, intent(in) :: n_passes
    integer :: num_max_floats, half_num_floats, max_block_size, block_size
    integer :: i, j, pass, elems_in_block, crt_idx, block_idx, curr_stream
    real(real32), allocatable :: h_mem(:), a0(:), b0(:), c0(:), a1(:), b1(:), c1(:)
    real(real32), parameter :: scalar = 1.75_real32
    real(real64) :: t0, t1, time, triad, bdwth
    logical :: ok
    num_max_floats = 1024 * mem_size / storage_size(1.0_real32) * 8
    half_num_floats = num_max_floats / 2
    max_block_size = block_sizes(n_sizes - 1) * 1024 / storage_size(1.0_real32) * 8
    block_size = 128
    allocate(h_mem(0:num_max_floats-1), a0(0:max_block_size-1), b0(0:max_block_size-1), c0(0:max_block_size-1))
    allocate(a1(0:max_block_size-1), b1(0:max_block_size-1), c1(0:max_block_size-1))
!$omp target data map(alloc:a0,b0,c0,a1,b1,c1)
    do i = 0, n_sizes - 1
      do j = 0, num_max_floats - 1
        if (j < max_block_size) then
          c0(j) = 0.0_real32
          c1(j) = 0.0_real32
        end if
      end do
      do j = 0, half_num_floats - 1
        if (j < max_block_size .and. half_num_floats + j < max_block_size) then
          a0(j) = drand48_like(j); a0(half_num_floats + j) = a0(j)
          b0(j) = a0(j); b0(half_num_floats + j) = a0(j)
          a1(j) = a0(j); a1(half_num_floats + j) = a0(j)
          b1(j) = a0(j); b1(half_num_floats + j) = a0(j)
        end if
      end do
      elems_in_block = block_sizes(i) * 1024 / storage_size(1.0_real32) * 8
      if (verbose) then
        print '(a,i0,a,i0,a)', '>> Executing Triad with vectors of length ', num_max_floats, ' and block size of ', elems_in_block, ' elements.'
        print '(a,i0,a)', 'Block: ', block_sizes(i), 'KB'
      end if
      crt_idx = 0
      t0 = wall_seconds()
      do pass = 0, n_passes - 1
!$omp target update to(a0(0:elems_in_block-1)) nowait
!$omp target update to(b0(0:elems_in_block-1)) nowait
!$omp target teams distribute parallel do thread_limit(block_size) nowait
        do j = 0, elems_in_block - 1
          c0(j) = a0(j) + scalar * b0(j)
        end do
!$omp end target teams distribute parallel do
        if (elems_in_block < num_max_floats) then
!$omp target update to(a1(elems_in_block:2*elems_in_block-1)) nowait
!$omp target update to(b1(elems_in_block:2*elems_in_block-1)) nowait
        end if
        block_idx = 1
        do while (crt_idx < num_max_floats)
          curr_stream = iand(block_idx, 1)
          if (curr_stream /= 0) then
!$omp target update from(c0(crt_idx:crt_idx+elems_in_block-1)) nowait
          else
!$omp target update from(c1(crt_idx:crt_idx+elems_in_block-1)) nowait
          end if
          crt_idx = crt_idx + elems_in_block
          if (crt_idx < num_max_floats) then
            if (curr_stream /= 0) then
!$omp target teams distribute parallel do thread_limit(block_size) nowait
              do j = 0, elems_in_block - 1
                c1(crt_idx+j) = a1(crt_idx+j) + scalar * b1(crt_idx+j)
              end do
!$omp end target teams distribute parallel do
            else
!$omp target teams distribute parallel do thread_limit(block_size) nowait
              do j = 0, elems_in_block - 1
                c0(crt_idx+j) = a0(crt_idx+j) + scalar * b0(crt_idx+j)
              end do
!$omp end target teams distribute parallel do
            end if
          end if
          if (crt_idx + elems_in_block < num_max_floats) then
            if (curr_stream /= 0) then
!$omp target update to(a0(crt_idx+elems_in_block:crt_idx+2*elems_in_block-1)) nowait
!$omp target update to(b0(crt_idx+elems_in_block:crt_idx+2*elems_in_block-1)) nowait
            else
!$omp target update to(a1(crt_idx+elems_in_block:crt_idx+2*elems_in_block-1)) nowait
!$omp target update to(b1(crt_idx+elems_in_block:crt_idx+2*elems_in_block-1)) nowait
            end if
          end if
          block_idx = block_idx + 1
        end do
      end do
      t1 = wall_seconds()
      time = t1 - t0
      triad = (real(num_max_floats, real64) * 2.0_real64 * real(n_passes, real64)) / (time * 1.0e9_real64)
      if (verbose) print '(a,f12.6,a)', 'Average TriadFlops ', triad, ' GFLOPS/s'
      bdwth = (real(num_max_floats, real64) * storage_size(1.0_real32) / 8.0_real64 * 3.0_real64 * real(n_passes, real64)) / (time * 1000.0_real64 * 1000.0_real64 * 1000.0_real64)
      if (verbose) print '(a,f12.6,a)', 'Average TriadBdwth ', bdwth, ' GB/s'
      ok = .true.
      do j = 0, num_max_floats - 1, elems_in_block
        if (iand(j / elems_in_block, 1) == 0) then
          h_mem(j:min(j+elems_in_block-1,num_max_floats-1)) = c0(j:min(j+elems_in_block-1,num_max_floats-1))
        else
          h_mem(j:min(j+elems_in_block-1,num_max_floats-1)) = c1(j:min(j+elems_in_block-1,num_max_floats-1))
        end if
      end do
      do j = 0, half_num_floats - 1
        if (h_mem(j) /= h_mem(j + half_num_floats)) then
          print '(a,i0,a,es12.5,a,i0,a,es12.5,a)', 'hostMem[', j, ']=', h_mem(j), ' is different from its twin element hostMem[', j + half_num_floats, ']: ', h_mem(j + half_num_floats), 'stopping check'
          ok = .false.
          exit
        end if
      end do
      print '(a)', merge('PASS', 'FAIL', ok)
    end do
!$omp end target data
  end subroutine run_benchmark

  pure real(real32) function drand48_like(i)
    integer, intent(in) :: i
    drand48_like = real(mod(i * 252149039 + 11, 2147483647), real32) / 2147483647.0_real32 * 10.0_real32
  end function drand48_like
end module triad_kernel_mod
