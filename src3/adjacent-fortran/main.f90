program adjacent
  use iso_fortran_env, only: int32, real64
  use omp_lib
  implicit none
  integer :: nelems, repeat
  character(len=64) :: arg

  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: ./main <number of elements> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) nelems
  call get_command_argument(2, arg); read(arg, *) repeat

  call test(64, nelems, repeat)
  call test(128, nelems, repeat)
  call test(256, nelems, repeat)
  call test(512, nelems, repeat)
  call test(1024, nelems, repeat)
contains
  subroutine test(block_threads, requested_nelems, repeat_count)
    integer, intent(in) :: block_threads, requested_nelems, repeat_count
    integer :: items_per_block, num_items, grid_size, b, i, j, iter, errors
    real(real64) :: start_time, elapsed
    integer(int32), allocatable :: h_in(:), h_out(:), r_out(:)

    items_per_block = block_threads * 4
    num_items = ((requested_nelems + items_per_block - 1) / items_per_block) * items_per_block
    grid_size = num_items / items_per_block
    allocate(h_in(0:num_items-1), h_out(0:num_items-1), r_out(0:num_items-1))
    do i = 0, num_items - 1
      h_in(i) = modulo(i, 17)
    end do

!$omp target data map(to: h_in(0:num_items-1)) map(alloc: h_out(0:num_items-1))
    do iter = 1, repeat_count
!$omp target teams distribute
      do b = 0, grid_size - 1
!$omp parallel do
        do j = 0, items_per_block - 1
          if (j == 0) then
            h_out(b * items_per_block + j) = h_in(b * items_per_block + j)
          else
            h_out(b * items_per_block + j) = h_in(b * items_per_block + j) - h_in(b * items_per_block + j - 1)
          end if
        end do
!$omp end parallel do
      end do
!$omp end target teams distribute
    end do
!$omp target update from(h_out(0:num_items-1))
    do b = 0, grid_size - 1
      do j = 0, items_per_block - 1
        if (j == 0) then
          r_out(b * items_per_block + j) = h_in(b * items_per_block + j)
        else
          r_out(b * items_per_block + j) = h_in(b * items_per_block + j) - h_in(b * items_per_block + j - 1)
        end if
      end do
    end do
    errors = count(r_out /= h_out)
    if (errors == 0) then; print '(a)', 'PASS'; else; print '(a)', 'FAIL'; end if

    do iter = 1, repeat_count
!$omp target teams distribute
      do b = 0, grid_size - 1
!$omp parallel do
        do j = 0, items_per_block - 1
          if (j == items_per_block - 1) then
            h_out(b * items_per_block + j) = h_in(b * items_per_block + j)
          else
            h_out(b * items_per_block + j) = h_in(b * items_per_block + j) - h_in(b * items_per_block + j + 1)
          end if
        end do
!$omp end parallel do
      end do
!$omp end target teams distribute
    end do
!$omp target update from(h_out(0:num_items-1))
    do b = 0, grid_size - 1
      do j = 0, items_per_block - 1
        if (j == items_per_block - 1) then
          r_out(b * items_per_block + j) = h_in(b * items_per_block + j)
        else
          r_out(b * items_per_block + j) = h_in(b * items_per_block + j) - h_in(b * items_per_block + j + 1)
        end if
      end do
    end do
    errors = count(r_out /= h_out)
    if (errors == 0) then; print '(a)', 'PASS'; else; print '(a)', 'FAIL'; end if

    start_time = omp_get_wtime()
    do iter = 1, repeat_count
!$omp target teams distribute
      do b = 0, grid_size - 1
!$omp parallel do
        do j = 0, items_per_block - 1
          if (j == 0) then
            h_out(b * items_per_block + j) = h_in(b * items_per_block + j)
          else
            h_out(b * items_per_block + j) = h_in(b * items_per_block + j) - h_in(b * items_per_block + j - 1)
          end if
        end do
!$omp end parallel do
      end do
!$omp end target teams distribute
!$omp target teams distribute
      do b = 0, grid_size - 1
!$omp parallel do
        do j = 0, items_per_block - 1
          if (j == items_per_block - 1) then
            h_out(b * items_per_block + j) = h_out(b * items_per_block + j)
          else
            h_out(b * items_per_block + j) = h_out(b * items_per_block + j) - h_out(b * items_per_block + j + 1)
          end if
        end do
!$omp end parallel do
      end do
!$omp end target teams distribute
    end do
    elapsed = omp_get_wtime() - start_time
    print '(a,i4,a,f12.6,a)', 'Average execution time of the kernels (thread block size = ', &
      block_threads, '): ', elapsed * 1.0e6_real64 / real(repeat_count, real64), ' (us)'
!$omp end target data
    deallocate(h_in, h_out, r_out)
  end subroutine test
end program adjacent
