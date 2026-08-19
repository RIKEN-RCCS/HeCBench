module ga_port
  use, intrinsic :: iso_c_binding, only : c_char, c_int
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  implicit none
contains
  subroutine ga_kernel(target, query, batch_result, length, query_length, coarse_length, threshold, position)
    integer, intent(in) :: length, query_length, coarse_length, threshold, position
    character(kind=c_char), intent(in) :: target(0:), query(0:)
    character(kind=c_char), intent(inout) :: batch_result(0:)
    integer :: tid, i, j, distance, max_length
    logical :: match

    !$omp target teams distribute parallel do thread_limit(256) private(i,j,distance,max_length,match)
    do tid = 0, length - 1
      match = .false.
      max_length = query_length - coarse_length
      do i = 0, max_length
        distance = 0
        do j = 0, coarse_length - 1
          if (target(position + tid + j) /= query(i + j)) distance = distance + 1
        end do
        if (distance < threshold) then
          match = .true.
          exit
        end if
      end do
      if (match) batch_result(tid) = achar(1, kind=c_char)
    end do
    !$omp end target teams distribute parallel do
  end subroutine ga_kernel

  subroutine ga_reference(target, query, result, length, query_length, coarse_length, threshold, position)
    integer, intent(in) :: length, query_length, coarse_length, threshold, position
    character(kind=c_char), intent(in) :: target(0:), query(0:)
    character(kind=c_char), intent(inout) :: result(0:)
    integer :: tid, i, j, distance, max_length
    logical :: match
    do tid = 0, length - 1
      match = .false.; max_length = query_length - coarse_length
      do i = 0, max_length
        distance = 0
        do j = 0, coarse_length - 1
          if (target(position + tid + j) /= query(i + j)) distance = distance + 1
        end do
        if (distance < threshold) then; match = .true.; exit; end if
      end do
      if (match) result(tid) = achar(1, kind=c_char)
    end do
  end subroutine ga_reference
end module ga_port

program main
  use, intrinsic :: iso_c_binding, only : c_char, c_int
  use, intrinsic :: iso_fortran_env, only : int64, real64
  use ga_port
  implicit none
  interface
    subroutine srand(seed) bind(C, name='srand')
      import c_int
      integer(c_int), value :: seed
    end subroutine srand
    function rand() bind(C, name='rand') result(value)
      import c_int
      integer(c_int) :: value
    end function rand
  end interface
  integer, parameter :: batch_size = 1024
  integer :: argc, tseq_size, qseq_size, coarse_length, threshold, i, current, end_position, length, error
  integer(int64) :: start_count, end_count, count_rate
  real(real32) :: total_time
  real(real64) :: elapsed_nanoseconds
  character(len=64) :: argument
  character(kind=c_char), allocatable :: target(:), query(:)
  character(kind=c_char) :: device_result(0:batch_size-1), reference_result(0:batch_size-1)
  character(kind=c_char), parameter :: bases(0:3) = [achar(65,kind=c_char), achar(67,kind=c_char), achar(84,kind=c_char), achar(71,kind=c_char)]

  argc = command_argument_count()
  if (argc /= 4) then
    write(*,'(a)') 'Usage: ./main <target sequence length> <query sequence length> <coarse match length> <coarse match threshold>'
    stop 1
  end if
  call get_command_argument(1,argument); read(argument,*) tseq_size
  call get_command_argument(2,argument); read(argument,*) qseq_size
  call get_command_argument(3,argument); read(argument,*) coarse_length
  call get_command_argument(4,argument); read(argument,*) threshold
  allocate(target(0:tseq_size-1), query(0:qseq_size-1))
  call srand(123_c_int)
  do i = 0, tseq_size - 1; target(i) = bases(mod(rand(), 4_c_int)); end do
  do i = 0, qseq_size - 1; query(i) = bases(mod(rand(), 4_c_int)); end do
  error = 0; current = 0; total_time = 0.0_real32

  !$omp target data map(to: target(0:tseq_size-1), query(0:qseq_size-1)) map(alloc: device_result(0:batch_size-1))
  do while (current < tseq_size - coarse_length)
    !$omp target teams distribute parallel do thread_limit(256)
    do i = 0, batch_size - 1
      device_result(i) = achar(0, kind=c_char)
    end do
    !$omp end target teams distribute parallel do
    reference_result = achar(0, kind=c_char)
    end_position = min(current + batch_size, tseq_size - coarse_length)
    length = end_position - current
    call system_clock(start_count, count_rate)
    call ga_kernel(target, query, device_result, length, qseq_size, coarse_length, threshold, current)
    call system_clock(end_count)
    elapsed_nanoseconds = real(end_count-start_count,real64)*1.0e9_real64/real(count_rate,real64)
    total_time = total_time + real(elapsed_nanoseconds,real32)
    call ga_reference(target, query, reference_result, length, qseq_size, coarse_length, threshold, current)
    !$omp target update from(device_result(0:batch_size-1))
    if (any(device_result /= reference_result)) then
      error = 1; exit
    end if
    current = end_position
  end do
  !$omp end target data
  write(*,'(a,f0.6,a)') 'Total kernel execution time ', real(total_time*1.0e-9_real32,real64), ' (s)'
  if (error /= 0) then; write(*,'(a)') 'FAIL'; else; write(*,'(a)') 'PASS'; end if
  deallocate(target, query)
end program main
