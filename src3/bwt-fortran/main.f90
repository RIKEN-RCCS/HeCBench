module bwt_support
  use, intrinsic :: iso_c_binding, only : c_char, c_int
  use, intrinsic :: iso_fortran_env, only : int64, real64
  implicit none

  integer, parameter :: block_size = 256
  character(kind=c_char), parameter :: etx = achar(0, kind=c_char)

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    function c_rand() bind(C, name="rand") result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

contains

  logical function rotation_less(a, b, sequence, n)
    integer(c_int), intent(in) :: a, b
    integer, intent(in) :: n
    character(kind=c_char), intent(in) :: sequence(0:n-1)
    integer :: i, ia, ib

    rotation_less = .false.
    do i = 0, n - 1
      ia = modulo(int(a) + i, n)
      ib = modulo(int(b) + i, n)
      if (sequence(ia) /= sequence(ib)) then
        rotation_less = sequence(ia) < sequence(ib)
        return
      end if
    end do
  end function rotation_less

  subroutine bwt_cpu(sequence, n, transformed)
    integer, intent(in) :: n
    character(kind=c_char), intent(in) :: sequence(0:n-1)
    character(kind=c_char), intent(out) :: transformed(0:n-1)
    integer(c_int), allocatable :: table(:), work(:)
    integer :: width, left, middle, right, p, q, destination, i

    allocate(table(0:n-1), work(0:n-1))
    do i = 0, n - 1
      table(i) = int(i, c_int)
    end do

    ! std::list::sort in the C++ reference is a stable merge sort.  This
    ! bottom-up merge sort has the same comparator and tie behavior.
    width = 1
    do while (width < n)
      left = 0
      do while (left < n)
        middle = min(left + width, n)
        right = min(left + 2 * width, n)
        p = left
        q = middle
        destination = left
        do while (p < middle .and. q < right)
          if (rotation_less(table(q), table(p), sequence, n)) then
            work(destination) = table(q)
            q = q + 1
          else
            work(destination) = table(p)
            p = p + 1
          end if
          destination = destination + 1
        end do
        do while (p < middle)
          work(destination) = table(p)
          p = p + 1
          destination = destination + 1
        end do
        do while (q < right)
          work(destination) = table(q)
          q = q + 1
          destination = destination + 1
        end do
        left = left + 2 * width
      end do
      table = work
      if (width > n / 2) exit
      width = width * 2
    end do

    do i = 0, n - 1
      transformed(i) = sequence(modulo(n + int(table(i)) - 1, n))
    end do
    deallocate(work, table)
  end subroutine bwt_cpu

  subroutine bwt_device(sequence, n, transformed)
    integer, intent(in) :: n
    character(kind=c_char), intent(in) :: sequence(0:n-1)
    character(kind=c_char), intent(out) :: transformed(0:n-1)
    integer(c_int), allocatable :: table(:)
    integer :: table_size, i, ixj, j, k, l
    integer(c_int) :: t1, t2, first_rotation, second_rotation
    logical :: ascending, less

    table_size = n
    table_size = table_size - 1
    table_size = ior(table_size, ishft(table_size, -1))
    table_size = ior(table_size, ishft(table_size, -2))
    table_size = ior(table_size, ishft(table_size, -4))
    table_size = ior(table_size, ishft(table_size, -8))
    table_size = ior(table_size, ishft(table_size, -16))
    table_size = table_size + 1
    allocate(table(0:table_size-1))

    !$omp target data map(from: table(0:table_size-1), transformed(0:n-1)) map(to: sequence(0:n-1))
    !$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, table_size - 1
      if (i < n) then
        table(i) = int(i, c_int)
      else
        table(i) = -1_c_int
      end if
    end do
    !$omp end target teams distribute parallel do

    k = 2
    do while (k <= table_size)
      j = ishft(k, -1)
      do while (j > 0)
        !$omp target teams distribute parallel do thread_limit(block_size) &
        !$omp& private(ixj, ascending, t1, t2, first_rotation, second_rotation, l, less)
        do i = 0, table_size - 1
          ixj = ieor(i, j)
          if (i < ixj) then
            ascending = iand(i, k) == 0
            t1 = table(i)
            t2 = table(ixj)
            less = .false.
            if (ascending) then
              first_rotation = t2
              second_rotation = t1
            else
              first_rotation = t1
              second_rotation = t2
            end if
            if (first_rotation < 0_c_int) then
              less = .false.
            else if (second_rotation < 0_c_int) then
              less = .true.
            else
              do l = 0, n - 1
                if (sequence(modulo(int(first_rotation) + l, n)) /= &
                    sequence(modulo(int(second_rotation) + l, n))) then
                  less = sequence(modulo(int(first_rotation) + l, n)) < &
                         sequence(modulo(int(second_rotation) + l, n))
                  exit
                end if
              end do
            end if
            if (less) then
              table(i) = t2
              table(ixj) = t1
            end if
          end if
        end do
        !$omp end target teams distribute parallel do
        j = ishft(j, -1)
      end do
      if (k > table_size / 2) exit
      k = ishft(k, 1)
    end do

    !$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, n - 1
      transformed(i) = sequence(modulo(n + int(table(i)) - 1, n))
    end do
    !$omp end target teams distribute parallel do
    !$omp end target data

    deallocate(table)
  end subroutine bwt_device

end module bwt_support

program main
  use, intrinsic :: iso_c_binding, only : c_char, c_int
  use, intrinsic :: iso_fortran_env, only : int64, real64
  use bwt_support
  implicit none

  integer :: n_input, n, i, ios
  integer(int64) :: start_count, stop_count, clock_rate, host_ms, device_ms
  character(len=64) :: argument
  character(kind=c_char), allocatable :: sequence(:), cpu_sequence(:), gpu_sequence(:)

  n_input = 1000000
  if (command_argument_count() > 0) then
    call get_command_argument(1, argument)
    read(argument, *, iostat=ios) n_input
    if (ios /= 0) n_input = 0
  end if
  if (n_input < 0) n_input = 0
  n = n_input + 1

  write(*,'(A,I0)') 'running a sample sequence of length ', n_input
  allocate(sequence(0:n-1), cpu_sequence(0:n-1), gpu_sequence(0:n-1))
  call c_srand(123_c_int)
  do i = 0, n_input - 1
    select case (modulo(c_rand(), 4_c_int))
    case (0)
      sequence(i) = 'A'
    case (1)
      sequence(i) = 'T'
    case (2)
      sequence(i) = 'C'
    case default
      sequence(i) = 'G'
    end select
  end do
  sequence(n_input) = etx

  call system_clock(count=start_count, count_rate=clock_rate)
  call bwt_cpu(sequence, n, cpu_sequence)
  call system_clock(count=stop_count)
  host_ms = int(1000.0_real64 * real(stop_count - start_count, real64) / real(clock_rate, real64), int64)

  call system_clock(count=start_count)
  call bwt_device(sequence, n, gpu_sequence)
  call system_clock(count=stop_count)
  device_ms = int(1000.0_real64 * real(stop_count - start_count, real64) / real(clock_rate, real64), int64)

  write(*,'(A,I0,A)') 'Host time: ', host_ms, ' ms'
  write(*,'(A,I0,A)') 'Device time: ', device_ms, ' ms'
  if (all(cpu_sequence == gpu_sequence)) then
    write(*,'(A)') 'PASS'
  else
    write(*,'(A)') 'FAIL'
  end if
  deallocate(gpu_sequence, cpu_sequence, sequence)
end program main
