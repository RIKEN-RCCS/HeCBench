module bsearch_kernels
  use iso_c_binding, only: c_size_t
  use iso_fortran_env, only: real32, int64
  use omp_lib
  implicit none

contains

  subroutine elapsed_seconds(start_count, stop_count, count_rate, repeat, elapsed)
    integer(int64), intent(in) :: start_count, stop_count, count_rate
    integer, intent(in) :: repeat
    real(real32), intent(out) :: elapsed

    elapsed = real(stop_count - start_count, real32) / real(count_rate, real32)
    elapsed = elapsed / real(repeat, real32)
  end subroutine elapsed_seconds

  subroutine bs1(a_size, z_size, a, z, r, n, repeat, elapsed)
    integer(c_size_t), intent(in) :: a_size, z_size, n
    integer, intent(in) :: repeat
    real(real32), intent(in) :: a(0:a_size - 1_c_size_t), z(0:z_size - 1_c_size_t)
    integer(c_size_t), intent(inout) :: r(0:z_size - 1_c_size_t)
    real(real32), intent(out) :: elapsed
    integer(c_size_t) :: element, low, high, mid
    integer :: iteration
    integer(int64) :: start_count, stop_count, count_rate
    real(real32) :: value

    call system_clock(start_count, count_rate)
    do iteration = 1, repeat
!$omp target teams distribute parallel do thread_limit(256) private(value, low, high, mid)
      do element = 0_c_size_t, z_size - 1_c_size_t
        value = z(element)
        low = 0_c_size_t
        high = n
        do while (high - low > 1_c_size_t)
          mid = low + (high - low) / 2_c_size_t
          if (value < a(mid)) then
            high = mid
          else
            low = mid
          end if
        end do
        r(element) = low
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(stop_count)
    call elapsed_seconds(start_count, stop_count, count_rate, repeat, elapsed)
  end subroutine bs1

  subroutine bs2(a_size, z_size, a, z, r, n, repeat, elapsed)
    integer(c_size_t), intent(in) :: a_size, z_size, n
    integer, intent(in) :: repeat
    real(real32), intent(in) :: a(0:a_size - 1_c_size_t), z(0:z_size - 1_c_size_t)
    integer(c_size_t), intent(inout) :: r(0:z_size - 1_c_size_t)
    real(real32), intent(out) :: elapsed
    integer(c_size_t) :: element, k, index_value, candidate
    integer :: iteration, nbits
    integer(int64) :: start_count, stop_count, count_rate
    real(real32) :: value

    call system_clock(start_count, count_rate)
    do iteration = 1, repeat
!$omp target teams distribute parallel do thread_limit(256) private(nbits, k, value, index_value, candidate)
      do element = 0_c_size_t, z_size - 1_c_size_t
        nbits = 0
        do while (shiftr(n, nbits) /= 0_c_size_t)
          nbits = nbits + 1
        end do
        k = shiftl(1_c_size_t, nbits - 1)
        value = z(element)
        if (a(k) <= value) then
          index_value = k
        else
          index_value = 0_c_size_t
        end if
        do while (k > 1_c_size_t)
          k = shiftr(k, 1)
          candidate = ior(index_value, k)
          if (candidate < n) then
            if (value >= a(candidate)) index_value = candidate
          end if
        end do
        r(element) = index_value
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(stop_count)
    call elapsed_seconds(start_count, stop_count, count_rate, repeat, elapsed)
  end subroutine bs2

  subroutine bs3(a_size, z_size, a, z, r, n, repeat, elapsed)
    integer(c_size_t), intent(in) :: a_size, z_size, n
    integer, intent(in) :: repeat
    real(real32), intent(in) :: a(0:a_size - 1_c_size_t), z(0:z_size - 1_c_size_t)
    integer(c_size_t), intent(inout) :: r(0:z_size - 1_c_size_t)
    real(real32), intent(out) :: elapsed
    integer(c_size_t) :: element, k, index_value, candidate, wrapped
    integer :: iteration, nbits
    integer(int64) :: start_count, stop_count, count_rate
    real(real32) :: value

    call system_clock(start_count, count_rate)
    do iteration = 1, repeat
!$omp target teams distribute parallel do thread_limit(256) private(nbits, k, value, index_value, candidate, wrapped)
      do element = 0_c_size_t, z_size - 1_c_size_t
        nbits = 0
        do while (shiftr(n, nbits) /= 0_c_size_t)
          nbits = nbits + 1
        end do
        k = shiftl(1_c_size_t, nbits - 1)
        value = z(element)
        if (a(k) <= value) then
          index_value = k
        else
          index_value = 0_c_size_t
        end if
        do while (k > 1_c_size_t)
          k = shiftr(k, 1)
          candidate = ior(index_value, k)
          if (candidate < n) then
            wrapped = candidate
          else
            wrapped = n
          end if
          if (value >= a(wrapped)) index_value = candidate
        end do
        r(element) = index_value
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(stop_count)
    call elapsed_seconds(start_count, stop_count, count_rate, repeat, elapsed)
  end subroutine bs3

  subroutine bs4(a_size, z_size, a, z, r, n, repeat, elapsed)
    integer(c_size_t), intent(in) :: a_size, z_size, n
    integer, intent(in) :: repeat
    real(real32), intent(in) :: a(0:a_size - 1_c_size_t), z(0:z_size - 1_c_size_t)
    integer(c_size_t), intent(inout) :: r(0:z_size - 1_c_size_t)
    real(real32), intent(out) :: elapsed
    integer(c_size_t) :: k, local_id, global_id, search_bit, index_value, candidate, wrapped
    integer :: iteration, nbits
    integer(int64) :: start_count, stop_count, count_rate
    real(real32) :: value

    call system_clock(start_count, count_rate)
    do iteration = 1, repeat
!$omp target teams num_teams(z_size / 256_c_size_t) thread_limit(256) private(k)
!$omp parallel private(local_id, global_id, nbits, search_bit, value, index_value, candidate, wrapped) shared(k)
      local_id = int(omp_get_thread_num(), c_size_t)
      global_id = int(omp_get_team_num(), c_size_t) * int(omp_get_num_threads(), c_size_t) + local_id
      if (local_id == 0_c_size_t) then
        nbits = 0
        do while (shiftr(n, nbits) /= 0_c_size_t)
          nbits = nbits + 1
        end do
        k = shiftl(1_c_size_t, nbits - 1)
      end if
!$omp barrier
      search_bit = k
      value = z(global_id)
      if (a(search_bit) <= value) then
        index_value = search_bit
      else
        index_value = 0_c_size_t
      end if
      do while (search_bit > 1_c_size_t)
        search_bit = shiftr(search_bit, 1)
        candidate = ior(index_value, search_bit)
        if (candidate < n) then
          wrapped = candidate
        else
          wrapped = n
        end if
        if (value >= a(wrapped)) index_value = candidate
      end do
      r(global_id) = index_value
!$omp end parallel
!$omp end target teams
    end do
    call system_clock(stop_count)
    call elapsed_seconds(start_count, stop_count, count_rate, repeat, elapsed)
  end subroutine bs4

end module bsearch_kernels

program main
  use iso_c_binding, only: c_int, c_size_t
  use iso_fortran_env, only: real32
  use bsearch_kernels
  implicit none

  integer(c_size_t) :: a_size, z_size, n, element
  integer :: repeat, argument_status
  character(len=64) :: argument
  real(real32), allocatable :: a(:), z(:)
  integer(c_size_t), allocatable :: r(:)
  real(real32) :: elapsed

  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    function c_rand() bind(C, name='rand') result(value)
      import c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  if (command_argument_count() /= 2) then
    write(*, '(A)') 'Usage ./main <number of elements> <repeat>'
    error stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *, iostat=argument_status) a_size
  if (argument_status /= 0) error stop 1
  call get_command_argument(2, argument)
  read(argument, *, iostat=argument_status) repeat
  if (argument_status /= 0) error stop 1

  z_size = 2_c_size_t * a_size
  n = a_size - 1_c_size_t
  allocate(a(0:a_size - 1_c_size_t), z(0:z_size - 1_c_size_t), r(0:z_size - 1_c_size_t))

  do element = 0_c_size_t, a_size - 1_c_size_t
    a(element) = real(element, real32)
  end do

  call c_srand(2_c_int)
  do element = 0_c_size_t, z_size - 1_c_size_t
    z(element) = real(modulo(int(c_rand(), c_size_t), n), real32)
  end do

!$omp target data map(to: a(0:a_size - 1_c_size_t), z(0:z_size - 1_c_size_t)) map(from: r(0:z_size - 1_c_size_t))
  call bs1(a_size, z_size, a, z, r, n, repeat, elapsed)
  write(*, '(A,G0.6,A)') 'Average device execution time (bs1) ', elapsed, ' (s)'

  call bs2(a_size, z_size, a, z, r, n, repeat, elapsed)
  write(*, '(A,G0.6,A)') 'Average device execution time (bs2) ', elapsed, ' (s)'

  call bs3(a_size, z_size, a, z, r, n, repeat, elapsed)
  write(*, '(A,G0.6,A)') 'Average device execution time (bs3) ', elapsed, ' (s)'

  call bs4(a_size, z_size, a, z, r, n, repeat, elapsed)
  write(*, '(A,G0.6,A)') 'Average device execution time (bs4) ', elapsed, ' (s)'
!$omp end target data

  deallocate(a, z, r)
end program main
