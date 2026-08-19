program main
  use iso_fortran_env, only: int64, real64
  use iso_c_binding, only: c_int
  implicit none
  integer, parameter :: nil = -1
  integer(int64), parameter :: mask = int(z'00000000FFFFFFFF', int64)
  integer :: argc, elems, set_random_list, repeat, i, r, teams, rep, stat
  character(len=128) :: arg
  integer, allocatable :: next(:), rank(:)
  integer(int64), allocatable :: list(:), d_res(:), h_res(:)
  real(real64) :: time_sum, t0, t1
  logical :: ok

  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage: ./main <list size> <0 or 1> <repeat>'
    print '(a)', '0 and 1 indicate an ordered list and a random list, respectively'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) elems
  call get_command_argument(2, arg); read(arg, *, iostat=stat) set_random_list
  call get_command_argument(3, arg); read(arg, *, iostat=stat) repeat
  allocate(next(0:elems-1), rank(0:elems-1), list(0:elems-1), d_res(0:elems-1), h_res(0:elems-1))
  if (set_random_list /= 0) then
    call random_list(next)
  else
    call ordered_list(next)
  end if
  do i = 0, elems - 1
    if (next(i) == nil) then
      rank(i) = 0
    else
      rank(i) = 1
    end if
    list(i) = ior(shiftl(int(next(i), int64), 32), int(rank(i), int64))
  end do
  teams = (elems + 255) / 256
!$omp target data map(tofrom:list(0:elems-1))
  time_sum = 0.0_real64
  do rep = 0, repeat
!$omp target update to(list(0:elems-1))
    t0 = wall_seconds()
!$omp target teams num_teams(teams) thread_limit(256)
!$omp parallel
    call pointer_jump(list, elems)
!$omp end parallel
!$omp end target teams
    t1 = wall_seconds()
    if (rep > 0) time_sum = time_sum + (t1 - t0)
  end do
  print '(a,f12.6,a)', 'Average kernel execution time: ', time_sum * 1000.0_real64 / real(repeat, real64), ' (ms)'
!$omp end target data
  do i = 0, elems - 1
    d_res(i) = iand(list(i), mask)
  end do
  h_res(0) = elems - 1
  i = 0
  do r = 1, elems - 1
    h_res(next(i)) = elems - 1 - r
    i = next(i)
  end do
  ok = all(h_res == d_res)
  print '(a)', merge('PASS', 'FAIL', ok)
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine ordered_list(next)
    integer, intent(out) :: next(0:)
    integer :: i, n
    n = size(next)
    do i = 0, n - 2
      next(i) = i + 1
    end do
    next(n - 1) = nil
  end subroutine ordered_list

  subroutine random_list(next)
    integer, intent(out) :: next(0:)
    logical, allocatable :: free_list(:)
    integer :: n, remaining, nil_pos, prev_pos, pos
    n = size(next)
    allocate(free_list(0:n-1))
    free_list = .true.
    call c_srand(123_c_int)
    nil_pos = modulo(c_rand(), n - 1) + 1
    free_list(nil_pos) = .false.
    next(nil_pos) = nil
    prev_pos = nil_pos
    remaining = n
    do while (remaining - 1 > 1)
      remaining = remaining - 1
      pos = modulo(c_rand(), n - 1) + 1
      do while (.not. free_list(pos))
        pos = modulo(c_rand(), n - 1) + 1
      end do
      free_list(pos) = .false.
      next(pos) = prev_pos
      prev_pos = pos
    end do
    next(0) = prev_pos
  end subroutine random_list

  subroutine pointer_jump(plist, elems)
    integer(int64), intent(inout) :: plist(0:)
    integer, intent(in) :: elems
    integer :: index
    integer(int64) :: node, nxt, temp
    index = omp_team_thread_index()
    if (index < elems) then
      do
        node = plist(index)
        if (shifta(node, 32) == nil) exit
        nxt = plist(shifta(node, 32))
        if (shifta(nxt, 32) == nil) exit
        temp = iand(node, mask)
        temp = temp + iand(nxt, mask)
        temp = temp + shiftl(shifta(nxt, 32), 32)
!$omp barrier
        plist(index) = temp
      end do
    end if
  end subroutine pointer_jump

  integer function omp_team_thread_index()
    !$omp declare target
    use omp_lib
    omp_team_thread_index = omp_get_team_num() * 256 + omp_get_thread_num()
  end function omp_team_thread_index
end program main
