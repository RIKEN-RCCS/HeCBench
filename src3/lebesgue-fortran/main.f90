module lebesgue_mod
  use iso_fortran_env, only: real32, real64
  implicit none

contains

  subroutine make_nodes(point_kind, n, x)
    integer, intent(in) :: point_kind, n
    real(real64), intent(out) :: x(0:n-1)
    integer :: i
    real(real64) :: angle
    real(real64), parameter :: pi = 3.141592653589793_real64

    do i = 0, n - 1
      select case (point_kind)
      case (1)
        angle = pi * real(2 * i + 1, real64) / real(2 * n, real64)
        x(i) = cos(angle)
      case (2)
        if (n == 1) then
          x(i) = 0.0_real64
        else
          angle = pi * real(n - i - 1, real64) / real(n - 1, real64)
          x(i) = cos(angle)
        end if
      case (3)
        angle = pi * real(2 * n - 2 * i - 1, real64) / real(2 * n + 1, real64)
        x(i) = cos(angle)
      case (4)
        angle = pi * real(2 * n - 2 * i, real64) / real(2 * n + 1, real64)
        x(i) = cos(angle)
      case (5)
        x(i) = real(-n + 1 + 2 * i, real64) / real(n + 1, real64)
      case (6)
        if (n == 1) then
          x(i) = 0.0_real64
        else
          x(i) = real(-n + 1 + 2 * i, real64) / real(n - 1, real64)
        end if
      case (7)
        x(i) = real(-n + 1 + 2 * i, real64) / real(n, real64)
      case (8)
        angle = pi * real(2 * n - 1 - 2 * i, real64) / real(2 * n, real64)
        x(i) = cos(angle)
      case (9)
        angle = pi * real(n - i, real64) / real(n + 1, real64)
        x(i) = cos(angle)
      end select
    end do
  end subroutine make_nodes

  function lebesgue_constant(n, x, nfun, xfun) result(lmax)
    integer, intent(in) :: n, nfun
    real(real64), intent(in) :: x(0:n-1), xfun(0:nfun-1)
    real(real64) :: lmax
    real(real64), allocatable :: linterp(:)
    real(real64) :: t
    integer :: j, i1, i2

    if (n == 1) then
      lmax = 1.0_real64
      return
    end if

    allocate(linterp(0:n*nfun-1))
    lmax = 0.0_real64
    !$omp target data map(tofrom:lmax) map(to:x(0:n-1),xfun(0:nfun-1)) &
    !$omp& map(alloc:linterp(0:n*nfun-1))
    !$omp target teams distribute parallel do thread_limit(256) reduction(max:lmax) private(t,i1,i2)
    do j = 0, nfun - 1
      t = 0.0_real64
      do i1 = 0, n - 1
        linterp(i1*nfun+j) = 1.0_real64
        do i2 = 0, n - 1
          if (i1 /= i2) then
            linterp(i1*nfun+j) = linterp(i1*nfun+j) * &
              (xfun(j) - x(i2)) / (x(i1) - x(i2))
          end if
        end do
        t = t + abs(linterp(i1*nfun+j))
      end do
      lmax = max(lmax, t)
    end do
    !$omp end target teams distribute parallel do
    !$omp end target data
    deallocate(linterp)
  end function lebesgue_constant

  logical function verify_result(res, n, x, nfun, xfun) result(ok)
    integer, intent(in) :: n, nfun
    real(real64), intent(in) :: res, x(0:n-1), xfun(0:nfun-1)
    real(real64), allocatable :: linterp(:)
    real(real64) :: lmax, t
    integer :: j, i1, i2

    allocate(linterp(0:n*nfun-1))
    lmax = 0.0_real64
    do j = 0, nfun - 1
      do i1 = 0, n - 1
        linterp(i1*nfun+j) = 1.0_real64
      end do
      do i1 = 0, n - 1
        do i2 = 0, n - 1
          if (i1 /= i2) then
            linterp(i1*nfun+j) = linterp(i1*nfun+j) * &
              (xfun(j) - x(i2)) / (x(i1) - x(i2))
          end if
        end do
      end do
      t = 0.0_real64
      do i1 = 0, n - 1
        t = t + abs(linterp(i1*nfun+j))
      end do
      lmax = max(lmax, t)
    end do
    ok = abs(res - lmax) <= 1.0e-6_real64
    deallocate(linterp)
  end function verify_result

  subroutine run_test(point_kind, nfun, xfun)
    integer, intent(in) :: point_kind, nfun
    real(real64), intent(in) :: xfun(0:nfun-1)
    real(real64), allocatable :: x(:)
    real(real64) :: l(0:10)
    real(real32) :: total_time
    integer :: n, c0, c1, rate
    logical :: ok
    character(len=16), parameter :: names(9) = [ character(len=16) :: &
      'Chebyshev1', 'Chebyshev2', 'Chebyshev3', 'Chebyshev4', &
      'Equidistant1', 'Equidistant2', 'Equidistant3', 'Fejer1', 'Fejer2' ]

    write(*,'(/,a,i2,a)') 'LEBESGUE_TEST', point_kind, ':'
    write(*,'(a,a,a)') '  Analyze ', trim(names(point_kind)), ' points.'
    total_time = 0.0_real32
    ok = .true.
    do n = 1, 11
      allocate(x(0:n-1))
      call make_nodes(point_kind, n, x)
      call system_clock(c0, rate)
      l(n-1) = lebesgue_constant(n, x, nfun, xfun)
      call system_clock(c1)
      total_time = total_time + real(c1-c0, real32) / real(rate, real32)
      ok = ok .and. verify_result(l(n-1), n, x, nfun, xfun)
      deallocate(x)
    end do
    write(*,'(a,f12.6,a)') '  Total kernel execution time ', total_time, ' (s)'
    if (ok) then
      write(*,'(a)') '  PASS'
    else
      write(*,'(a)') '  FAIL'
      error stop 2
    end if
  end subroutine run_test

  subroutine timestamp()
    integer :: values(8)
    call date_and_time(values=values)
    write(*,'(i4.4,a,i2.2,a,i2.2,1x,i2.2,a,i2.2,a,i2.2)') &
      values(1), '-', values(2), '-', values(3), values(5), ':', values(6), ':', values(7)
  end subroutine timestamp
end module lebesgue_mod

program main
  use iso_fortran_env, only: real64
  use lebesgue_mod
  implicit none
  integer :: nfun, repeat, iteration, point_kind, i
  real(real64), allocatable :: xfun(:)
  character(len=64) :: arg

  if (command_argument_count() /= 2) then
    write(*,'(a)') 'Usage: ./main <number of points in an interval> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg,*) nfun
  call get_command_argument(2, arg)
  read(arg,*) repeat
  if (nfun < 1 .or. repeat < 1) error stop 'arguments must be positive'

  allocate(xfun(0:nfun-1))
  if (nfun == 1) then
    xfun(0) = 0.0_real64
  else
    do i = 0, nfun - 1
      xfun(i) = (real(nfun-1-i,real64) * (-1.0_real64) + &
                 real(i,real64)) / real(nfun-1,real64)
    end do
  end if

  write(*,'(/,a)') 'LEBESGUE_TEST'
  do iteration = 1, repeat
    call timestamp()
    do point_kind = 1, 9
      call run_test(point_kind, nfun, xfun)
    end do
    call timestamp()
  end do
end program main
