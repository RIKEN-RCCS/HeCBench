module libor_device
  use iso_fortran_env, only: real32
  implicit none
  integer, parameter :: NN=80, NMAT=40, L2_SIZE=3280, NOPT=15, NPATH=96000
!$omp declare target(path_calc,path_calc_b1,path_calc_b2,portfolio,portfolio_b)
contains
  subroutine path_calc(l, z, lambda, delta, nmat, n)
    real(real32), intent(inout) :: l(0:*)
    real(real32), intent(in) :: z(0:*), lambda(0:*), delta
    integer, intent(in) :: nmat, n
    integer :: i, m
    real(real32) :: sqez, lam, con1, v, vrat
    do m = 0, nmat-1
      sqez = sqrt(delta) * z(m)
      v = 0.0_real32
      do i = m+1, n-1
        lam = lambda(i-m-1)
        con1 = delta * lam
        v = v + con1*l(i) / (1.0_real32 + delta*l(i))
        vrat = exp(con1*v + lam*(sqez - 0.5_real32*con1))
        l(i) = l(i) * vrat
      end do
    end do
  end subroutine path_calc

  subroutine path_calc_b1(l, z, l2, lambda, delta, nmat, n)
    real(real32), intent(inout) :: l(0:*)
    real(real32), intent(in) :: z(0:*), lambda(0:*), delta
    real(real32), intent(out) :: l2(0:*)
    integer, intent(in) :: nmat, n
    integer :: i, m
    real(real32) :: sqez, lam, con1, v, vrat
    do i = 0, n-1
      l2(i) = l(i)
    end do
    do m = 0, nmat-1
      sqez = sqrt(delta) * z(m)
      v = 0.0_real32
      do i = m+1, n-1
        lam = lambda(i-m-1)
        con1 = delta * lam
        v = v + con1*l(i) / (1.0_real32 + delta*l(i))
        vrat = exp(con1*v + lam*(sqez - 0.5_real32*con1))
        l(i) = l(i) * vrat
        l2(i+(m+1)*n) = l(i)
      end do
    end do
  end subroutine path_calc_b1

  subroutine path_calc_b2(l_b, z, l2, lambda, delta, nmat, n)
    real(real32), intent(inout) :: l_b(0:*)
    real(real32), intent(in) :: z(0:*), l2(0:*), lambda(0:*), delta
    integer, intent(in) :: nmat, n
    integer :: i, m
    real(real32) :: faci, v1
    do m = nmat-1, 0, -1
      v1 = 0.0_real32
      do i = n-1, m+1, -1
        v1 = v1 + lambda(i-m-1) * l2(i+(m+1)*n) * l_b(i)
        faci = delta / (1.0_real32 + delta*l2(i+m*n))
        l_b(i) = l_b(i) * (l2(i+(m+1)*n)/l2(i+m*n)) + v1 * lambda(i-m-1) * faci*faci
      end do
    end do
  end subroutine path_calc_b2

  real(real32) function portfolio(l, lambda, maturities, swaprates, delta, nmat, n, nopt)
    real(real32), intent(in) :: l(0:*), lambda(0:*), swaprates(0:*), delta
    integer, intent(in) :: maturities(0:*), nmat, n, nopt
    integer :: i, m, nn
    real(real32) :: b, s, swapval, bb(0:39), ss(0:39)
    b = 1.0_real32
    s = 0.0_real32
    do nn = nmat, n-1
      b = b / (1.0_real32 + delta*l(nn))
      s = s + delta*b
      bb(nn-nmat) = b
      ss(nn-nmat) = s
    end do
    portfolio = 0.0_real32
    do i = 0, nopt-1
      m = maturities(i) - 1
      swapval = bb(m) + swaprates(i)*ss(m) - 1.0_real32
      if (swapval < 0.0_real32) portfolio = portfolio - 100.0_real32*swapval
    end do
    b = 1.0_real32
    do nn = 0, nmat-1
      b = b / (1.0_real32 + delta*l(nn))
    end do
    portfolio = b * portfolio
  end function portfolio

  real(real32) function portfolio_b(l, lambda, maturities, swaprates, delta, nmat, n, nopt)
    real(real32), intent(inout) :: l(0:*)
    real(real32), intent(in) :: lambda(0:*), swaprates(0:*), delta
    integer, intent(in) :: maturities(0:*), nmat, n, nopt
    integer :: mm, nn
    real(real32) :: b, s, swapval, bb(0:39), ss(0:39), b_b(0:39), s_b(0:39)
    b = 1.0_real32
    s = 0.0_real32
    do mm = 0, n-nmat-1
      nn = mm + nmat
      b = b / (1.0_real32 + delta*l(nn))
      s = s + delta*b
      bb(mm) = b
      ss(mm) = s
    end do
    portfolio_b = 0.0_real32
    b_b = 0.0_real32
    s_b = 0.0_real32
    do nn = 0, nopt-1
      mm = maturities(nn) - 1
      swapval = bb(mm) + swaprates(nn)*ss(mm) - 1.0_real32
      if (swapval < 0.0_real32) then
        portfolio_b = portfolio_b - 100.0_real32*swapval
        s_b(mm) = s_b(mm) - 100.0_real32*swaprates(nn)
        b_b(mm) = b_b(mm) - 100.0_real32
      end if
    end do
    do mm = n-nmat-1, 0, -1
      nn = mm + nmat
      b_b(mm) = b_b(mm) + delta*s_b(mm)
      l(nn) = -b_b(mm)*bb(mm)*(delta/(1.0_real32+delta*l(nn)))
      if (mm > 0) then
        s_b(mm-1) = s_b(mm-1) + s_b(mm)
        b_b(mm-1) = b_b(mm-1) + b_b(mm)/(1.0_real32+delta*l(nn))
      end if
    end do
    b = 1.0_real32
    do nn = 0, nmat-1
      b = b / (1.0_real32 + delta*l(nn))
    end do
    portfolio_b = b * portfolio_b
    do nn = 0, nmat-1
      l(nn) = -portfolio_b * delta / (1.0_real32 + delta*l(nn))
    end do
    do nn = nmat, n-1
      l(nn) = b * l(nn)
    end do
  end function portfolio_b
end module libor_device

program main
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use libor_device
  implicit none
  integer, parameter :: BLOCK_SIZE=64, GRID_SIZE=1500
  integer :: repeat, tid, path, threadn, i, iter
  integer :: h_n, h_nmat, h_nopt
  integer :: h_maturities(0:NOPT-1)
  character(len=64) :: arg
  real(real32) :: h_lambda(0:NN-1), h_swaprates(0:NOPT-1), h_delta
  real(real32), allocatable :: h_v(:), h_lb(:)
  real(real32) :: l(0:NN-1), z(0:NN-1), l2(0:L2_SIZE-1)
  real(real64) :: start_time, elapsed, v, lb
  logical :: ok

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) repeat
  h_delta = 0.25_real32
  h_n = NN; h_nmat = NMAT; h_nopt = NOPT
  h_maturities = [4,4,4,8,8,8,20,20,20,28,28,28,40,40,40]
  h_swaprates = [.045_real32,.05_real32,.055_real32,.045_real32,.05_real32,.055_real32,.045_real32,.05_real32, &
                 .055_real32,.045_real32,.05_real32,.055_real32,.045_real32,.05_real32,.055_real32]
  h_lambda = 0.2_real32
  allocate(h_v(0:NPATH-1), h_lb(0:NPATH-1))
  ok = .true.

!$omp target data map(to:h_maturities(0:NOPT-1),h_swaprates(0:NOPT-1),h_lambda(0:NN-1)) map(alloc:h_v(0:NPATH-1),h_lb(0:NPATH-1))
  start_time = omp_get_wtime()
  do iter = 1, repeat
!$omp target teams distribute parallel do num_teams(GRID_SIZE) thread_limit(BLOCK_SIZE) &
!$omp& map(to:h_lambda(0:NN-1),h_maturities(0:NOPT-1),h_swaprates(0:NOPT-1)) &
!$omp& map(tofrom:h_v(0:NPATH-1)) private(threadn,path,i,l,z)
    do tid = 0, GRID_SIZE*BLOCK_SIZE-1
      threadn = GRID_SIZE * BLOCK_SIZE
      do path = tid, NPATH-1, threadn
        do i = 0, h_n-1
          z(i) = 0.3_real32
          l(i) = 0.05_real32
        end do
        call path_calc(l, z, h_lambda, h_delta, h_nmat, h_n)
        h_v(path) = portfolio(l, h_lambda, h_maturities, h_swaprates, h_delta, h_nmat, h_n, h_nopt)
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average kernel execution time : ', elapsed / repeat, ' (s)'
!$omp target update from(h_v(0:NPATH-1))
  v = sum(real(h_v, real64)) / real(NPATH, real64)
  if (abs(v - 224.323_real64) > 1.0e-3_real64) then
    ok = .false.
    print '(a,f15.3)', 'Expected: 224.323 Actual ', v
  end if

  start_time = omp_get_wtime()
  do iter = 1, repeat
!$omp target teams distribute parallel do num_teams(GRID_SIZE) thread_limit(BLOCK_SIZE) &
!$omp& map(to:h_lambda(0:NN-1),h_maturities(0:NOPT-1),h_swaprates(0:NOPT-1)) &
!$omp& map(tofrom:h_v(0:NPATH-1),h_lb(0:NPATH-1)) private(threadn,path,i,l,z,l2)
    do tid = 0, GRID_SIZE*BLOCK_SIZE-1
      threadn = GRID_SIZE * BLOCK_SIZE
      do path = tid, NPATH-1, threadn
        do i = 0, h_n-1
          z(i) = 0.3_real32
          l(i) = 0.05_real32
        end do
        call path_calc_b1(l, z, l2, h_lambda, h_delta, h_nmat, h_n)
        h_v(path) = portfolio_b(l, h_lambda, h_maturities, h_swaprates, h_delta, h_nmat, h_n, h_nopt)
        call path_calc_b2(l, z, l2, h_lambda, h_delta, h_nmat, h_n)
        h_lb(path) = l(NN-1)
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average kernel execution time : ', elapsed / repeat, ' (s)'
!$omp target update from(h_lb(0:NPATH-1))
!$omp target update from(h_v(0:NPATH-1))
!$omp end target data

  v = sum(real(h_v, real64)) / real(NPATH, real64)
  lb = sum(real(h_lb, real64)) / real(NPATH, real64)
  if (abs(v - 224.323_real64) > 1.0e-3_real64) then
    ok = .false.
    print '(a,f15.3)', 'Expected: 224.323 Actual ', v
  end if
  if (abs(lb - 21.348_real64) > 1.0e-3_real64) then
    ok = .false.
    print '(a,f15.3)', 'Expected:  21.348 Actual ', lb
  end if
end program main
