module perplexity_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine perplexity_search(distances, p, perplexity, epochs, tol, n, k, ktime)
    integer, intent(in) :: n, k, epochs
    real(real32), intent(in) :: distances(0:n*k-1), perplexity, tol
    real(real32), intent(inout) :: p(0:n*k-1)
    real(real64), intent(inout) :: ktime
    integer :: i, j, step, ik
    real(real32) :: desired_entropy, beta_min, beta_max, beta, sum_pi
    real(real32) :: sum_disti_pi, divv, entropy, entropy_diff
    real(real64) :: t0, t1

    desired_entropy = log(perplexity)
    t0 = seconds()
    !$omp target teams distribute parallel do private(j,step,ik,beta_min,beta_max,beta,sum_pi,sum_disti_pi,divv,entropy,entropy_diff) thread_limit(256)
    do i = 0, n-1
      beta_min = -huge(1.0_real32)
      beta_max = huge(1.0_real32)
      beta = 1.0_real32
      ik = i * k
      do step = 0, epochs-1
        sum_pi = epsilon(1.0_real32)
        do j = 0, k-1
          p(ik+j) = exp(-distances(ik+j) * beta)
          sum_pi = sum_pi + p(ik+j)
        end do
        sum_disti_pi = 0.0_real32
        divv = 1.0_real32 / sum_pi
        do j = 0, k-1
          p(ik+j) = p(ik+j) * divv
          sum_disti_pi = sum_disti_pi + distances(ik+j) * p(ik+j)
        end do
        entropy = log(sum_pi) + beta * sum_disti_pi
        entropy_diff = entropy - desired_entropy
        if (abs(entropy_diff) <= tol) exit
        if (entropy_diff > 0.0_real32) then
          beta_min = beta
          if (beta_max >= huge(1.0_real32) * 0.5_real32) then
            beta = beta * 2.0_real32
          else
            beta = (beta + beta_max) * 0.5_real32
          end if
        else
          beta_max = beta
          if (beta_min <= -huge(1.0_real32) * 0.5_real32) then
            beta = beta * 0.5_real32
          else
            beta = (beta + beta_min) * 0.5_real32
          end if
        end if
      end do
    end do
    !$omp end target teams distribute parallel do
    t1 = seconds()
    ktime = ktime + (t1 - t0) * 1.0e9_real64
  end subroutine

  subroutine reference(distances, p, perplexity, epochs, tol, n, k)
    integer, intent(in) :: n, k, epochs
    real(real32), intent(in) :: distances(0:n*k-1), perplexity, tol
    real(real32), intent(out) :: p(0:n*k-1)
    real(real64) :: dummy
    dummy = 0.0_real64
    call perplexity_search_host(distances, p, perplexity, epochs, tol, n, k)
  end subroutine

  subroutine perplexity_search_host(distances, p, perplexity, epochs, tol, n, k)
    integer, intent(in) :: n, k, epochs
    real(real32), intent(in) :: distances(0:n*k-1), perplexity, tol
    real(real32), intent(out) :: p(0:n*k-1)
    integer :: i, j, step, ik
    real(real32) :: desired_entropy, beta_min, beta_max, beta, sum_pi
    real(real32) :: sum_disti_pi, divv, entropy, entropy_diff
    desired_entropy = log(perplexity)
    do i = 0, n-1
      beta_min = -huge(1.0_real32); beta_max = huge(1.0_real32); beta = 1.0_real32; ik = i*k
      do step = 0, epochs-1
        sum_pi = epsilon(1.0_real32)
        do j = 0, k-1
          p(ik+j) = exp(-distances(ik+j) * beta); sum_pi = sum_pi + p(ik+j)
        end do
        sum_disti_pi = 0.0_real32; divv = 1.0_real32 / sum_pi
        do j = 0, k-1
          p(ik+j) = p(ik+j) * divv; sum_disti_pi = sum_disti_pi + distances(ik+j) * p(ik+j)
        end do
        entropy = log(sum_pi) + beta * sum_disti_pi
        entropy_diff = entropy - desired_entropy
        if (abs(entropy_diff) <= tol) exit
        if (entropy_diff > 0.0_real32) then
          beta_min = beta
          if (beta_max >= huge(1.0_real32) * 0.5_real32) then; beta = beta*2.0_real32; else; beta = (beta+beta_max)*0.5_real32; end if
        else
          beta_max = beta
          if (beta_min <= -huge(1.0_real32) * 0.5_real32) then; beta = beta*0.5_real32; else; beta = (beta+beta_min)*0.5_real32; end if
        end if
      end do
    end do
  end subroutine
end module

program main
  use perplexity_mod
  implicit none
  integer :: n, perpl, repeat, n_nbrs, max_iter, i, ios
  character(len=64) :: arg
  real(real32), parameter :: tol = 1.0e-8_real32
  real(real32), allocatable :: data(:), h_data(:), distance(:)
  real(real64) :: ktime
  logical :: ok

  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: main <number of points> <perplexity> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) n
  call get_command_argument(2,arg); read(arg,*,iostat=ios) perpl
  call get_command_argument(3,arg); read(arg,*,iostat=ios) repeat
  n_nbrs = 4 * perpl
  max_iter = 100
  allocate(data(0:n*n_nbrs-1), h_data(0:n*n_nbrs-1), distance(0:n*n_nbrs-1))
  call c_srand(123_int32)
  do i = 0, n*n_nbrs-1
    distance(i) = real(c_rand(), real32) / 2147483647.0_real32
  end do

  !$omp target data map(from:data(0:n*n_nbrs-1)) map(to:distance(0:n*n_nbrs-1))
  ktime = 0.0_real64
  do i = 1, repeat
    call perplexity_search(distance, data, real(perpl,real32), max_iter, tol, n, n_nbrs, ktime)
  end do
  print '(a,f10.6,a)', 'Average kernel execution time: ', (ktime * 1.0e-9_real64) / real(repeat,real64), ' (s)'
  !$omp end target data

  call reference(distance, h_data, real(perpl,real32), max_iter, tol, n, n_nbrs)
  ok = .true.
  do i = 0, n*n_nbrs-1
    if (abs(data(i) - h_data(i)) > 1.0e-3_real32) then
      print '(i0,1x,f0.6,1x,f0.6)', i, data(i), h_data(i)
      ok = .false.
      exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
end program
