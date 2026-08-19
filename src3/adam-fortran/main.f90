program adam
  use iso_fortran_env, only: real32, real64
  use omp_lib
  implicit none
  integer :: n, steps, repeat, i, t, r
  character(len=64) :: arg
  real(real32), allocatable :: m(:), v(:), g(:), p(:), refp(:), refm(:), refv(:)
  real(real32), parameter :: step_size = 1.e-3_real32, decay = .5_real32
  real(real32), parameter :: beta1 = .9_real32, beta2 = .999_real32
  real(real32), parameter :: eps = 1.e-10_real32, grad_scale = 256._real32
  real(real32) :: scaled_grad, mc, vc, denom, update
  real(real64) :: start_time, elapsed, cr, cp

  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: ./main <vector size> <number of time steps> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) n
  call get_command_argument(2, arg); read(arg, *) steps
  call get_command_argument(3, arg); read(arg, *) repeat
  allocate(m(0:n-1), v(0:n-1), g(0:n-1), p(0:n-1), refp(0:n-1), refm(0:n-1), refv(0:n-1))
  call random_seed(); call random_number(m); call random_number(v)
  call random_number(g); call random_number(p)
  refp = p; refm = m; refv = v

!$omp target data map(to: m(0:n-1), v(0:n-1), g(0:n-1)) map(tofrom: p(0:n-1))
  start_time = omp_get_wtime()
  do r = 1, repeat
!$omp target teams distribute parallel do thread_limit(256) private(t,scaled_grad,mc,vc,denom,update)
    do i = 0, n - 1
      do t = 1, steps
        scaled_grad = g(i) / grad_scale
        m(i) = beta1 * m(i) + (1._real32 - beta1) * scaled_grad
        v(i) = beta2 * v(i) + (1._real32 - beta2) * scaled_grad * scaled_grad
        mc = m(i) / (1._real32 - beta1 ** t)
        vc = v(i) / (1._real32 - beta2 ** t)
        denom = sqrt(vc + eps)
        update = mc / denom + decay * p(i)
        p(i) = p(i) - step_size * update
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f12.6,a)', 'Average kernel execution time ', elapsed * 1.e3_real64 / real(repeat, real64), ' (ms)'
!$omp end target data

  do r = 1, repeat
    do i = 0, n - 1
      do t = 1, steps
        scaled_grad = g(i) / grad_scale
        refm(i) = beta1 * refm(i) + (1._real32 - beta1) * scaled_grad
        refv(i) = beta2 * refv(i) + (1._real32 - beta2) * scaled_grad * scaled_grad
        mc = refm(i) / (1._real32 - beta1 ** t)
        vc = refv(i) / (1._real32 - beta2 ** t)
        denom = sqrt(vc + eps)
        update = mc / denom + decay * refp(i)
        refp(i) = refp(i) - step_size * update
      end do
    end do
  end do
  if (all(abs(refp - p) <= 1.e-3_real32)) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
  cr = sum(real(refp, real64)) / real(n, real64)
  cp = sum(real(p, real64)) / real(n, real64)
  print '(a,2f16.8)', 'Checksum: ', cr, cp
end program adam
