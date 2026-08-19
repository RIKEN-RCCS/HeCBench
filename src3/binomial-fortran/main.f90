! Faithful Fortran/OpenMP target port of binomial-omp.
module binomial_mod
  use, intrinsic :: iso_fortran_env, only : real32
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib
  implicit none

  integer, parameter :: sp = real32
  integer, parameter :: num_steps = 2048
  integer, parameter :: max_options = 1024
  integer, parameter :: num_iterations = 1000
  integer, parameter :: threadblock_size = 128
  integer, parameter :: elems_per_thread = num_steps / threadblock_size

  type :: option_data_t
    real(sp) :: s, x, t, r, v
  end type option_data_t

  type :: device_option_data_t
    real(sp) :: s, x, vdt, pu_by_df, pd_by_df
  end type device_option_data_t

  !$omp declare target (expiry_call_value)

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    integer(c_int) function c_rand() bind(C, name="rand")
      import :: c_int
    end function c_rand
  end interface

contains

  function rand_data(low, high) result(value)
    real(sp), intent(in) :: low, high
    real(sp) :: value, t

    t = real(c_rand(), sp) / real(2147483647_c_int, sp)
    value = (1.0_sp - t) * low + t * high
  end function rand_data

  function cnd(d) result(value)
    real(sp), intent(in) :: d
    real(sp) :: value, k
    real(sp), parameter :: a1 = 0.31938153_sp
    real(sp), parameter :: a2 = -0.356563782_sp
    real(sp), parameter :: a3 = 1.781477937_sp
    real(sp), parameter :: a4 = -1.821255978_sp
    real(sp), parameter :: a5 = 1.330274429_sp
    real(sp), parameter :: rsqrt2pi = 0.39894228040143267793994605993438_sp

    k = 1.0_sp / (1.0_sp + 0.2316419_sp * abs(d))
    value = rsqrt2pi * exp(-0.5_sp * d * d) * &
      (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))
    if (d > 0.0_sp) value = 1.0_sp - value
  end function cnd

  subroutine black_scholes_call(call_result, option_data)
    real(sp), intent(out) :: call_result
    type(option_data_t), intent(in) :: option_data
    real(sp) :: sqrt_t, d1, d2, cnd_d1, cnd_d2, exp_rt

    sqrt_t = sqrt(option_data%t)
    d1 = (log(option_data%s / option_data%x) + &
      (option_data%r + 0.5_sp * option_data%v * option_data%v) * option_data%t) / &
      (option_data%v * sqrt_t)
    d2 = d1 - option_data%v * sqrt_t
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)
    exp_rt = exp(-option_data%r * option_data%t)
    call_result = option_data%s * cnd_d1 - option_data%x * exp_rt * cnd_d2
  end subroutine black_scholes_call

  function expiry_call_value(s, x, vdt, index) result(value)
    real(sp), intent(in) :: s, x, vdt
    integer, intent(in) :: index
    real(sp) :: value, d

    d = s * exp(vdt * real(2 * index - num_steps, sp)) - x
    if (d > 0.0_sp) then
      value = d
    else
      value = 0.0_sp
    end if
  end function expiry_call_value

  subroutine binomial_options_cpu(call_result, option_data)
    real(sp), intent(out) :: call_result
    type(option_data_t), intent(in) :: option_data
    real(sp) :: call(0:num_steps)
    real(sp) :: dt, vdt, rdt, interest_factor, discount_factor
    real(sp) :: u, d, pu, pd, pu_by_df, pd_by_df
    integer :: i, j

    dt = option_data%t / real(num_steps, sp)
    vdt = option_data%v * sqrt(dt)
    rdt = option_data%r * dt
    interest_factor = exp(rdt)
    discount_factor = exp(-rdt)
    u = exp(vdt)
    d = exp(-vdt)
    pu = (interest_factor - d) / (u - d)
    pd = 1.0_sp - pu
    pu_by_df = pu * discount_factor
    pd_by_df = pd * discount_factor

    do i = 0, num_steps
      call(i) = expiry_call_value(option_data%s, option_data%x, vdt, i)
    end do
    do i = num_steps, 1, -1
      do j = 0, i - 1
        call(j) = pu_by_df * call(j + 1) + pd_by_df * call(j)
      end do
    end do
    call_result = call(0)
  end subroutine binomial_options_cpu

  subroutine binomial_options_gpu(call_value, option_data, opt_n, iterations)
    real(sp), intent(out) :: call_value(0:max_options-1)
    type(option_data_t), intent(in) :: option_data(0:max_options-1)
    integer, intent(in) :: opt_n, iterations
    type(device_option_data_t) :: device_option_data(0:max_options-1)
    real(sp) :: dt, vdt, rdt, interest_factor, discount_factor
    real(sp) :: u, d, pu, pd, gpu_time
    real(sp) :: call_exchange(0:threadblock_size)
    real(sp) :: call(0:elems_per_thread)
    real(sp) :: s, x, pu_by_df, pd_by_df
    real(8) :: start_time, end_time
    integer :: option_index, iteration, tid, final_it, i, j

    do option_index = 0, opt_n - 1
      dt = option_data(option_index)%t / real(num_steps, sp)
      vdt = option_data(option_index)%v * sqrt(dt)
      rdt = option_data(option_index)%r * dt
      interest_factor = exp(rdt)
      discount_factor = exp(-rdt)
      u = exp(vdt)
      d = exp(-vdt)
      pu = (interest_factor - d) / (u - d)
      pd = 1.0_sp - pu
      device_option_data(option_index)%s = option_data(option_index)%s
      device_option_data(option_index)%x = option_data(option_index)%x
      device_option_data(option_index)%vdt = vdt
      device_option_data(option_index)%pu_by_df = pu * discount_factor
      device_option_data(option_index)%pd_by_df = pd * discount_factor
    end do

    !$omp target data map(to: device_option_data) map(from: call_value)
      start_time = omp_get_wtime()
      do iteration = 1, iterations
        ! One team per option and 128 threads per team, matching kernel.cpp.
        !$omp target teams distribute num_teams(opt_n) thread_limit(threadblock_size) &
        !$omp& private(call_exchange)
        do option_index = 0, opt_n - 1
          !$omp parallel default(shared) private(tid, call, s, x, vdt, pu_by_df, &
          !$omp& pd_by_df, final_it, i, j)
            tid = omp_get_thread_num()
            s = device_option_data(option_index)%s
            x = device_option_data(option_index)%x
            vdt = device_option_data(option_index)%vdt
            pu_by_df = device_option_data(option_index)%pu_by_df
            pd_by_df = device_option_data(option_index)%pd_by_df

            do i = 0, elems_per_thread - 1
              call(i) = expiry_call_value(s, x, vdt, tid * elems_per_thread + i)
            end do
            if (tid == 0) then
              call_exchange(threadblock_size) = expiry_call_value(s, x, vdt, num_steps)
            end if

            final_it = max(0, tid * elems_per_thread - 1)
            do i = num_steps, 1, -1
              call_exchange(tid) = call(0)
              !$omp barrier
              call(elems_per_thread) = call_exchange(tid + 1)
              !$omp barrier
              if (i > final_it) then
                do j = 0, elems_per_thread - 1
                  call(j) = pu_by_df * call(j + 1) + pd_by_df * call(j)
                end do
              end if
            end do
            if (tid == 0) call_value(option_index) = call(0)
          !$omp end parallel
        end do
        !$omp end target teams distribute
      end do
      end_time = omp_get_wtime()
      gpu_time = real((end_time - start_time) * 1.0e6_8 / real(iterations, 8), sp)
      write (*, '(A,F0.6,A)') 'Average kernel execution time : ', gpu_time, ' (us)'
    !$omp end target data
  end subroutine binomial_options_gpu

end module binomial_mod

program binomial
  use, intrinsic :: iso_c_binding, only : c_int
  use binomial_mod
  use omp_lib
  implicit none

  type(option_data_t) :: option_data(0:max_options-1)
  real(sp) :: call_value_bs(0:max_options-1), call_value_gpu(0:max_options-1)
  real(sp) :: call_value_cpu(0:max_options-1)
  real(sp) :: sum_delta, sum_ref, error_val, gpu_time
  real(8) :: start_time, end_time
  integer :: i

  write (*, '(A)') '[./main] - Starting...'
  write (*, '(A)') 'Generating input data...'
  call c_srand(123_c_int)
  do i = 0, max_options - 1
    option_data(i)%s = rand_data(5.0_sp, 30.0_sp)
    option_data(i)%x = rand_data(1.0_sp, 100.0_sp)
    option_data(i)%t = rand_data(0.25_sp, 10.0_sp)
    option_data(i)%r = 0.06_sp
    option_data(i)%v = 0.10_sp
    call black_scholes_call(call_value_bs(i), option_data(i))
  end do

  write (*, '(A)') 'Running GPU binomial tree...'
  start_time = omp_get_wtime()
  call binomial_options_gpu(call_value_gpu, option_data, max_options, num_iterations)
  end_time = omp_get_wtime()
  gpu_time = real(end_time - start_time, sp)
  write (*, '(A,I0)') 'Options count            : ', max_options
  write (*, '(A,I0)') 'Time steps               : ', num_steps
  write (*, '(A,F0.6,A)') 'Total binomialOptionsGPU() time: ', gpu_time * 1000.0_sp, ' msec'
  write (*, '(A,F0.6)') 'Options per second       : ', real(max_options, sp) / gpu_time

  write (*, '(A)') 'Running CPU binomial tree...'
  do i = 0, max_options - 1
    call binomial_options_cpu(call_value_cpu(i), option_data(i))
  end do

  write (*, '(A)') 'Comparing the results...'
  write (*, '(A)') 'GPU binomial vs. Black-Scholes'
  sum_delta = sum(abs(call_value_bs - call_value_gpu))
  sum_ref = sum(abs(call_value_bs))
  if (sum_ref > 1.0e-5_sp) then
    write (*, '(A,ES12.5)') 'L1 norm: ', sum_delta / sum_ref
  else
    write (*, '(A,ES12.5)') 'Avg. diff: ', sum_delta / real(max_options, sp)
  end if

  write (*, '(A)') 'CPU binomial vs. Black-Scholes'
  sum_delta = sum(abs(call_value_bs - call_value_cpu))
  sum_ref = sum(abs(call_value_bs))
  if (sum_ref > 1.0e-5_sp) then
    write (*, '(A,ES12.5)') 'L1 norm: ', sum_delta / sum_ref
  else
    write (*, '(A,ES12.5)') 'Avg. diff: ', sum_delta / real(max_options, sp)
  end if

  write (*, '(A)') 'CPU binomial vs. GPU binomial'
  sum_delta = sum(abs(call_value_gpu - call_value_cpu))
  sum_ref = sum(call_value_cpu)
  error_val = sum_delta / sum_ref
  write (*, '(A,ES12.5)') 'Avg. diff: ', sum_delta / real(max_options, sp)
  write (*, '(A,ES12.5)') 'L1 norm: ', error_val
  if (error_val > 5.0e-4_sp) then
    write (*, '(A)') 'Test failed!'
    error stop 1
  end if
  write (*, '(A)') 'Test passed'
end program binomial
