module black_scholes_types
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32
  implicit none

  integer(int32), parameter :: call_option = 0_int32, put_option = 1_int32
  integer(int32), parameter :: num_diff_settings = 37_int32
  integer(int32), parameter :: thread_block_size = 256_int32

  type :: option_input
    integer(int32) :: option_type
    real(real32) :: strike, spot, q, r, t, vol, value, tol
  end type option_input

!$omp declare target (erf_approx, price_option)
contains

  pure real(real32) function erf_approx(x) result(value)
    real(real32), intent(in) :: x
    real(real32) :: r, s, p, q, z, ax
    real(real32), parameter :: tiny = 1.0e-21_real32
    real(real32), parameter :: one = 1.0_real32
    real(real32), parameter :: erx = 8.45062911510467529297e-1_real32
    real(real32), parameter :: efx = 1.28379167095512586316e-1_real32
    real(real32), parameter :: efx8 = 1.02703333676410069053_real32
    real(real32), parameter :: pp0 = 1.28379167095512558561e-1_real32
    real(real32), parameter :: pp1 = -3.25042107247001499370e-1_real32
    real(real32), parameter :: pp2 = -2.84817495755985104766e-2_real32
    real(real32), parameter :: pp3 = -5.77027029648944159157e-3_real32
    real(real32), parameter :: pp4 = -2.37630166566501626084e-5_real32
    real(real32), parameter :: qq1 = 3.97917223959155352819e-1_real32
    real(real32), parameter :: qq2 = 6.50222499887672944485e-2_real32
    real(real32), parameter :: qq3 = 5.08130628187576562776e-3_real32
    real(real32), parameter :: qq4 = 1.32494738004321644526e-4_real32
    real(real32), parameter :: qq5 = -3.96022827877536812320e-6_real32
    real(real32), parameter :: pa0 = -2.36211856075265944077e-3_real32
    real(real32), parameter :: pa1 = 4.14856118683748331666e-1_real32
    real(real32), parameter :: pa2 = -3.72207876035701323847e-1_real32
    real(real32), parameter :: pa3 = 3.18346619901161753674e-1_real32
    real(real32), parameter :: pa4 = -1.10894694282396677476e-1_real32
    real(real32), parameter :: pa5 = 3.54783043256182359371e-2_real32
    real(real32), parameter :: pa6 = -2.16637559486879084300e-3_real32
    real(real32), parameter :: qa1 = 1.06420880400844228286e-1_real32
    real(real32), parameter :: qa2 = 5.40397917702171048937e-1_real32
    real(real32), parameter :: qa3 = 7.18286544141962662868e-2_real32
    real(real32), parameter :: qa4 = 1.26171219808761642112e-1_real32
    real(real32), parameter :: qa5 = 1.36370839120290507362e-2_real32
    real(real32), parameter :: qa6 = 1.19844998467991074170e-2_real32
    real(real32), parameter :: ra0 = -9.86494403484714822705e-3_real32
    real(real32), parameter :: ra1 = -6.93858572707181764372e-1_real32
    real(real32), parameter :: ra2 = -1.05586262253232909814e1_real32
    real(real32), parameter :: ra3 = -6.23753324503260060396e1_real32
    real(real32), parameter :: ra4 = -1.62396669462573470355e2_real32
    real(real32), parameter :: ra5 = -1.84605092906711035994e2_real32
    real(real32), parameter :: ra6 = -8.12874355063065934246e1_real32
    real(real32), parameter :: ra7 = -9.81432934416914548592_real32
    real(real32), parameter :: sa1 = 1.96512716674392571292e1_real32
    real(real32), parameter :: sa2 = 1.37657754143519042600e2_real32
    real(real32), parameter :: sa3 = 4.34565877475229228821e2_real32
    real(real32), parameter :: sa4 = 6.45387271733267880336e2_real32
    real(real32), parameter :: sa5 = 4.29008140027567833386e2_real32
    real(real32), parameter :: sa6 = 1.08635005541779435134e2_real32
    real(real32), parameter :: sa7 = 6.57024977031928170135_real32
    real(real32), parameter :: sa8 = -6.04244152148580987438e-2_real32
    real(real32), parameter :: rb0 = -9.86494292470009928597e-3_real32
    real(real32), parameter :: rb1 = -7.99283237680523006574e-1_real32
    real(real32), parameter :: rb2 = -1.77579549177547519889e1_real32
    real(real32), parameter :: rb3 = -1.60636384855821916062e2_real32
    real(real32), parameter :: rb4 = -6.37566443368389627722e2_real32
    real(real32), parameter :: rb5 = -1.02509513161107724954e3_real32
    real(real32), parameter :: rb6 = -4.83519191608651397019e2_real32
    real(real32), parameter :: sb1 = 3.03380607434824582924e1_real32
    real(real32), parameter :: sb2 = 3.25792512996573918826e2_real32
    real(real32), parameter :: sb3 = 1.53672958608443695994e3_real32
    real(real32), parameter :: sb4 = 3.19985821950859553908e3_real32
    real(real32), parameter :: sb5 = 2.55305040643316442583e3_real32
    real(real32), parameter :: sb6 = 4.74528541206955367215e2_real32
    real(real32), parameter :: sb7 = -2.24409524465858183362e1_real32

    ax = abs(x)
    if (ax < 0.84375_real32) then
      if (ax < 3.7252902984e-9_real32) then
        if (ax < tiny * 16.0_real32) then
          value = 0.125_real32 * (8.0_real32*x + efx8*x)
        else
          value = x + efx*x
        end if
        return
      end if
      z = x*x
      r = pp0+z*(pp1+z*(pp2+z*(pp3+z*pp4)))
      s = one+z*(qq1+z*(qq2+z*(qq3+z*(qq4+z*qq5))))
      value = x + x*(r/s)
      return
    end if
    if (ax < 1.25_real32) then
      s = ax-one
      p = pa0+s*(pa1+s*(pa2+s*(pa3+s*(pa4+s*(pa5+s*pa6)))))
      q = one+s*(qa1+s*(qa2+s*(qa3+s*(qa4+s*(qa5+s*qa6)))))
      if (x >= 0.0_real32) then
        value = erx+p/q
      else
        value = -erx-p/q
      end if
      return
    end if
    if (ax >= 6.0_real32) then
      if (x >= 0.0_real32) then
        value = one-tiny
      else
        value = tiny-one
      end if
      return
    end if
    s = one/(ax*ax)
    if (ax < 2.85714285714285_real32) then
      r = ra0+s*(ra1+s*(ra2+s*(ra3+s*(ra4+s*(ra5+s*(ra6+s*ra7))))))
      q = one+s*(sa1+s*(sa2+s*(sa3+s*(sa4+s*(sa5+s*(sa6+s*(sa7+s*sa8)))))))
    else
      r = rb0+s*(rb1+s*(rb2+s*(rb3+s*(rb4+s*(rb5+s*rb6)))))
      q = one+s*(sb1+s*(sb2+s*(sb3+s*(sb4+s*(sb5+s*(sb6+s*sb7))))))
    end if
    r = exp(-ax*ax-0.5625_real32+r/q)
    if (x >= 0.0_real32) then
      value = one-r/ax
    else
      value = r/ax-one
    end if
  end function erf_approx

  pure real(real32) function price_option(thread_option) result(result_value)
    type(option_input), intent(in) :: thread_option
    real(real32) :: variance, dividend_discount, risk_free_discount, forward_price
    real(real32) :: std_dev, d1, d2, cum_d1, cum_d2, alpha, beta, discount
    real(real32) :: n_d1, n_d2, dalpha_dd1, dbeta_dd2
    real(real32), parameter :: inv_sqrt_pi = 0.564189583547756286948_real32
    real(real32), parameter :: sqrt_half = 0.7071067811865475244008443621048490392848359376887_real32

    variance = thread_option%vol*thread_option%vol*thread_option%t
    dividend_discount = 1.0_real32 / exp(thread_option%q*thread_option%t)
    risk_free_discount = 1.0_real32 / exp(thread_option%r*thread_option%t)
    forward_price = thread_option%spot * dividend_discount / risk_free_discount
    std_dev = sqrt(variance)
    d1 = log(forward_price/thread_option%strike)/std_dev + 0.5_real32*std_dev
    d2 = d1-std_dev
    cum_d1 = 0.5_real32*(1.0_real32+erf_approx(d1*sqrt_half))
    cum_d2 = 0.5_real32*(1.0_real32+erf_approx(d2*sqrt_half))
    ! The calculator builds these derivatives before selecting call/put alpha/beta.
    n_d1 = sqrt_half*inv_sqrt_pi*exp(-(d1*d1)/2.0_real32)
    n_d2 = sqrt_half*inv_sqrt_pi*exp(-(d2*d2)/2.0_real32)
    if (thread_option%option_type == call_option) then
      alpha = cum_d1
      dalpha_dd1 = n_d1
      beta = -cum_d2
      dbeta_dd2 = -n_d2
    else
      alpha = -1.0_real32+cum_d1
      dalpha_dd1 = n_d1
      beta = 1.0_real32-cum_d2
      dbeta_dd2 = -n_d2
    end if
    discount = risk_free_discount
    result_value = discount*(forward_price*alpha+thread_option%strike*beta)
    ! Preserve the source calculator's derivative evaluation without changing output.
    result_value = result_value + 0.0_real32*(dalpha_dd1+dbeta_dd2)
  end function price_option

  subroutine initialize_option(value, setting)
    type(option_input), intent(out) :: value
    integer(int32), intent(in) :: setting
    select case (setting)
    case (0);  value = option_input(call_option,40.00_real32,42.00_real32,0.08_real32,0.04_real32,0.75_real32,0.35_real32,5.0975_real32,1.0e-4_real32)
    case (1);  value = option_input(call_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.10_real32,0.15_real32,0.0205_real32,1.0e-4_real32)
    case (2);  value = option_input(call_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.10_real32,0.15_real32,1.8734_real32,1.0e-4_real32)
    case (3);  value = option_input(call_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.10_real32,0.15_real32,9.9413_real32,1.0e-4_real32)
    case (4);  value = option_input(call_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.10_real32,0.25_real32,0.3150_real32,1.0e-4_real32)
    case (5);  value = option_input(call_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.10_real32,0.25_real32,3.1217_real32,1.0e-4_real32)
    case (6);  value = option_input(call_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.10_real32,0.25_real32,10.3556_real32,1.0e-4_real32)
    case (7);  value = option_input(call_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.10_real32,0.35_real32,0.9474_real32,1.0e-4_real32)
    case (8);  value = option_input(call_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.10_real32,0.35_real32,4.3693_real32,1.0e-4_real32)
    case (9);  value = option_input(call_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.10_real32,0.35_real32,11.1381_real32,1.0e-4_real32)
    case (10); value = option_input(call_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.50_real32,0.15_real32,0.8069_real32,1.0e-4_real32)
    case (11); value = option_input(call_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.50_real32,0.15_real32,4.0232_real32,1.0e-4_real32)
    case (12); value = option_input(call_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.50_real32,0.15_real32,10.5769_real32,1.0e-4_real32)
    case (13); value = option_input(call_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.50_real32,0.25_real32,2.7026_real32,1.0e-4_real32)
    case (14); value = option_input(call_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.50_real32,0.25_real32,6.6997_real32,1.0e-4_real32)
    case (15); value = option_input(call_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.50_real32,0.25_real32,12.7857_real32,1.0e-4_real32)
    case (16); value = option_input(call_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.50_real32,0.35_real32,4.9329_real32,1.0e-4_real32)
    case (17); value = option_input(call_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.50_real32,0.35_real32,9.3679_real32,1.0e-4_real32)
    case (18); value = option_input(call_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.50_real32,0.35_real32,15.3086_real32,1.0e-4_real32)
    case (19); value = option_input(put_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.10_real32,0.15_real32,9.9210_real32,1.0e-4_real32)
    case (20); value = option_input(put_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.10_real32,0.15_real32,1.8734_real32,1.0e-4_real32)
    case (21); value = option_input(put_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.10_real32,0.15_real32,0.0408_real32,1.0e-4_real32)
    case (22); value = option_input(put_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.10_real32,0.25_real32,10.2155_real32,1.0e-4_real32)
    case (23); value = option_input(put_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.10_real32,0.25_real32,3.1217_real32,1.0e-4_real32)
    case (24); value = option_input(put_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.10_real32,0.25_real32,0.4551_real32,1.0e-4_real32)
    case (25); value = option_input(put_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.10_real32,0.35_real32,10.8479_real32,1.0e-4_real32)
    case (26); value = option_input(put_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.10_real32,0.35_real32,4.3693_real32,1.0e-4_real32)
    case (27); value = option_input(put_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.10_real32,0.35_real32,1.2376_real32,1.0e-4_real32)
    case (28); value = option_input(put_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.50_real32,0.15_real32,10.3192_real32,1.0e-4_real32)
    case (29); value = option_input(put_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.50_real32,0.15_real32,4.0232_real32,1.0e-4_real32)
    case (30); value = option_input(put_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.50_real32,0.15_real32,1.0646_real32,1.0e-4_real32)
    case (31); value = option_input(put_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.50_real32,0.25_real32,12.2149_real32,1.0e-4_real32)
    case (32); value = option_input(put_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.50_real32,0.25_real32,6.6997_real32,1.0e-4_real32)
    case (33); value = option_input(put_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.50_real32,0.25_real32,3.2734_real32,1.0e-4_real32)
    case (34); value = option_input(put_option,100.00_real32,90.00_real32,0.10_real32,0.10_real32,0.50_real32,0.35_real32,14.4452_real32,1.0e-4_real32)
    case (35); value = option_input(put_option,100.00_real32,100.00_real32,0.10_real32,0.10_real32,0.50_real32,0.35_real32,9.3679_real32,1.0e-4_real32)
    case (36); value = option_input(put_option,100.00_real32,110.00_real32,0.10_real32,0.10_real32,0.50_real32,0.35_real32,5.7963_real32,1.0e-4_real32)
    end select
  end subroutine initialize_option
end module black_scholes_types

program black_scholes
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32
  use black_scholes_types
  implicit none
  integer(int32), parameter :: num_vals = 50000000_int32
  character(len=64) :: argument
  integer :: argument_status, ios
  integer(int32) :: repeat, option_num, iteration
  integer(int64) :: start_count, end_count, kernel_start, kernel_end, clock_rate
  real(real32) :: mtime_cpu, mtime_gpu, ktime_gpu, total_result
  type(option_input), allocatable :: values(:)
  real(real32), allocatable :: output_vals(:)

  if (command_argument_count() /= 1) then
    print '(A)', 'Usage: ./main <repeat>'
    error stop 1
  end if
  call get_command_argument(1, argument, status=argument_status)
  if (argument_status /= 0) error stop 1
  read(argument, *, iostat=ios) repeat
  if (ios /= 0 .or. repeat <= 0_int32) then
    print '(A)', 'Usage: ./main <repeat>'
    error stop 1
  end if

  allocate(values(1:num_vals), output_vals(1:num_vals))
  do option_num = 1_int32, num_vals
    call initialize_option(values(option_num), modulo(option_num-1_int32, num_diff_settings))
  end do

  print '(A,I0)', 'Number of options: ', num_vals
  print *
  call system_clock(start_count, clock_rate)
!$omp target data map(to: values(1:num_vals)) map(from: output_vals(1:num_vals))
  call system_clock(kernel_start)
  do iteration = 1_int32, repeat
!$omp target teams distribute parallel do simd thread_limit(thread_block_size)
    do option_num = 1_int32, num_vals
      output_vals(option_num) = price_option(values(option_num))
    end do
!$omp end target teams distribute parallel do simd
  end do
  call system_clock(kernel_end)
!$omp end target data
  call system_clock(end_count)

  ktime_gpu = elapsed_ms(kernel_start, kernel_end, clock_rate) + 0.5_real32
  mtime_gpu = elapsed_ms(start_count, end_count, clock_rate) + 0.5_real32
  print '(A)', 'Run on GPU'
  print '(A,F0.6,A)', 'Average kernel execution time on GPU: ', ktime_gpu/real(repeat, real32), ' (ms)'
  mtime_gpu = mtime_gpu-ktime_gpu-ktime_gpu/real(repeat, real32)
  print '(A,F0.6,A)', 'Processing time on GPU: ', mtime_gpu, ' (ms)'
  total_result = sum(output_vals)
  print '(A,F0.6)', 'Summation of output prices on GPU: ', total_result
  print '(A,I0,A,F0.6)', 'Output price at index ', num_vals/2_int32, ' on GPU: ', output_vals(num_vals/2_int32+1_int32)
  print *

  call system_clock(start_count)
  do option_num = 1_int32, num_vals
    output_vals(option_num) = price_option(values(option_num))
  end do
  call system_clock(end_count)
  mtime_cpu = elapsed_ms(start_count, end_count, clock_rate) + 0.5_real32
  print '(A)', 'Run on CPU'
  print '(A,F0.6,A)', 'Processing time on CPU: ', mtime_cpu, ' (ms)'
  total_result = sum(output_vals)
  print '(A,F0.6)', 'Summation of output prices on CPU: ', total_result
  print '(A,I0,A,F0.6)', 'Output price at index ', num_vals/2_int32, ' on CPU: ', output_vals(num_vals/2_int32+1_int32)
  print *
  print '(A,F0.6)', 'Speedup on GPU: ', mtime_cpu/mtime_gpu

contains
  pure real(real32) function elapsed_ms(first, last, rate) result(milliseconds)
    integer(int64), intent(in) :: first, last, rate
    milliseconds = real(last-first, real32)*1000.0_real32/real(rate, real32)
  end function elapsed_ms
end program black_scholes
