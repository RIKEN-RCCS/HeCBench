program cooling
  use, intrinsic :: iso_fortran_env, only : int32, real64
  use omp_lib, only : omp_get_wtime
  implicit none
!$omp declare target (primordial_cool)

  integer(int32) :: num, repeat, i, ios
  real(real64), parameter :: density = 0.0899_real64
  real(real64), allocatable :: temperature(:), host_result(:), device_result(:)
  real(real64) :: begin_time, elapsed_ms
  logical :: error
  character(len=64) :: argument

  if (command_argument_count() /= 2) then
    write(*,'(a)') 'Usage: ./main <number of points> <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument); read(argument,*,iostat=ios) num
  if (ios /= 0 .or. num <= 0) error stop 'number of points must be positive'
  call get_command_argument(2, argument); read(argument,*,iostat=ios) repeat
  if (ios /= 0 .or. repeat <= 0) error stop 'repeat must be positive'

  allocate(temperature(0:num-1), host_result(0:num-1), device_result(0:num-1))
  do i = 0, num - 1
    temperature(i) = -275.0_real64 + real(i, real64) * 275.0_real64 * 2.0_real64 / real(num, real64)
  end do

!$omp target data map(to:temperature(0:num-1)) map(from:device_result(0:num-1))
  do i = 1, repeat
    call cool_kernel(num, density, temperature, device_result, 0_int32)
  end do
  begin_time = omp_get_wtime()
  do i = 1, repeat
    call cool_kernel(num, density, temperature, device_result, 1_int32)
  end do
  elapsed_ms = (omp_get_wtime() - begin_time) * 1.0e3_real64 / real(repeat, real64)
!$omp end target data

  write(*,'(a,f0.6,a)') 'Average kernel execution time ', elapsed_ms, ' (ms)'
  call reference(num, density, temperature, host_result, 1_int32)
  error = .false.
  do i = 0, num - 1
    if (abs(device_result(i) - host_result(i)) > 1.0e-3_real64) then
      error = .true.
      exit
    end if
  end do
  if (error) then
    write(*,'(a)') 'FAIL'
  else
    write(*,'(a)') 'PASS'
  end if
  deallocate(temperature, host_result, device_result)

contains

  function primordial_cool(n, t, heat_flag) result(cool)
    real(real64), intent(in) :: n, t
    integer(int32), intent(in) :: heat_flag
    real(real64) :: cool
    real(real64) :: n_h, y, g_ff, n_h0, n_hp, n_he0, n_hep, n_hepp, n_e, n_e_old
    real(real64) :: alpha_hp, alpha_hep, alpha_d, alpha_hepp, gamma_eh0, gamma_ehe0, gamma_ehep
    real(real64) :: le_h0, le_hep, li_h0, li_he0, li_hep, lr_hp, lr_hep, lr_hepp, ld_hep, l_ff
    real(real64) :: gamma_lh0, gamma_lhe0, gamma_lhep, e_h0, e_he0, e_hep, heating, diff
    integer(int32) :: iteration

    y = 0.24_real64 / (4.0_real64 - 4.0_real64 * 0.24_real64)
    n_h = n
    alpha_hp = 8.4e-11_real64 * (1.0_real64 / sqrt(t)) * (t / 1.0e3_real64)**(-0.2_real64) &
      * (1.0_real64 / (1.0_real64 + (t / 1.0e6_real64)**0.7_real64))
    alpha_hep = 1.5e-10_real64 * t**(-0.6353_real64)
    alpha_d = 1.9e-3_real64 * t**(-1.5_real64) * exp(-470000.0_real64/t) &
      * (1.0_real64 + 0.3_real64 * exp(-94000.0_real64/t))
    alpha_hepp = 3.36e-10_real64 * (1.0_real64 / sqrt(t)) * (t / 1.0e3_real64)**(-0.2_real64) &
      * (1.0_real64 / (1.0_real64 + (t / 1.0e6_real64)**0.7_real64))
    gamma_eh0 = 5.85e-11_real64 * sqrt(t) * exp(-157809.1_real64/t) &
      * (1.0_real64 / (1.0_real64 + sqrt(t/1.0e5_real64)))
    gamma_ehe0 = 2.38e-11_real64 * sqrt(t) * exp(-285335.4_real64/t) &
      * (1.0_real64 / (1.0_real64 + sqrt(t/1.0e5_real64)))
    gamma_ehep = 5.68e-12_real64 * sqrt(t) * exp(-631515.0_real64/t) &
      * (1.0_real64 / (1.0_real64 + sqrt(t/1.0e5_real64)))
    gamma_lh0 = 3.19851e-13_real64; gamma_lhe0 = 3.13029e-13_real64; gamma_lhep = 2.00541e-14_real64
    e_h0 = 2.4796e-24_real64; e_he0 = 6.86167e-24_real64; e_hep = 6.21868e-25_real64
    n_e = n_h
    if (heat_flag /= 0_int32) then
      do iteration = 1, 20
        n_e_old = n_e
        n_h0 = n_h * alpha_hp / (alpha_hp + gamma_eh0 + gamma_lh0/n_e)
        n_hp = n_h - n_h0
        n_hep = y*n_h / (1.0_real64 + (alpha_hep + alpha_d)/(gamma_ehe0 + gamma_lhe0/n_e) &
          + (gamma_ehep + gamma_lhep/n_e)/alpha_hepp)
        n_he0 = n_hep*(alpha_hep + alpha_d)/(gamma_ehe0 + gamma_lhe0/n_e)
        n_hepp = n_hep*(gamma_ehep + gamma_lhep/n_e)/alpha_hepp
        n_e = n_hp + n_hep + 2.0_real64*n_hepp
        diff = abs(n_e_old - n_e)
        if (diff < 1.0e-6_real64) exit
      end do
    else
      n_h0 = n_h*alpha_hp/(alpha_hp + gamma_eh0)
      n_hp = n_h - n_h0
      n_hep = y*n_h/(1.0_real64 + (alpha_hep+alpha_d)/gamma_ehe0 + gamma_ehep/alpha_hepp)
      n_he0 = n_hep*(alpha_hep+alpha_d)/gamma_ehe0
      n_hepp = n_hep*gamma_ehep/alpha_hepp
      n_e = n_hp + n_hep + 2.0_real64*n_hepp
    end if
    le_h0 = 7.50e-19_real64*exp(-118348.0_real64/t)/(1.0_real64+sqrt(t/1.0e5_real64))*n_e*n_h0
    le_hep = 5.54e-17_real64*t**(-0.397_real64)*exp(-473638.0_real64/t) &
      /(1.0_real64+sqrt(t/1.0e5_real64))*n_e*n_hep
    li_h0 = 1.27e-21_real64*sqrt(t)*exp(-157809.1_real64/t)/(1.0_real64+sqrt(t/1.0e5_real64))*n_e*n_h0
    li_he0 = 9.38e-22_real64*sqrt(t)*exp(-285335.4_real64/t)/(1.0_real64+sqrt(t/1.0e5_real64))*n_e*n_he0
    li_hep = 4.95e-22_real64*sqrt(t)*exp(-631515.0_real64/t)/(1.0_real64+sqrt(t/1.0e5_real64))*n_e*n_hep
    lr_hp = 8.70e-27_real64*sqrt(t)*(t/1.0e3_real64)**(-0.2_real64) &
      /(1.0_real64+(t/1.0e6_real64)**0.7_real64)*n_e*n_hp
    lr_hep = 1.55e-26_real64*t**0.3647_real64*n_e*n_hep
    lr_hepp = 3.48e-26_real64*sqrt(t)*(t/1.0e3_real64)**(-0.2_real64) &
      /(1.0_real64+(t/1.0e6_real64)**0.7_real64)*n_e*n_hepp
    ld_hep = 1.24e-13_real64*t**(-1.5_real64)*exp(-470000.0_real64/t) &
      *(1.0_real64+0.3_real64*exp(-94000.0_real64/t))*n_e*n_hep
    g_ff = 1.1_real64 + 0.34_real64*exp(-(5.5_real64-log(t))*(5.5_real64-log(t))/3.0_real64)
    l_ff = 1.42e-27_real64*g_ff*sqrt(t)*(n_hp+n_hep+4.0_real64*n_hepp)*n_e
    cool = le_h0+le_hep+li_h0+li_he0+li_hep+lr_hp+lr_hep+lr_hepp+ld_hep+l_ff
    heating = 0.0_real64
    if (heat_flag /= 0_int32) heating = n_h0*e_h0+n_he0*e_he0+n_hep*e_hep
    cool = cool - heating
  end function primordial_cool
  subroutine cool_kernel(num, n, temperature, result, heat_flag)
    integer(int32), intent(in) :: num, heat_flag
    real(real64), intent(in) :: n, temperature(0:num-1)
    real(real64), intent(out) :: result(0:num-1)
    integer(int32) :: i
!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, num - 1
      result(i) = primordial_cool(n, temperature(i), heat_flag)
    end do
!$omp end target teams distribute parallel do
  end subroutine cool_kernel

  subroutine reference(num, n, temperature, result, heat_flag)
    integer(int32), intent(in) :: num, heat_flag
    real(real64), intent(in) :: n, temperature(0:num-1)
    real(real64), intent(out) :: result(0:num-1)
    integer(int32) :: i
    do i = 0, num - 1
      result(i) = primordial_cool(n, temperature(i), heat_flag)
    end do
  end subroutine reference
end program cooling
