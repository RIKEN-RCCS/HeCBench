program fhd
  use, intrinsic :: iso_fortran_env, only : real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib
  implicit none
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
  integer :: argc, samples, voxels, verify, n, m, sample_index
  integer(c_int) :: random_value
  real(real32), allocatable :: h_rmu(:), h_imu(:), h_kx(:), h_ky(:), h_kz(:)
  real(real32), allocatable :: h_rfhd(:), h_ifhd(:), h_x(:), h_y(:), h_z(:), rfhd(:), ifhd(:)
  real(real32) :: r, im, xn, yn, zn, exponent, cosine_value, sine_value, error_sum
  real(real32), parameter :: pi = acos(-1.0_real32)
  real(real64) :: start_time, end_time
  character(len=64) :: argument
  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage: ./main <#samples> <#voxels> <verify>'
    stop 1
  end if
  call get_command_argument(1, argument); read(argument, *) samples
  call get_command_argument(2, argument); read(argument, *) voxels
  call get_command_argument(3, argument); read(argument, *) verify
  allocate(h_rmu(0:voxels-1),h_imu(0:voxels-1),h_kx(0:voxels-1),h_ky(0:voxels-1),h_kz(0:voxels-1))
  allocate(h_rfhd(0:samples-1),h_ifhd(0:samples-1),h_x(0:samples-1),h_y(0:samples-1),h_z(0:samples-1))
  allocate(rfhd(0:samples-1),ifhd(0:samples-1))
  call c_srand(2_c_int)
  do n = 0, samples-1
    rfhd(n) = real(n,real32)/real(samples,real32); h_rfhd(n) = rfhd(n)
    ifhd(n) = real(n,real32)/real(samples,real32); h_ifhd(n) = ifhd(n)
    random_value = c_rand(); h_x(n) = 0.3_real32 + merge(0.1_real32,-0.1_real32,modulo(random_value,2_c_int)/=0_c_int)
    random_value = c_rand(); h_y(n) = 0.2_real32 + merge(0.1_real32,-0.1_real32,modulo(random_value,2_c_int)/=0_c_int)
    random_value = c_rand(); h_z(n) = 0.1_real32 + merge(0.1_real32,-0.1_real32,modulo(random_value,2_c_int)/=0_c_int)
  end do
  do m = 0, voxels-1
    h_rmu(m) = real(m,real32)/real(voxels,real32); h_imu(m) = real(m,real32)/real(voxels,real32)
    random_value = c_rand(); h_kx(m) = 0.1_real32 + merge(0.1_real32,-0.1_real32,modulo(random_value,2_c_int)/=0_c_int)
    random_value = c_rand(); h_ky(m) = 0.2_real32 + merge(0.1_real32,-0.1_real32,modulo(random_value,2_c_int)/=0_c_int)
    random_value = c_rand(); h_kz(m) = 0.3_real32 + merge(0.1_real32,-0.1_real32,modulo(random_value,2_c_int)/=0_c_int)
  end do
  print '(a)', 'Run FHd on a device'
!$omp target data map(to:h_rmu,h_imu,h_kx,h_ky,h_kz,h_x,h_y,h_z) map(tofrom:rfhd,ifhd)
  start_time = omp_get_wtime()
!$omp target teams distribute parallel do private(r,im,xn,yn,zn,exponent,cosine_value,sine_value,m)
  do n = 0, samples-1
    r = rfhd(n); im = ifhd(n); xn = h_x(n); yn = h_y(n); zn = h_z(n)
    do m = 0, voxels-1
      exponent = 2.0_real32*pi*(h_kx(m)*xn+h_ky(m)*yn+h_kz(m)*zn)
      cosine_value = cos(exponent); sine_value = sin(exponent)
      r = r + h_rmu(m)*cosine_value-h_imu(m)*sine_value
      im = im + h_imu(m)*cosine_value+h_rmu(m)*sine_value
    end do
    rfhd(n) = r; ifhd(n) = im
  end do
!$omp end target teams distribute parallel do
  end_time = omp_get_wtime()
  print '(a,f0.6,a)', 'Device execution time ', end_time-start_time, ' (s)'
!$omp end target data
  if (verify /= 0) then
    print '(a)', 'Computing root mean square error between host and device results.'
    print '(a)', 'This will take a while..'
!$omp parallel do private(r,im,exponent,cosine_value,sine_value,m)
    do n = 0, samples-1
      r = h_rfhd(n); im = h_ifhd(n)
      do m = 0, voxels-1
        exponent = 2.0_real32*pi*(h_kx(m)*h_x(n)+h_ky(m)*h_y(n)+h_kz(m)*h_z(n))
        cosine_value = cos(exponent); sine_value = sin(exponent)
        r = r + h_rmu(m)*cosine_value-h_imu(m)*sine_value
        im = im + h_imu(m)*cosine_value+h_rmu(m)*sine_value
      end do
      h_rfhd(n) = r; h_ifhd(n) = im
    end do
!$omp end parallel do
    error_sum = 0.0_real32
    do sample_index = 0, samples-1
      error_sum = error_sum + (h_rfhd(sample_index)-rfhd(sample_index))**2 + &
        (h_ifhd(sample_index)-ifhd(sample_index))**2
    end do
    print '(a,f0.6)', 'RMSE = ', sqrt(error_sum/real(2*samples,real32))
  end if
  deallocate(h_rmu,h_imu,h_kx,h_ky,h_kz,h_rfhd,h_ifhd,h_x,h_y,h_z,rfhd,ifhd)
end program fhd
