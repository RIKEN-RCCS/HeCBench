program main
  use iso_fortran_env, only: int32, real32, real64
  use mriq_kernels
  implicit none
  integer(int32) :: num_x, num_k
  integer :: i
  real(real32), allocatable :: kx(:), ky(:), kz(:), x(:), y(:), z(:), phi_r(:), phi_i(:), phi_mag(:), qr(:), qi(:)
  type(kvalues), allocatable :: kvals(:), ck(:)
  real(real64) :: t0, t1
  character(len=512) :: input_name, output_name
  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: main <input filename> <output filename>'
    stop 1
  end if
  call get_command_argument(1, input_name)
  call get_command_argument(2, output_name)
  call input_data(input_name, num_k, num_x, kx, ky, kz, x, y, z, phi_r, phi_i)
  print '(i0,a,i0,a)', num_x, ' pixels in output; ', num_k, ' samples in trajectory'
  allocate(phi_mag(0:num_k-1), qr(0:num_x-1), qi(0:num_x-1), kvals(0:num_k-1), ck(0:kernel_q_k_elems_per_grid-1))

  !$omp target data map(to:phi_r(0:num_k-1),phi_i(0:num_k-1)) map(from:phi_mag(0:num_k-1))
  t0 = seconds()
  call compute_phi_mag(num_k, phi_r, phi_i, phi_mag)
  t1 = seconds()
  print '(a,f10.6,a)', 'computePhiMag time: ', t1-t0, ' s'
  !$omp end target data

  do i = 0, num_k-1
    kvals(i)%kx = kx(i); kvals(i)%ky = ky(i); kvals(i)%kz = kz(i); kvals(i)%phi_mag = phi_mag(i)
  end do
  !$omp target data map(to:x(0:num_x-1),y(0:num_x-1),z(0:num_x-1)) &
  !$omp& map(from:qr(0:num_x-1),qi(0:num_x-1)) map(alloc:ck(0:kernel_q_k_elems_per_grid-1))
  !$omp target teams distribute parallel do thread_limit(256)
  do i = 0, num_x-1
    qr(i) = 0.0_real32
    qi(i) = 0.0_real32
  end do
  !$omp end target teams distribute parallel do
  t0 = seconds()
  call compute_q_gpu(num_k, num_x, x, y, z, kvals, ck, qr, qi)
  t1 = seconds()
  print '(a,f10.6,a)', 'computeQ time: ', t1-t0, ' s'
  !$omp end target data
  call output_data(output_name, qr, qi, num_x)
end program
