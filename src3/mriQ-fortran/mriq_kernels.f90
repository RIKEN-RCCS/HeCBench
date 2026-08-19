module mriq_kernels
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  real(real32), parameter :: pix2 = 6.2831853071795864769_real32
  integer, parameter :: kernel_phi_mag_threads_per_block = 256
  integer, parameter :: kernel_q_threads_per_block = 256
  integer, parameter :: kernel_q_k_elems_per_grid = 1024
  type :: kvalues
    real(real32) :: kx, ky, kz, phi_mag
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine input_data(fname, num_k, num_x, kx, ky, kz, x, y, z, phi_r, phi_i)
    character(len=*), intent(in) :: fname
    integer(int32), intent(out) :: num_k, num_x
    real(real32), allocatable, intent(out) :: kx(:), ky(:), kz(:), x(:), y(:), z(:), phi_r(:), phi_i(:)
    integer :: u
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old', action='read')
    read(u) num_k
    read(u) num_x
    allocate(kx(0:num_k-1), ky(0:num_k-1), kz(0:num_k-1), x(0:num_x-1), y(0:num_x-1), z(0:num_x-1), &
      phi_r(0:num_k-1), phi_i(0:num_k-1))
    read(u) kx
    read(u) ky
    read(u) kz
    read(u) x
    read(u) y
    read(u) z
    read(u) phi_r
    read(u) phi_i
    close(u)
  end subroutine

  subroutine output_data(fname, qr, qi, num_x)
    character(len=*), intent(in) :: fname
    integer(int32), intent(in) :: num_x
    real(real32), intent(in) :: qr(0:num_x-1), qi(0:num_x-1)
    integer :: u
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='replace', action='write')
    write(u) num_x
    write(u) qr
    write(u) qi
    close(u)
  end subroutine

  subroutine compute_phi_mag(num_k, phi_r, phi_i, phi_mag)
    integer(int32), intent(in) :: num_k
    real(real32), intent(in) :: phi_r(0:num_k-1), phi_i(0:num_k-1)
    real(real32), intent(out) :: phi_mag(0:num_k-1)
    integer :: index_k, phi_mag_blocks
    phi_mag_blocks = (num_k + kernel_phi_mag_threads_per_block - 1) / kernel_phi_mag_threads_per_block
    !$omp target teams distribute parallel do num_teams(phi_mag_blocks) thread_limit(kernel_phi_mag_threads_per_block)
    do index_k = 0, num_k-1
      phi_mag(index_k) = phi_r(index_k) * phi_r(index_k) + phi_i(index_k) * phi_i(index_k)
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine compute_q_gpu(num_k, num_x, x, y, z, kvals, ck, qr, qi)
    integer(int32), intent(in) :: num_k, num_x
    real(real32), intent(in) :: x(0:num_x-1), y(0:num_x-1), z(0:num_x-1)
    type(kvalues), intent(in) :: kvals(0:num_k-1)
    type(kvalues), intent(inout) :: ck(0:kernel_q_k_elems_per_grid-1)
    real(real32), intent(inout) :: qr(0:num_x-1), qi(0:num_x-1)
    integer :: qgrids, qblocks, qgrid, qgrid_base, num_elems
    integer :: x_index, k_index, k_global_index, k_index1
    real(real32) :: sx, sy, sz, sqr, sqi, exp_arg, exp_arg1
    qgrids = (num_k + kernel_q_k_elems_per_grid - 1) / kernel_q_k_elems_per_grid
    qblocks = (num_x + kernel_q_threads_per_block - 1) / kernel_q_threads_per_block
    do qgrid = 0, qgrids-1
      qgrid_base = qgrid * kernel_q_k_elems_per_grid
      num_elems = min(kernel_q_k_elems_per_grid, num_k - qgrid_base)
      ck(0:num_elems-1) = kvals(qgrid_base:qgrid_base+num_elems-1)
      !$omp target update to(ck(0:num_elems-1))
      !$omp target teams distribute parallel do num_teams(qblocks) thread_limit(kernel_q_threads_per_block) &
      !$omp& private(sx,sy,sz,sqr,sqi,k_index,k_global_index,k_index1,exp_arg,exp_arg1)
      do x_index = 0, num_x-1
        sx = x(x_index); sy = y(x_index); sz = z(x_index)
        sqr = qr(x_index); sqi = qi(x_index)
        k_index = 0
        k_global_index = qgrid_base
        if (mod(num_k, 2) /= 0 .and. num_elems > 0) then
          exp_arg = pix2 * (ck(0)%kx*sx + ck(0)%ky*sy + ck(0)%kz*sz)
          sqr = sqr + ck(0)%phi_mag * cos(exp_arg)
          sqi = sqi + ck(0)%phi_mag * sin(exp_arg)
          k_index = k_index + 1
          k_global_index = k_global_index + 1
        end if
        do while (k_index < kernel_q_k_elems_per_grid .and. k_global_index < num_k .and. k_index+1 < num_elems)
          exp_arg = pix2 * (ck(k_index)%kx*sx + ck(k_index)%ky*sy + ck(k_index)%kz*sz)
          sqr = sqr + ck(k_index)%phi_mag * cos(exp_arg)
          sqi = sqi + ck(k_index)%phi_mag * sin(exp_arg)
          k_index1 = k_index + 1
          exp_arg1 = pix2 * (ck(k_index1)%kx*sx + ck(k_index1)%ky*sy + ck(k_index1)%kz*sz)
          sqr = sqr + ck(k_index1)%phi_mag * cos(exp_arg1)
          sqi = sqi + ck(k_index1)%phi_mag * sin(exp_arg1)
          k_index = k_index + 2
          k_global_index = k_global_index + 2
        end do
        qr(x_index) = sqr; qi(x_index) = sqi
      end do
      !$omp end target teams distribute parallel do
    end do
  end subroutine
end module
