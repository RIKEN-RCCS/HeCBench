module ace_kernels
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none

  integer, parameter :: nx = 400, ny = 400, nz = 400

!$omp declare target (dfphi, gradient_x, gradient_y, gradient_z, divergence, laplacian, anisotropy, wn, taun, dfunc)
contains

  pure real(real64) function dfphi(phi, u, lambda)
    real(real64), intent(in) :: phi, u, lambda
    dfphi = -phi * (1.0_real64 - phi * phi) + lambda * u * &
            (1.0_real64 - phi * phi) * (1.0_real64 - phi * phi)
  end function dfphi

  pure real(real64) function gradient_x(field, dx, x, y, z)
    real(real64), intent(in) :: field(0:nz-1, 0:ny-1, 0:nx-1), dx
    integer, intent(in) :: x, y, z
    gradient_x = (field(z, y, x + 1) - field(z, y, x - 1)) / (2.0_real64 * dx)
  end function gradient_x

  pure real(real64) function gradient_y(field, dy, x, y, z)
    real(real64), intent(in) :: field(0:nz-1, 0:ny-1, 0:nx-1), dy
    integer, intent(in) :: x, y, z
    gradient_y = (field(z, y + 1, x) - field(z, y - 1, x)) / (2.0_real64 * dy)
  end function gradient_y

  pure real(real64) function gradient_z(field, dz, x, y, z)
    real(real64), intent(in) :: field(0:nz-1, 0:ny-1, 0:nx-1), dz
    integer, intent(in) :: x, y, z
    gradient_z = (field(z + 1, y, x) - field(z - 1, y, x)) / (2.0_real64 * dz)
  end function gradient_z

  pure real(real64) function divergence(fx, fy, fz, dx, dy, dz, x, y, z)
    real(real64), intent(in) :: fx(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: fy(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: fz(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: dx, dy, dz
    integer, intent(in) :: x, y, z
    divergence = gradient_x(fx, dx, x, y, z) + gradient_y(fy, dy, x, y, z) + &
                 gradient_z(fz, dz, x, y, z)
  end function divergence

  pure real(real64) function laplacian(field, dx, dy, dz, x, y, z)
    real(real64), intent(in) :: field(0:nz-1, 0:ny-1, 0:nx-1), dx, dy, dz
    integer, intent(in) :: x, y, z
    laplacian = (field(z, y, x + 1) + field(z, y, x - 1) - 2.0_real64 * field(z, y, x)) / (dx * dx) + &
                (field(z, y + 1, x) + field(z, y - 1, x) - 2.0_real64 * field(z, y, x)) / (dy * dy) + &
                (field(z + 1, y, x) + field(z - 1, y, x) - 2.0_real64 * field(z, y, x)) / (dz * dz)
  end function laplacian

  pure real(real64) function anisotropy(phix, phiy, phiz, epsilon)
    real(real64), intent(in) :: phix, phiy, phiz, epsilon
    real(real64) :: gradient_squared
    if (phix /= 0.0_real64 .or. phiy /= 0.0_real64 .or. phiz /= 0.0_real64) then
      gradient_squared = phix * phix + phiy * phiy + phiz * phiz
      anisotropy = (1.0_real64 - 3.0_real64 * epsilon) * &
        (1.0_real64 + (4.0_real64 * epsilon / (1.0_real64 - 3.0_real64 * epsilon)) * &
        ((phix * phix * phix * phix + phiy * phiy * phiy * phiy + phiz * phiz * phiz * phiz) / &
        (gradient_squared * gradient_squared)))
    else
      anisotropy = 1.0_real64 - (5.0_real64 / 3.0_real64) * epsilon
    end if
  end function anisotropy

  pure real(real64) function wn(phix, phiy, phiz, epsilon, w0)
    real(real64), intent(in) :: phix, phiy, phiz, epsilon, w0
    wn = w0 * anisotropy(phix, phiy, phiz, epsilon)
  end function wn

  pure real(real64) function taun(phix, phiy, phiz, epsilon, tau0)
    real(real64), intent(in) :: phix, phiy, phiz, epsilon, tau0
    taun = tau0 * anisotropy(phix, phiy, phiz, epsilon) ** 2
  end function taun

  pure real(real64) function dfunc(l, m, n)
    real(real64), intent(in) :: l, m, n
    real(real64) :: sum_squared
    if (l /= 0.0_real64 .or. m /= 0.0_real64 .or. n /= 0.0_real64) then
      sum_squared = l * l + m * m + n * n
      dfunc = (l * l * l * (m * m + n * n) - l * (m * m * m * m + n * n * n * n)) / &
              (sum_squared * sum_squared)
    else
      dfunc = 0.0_real64
    end if
  end function dfunc

  subroutine calculate_force(phi, fx, fy, fz, dx, dy, dz, epsilon, w0, tau0)
    real(real64), intent(in) :: phi(0:nz-1, 0:ny-1, 0:nx-1), dx, dy, dz, epsilon, w0, tau0
    real(real64), intent(out) :: fx(0:nz-1, 0:ny-1, 0:nx-1), fy(0:nz-1, 0:ny-1, 0:nx-1), fz(0:nz-1, 0:ny-1, 0:nx-1)
    integer :: x, y, z
    real(real64) :: phix, phiy, phiz, sqgphi, c, w, w2
!$omp target teams distribute parallel do collapse(3) thread_limit(256) private(phix, phiy, phiz, sqgphi, c, w, w2)
    do x = 0, nx - 1
      do y = 0, ny - 1
        do z = 0, nz - 1
          if (x < nx - 1 .and. y < ny - 1 .and. z < nz - 1 .and. x > 0 .and. y > 0 .and. z > 0) then
            phix = gradient_x(phi, dx, x, y, z); phiy = gradient_y(phi, dy, x, y, z); phiz = gradient_z(phi, dz, x, y, z)
            sqgphi = phix * phix + phiy * phiy + phiz * phiz; c = 16.0_real64 * w0 * epsilon
            w = wn(phix, phiy, phiz, epsilon, w0); w2 = w * w
            fx(z, y, x) = w2 * phix + sqgphi * w * c * dfunc(phix, phiy, phiz)
            fy(z, y, x) = w2 * phiy + sqgphi * w * c * dfunc(phiy, phiz, phix)
            fz(z, y, x) = w2 * phiz + sqgphi * w * c * dfunc(phiz, phix, phiy)
          else
            fx(z, y, x) = 0.0_real64; fy(z, y, x) = 0.0_real64; fz(z, y, x) = 0.0_real64
          end if
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine calculate_force

  subroutine allen_cahn(phinew, phiold, uold, fx, fy, fz, epsilon, w0, tau0, lambda, dt, dx, dy, dz)
    real(real64), intent(out) :: phinew(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: phiold(0:nz-1, 0:ny-1, 0:nx-1), uold(0:nz-1, 0:ny-1, 0:nx-1), fx(0:nz-1, 0:ny-1, 0:nx-1), fy(0:nz-1, 0:ny-1, 0:nx-1), fz(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: epsilon, w0, tau0, lambda, dt, dx, dy, dz
    integer :: x, y, z
    real(real64) :: phix, phiy, phiz
!$omp target teams distribute parallel do collapse(3) thread_limit(256) private(phix, phiy, phiz)
    do x = 1, nx - 2
      do y = 1, ny - 2
        do z = 1, nz - 2
          phix = gradient_x(phiold, dx, x, y, z); phiy = gradient_y(phiold, dy, x, y, z); phiz = gradient_z(phiold, dz, x, y, z)
          phinew(z, y, x) = phiold(z, y, x) + (dt / taun(phix, phiy, phiz, epsilon, tau0)) * &
            (divergence(fx, fy, fz, dx, dy, dz, x, y, z) - dfphi(phiold(z, y, x), uold(z, y, x), lambda))
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine allen_cahn

  subroutine boundary_phi(phinew)
    real(real64), intent(inout) :: phinew(0:nz-1, 0:ny-1, 0:nx-1)
    integer :: x, y, z
!$omp target teams distribute parallel do collapse(3) thread_limit(256)
    do x = 0, nx - 1
      do y = 0, ny - 1
        do z = 0, nz - 1
          if (x == 0 .or. x == nx - 1 .or. y == 0 .or. y == ny - 1 .or. z == 0 .or. z == nz - 1) phinew(z, y, x) = -1.0_real64
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine boundary_phi

  subroutine thermal_equation(unew, uold, phinew, phiold, diffusivity, dt, dx, dy, dz)
    real(real64), intent(out) :: unew(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: uold(0:nz-1, 0:ny-1, 0:nx-1), phinew(0:nz-1, 0:ny-1, 0:nx-1), phiold(0:nz-1, 0:ny-1, 0:nx-1), diffusivity, dt, dx, dy, dz
    integer :: x, y, z
!$omp target teams distribute parallel do collapse(3) thread_limit(256)
    do x = 1, nx - 2
      do y = 1, ny - 2
        do z = 1, nz - 2
          unew(z, y, x) = uold(z, y, x) + 0.5_real64 * (phinew(z, y, x) - phiold(z, y, x)) + dt * diffusivity * laplacian(uold, dx, dy, dz, x, y, z)
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine thermal_equation

  subroutine boundary_u(unew, delta)
    real(real64), intent(inout) :: unew(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: delta
    integer :: x, y, z
!$omp target teams distribute parallel do collapse(3) thread_limit(256)
    do x = 0, nx - 1
      do y = 0, ny - 1
        do z = 0, nz - 1
          if (x == 0 .or. x == nx - 1 .or. y == 0 .or. y == ny - 1 .or. z == 0 .or. z == nz - 1) unew(z, y, x) = -delta
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine boundary_u

  subroutine swap_grid(cnew, cold)
    real(real64), intent(inout) :: cnew(0:nz-1, 0:ny-1, 0:nx-1), cold(0:nz-1, 0:ny-1, 0:nx-1)
    integer :: x, y, z
    real(real64) :: temporary
!$omp target teams distribute parallel do collapse(3) thread_limit(256) private(temporary)
    do x = 0, nx - 1
      do y = 0, ny - 1
        do z = 0, nz - 1
          temporary = cnew(z, y, x); cnew(z, y, x) = cold(z, y, x); cold(z, y, x) = temporary
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine swap_grid

  subroutine initialize_phi(phi, radius)
    real(real64), intent(out) :: phi(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: radius
    integer :: x, y, z
    real(real64) :: distance
!$omp parallel do collapse(3) private(distance)
    do x = 0, nx - 1
      do y = 0, ny - 1
        do z = 0, nz - 1
          distance = sqrt(real((x - 0.5_real64 * nx)**2 + (y - 0.5_real64 * ny)**2 + (z - 0.5_real64 * nz)**2, real64))
          if (distance < radius) then; phi(z, y, x) = 1.0_real64; else; phi(z, y, x) = -1.0_real64; end if
        end do
      end do
    end do
!$omp end parallel do
  end subroutine initialize_phi

  subroutine initialize_u(u, radius, delta)
    real(real64), intent(out) :: u(0:nz-1, 0:ny-1, 0:nx-1)
    real(real64), intent(in) :: radius, delta
    integer :: x, y, z
    real(real64) :: distance
!$omp parallel do collapse(3) private(distance)
    do x = 0, nx - 1
      do y = 0, ny - 1
        do z = 0, nz - 1
          distance = sqrt(real((x - 0.5_real64 * nx)**2 + (y - 0.5_real64 * ny)**2 + (z - 0.5_real64 * nz)**2, real64))
          if (distance < radius) then; u(z, y, x) = 0.0_real64; else; u(z, y, x) = -delta * (1.0_real64 - exp(-(distance - radius))); end if
        end do
      end do
    end do
!$omp end parallel do
  end subroutine initialize_u
end module ace_kernels

program ace
  use, intrinsic :: iso_fortran_env, only : real64, int64
  use ace_kernels
  implicit none
  integer :: num_steps, step, ios
  integer(int64) :: offload_start, offload_end, kernel_start, kernel_end, clock_rate
  character(len=64) :: argument
  real(real64), parameter :: dx = 0.4_real64, dy = 0.4_real64, dz = 0.4_real64, dt = 0.01_real64
  real(real64), parameter :: delta = 0.8_real64, radius = 5.0_real64, epsilon = 0.07_real64, w0 = 1.0_real64
  real(real64), parameter :: beta0 = 0.0_real64, diffusivity = 2.0_real64, d0 = 0.5_real64, a1 = 1.25_real64 / sqrt(2.0_real64), a2 = 0.64_real64
  real(real64) :: lambda, tau0, kernel_ms, offload_ms
  real(real64), allocatable :: phiold(:, :, :), uold(:, :, :), phinew(:, :, :), unew(:, :, :), fx(:, :, :), fy(:, :, :), fz(:, :, :)

  call get_command_argument(1, argument)
  read(argument, *, iostat=ios) num_steps
  if (ios /= 0) error stop 'Usage: ./main <num_steps>'
  lambda = w0 * a1 / d0
  tau0 = (w0 * w0 * w0 * a1 * a2 / (d0 * diffusivity)) + (w0 * w0 * beta0 / d0)

  allocate(phiold(0:nz-1, 0:ny-1, 0:nx-1), uold(0:nz-1, 0:ny-1, 0:nx-1))
  call initialize_phi(phiold, radius)
  call initialize_u(uold, radius, delta)
  call system_clock(offload_start, clock_rate)
  allocate(phinew(0:nz-1, 0:ny-1, 0:nx-1), unew(0:nz-1, 0:ny-1, 0:nx-1), fx(0:nz-1, 0:ny-1, 0:nx-1), fy(0:nz-1, 0:ny-1, 0:nx-1), fz(0:nz-1, 0:ny-1, 0:nx-1))

!$omp target data map(tofrom: phiold, uold) map(alloc: phinew, unew, fx, fy, fz)
  call system_clock(kernel_start)
  do step = 0, num_steps
    call calculate_force(phiold, fx, fy, fz, dx, dy, dz, epsilon, w0, tau0)
    call allen_cahn(phinew, phiold, uold, fx, fy, fz, epsilon, w0, tau0, lambda, dt, dx, dy, dz)
    call boundary_phi(phinew)
    call thermal_equation(unew, uold, phinew, phiold, diffusivity, dt, dx, dy, dz)
    call boundary_u(unew, delta)
    call swap_grid(phinew, phiold)
    call swap_grid(unew, uold)
  end do
  call system_clock(kernel_end)
!$omp end target data
  call system_clock(offload_end)
  kernel_ms = real(kernel_end - kernel_start, real64) * 1000.0_real64 / real(clock_rate, real64)
  offload_ms = real(offload_end - offload_start, real64) * 1000.0_real64 / real(clock_rate, real64)
  write(*, '(A,F0.3,A)') 'Total kernel execution time: ', kernel_ms, ' (ms)'
  write(*, '(A,F0.3,A)') 'Offload time: ', offload_ms, ' (ms)'
  deallocate(phiold, uold, phinew, unew, fx, fy, fz)
end program ace
