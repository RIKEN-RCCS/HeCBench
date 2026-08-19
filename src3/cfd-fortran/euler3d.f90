module cfd_kernels
  use iso_fortran_env, only: real32, real64
  implicit none

  integer, parameter :: ndim = 3, nnb = 4, rk = 3, nvar = 5
  integer, parameter :: var_density = 0, var_momentum = 1, var_density_energy = 4
  integer, parameter :: block_size_1 = 192, block_size_2 = 192, block_size_3 = 192, block_size_4 = 192
  real(real32), parameter :: gamma = 1.4_real32

  type :: float3
    real(real32) :: x, y, z
  end type float3

!$omp declare target (compute_velocity, compute_speed_sqd, compute_pressure, &
!$omp& compute_speed_of_sound, compute_flux_contribution, initialize_buffer, &
!$omp& initialize_variables, copy_buffer, compute_step_factor, compute_flux, time_step)
contains

  subroutine compute_velocity(density, momentum, velocity)
    real(real32), intent(in) :: density
    type(float3), intent(in) :: momentum
    type(float3), intent(out) :: velocity
    velocity%x = momentum%x / density
    velocity%y = momentum%y / density
    velocity%z = momentum%z / density
  end subroutine compute_velocity

  real(real32) function compute_speed_sqd(velocity)
    type(float3), intent(in) :: velocity
    compute_speed_sqd = velocity%x * velocity%x + velocity%y * velocity%y + velocity%z * velocity%z
  end function compute_speed_sqd

  real(real32) function compute_pressure(density, density_energy, speed_sqd)
    real(real32), intent(in) :: density, density_energy, speed_sqd
    compute_pressure = (gamma - 1.0_real32) * (density_energy - 0.5_real32 * density * speed_sqd)
  end function compute_pressure

  real(real32) function compute_speed_of_sound(density, pressure)
    real(real32), intent(in) :: density, pressure
    compute_speed_of_sound = sqrt(gamma * pressure / density)
  end function compute_speed_of_sound

  subroutine compute_flux_contribution(density, momentum, density_energy, pressure, velocity, fc_mx, fc_my, fc_mz, fc_de)
    real(real32), intent(in) :: density, density_energy, pressure
    type(float3), intent(in) :: momentum, velocity
    type(float3), intent(out) :: fc_mx, fc_my, fc_mz, fc_de
    real(real32) :: de_p
    fc_mx%x = velocity%x * momentum%x + pressure
    fc_mx%y = velocity%x * momentum%y
    fc_mx%z = velocity%x * momentum%z
    fc_my%x = fc_mx%y
    fc_my%y = velocity%y * momentum%y + pressure
    fc_my%z = velocity%y * momentum%z
    fc_mz%x = fc_mx%z
    fc_mz%y = fc_my%z
    fc_mz%z = velocity%z * momentum%z + pressure
    de_p = density_energy + pressure
    fc_de%x = velocity%x * de_p
    fc_de%y = velocity%y * de_p
    fc_de%z = velocity%z * de_p
  end subroutine compute_flux_contribution

  subroutine initialize_buffer(buffer, value, number_words)
    integer, intent(in) :: number_words
    real(real32), intent(in) :: value
    real(real32), intent(inout) :: buffer(0:number_words-1)
    integer :: i
!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, number_words - 1
      buffer(i) = value
    end do
!$omp end target teams distribute parallel do
  end subroutine initialize_buffer

  subroutine initialize_variables(nelr, variables, ff_variable)
    integer, intent(in) :: nelr
    real(real32), intent(inout) :: variables(0:nelr*nvar-1)
    real(real32), intent(in) :: ff_variable(0:nvar-1)
    integer :: i, j
!$omp target teams distribute parallel do thread_limit(block_size_1) private(j)
    do i = 0, nelr - 1
      do j = 0, nvar - 1
        variables(i + j * nelr) = ff_variable(j)
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine initialize_variables

  subroutine copy_buffer(destination, source, number_words)
    integer, intent(in) :: number_words
    real(real32), intent(out) :: destination(0:number_words-1)
    real(real32), intent(in) :: source(0:number_words-1)
    integer :: i
!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, number_words - 1
      destination(i) = source(i)
    end do
!$omp end target teams distribute parallel do
  end subroutine copy_buffer

  subroutine compute_step_factor(nelr, variables, areas, step_factors)
    integer, intent(in) :: nelr
    real(real32), intent(in) :: variables(0:nelr*nvar-1), areas(0:nelr-1)
    real(real32), intent(inout) :: step_factors(0:nelr-1)
    integer :: i
    real(real32) :: density, density_energy, speed_sqd, pressure, speed_of_sound
    type(float3) :: momentum, velocity
!$omp target teams distribute parallel do thread_limit(block_size_2) &
!$omp& private(density, density_energy, speed_sqd, pressure, speed_of_sound, momentum, velocity)
    do i = 0, nelr - 1
      density = variables(i + var_density * nelr)
      momentum%x = variables(i + (var_momentum + 0) * nelr)
      momentum%y = variables(i + (var_momentum + 1) * nelr)
      momentum%z = variables(i + (var_momentum + 2) * nelr)
      density_energy = variables(i + var_density_energy * nelr)
      call compute_velocity(density, momentum, velocity)
      speed_sqd = compute_speed_sqd(velocity)
      pressure = compute_pressure(density, density_energy, speed_sqd)
      speed_of_sound = compute_speed_of_sound(density, pressure)
      step_factors(i) = 0.5_real32 / (sqrt(areas(i)) * (sqrt(speed_sqd) + speed_of_sound))
    end do
!$omp end target teams distribute parallel do
  end subroutine compute_step_factor

  subroutine compute_flux(nelr, elements, normals, variables, ff_variable, fluxes, ff_de, ff_mx, ff_my, ff_mz)
    integer, intent(in) :: nelr
    integer, intent(in) :: elements(0:nelr*nnb-1)
    real(real32), intent(in) :: normals(0:nelr*ndim*nnb-1), variables(0:nelr*nvar-1), ff_variable(0:nvar-1)
    real(real32), intent(inout) :: fluxes(0:nelr*nvar-1)
    type(float3), intent(in) :: ff_de, ff_mx, ff_my, ff_mz
    integer :: i, j, nb
    real(real32) :: normal_len, factor, density_i, density_energy_i, speed_sqd_i, speed_i, pressure_i, speed_sound_i
    real(real32) :: density_nb, density_energy_nb, speed_sqd_nb, pressure_nb, speed_sound_nb, flux_i_density, flux_i_de
    type(float3) :: normal, momentum_i, velocity_i, fci_mx, fci_my, fci_mz, fci_de
    type(float3) :: momentum_nb, velocity_nb, fcn_mx, fcn_my, fcn_mz, fcn_de, flux_i_momentum
!$omp target teams distribute parallel do thread_limit(block_size_3) &
!$omp& private(j, nb, normal, normal_len, factor, density_i, density_energy_i, speed_sqd_i, speed_i, pressure_i, speed_sound_i, &
!$omp& density_nb, density_energy_nb, speed_sqd_nb, pressure_nb, speed_sound_nb, momentum_i, velocity_i, &
!$omp& fci_mx, fci_my, fci_mz, fci_de, momentum_nb, velocity_nb, fcn_mx, fcn_my, fcn_mz, fcn_de, &
!$omp& flux_i_density, flux_i_de, flux_i_momentum)
    do i = 0, nelr - 1
      density_i = variables(i + var_density * nelr)
      momentum_i%x = variables(i + (var_momentum + 0) * nelr)
      momentum_i%y = variables(i + (var_momentum + 1) * nelr)
      momentum_i%z = variables(i + (var_momentum + 2) * nelr)
      density_energy_i = variables(i + var_density_energy * nelr)
      call compute_velocity(density_i, momentum_i, velocity_i)
      speed_sqd_i = compute_speed_sqd(velocity_i)
      speed_i = sqrt(speed_sqd_i)
      pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i)
      speed_sound_i = compute_speed_of_sound(density_i, pressure_i)
      call compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, &
                                     fci_mx, fci_my, fci_mz, fci_de)
      flux_i_density = 0.0_real32
      flux_i_momentum%x = 0.0_real32; flux_i_momentum%y = 0.0_real32; flux_i_momentum%z = 0.0_real32
      flux_i_de = 0.0_real32
      do j = 0, nnb - 1
        nb = elements(i + j * nelr)
        normal%x = normals(i + (j + 0 * nnb) * nelr)
        normal%y = normals(i + (j + 1 * nnb) * nelr)
        normal%z = normals(i + (j + 2 * nnb) * nelr)
        normal_len = sqrt(normal%x * normal%x + normal%y * normal%y + normal%z * normal%z)
        if (nb >= 0) then
          density_nb = variables(nb + var_density * nelr)
          momentum_nb%x = variables(nb + (var_momentum + 0) * nelr)
          momentum_nb%y = variables(nb + (var_momentum + 1) * nelr)
          momentum_nb%z = variables(nb + (var_momentum + 2) * nelr)
          density_energy_nb = variables(nb + var_density_energy * nelr)
          call compute_velocity(density_nb, momentum_nb, velocity_nb)
          speed_sqd_nb = compute_speed_sqd(velocity_nb)
          pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb)
          speed_sound_nb = compute_speed_of_sound(density_nb, pressure_nb)
          call compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, &
                                         fcn_mx, fcn_my, fcn_mz, fcn_de)
          factor = -normal_len * 0.2_real32 * 0.5_real32 * (speed_i + sqrt(speed_sqd_nb) + speed_sound_i + speed_sound_nb)
          flux_i_density = flux_i_density + factor * (density_i - density_nb)
          flux_i_de = flux_i_de + factor * (density_energy_i - density_energy_nb)
          flux_i_momentum%x = flux_i_momentum%x + factor * (momentum_i%x - momentum_nb%x)
          flux_i_momentum%y = flux_i_momentum%y + factor * (momentum_i%y - momentum_nb%y)
          flux_i_momentum%z = flux_i_momentum%z + factor * (momentum_i%z - momentum_nb%z)
          factor = 0.5_real32 * normal%x
          flux_i_density = flux_i_density + factor * (momentum_nb%x + momentum_i%x)
          flux_i_de = flux_i_de + factor * (fcn_de%x + fci_de%x)
          flux_i_momentum%x = flux_i_momentum%x + factor * (fcn_mx%x + fci_mx%x)
          flux_i_momentum%y = flux_i_momentum%y + factor * (fcn_my%x + fci_my%x)
          flux_i_momentum%z = flux_i_momentum%z + factor * (fcn_mz%x + fci_mz%x)
          factor = 0.5_real32 * normal%y
          flux_i_density = flux_i_density + factor * (momentum_nb%y + momentum_i%y)
          flux_i_de = flux_i_de + factor * (fcn_de%y + fci_de%y)
          flux_i_momentum%x = flux_i_momentum%x + factor * (fcn_mx%y + fci_mx%y)
          flux_i_momentum%y = flux_i_momentum%y + factor * (fcn_my%y + fci_my%y)
          flux_i_momentum%z = flux_i_momentum%z + factor * (fcn_mz%y + fci_mz%y)
          factor = 0.5_real32 * normal%z
          flux_i_density = flux_i_density + factor * (momentum_nb%z + momentum_i%z)
          flux_i_de = flux_i_de + factor * (fcn_de%z + fci_de%z)
          flux_i_momentum%x = flux_i_momentum%x + factor * (fcn_mx%z + fci_mx%z)
          flux_i_momentum%y = flux_i_momentum%y + factor * (fcn_my%z + fci_my%z)
          flux_i_momentum%z = flux_i_momentum%z + factor * (fcn_mz%z + fci_mz%z)
        else if (nb == -1) then
          flux_i_momentum%x = flux_i_momentum%x + normal%x * pressure_i
          flux_i_momentum%y = flux_i_momentum%y + normal%y * pressure_i
          flux_i_momentum%z = flux_i_momentum%z + normal%z * pressure_i
        else if (nb == -2) then
          factor = 0.5_real32 * normal%x
          flux_i_density = flux_i_density + factor * (ff_variable(var_momentum + 0) + momentum_i%x)
          flux_i_de = flux_i_de + factor * (ff_de%x + fci_de%x)
          flux_i_momentum%x = flux_i_momentum%x + factor * (ff_mx%x + fci_mx%x)
          flux_i_momentum%y = flux_i_momentum%y + factor * (ff_my%x + fci_my%x)
          flux_i_momentum%z = flux_i_momentum%z + factor * (ff_mz%x + fci_mz%x)
          factor = 0.5_real32 * normal%y
          flux_i_density = flux_i_density + factor * (ff_variable(var_momentum + 1) + momentum_i%y)
          flux_i_de = flux_i_de + factor * (ff_de%y + fci_de%y)
          flux_i_momentum%x = flux_i_momentum%x + factor * (ff_mx%y + fci_mx%y)
          flux_i_momentum%y = flux_i_momentum%y + factor * (ff_my%y + fci_my%y)
          flux_i_momentum%z = flux_i_momentum%z + factor * (ff_mz%y + fci_mz%y)
          factor = 0.5_real32 * normal%z
          flux_i_density = flux_i_density + factor * (ff_variable(var_momentum + 2) + momentum_i%z)
          flux_i_de = flux_i_de + factor * (ff_de%z + fci_de%z)
          flux_i_momentum%x = flux_i_momentum%x + factor * (ff_mx%z + fci_mx%z)
          flux_i_momentum%y = flux_i_momentum%y + factor * (ff_my%z + fci_my%z)
          flux_i_momentum%z = flux_i_momentum%z + factor * (ff_mz%z + fci_mz%z)
        end if
      end do
      fluxes(i + var_density * nelr) = flux_i_density
      fluxes(i + (var_momentum + 0) * nelr) = flux_i_momentum%x
      fluxes(i + (var_momentum + 1) * nelr) = flux_i_momentum%y
      fluxes(i + (var_momentum + 2) * nelr) = flux_i_momentum%z
      fluxes(i + var_density_energy * nelr) = flux_i_de
    end do
!$omp end target teams distribute parallel do
  end subroutine compute_flux

  subroutine time_step(stage, nelr, old_variables, variables, step_factors, fluxes)
    integer, intent(in) :: stage, nelr
    real(real32), intent(in) :: old_variables(0:nelr*nvar-1), step_factors(0:nelr-1), fluxes(0:nelr*nvar-1)
    real(real32), intent(inout) :: variables(0:nelr*nvar-1)
    integer :: i
    real(real32) :: factor
!$omp target teams distribute parallel do thread_limit(block_size_4) private(factor)
    do i = 0, nelr - 1
      factor = step_factors(i) / real(rk + 1 - stage, real32)
      variables(i + var_density * nelr) = old_variables(i + var_density * nelr) + factor * fluxes(i + var_density * nelr)
      variables(i + var_density_energy * nelr) = old_variables(i + var_density_energy * nelr) &
          + factor * fluxes(i + var_density_energy * nelr)
      variables(i + (var_momentum + 0) * nelr) = old_variables(i + (var_momentum + 0) * nelr) &
          + factor * fluxes(i + (var_momentum + 0) * nelr)
      variables(i + (var_momentum + 1) * nelr) = old_variables(i + (var_momentum + 1) * nelr) &
          + factor * fluxes(i + (var_momentum + 1) * nelr)
      variables(i + (var_momentum + 2) * nelr) = old_variables(i + (var_momentum + 2) * nelr) &
          + factor * fluxes(i + (var_momentum + 2) * nelr)
    end do
!$omp end target teams distribute parallel do
  end subroutine time_step
  subroutine dump_solution(variables, nel, nelr)
    integer, intent(in) :: nel, nelr
    real(real32), intent(in) :: variables(0:nelr*nvar-1)
    integer :: unit_number, i, j
    open(newunit=unit_number, file='density', status='replace', action='write')
    write(unit_number, *) nel, nelr
    do i = 0, nel - 1
      write(unit_number, *) variables(i + var_density * nelr)
    end do
    close(unit_number)
    open(newunit=unit_number, file='momentum', status='replace', action='write')
    write(unit_number, *) nel, nelr
    do i = 0, nel - 1
      write(unit_number, *) (variables(i + (var_momentum + j) * nelr), j = 0, ndim - 1)
    end do
    close(unit_number)
    open(newunit=unit_number, file='density_energy', status='replace', action='write')
    write(unit_number, *) nel, nelr
    do i = 0, nel - 1
      write(unit_number, *) variables(i + var_density_energy * nelr)
    end do
    close(unit_number)
  end subroutine dump_solution

end module cfd_kernels

program euler3d
  use iso_fortran_env, only: real32, real64
  use omp_lib, only: omp_get_wtime
  use cfd_kernels
  implicit none
  integer, parameter :: iterations = 2000, block_length = 192
  integer :: nel, nelr, input_unit, io_status, i, j, k, nb, last, stage, iteration
  character(len=1024) :: data_file_name
  real(real32) :: ff_variable(0:nvar-1), ff_pressure, ff_speed_sound, ff_speed, angle_of_attack
  real(real32), allocatable :: areas(:), normals(:), variables(:), old_variables(:), step_factors(:), fluxes(:)
  integer, allocatable :: elements(:)
  type(float3) :: ff_velocity, ff_momentum, ff_mx, ff_my, ff_mz, ff_de
  real(real64) :: offload_start, offload_end, kernel_start, kernel_end

  write(*, '(A,I0)') 'WG size of kernel:initialize = ', block_size_1
  write(*, '(A,I0)') 'WG size of kernel:compute_step_factor = ', block_size_2
  write(*, '(A,I0)') 'WG size of kernel:compute_flux = ', block_size_3
  write(*, '(A,I0)') 'WG size of kernel:time_step = ', block_size_4
  if (command_argument_count() < 1) then
    write(*, '(A)') 'Please specify data file name'
    stop
  end if
  call get_command_argument(1, data_file_name)
  angle_of_attack = real(3.1415926535897931_real64 / 180.0_real64, real32) * 0.0_real32
  ff_variable(var_density) = 1.4_real32
  ff_pressure = 1.0_real32
  ff_speed_sound = sqrt(gamma * ff_pressure / ff_variable(var_density))
  ff_speed = 1.2_real32 * ff_speed_sound
  ff_velocity%x = ff_speed * cos(angle_of_attack); ff_velocity%y = ff_speed * sin(angle_of_attack); ff_velocity%z = 0.0_real32
  ff_variable(var_momentum + 0) = ff_variable(var_density) * ff_velocity%x
  ff_variable(var_momentum + 1) = ff_variable(var_density) * ff_velocity%y
  ff_variable(var_momentum + 2) = ff_variable(var_density) * ff_velocity%z
  ff_variable(var_density_energy) = ff_variable(var_density) * (0.5_real32 * ff_speed * ff_speed) &
      + ff_pressure / (gamma - 1.0_real32)
  ff_momentum%x = ff_variable(var_momentum + 0)
  ff_momentum%y = ff_variable(var_momentum + 1)
  ff_momentum%z = ff_variable(var_momentum + 2)
  call compute_flux_contribution(ff_variable(var_density), ff_momentum, ff_variable(var_density_energy), &
                                 ff_pressure, ff_velocity, ff_mx, ff_my, ff_mz, ff_de)
  open(newunit=input_unit, file=trim(data_file_name), status='old', action='read', iostat=io_status)
  if (io_status /= 0) error stop 'can not find/open file!'
  read(input_unit, *) nel
  nelr = block_length * (nel / block_length + min(1, mod(nel, block_length)))
  write(*, '(A,I0,A,I0)') '--cambine: nel=', nel, ', nelr=', nelr
  allocate(areas(0:nelr-1), elements(0:nelr*nnb-1), normals(0:nelr*ndim*nnb-1))
  allocate(variables(0:nelr*nvar-1), old_variables(0:nelr*nvar-1), step_factors(0:nelr-1), fluxes(0:nelr*nvar-1))
  do i = 0, nel - 1
    read(input_unit, *) areas(i), (elements(i + j*nelr), (normals(i + (j + k*nnb)*nelr), k = 0, ndim-1), j = 0, nnb-1)
    do j = 0, nnb - 1
      if (elements(i + j*nelr) < 0) elements(i + j*nelr) = -1
      elements(i + j*nelr) = elements(i + j*nelr) - 1
      do k = 0, ndim - 1
        normals(i + (j + k*nnb)*nelr) = -normals(i + (j + k*nnb)*nelr)
      end do
    end do
  end do
  close(input_unit)
  last = nel - 1
  do i = nel, nelr - 1
    areas(i) = areas(last)
    do j = 0, nnb - 1
      elements(i + j*nelr) = elements(last + j*nelr)
    end do
  end do

  offload_start = omp_get_wtime()
!$omp target data map(to: ff_variable(0:nvar-1), areas(0:nelr-1), elements(0:nelr*nnb-1), &
!$omp& normals(0:nelr*ndim*nnb-1)) map(alloc: fluxes(0:nelr*nvar-1), old_variables(0:nelr*nvar-1), &
!$omp& step_factors(0:nelr-1)) map(from: variables(0:nelr*nvar-1))
  kernel_start = omp_get_wtime()
  call initialize_variables(nelr, variables, ff_variable)
  call initialize_variables(nelr, old_variables, ff_variable)
  call initialize_variables(nelr, fluxes, ff_variable)
  call initialize_buffer(step_factors, 0.0_real32, nelr)
  do iteration = 0, iterations - 1
    call copy_buffer(old_variables, variables, nelr*nvar)
    call compute_step_factor(nelr, variables, areas, step_factors)
    do stage = 0, rk - 1
      call compute_flux(nelr, elements, normals, variables, ff_variable, fluxes, ff_de, ff_mx, ff_my, ff_mz)
      call time_step(stage, nelr, old_variables, variables, step_factors, fluxes)
    end do
  end do
  kernel_end = omp_get_wtime()
!$omp end target data
  call dump_solution(variables, nel, nelr)
  offload_end = omp_get_wtime()
  write(*, '(A,F0.6,A)') 'Device offloading time = ', offload_end - offload_start, '(s)'
  write(*, '(A,F0.6,A)') 'Total execution time of kernels = ', kernel_end - kernel_start, '(s)'
  deallocate(areas, elements, normals, variables, old_variables, fluxes, step_factors)
  write(*, '(A)') 'Done...'
end program euler3d
