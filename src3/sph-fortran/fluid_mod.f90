module fluid_mod
  use iso_fortran_env, only: real64
  use sph_types
  use geometry_mod
  implicit none
contains

  !$omp declare target
  real(real64) function W(p_pos, q_pos, h) result(val)
    type(double3), intent(in) :: p_pos, q_pos
    real(real64), intent(in) :: h
    real(real64) :: r, c, u
    r = sqrt((p_pos%x - q_pos%x)**2 + (p_pos%y - q_pos%y)**2 + (p_pos%z - q_pos%z)**2)
    c = 1.0_real64 / (pi * h * h * h)
    u = r / h
    val = 0.0_real64
    if (u >= 2.0_real64) then
      return
    else if (u < 1.0_real64) then
      val = 1.0_real64 - 1.5_real64 * u * u + 0.75_real64 * u * u * u
    else if (u >= 1.0_real64 .and. u < 2.0_real64) then
      val = 0.25_real64 * (2.0_real64 - u)**3
    end if
    val = val * c
  end function W
  !$omp end declare target

  !$omp declare target
  real(real64) function del_W(p_pos, q_pos, h) result(val)
    type(double3), intent(in) :: p_pos, q_pos
    real(real64), intent(in) :: h
    real(real64) :: r, c, u
    r = sqrt((p_pos%x - q_pos%x)**2 + (p_pos%y - q_pos%y)**2 + (p_pos%z - q_pos%z)**2)
    c = 1.0_real64 / (pi * h * h * h)
    u = r / h
    val = 0.0_real64
    if (u >= 2.0_real64) then
      return
    else if (u < 1.0_real64) then
      val = -1.0_real64 / (h * h) * (3.0_real64 - 2.25_real64 * u)
    else if (u >= 1.0_real64 .and. u < 2.0_real64) then
      val = -3.0_real64 / (4.0_real64 * h * r) * (2.0_real64 - u)**2
    end if
    val = val * c
  end function del_W
  !$omp end declare target

  !$omp declare target
  real(real64) function boundary_gamma(p_pos, k_pos, k_n, h, speed_sound) result(val)
    type(double3), intent(in) :: p_pos, k_pos, k_n
    real(real64), intent(in) :: h, speed_sound
    real(real64) :: r, y, x, u, xi, c
    r = sqrt((p_pos%x - k_pos%x)**2 + (p_pos%y - k_pos%y)**2 + (p_pos%z - k_pos%z)**2)
    y = sqrt((p_pos%x - k_pos%x)**2 * k_n%x * k_n%x + &
             (p_pos%y - k_pos%y)**2 * k_n%y * k_n%y + &
             (p_pos%z - k_pos%z)**2 * k_n%z * k_n%z)
    x = r - y
    u = y / h
    xi = merge(1.0_real64, 0.0_real64, x < h)
    c = xi * 2.0_real64 * 0.02_real64 * speed_sound * speed_sound / y
    if (u > 0.0_real64 .and. u < 2.0_real64 / 3.0_real64) then
      val = 2.0_real64 / 3.0_real64
    else if (u < 1.0_real64 .and. u > 2.0_real64 / 3.0_real64) then
      val = 2.0_real64 * u - 1.5_real64 * u * u
    else if (u < 2.0_real64 .and. u > 1.0_real64) then
      val = 0.5_real64 * (2.0_real64 - u) * (2.0_real64 - u)
    else
      val = 0.0_real64
    end if
    val = val * c
  end function boundary_gamma
  !$omp end declare target

  !$omp declare target
  real(real64) function compute_density(p_pos, p_v, q_pos, q_v, params) result(density)
    type(double3), intent(in) :: p_pos, p_v, q_pos, q_v
    type(param), intent(in) :: params
    real(real64) :: kernel, density_x, density_y, density_z
    kernel = params%mass_particle * del_W(p_pos, q_pos, params%smoothing_radius)
    density_x = kernel * (p_v%x - q_v%x) * (p_pos%x - q_pos%x)
    density_y = kernel * (p_v%y - q_v%y) * (p_pos%y - q_pos%y)
    density_z = kernel * (p_v%z - q_v%z) * (p_pos%z - q_pos%z)
    density = (density_x + density_y + density_z) * params%time_step
  end function compute_density
  !$omp end declare target

  !$omp declare target
  real(real64) function compute_pressure(p_density, params) result(pressure)
    real(real64), intent(in) :: p_density
    type(param), intent(in) :: params
    real(real64) :: gam, b
    gam = 7.0_real64
    b = params%rest_density * params%speed_sound * params%speed_sound / gam
    pressure = b * ((p_density / params%rest_density)**gam - 1.0_real64)
  end function compute_pressure
  !$omp end declare target

  !$omp declare target
  type(double3) function compute_boundary_acceleration(p_pos, k_pos, k_n, h, speed_sound) result(p_a)
    type(double3), intent(in) :: p_pos, k_pos, k_n
    real(real64), intent(in) :: h, speed_sound
    real(real64) :: bg
    bg = boundary_gamma(p_pos, k_pos, k_n, h, speed_sound)
    p_a%x = bg * k_n%x
    p_a%y = bg * k_n%y
    p_a%z = bg * k_n%z
  end function compute_boundary_acceleration
  !$omp end declare target

  !$omp declare target
  type(double3) function compute_acceleration(p_pos, p_v, p_density, p_pressure, &
                                              q_pos, q_v, q_density, q_pressure, params) result(a)
    type(double3), intent(in) :: p_pos, p_v, q_pos, q_v
    real(real64), intent(in) :: p_density, p_pressure, q_density, q_pressure
    type(param), intent(in) :: params
    real(real64) :: accel, h, alpha, speed_sound, mass_particle, surface_tension
    real(real64) :: vdotr, nu, r2, eps, stress

    h = params%smoothing_radius
    alpha = params%alpha
    speed_sound = params%speed_sound
    mass_particle = params%mass_particle
    surface_tension = params%surface_tension

    accel = (p_pressure / (p_density * p_density) + q_pressure / (q_density * q_density)) * &
            mass_particle * del_W(p_pos, q_pos, h)
    a%x = -accel * (p_pos%x - q_pos%x)
    a%y = -accel * (p_pos%y - q_pos%y)
    a%z = -accel * (p_pos%z - q_pos%z)

    vdotr = (p_v%x - q_v%x) * (p_pos%x - q_pos%x) + &
            (p_v%y - q_v%y) * (p_pos%y - q_pos%y) + &
            (p_v%z - q_v%z) * (p_pos%z - q_pos%z)
    if (vdotr < 0.0_real64) then
      nu = 2.0_real64 * alpha * h * speed_sound / (p_density + q_density)
      r2 = (p_pos%x - q_pos%x)**2 + (p_pos%y - q_pos%y)**2 + (p_pos%z - q_pos%z)**2
      eps = h / 10.0_real64
      stress = nu * vdotr / (r2 + eps * h * h)
      accel = mass_particle * stress * del_W(p_pos, q_pos, h)
      a%x = a%x + accel * (p_pos%x - q_pos%x)
      a%y = a%y + accel * (p_pos%y - q_pos%y)
      a%z = a%z + accel * (p_pos%z - q_pos%z)
    end if

    accel = surface_tension * W(p_pos, q_pos, h)
    a%x = a%x + accel * (p_pos%x - q_pos%x)
    a%y = a%y + accel * (p_pos%y - q_pos%y)
    a%z = a%z + accel * (p_pos%z - q_pos%z)
  end function compute_acceleration
  !$omp end declare target

  subroutine euler_start(fluid_particles, boundary_particles, params)
    type(fluid_particle), intent(inout) :: fluid_particles(0:)
    type(boundary_particle), intent(in) :: boundary_particles(0:)
    type(param), intent(in) :: params
    real(real64) :: dt_half
    integer :: i

    dt_half = params%time_step / 2.0_real64
    do i = 0, params%number_fluid_particles - 1
      fluid_particles(i)%v_half%x = fluid_particles(i)%v%x
      fluid_particles(i)%v_half%y = fluid_particles(i)%v%y
      fluid_particles(i)%v_half%z = fluid_particles(i)%v%z - params%g * dt_half
    end do
  end subroutine euler_start

  subroutine init_particles(fluid_particles, boundary_particles, water, boundary, params)
    type(fluid_particle), allocatable, intent(out) :: fluid_particles(:)
    type(boundary_particle), allocatable, intent(out) :: boundary_particles(:)
    type(aabb), intent(inout) :: water
    type(aabb), intent(inout) :: boundary
    type(param), intent(inout) :: params
    real(real64) :: spacing, x, y, z
    integer :: i

    allocate(fluid_particles(0:params%number_fluid_particles - 1))
    allocate(boundary_particles(0:params%number_boundary_particles - 1))

    do i = 0, params%number_fluid_particles - 1
      fluid_particles(i)%a = double3(0.0_real64, 0.0_real64, 0.0_real64)
      fluid_particles(i)%v = double3(0.0_real64, 0.0_real64, 0.0_real64)
      fluid_particles(i)%density = params%rest_density
    end do

    spacing = params%spacing_particle
    i = 0
    z = water%min_z
    do while (z <= water%max_z)
      y = water%min_y
      do while (y <= water%max_y)
        x = water%min_x
        do while (x <= water%max_x)
          if (i < params%number_fluid_particles) then
            fluid_particles(i)%pos%x = x
            fluid_particles(i)%pos%y = y
            fluid_particles(i)%pos%z = z
            i = i + 1
          end if
          x = x + spacing
        end do
        y = y + spacing
      end do
      z = z + spacing
    end do
    params%number_fluid_particles = i

    call construct_boundary_box(boundary_particles, boundary, params)
  end subroutine init_particles

  subroutine init_params(water_volume, boundary_volume, params)
    type(aabb), intent(out) :: water_volume
    type(aabb), intent(out) :: boundary_volume
    type(param), intent(out) :: params
    real(real64) :: volume, max_height, max_velocity, recomend_step
    integer :: num_x, num_y, num_z, num_boundary_particles

    boundary_volume%min_x = 0.0_real64
    boundary_volume%max_x = 1.1_real64
    boundary_volume%min_y = 0.0_real64
    boundary_volume%max_y = 1.1_real64
    boundary_volume%min_z = 0.0_real64
    boundary_volume%max_z = 1.1_real64

    water_volume%min_x = 0.1_real64
    water_volume%max_x = 0.5_real64
    water_volume%min_y = 0.1_real64
    water_volume%max_y = 0.5_real64
    water_volume%min_z = 0.08_real64
    water_volume%max_z = 0.8_real64

    params%number_fluid_particles = 2048
    params%rest_density = 1000.0_real64
    params%g = 9.8_real64
    params%alpha = 0.02_real64
    params%surface_tension = 0.01_real64
    params%number_steps = 500
    params%time_step = 0.00035_real64

    volume = (water_volume%max_x - water_volume%min_x) * &
             (water_volume%max_y - water_volume%min_y) * &
             (water_volume%max_z - water_volume%min_z)
    params%mass_particle = params%rest_density * (volume / params%number_fluid_particles)
    params%spacing_particle = (volume / params%number_fluid_particles)**(1.0_real64 / 3.0_real64)
    params%smoothing_radius = params%spacing_particle

    num_x = ceiling((boundary_volume%max_x - boundary_volume%min_x) / params%spacing_particle)
    num_y = ceiling((boundary_volume%max_y - boundary_volume%min_y) / params%spacing_particle)
    num_z = ceiling((boundary_volume%max_z - boundary_volume%min_z) / params%spacing_particle)
    num_boundary_particles = (2 * num_x * num_z) + (2 * num_y * num_z) + (2 * num_y * num_z)
    params%number_boundary_particles = num_boundary_particles
    params%number_particles = params%number_boundary_particles + params%number_fluid_particles

    params%steps_per_frame = int(1.0_real64 / (params%time_step * 30.0_real64))

    max_height = water_volume%max_y
    max_velocity = sqrt(2.0_real64 * params%g * max_height)
    params%speed_sound = max_velocity / sqrt(0.01_real64)

    recomend_step = 0.4_real64 * params%smoothing_radius / &
                    (params%speed_sound * (1.0_real64 + 0.6_real64 * params%alpha))
    print '(a,f0.6,a,f0.6)', 'Using time step: ', params%time_step, ', Minimum recomended ', recomend_step
  end subroutine init_params

  subroutine finalize_particles(fluid_particles, boundary_particles)
    type(fluid_particle), allocatable, intent(inout) :: fluid_particles(:)
    type(boundary_particle), allocatable, intent(inout) :: boundary_particles(:)
    if (allocated(fluid_particles)) deallocate(fluid_particles)
    if (allocated(boundary_particles)) deallocate(boundary_particles)
  end subroutine finalize_particles
end module fluid_mod
