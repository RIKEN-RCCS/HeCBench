program sph
  use iso_fortran_env, only: real64
  use omp_lib
  use sph_types
  use fluid_mod
  use fileio_mod
  implicit none
  type(param) :: params
  type(aabb) :: water_volume, boundary_volume
  type(fluid_particle), allocatable :: fluid_particles(:)
  type(boundary_particle), allocatable :: boundary_particles(:)
  integer :: num_fluid_particles, num_boundary_particles, num_particles, n, i, j
  type(double3) :: p_pos, p_v, q_pos, q_v, tmp_a, k_pos, k_n, v_half, v, pos, a
  real(real64) :: density, p_density, p_pressure, q_density, q_pressure, ax, ay, az, dtv, start_time, end_time

  call init_params(water_volume, boundary_volume, params)
  call init_particles(fluid_particles, boundary_particles, water_volume, boundary_volume, params)
  call euler_start(fluid_particles, boundary_particles, params)
  num_fluid_particles = params%number_fluid_particles
  num_boundary_particles = params%number_boundary_particles
  num_particles = min(num_fluid_particles, num_boundary_particles)

  !$omp target data map(tofrom: fluid_particles(0:num_fluid_particles-1), boundary_particles(0:num_boundary_particles-1))
    start_time = omp_get_wtime()
    do n = 0, params%number_steps - 1
      !$omp target teams distribute parallel do simd thread_limit(256) private(p_pos,p_v,q_pos,q_v,density,j)
      do i = 0, num_fluid_particles - 1
        p_pos = fluid_particles(i)%pos; p_v = fluid_particles(i)%v; density = fluid_particles(i)%density
        do j = 0, num_fluid_particles - 1
          q_pos = fluid_particles(j)%pos; q_v = fluid_particles(j)%v
          density = density + compute_density(p_pos, p_v, q_pos, q_v, params)
        end do
        fluid_particles(i)%density = density
        fluid_particles(i)%pressure = compute_pressure(density, params)
      end do
      !$omp end target teams distribute parallel do simd

      !$omp target teams distribute parallel do simd thread_limit(256) private(ax,ay,az,p_pos,p_v,p_density,p_pressure,q_pos,q_v,q_density,q_pressure,tmp_a,j)
      do i = 0, num_fluid_particles - 1
        ax = 0.0_real64; ay = 0.0_real64; az = -9.8_real64
        p_pos = fluid_particles(i)%pos; p_v = fluid_particles(i)%v; p_density = fluid_particles(i)%density; p_pressure = fluid_particles(i)%pressure
        do j = 0, num_fluid_particles - 1
          if (i /= j) then
            q_pos = fluid_particles(j)%pos; q_v = fluid_particles(j)%v; q_density = fluid_particles(j)%density; q_pressure = fluid_particles(j)%pressure
            tmp_a = compute_acceleration(p_pos, p_v, p_density, p_pressure, q_pos, q_v, q_density, q_pressure, params)
            ax = ax + tmp_a%x; ay = ay + tmp_a%y; az = az + tmp_a%z
          end if
        end do
        fluid_particles(i)%a = double3(ax, ay, az)
      end do
      !$omp end target teams distribute parallel do simd

      !$omp target teams distribute parallel do simd thread_limit(256) private(ax,ay,az,p_pos,k_pos,k_n,tmp_a,j)
      do i = 0, num_particles - 1
        ax = fluid_particles(i)%a%x; ay = fluid_particles(i)%a%y; az = fluid_particles(i)%a%z; p_pos = fluid_particles(i)%pos
        do j = 0, num_boundary_particles - 1
          k_pos = boundary_particles(j)%pos; k_n = boundary_particles(j)%n
          tmp_a = compute_boundary_acceleration(p_pos, k_pos, k_n, params%smoothing_radius, params%speed_sound)
          ax = ax + tmp_a%x; ay = ay + tmp_a%y; az = az + tmp_a%z
        end do
        fluid_particles(i)%a = double3(ax, ay, az)
      end do
      !$omp end target teams distribute parallel do simd

      !$omp target teams distribute parallel do simd thread_limit(256) private(dtv,v_half,v,pos,a)
      do i = 0, num_fluid_particles - 1
        dtv = params%time_step
        v_half = fluid_particles(i)%v_half; v = fluid_particles(i)%v; pos = fluid_particles(i)%pos; a = fluid_particles(i)%a
        v_half%x = v_half%x + dtv*a%x; v_half%y = v_half%y + dtv*a%y; v_half%z = v_half%z + dtv*a%z
        v%x = v_half%x + a%x*(dtv/2.0_real64); v%y = v_half%y + a%y*(dtv/2.0_real64); v%z = v_half%z + a%z*(dtv/2.0_real64)
        pos%x = pos%x + dtv*v_half%x; pos%y = pos%y + dtv*v_half%y; pos%z = pos%z + dtv*v_half%z
        fluid_particles(i)%v_half = v_half; fluid_particles(i)%v = v; fluid_particles(i)%pos = pos
      end do
      !$omp end target teams distribute parallel do simd
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average execution time of sph kernels: ', ((end_time-start_time)*1.0e3_real64)/params%number_steps, ' (ms)'
  !$omp end target data

  call write_file(fluid_particles, params)
  call finalize_particles(fluid_particles, boundary_particles)
end program sph
