module geometry_mod
  use iso_fortran_env, only: real64
  use sph_types
  implicit none
contains

  subroutine construct_boundary_box(boundary_particles, boundary, params)
    type(boundary_particle), intent(inout) :: boundary_particles(0:)
    type(aabb), intent(inout) :: boundary
    type(param), intent(inout) :: params
    real(real64) :: spacing, min_x, min_y, min_z, max_x, max_y, max_z
    real(real64) :: recip_root_three, recip_root_two
    integer :: num_x, num_y, num_z, i, nx, ny, nz

    spacing = params%spacing_particle
    num_x = ceiling((boundary%max_x - boundary%min_x) / spacing)
    num_y = ceiling((boundary%max_y - boundary%min_y) / spacing)
    num_z = ceiling((boundary%max_z - boundary%min_z) / spacing)
    min_x = boundary%min_x
    min_y = boundary%min_y
    min_z = boundary%min_z
    max_x = min_x + real(num_x - 1, real64) * spacing
    max_y = min_y + real(num_y - 1, real64) * spacing
    max_z = min_z + real(num_z - 1, real64) * spacing
    boundary%max_x = max_x
    boundary%max_y = max_y
    boundary%max_z = max_z

    recip_root_three = 1.0_real64 / sqrt(3.0_real64)
    recip_root_two = 1.0_real64 / sqrt(2.0_real64)
    i = 0

    boundary_particles(i)%pos = double3(min_x, max_y, min_z)
    boundary_particles(i)%n = double3(recip_root_three, -recip_root_three, recip_root_three)
    i = i + 1
    boundary_particles(i)%pos = double3(max_x, max_y, min_z)
    boundary_particles(i)%n = double3(-recip_root_three, -recip_root_three, recip_root_three)
    i = i + 1
    boundary_particles(i)%pos = double3(min_x, max_y, max_z)
    boundary_particles(i)%n = double3(recip_root_three, -recip_root_three, -recip_root_three)
    i = i + 1
    boundary_particles(i)%pos = double3(max_x, max_y, max_z)
    boundary_particles(i)%n = double3(-recip_root_three, -recip_root_three, -recip_root_three)
    i = i + 1

    boundary_particles(i)%pos = double3(min_x, min_y, min_z)
    boundary_particles(i)%n = double3(recip_root_three, recip_root_three, recip_root_three)
    i = i + 1
    boundary_particles(i)%pos = double3(max_x, min_y, min_z)
    boundary_particles(i)%n = double3(-recip_root_three, recip_root_three, recip_root_three)
    i = i + 1
    boundary_particles(i)%pos = double3(min_x, min_y, max_z)
    boundary_particles(i)%n = double3(recip_root_three, recip_root_three, -recip_root_three)
    i = i + 1
    boundary_particles(i)%pos = double3(max_x, min_y, max_z)
    boundary_particles(i)%n = double3(-recip_root_three, recip_root_three, -recip_root_three)
    i = i + 1

    do nx = 0, num_x - 3
      boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, max_y, min_z)
      boundary_particles(i)%n = double3(0.0_real64, -recip_root_two, recip_root_two)
      i = i + 1
      boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, max_y, max_z)
      boundary_particles(i)%n = double3(0.0_real64, -recip_root_two, -recip_root_two)
      i = i + 1
      boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, min_y, min_z)
      boundary_particles(i)%n = double3(0.0_real64, recip_root_two, recip_root_two)
      i = i + 1
      boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, min_y, max_z)
      boundary_particles(i)%n = double3(0.0_real64, recip_root_two, -recip_root_two)
      i = i + 1
    end do

    do ny = 0, num_y - 3
      boundary_particles(i)%pos = double3(max_x, min_y + spacing + real(ny, real64) * spacing, min_z)
      boundary_particles(i)%n = double3(-recip_root_two, 0.0_real64, recip_root_two)
      i = i + 1
      boundary_particles(i)%pos = double3(max_x, min_y + spacing + real(ny, real64) * spacing, max_z)
      boundary_particles(i)%n = double3(-recip_root_two, 0.0_real64, -recip_root_two)
      i = i + 1
      boundary_particles(i)%pos = double3(min_x, min_y + spacing + real(ny, real64) * spacing, min_z)
      boundary_particles(i)%n = double3(recip_root_two, 0.0_real64, recip_root_two)
      i = i + 1
      boundary_particles(i)%pos = double3(min_x, min_y + spacing + real(ny, real64) * spacing, max_z)
      boundary_particles(i)%n = double3(recip_root_two, 0.0_real64, -recip_root_two)
      i = i + 1

      do nx = 0, num_x - 3
        boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, &
                                            min_y + spacing + real(ny, real64) * spacing, max_z)
        boundary_particles(i)%n = double3(0.0_real64, 0.0_real64, -1.0_real64)
        i = i + 1
        boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, &
                                            min_y + spacing + real(ny, real64) * spacing, min_z)
        boundary_particles(i)%n = double3(0.0_real64, 0.0_real64, 1.0_real64)
        i = i + 1
      end do
    end do

    do nz = 0, num_z - 3
      boundary_particles(i)%pos = double3(min_x, max_y, min_z + spacing + real(nz, real64) * spacing)
      boundary_particles(i)%n = double3(recip_root_two, -recip_root_two, 0.0_real64)
      i = i + 1
      boundary_particles(i)%pos = double3(max_x, max_y, min_z + spacing + real(nz, real64) * spacing)
      boundary_particles(i)%n = double3(-recip_root_two, -recip_root_two, 0.0_real64)
      i = i + 1
      boundary_particles(i)%pos = double3(min_x, min_y, min_z + spacing + real(nz, real64) * spacing)
      boundary_particles(i)%n = double3(recip_root_two, recip_root_two, 0.0_real64)
      i = i + 1
      boundary_particles(i)%pos = double3(max_x, min_y, min_z + spacing + real(nz, real64) * spacing)
      boundary_particles(i)%n = double3(-recip_root_two, recip_root_two, 0.0_real64)
      i = i + 1

      do nx = 0, num_x - 3
        boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, max_y, &
                                            min_z + spacing + real(nz, real64) * spacing)
        boundary_particles(i)%n = double3(0.0_real64, -1.0_real64, 0.0_real64)
        i = i + 1
        boundary_particles(i)%pos = double3(min_x + spacing + real(nx, real64) * spacing, min_y, &
                                            min_z + spacing + real(nz, real64) * spacing)
        boundary_particles(i)%n = double3(0.0_real64, 1.0_real64, 0.0_real64)
        i = i + 1
      end do

      do ny = 0, num_y - 3
        boundary_particles(i)%pos = double3(min_x, min_y + spacing + real(ny, real64) * spacing, &
                                            min_z + spacing + real(nz, real64) * spacing)
        boundary_particles(i)%n = double3(1.0_real64, 0.0_real64, 0.0_real64)
        i = i + 1
        boundary_particles(i)%pos = double3(max_x, min_y + spacing + real(ny, real64) * spacing, &
                                            min_z + spacing + real(nz, real64) * spacing)
        boundary_particles(i)%n = double3(-1.0_real64, 0.0_real64, 0.0_real64)
        i = i + 1
      end do
    end do

    params%number_boundary_particles = i
    params%number_particles = params%number_fluid_particles + params%number_boundary_particles
  end subroutine construct_boundary_box
end module geometry_mod
