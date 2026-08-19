module sph_types
  use iso_fortran_env, only: real64
  implicit none

  real(real64), parameter :: pi = 3.1415926535897932384626433832795_real64

  type :: double3
    real(real64) :: x
    real(real64) :: y
    real(real64) :: z
  end type double3

  type :: boundary_particle
    type(double3) :: pos
    type(double3) :: n
  end type boundary_particle

  type :: fluid_particle
    real(real64) :: density
    real(real64) :: pressure
    type(double3) :: pos
    type(double3) :: v
    type(double3) :: v_half
    type(double3) :: a
  end type fluid_particle

  type :: param
    real(real64) :: rest_density
    real(real64) :: mass_particle
    real(real64) :: spacing_particle
    real(real64) :: smoothing_radius
    real(real64) :: g
    real(real64) :: time_step
    real(real64) :: alpha
    real(real64) :: surface_tension
    real(real64) :: speed_sound
    integer :: number_particles
    integer :: number_fluid_particles
    integer :: number_boundary_particles
    integer :: number_steps
    integer :: steps_per_frame
  end type param

  type :: aabb
    real(real64) :: min_x
    real(real64) :: max_x
    real(real64) :: min_y
    real(real64) :: max_y
    real(real64) :: min_z
    real(real64) :: max_z
  end type aabb
end module sph_types
