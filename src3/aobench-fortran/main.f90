module aobench_types
  use, intrinsic :: iso_fortran_env, only : int8, int32, real32, real64
  implicit none

  integer, parameter :: width = 256
  integer, parameter :: height = 256
  integer, parameter :: nsubsamples = 2
  integer, parameter :: nao_samples = 8

  type :: vec
    real(real32) :: x
    real(real32) :: y
    real(real32) :: z
  end type vec

  type :: isect
    real(real32) :: t
    type(vec) :: p
    type(vec) :: n
    integer(int32) :: hit
  end type isect

  type :: sphere
    type(vec) :: center
    real(real32) :: radius
  end type sphere

  type :: plane
    type(vec) :: p
    type(vec) :: n
  end type plane

  type :: ray
    type(vec) :: org
    type(vec) :: dir
  end type ray
end module aobench_types

module aobench_device
  use, intrinsic :: iso_fortran_env, only : int8, int32, real32
  use aobench_types, only : vec, isect, sphere, plane, ray, nao_samples
  implicit none
  !$omp declare target (vdot, vcross, vnormalize, ray_sphere_intersect, ray_plane_intersect, ortho_basis, &
  !$omp& rng_init, rng_next, rng_uniform, ambient_occlusion, my_clamp)
contains

  pure real(real32) function vdot(v0, v1)
    type(vec), intent(in) :: v0, v1
    vdot = v0%x * v1%x + v0%y * v1%y + v0%z * v1%z
  end function vdot

  pure subroutine vcross(c, v0, v1)
    type(vec), intent(out) :: c
    type(vec), intent(in) :: v0, v1
    c%x = v0%y * v1%z - v0%z * v1%y
    c%y = v0%z * v1%x - v0%x * v1%z
    c%z = v0%x * v1%y - v0%y * v1%x
  end subroutine vcross

  subroutine vnormalize(c)
    type(vec), intent(inout) :: c
    real(real32) :: length
    length = sqrt(vdot(c, c))
    if (abs(length) > 1.0e-17_real32) then
      c%x = c%x / length
      c%y = c%y / length
      c%z = c%z / length
    end if
  end subroutine vnormalize

  subroutine ray_sphere_intersect(intersection, current_ray, current_sphere)
    type(isect), intent(inout) :: intersection
    type(ray), intent(in) :: current_ray
    type(sphere), intent(in) :: current_sphere
    type(vec) :: rs
    real(real32) :: b, c, d, t

    rs%x = current_ray%org%x - current_sphere%center%x
    rs%y = current_ray%org%y - current_sphere%center%y
    rs%z = current_ray%org%z - current_sphere%center%z
    b = vdot(rs, current_ray%dir)
    c = vdot(rs, rs) - current_sphere%radius * current_sphere%radius
    d = b * b - c

    if (d > 0.0_real32) then
      t = -b - sqrt(d)
      if (t > 0.0_real32 .and. t < intersection%t) then
        intersection%t = t
        intersection%hit = 1_int32
        intersection%p%x = current_ray%org%x + current_ray%dir%x * t
        intersection%p%y = current_ray%org%y + current_ray%dir%y * t
        intersection%p%z = current_ray%org%z + current_ray%dir%z * t
        intersection%n%x = intersection%p%x - current_sphere%center%x
        intersection%n%y = intersection%p%y - current_sphere%center%y
        intersection%n%z = intersection%p%z - current_sphere%center%z
        call vnormalize(intersection%n)
      end if
    end if
  end subroutine ray_sphere_intersect

  subroutine ray_plane_intersect(intersection, current_ray, current_plane)
    type(isect), intent(inout) :: intersection
    type(ray), intent(in) :: current_ray
    type(plane), intent(in) :: current_plane
    real(real32) :: d, v, t

    d = -vdot(current_plane%p, current_plane%n)
    v = vdot(current_ray%dir, current_plane%n)
    if (abs(v) < 1.0e-17_real32) return
    t = -(vdot(current_ray%org, current_plane%n) + d) / v
    if (t > 0.0_real32 .and. t < intersection%t) then
      intersection%t = t
      intersection%hit = 1_int32
      intersection%p%x = current_ray%org%x + current_ray%dir%x * t
      intersection%p%y = current_ray%org%y + current_ray%dir%y * t
      intersection%p%z = current_ray%org%z + current_ray%dir%z * t
      intersection%n = current_plane%n
    end if
  end subroutine ray_plane_intersect

  subroutine ortho_basis(basis, normal)
    type(vec), intent(out) :: basis(0:2)
    type(vec), intent(in) :: normal
    basis(2) = normal
    basis(1)%x = 0.0_real32
    basis(1)%y = 0.0_real32
    basis(1)%z = 0.0_real32
    if (normal%x < 0.6_real32 .and. normal%x > -0.6_real32) then
      basis(1)%x = 1.0_real32
    else if (normal%y < 0.6_real32 .and. normal%y > -0.6_real32) then
      basis(1)%y = 1.0_real32
    else if (normal%z < 0.6_real32 .and. normal%z > -0.6_real32) then
      basis(1)%z = 1.0_real32
    else
      basis(1)%x = 1.0_real32
    end if
    call vcross(basis(0), basis(1), basis(2))
    call vnormalize(basis(0))
    call vcross(basis(1), basis(2), basis(0))
    call vnormalize(basis(1))
  end subroutine ortho_basis

  subroutine rng_init(state, seed)
    integer(int32), intent(out) :: state
    integer(int32), intent(in) :: seed
    state = seed
  end subroutine rng_init

  integer(int32) function rng_next(state)
    integer(int32), intent(inout) :: state
    state = ieor(state, shiftr(state, 6))
    state = ieor(state, shiftl(state, 17))
    state = ieor(state, shiftr(state, 9))
    rng_next = state
  end function rng_next

  real(real32) function rng_uniform(state)
    integer(int32), intent(inout) :: state
    integer(int32) :: bits
    bits = ior(iand(rng_next(state), int(z'007fffff', int32)), int(z'3f800000', int32))
    rng_uniform = transfer(bits, rng_uniform) - 1.0_real32
  end function rng_uniform

  subroutine ambient_occlusion(col, intersection, spheres, current_plane, rng_state)
    type(vec), intent(out) :: col
    type(isect), intent(in) :: intersection
    type(sphere), intent(in) :: spheres(0:2)
    type(plane), intent(in) :: current_plane
    integer(int32), intent(inout) :: rng_state
    integer :: i, j
    type(vec) :: p, basis(0:2)
    type(ray) :: current_ray
    type(isect) :: occlusion_intersection
    real(real32) :: theta, phi, x, y, z, rx, ry, rz, occlusion
    real(real32), parameter :: pi = acos(-1.0_real32)
    real(real32), parameter :: eps = 0.0001_real32

    p%x = intersection%p%x + eps * intersection%n%x
    p%y = intersection%p%y + eps * intersection%n%y
    p%z = intersection%p%z + eps * intersection%n%z
    call ortho_basis(basis, intersection%n)
    occlusion = 0.0_real32
    do j = 0, nao_samples - 1
      do i = 0, nao_samples - 1
        theta = sqrt(rng_uniform(rng_state))
        phi = 2.0_real32 * pi * rng_uniform(rng_state)
        x = cos(phi) * theta
        y = sin(phi) * theta
        z = sqrt(1.0_real32 - theta * theta)
        rx = x * basis(0)%x + y * basis(1)%x + z * basis(2)%x
        ry = x * basis(0)%y + y * basis(1)%y + z * basis(2)%y
        rz = x * basis(0)%z + y * basis(1)%z + z * basis(2)%z
        current_ray%org = p
        current_ray%dir%x = rx
        current_ray%dir%y = ry
        current_ray%dir%z = rz
        occlusion_intersection%t = 1.0e17_real32
        occlusion_intersection%hit = 0_int32
        call ray_sphere_intersect(occlusion_intersection, current_ray, spheres(0))
        call ray_sphere_intersect(occlusion_intersection, current_ray, spheres(1))
        call ray_sphere_intersect(occlusion_intersection, current_ray, spheres(2))
        call ray_plane_intersect(occlusion_intersection, current_ray, current_plane)
        if (occlusion_intersection%hit /= 0_int32) occlusion = occlusion + 1.0_real32
      end do
    end do
    occlusion = (real(nao_samples * nao_samples, real32) - occlusion) / real(nao_samples * nao_samples, real32)
    col%x = occlusion
    col%y = occlusion
    col%z = occlusion
  end subroutine ambient_occlusion

  integer(int8) function my_clamp(value)
    real(real32), intent(in) :: value
    integer(int32) :: integer_value
    integer_value = int(value * 255.5_real32, int32)
    if (integer_value < 0_int32) integer_value = 0_int32
    if (integer_value > 255_int32) integer_value = 255_int32
    my_clamp = transfer(iand(integer_value, 255_int32), my_clamp)
  end function my_clamp

end module aobench_device

program aobench
  use, intrinsic :: iso_fortran_env, only : int8, int32, real32, real64
  use omp_lib, only : omp_get_wtime
  use aobench_types
  use aobench_device
  implicit none

  integer :: argc, loopmax, iteration, parse_status
  integer(int8), allocatable :: image(:)
  type(sphere) :: spheres(0:2)
  type(plane) :: current_plane
  real(real64) :: total_time
  character(len=64) :: argument

  argc = command_argument_count()
  if (argc /= 1) then
    write(*, '(A)') 'Usage: ./main <iterations>'
    error stop 1
  end if
  call get_command_argument(1, argument)
  read(argument, *, iostat=parse_status) loopmax
  if (parse_status /= 0) error stop 1

  call init_scene(spheres, current_plane)
  allocate(image(0:width * height * 3 - 1))
  total_time = 0.0_real64
  do iteration = 0, loopmax - 1
    total_time = total_time + render(image, width, height, nsubsamples, spheres, current_plane)
  end do
  write(*, '(A,F0.6,A)') 'Average kernel time: ', total_time * 1.0e6_real64 / real(loopmax, real64), ' usec.'
  call saveppm('ao.ppm', width, height, image)
  deallocate(image)

contains

  subroutine init_scene(scene_spheres, scene_plane)
    type(sphere), intent(out) :: scene_spheres(0:2)
    type(plane), intent(out) :: scene_plane
    scene_spheres(0)%center%x = -2.0_real32
    scene_spheres(0)%center%y = 0.0_real32
    scene_spheres(0)%center%z = -3.5_real32
    scene_spheres(0)%radius = 0.5_real32
    scene_spheres(1)%center%x = -0.5_real32
    scene_spheres(1)%center%y = 0.0_real32
    scene_spheres(1)%center%z = -3.0_real32
    scene_spheres(1)%radius = 0.5_real32
    scene_spheres(2)%center%x = 1.0_real32
    scene_spheres(2)%center%y = 0.0_real32
    scene_spheres(2)%center%z = -2.2_real32
    scene_spheres(2)%radius = 0.5_real32
    scene_plane%p%x = 0.0_real32
    scene_plane%p%y = -0.5_real32
    scene_plane%p%z = 0.0_real32
    scene_plane%n%x = 0.0_real32
    scene_plane%n%y = 1.0_real32
    scene_plane%n%z = 0.0_real32
  end subroutine init_scene

  real(real64) function render(img, w, h, samples, scene_spheres, scene_plane)
    integer(int8), intent(inout) :: img(0:)
    integer, intent(in) :: w, h, samples
    type(sphere), intent(in) :: scene_spheres(0:2)
    type(plane), intent(in) :: scene_plane
    integer :: x_index, y_index, u, v
    integer(int32) :: rng_state
    real(real32) :: s0, s1, s2, px, py
    type(ray) :: current_ray
    type(isect) :: intersection
    type(vec) :: col
    real(real64) :: start_time, end_time

    !$omp target data map(from: img(0:w*h*3-1)) map(to: scene_spheres(0:2), scene_plane)
    start_time = omp_get_wtime()
    !$omp target teams distribute parallel do collapse(2) thread_limit(256) &
    !$omp& private(rng_state, s0, s1, s2, u, v, px, py, current_ray, intersection, col)
    do x_index = 0, w - 1
      do y_index = 0, h - 1
        call rng_init(rng_state, int(y_index * w + x_index, int32))
        s0 = 0.0_real32
        s1 = 0.0_real32
        s2 = 0.0_real32
        do v = 0, samples - 1
          do u = 0, samples - 1
            px = (real(x_index, real32) + real(u, real32) / real(samples, real32) &
              - real(w, real32) / 2.0_real32) / (real(w, real32) / 2.0_real32)
            py = -(real(y_index, real32) + real(v, real32) / real(samples, real32) &
              - real(h, real32) / 2.0_real32) / (real(h, real32) / 2.0_real32)
            current_ray%org%x = 0.0_real32
            current_ray%org%y = 0.0_real32
            current_ray%org%z = 0.0_real32
            current_ray%dir%x = px
            current_ray%dir%y = py
            current_ray%dir%z = -1.0_real32
            call vnormalize(current_ray%dir)
            intersection%t = 1.0e17_real32
            intersection%hit = 0_int32
            call ray_sphere_intersect(intersection, current_ray, scene_spheres(0))
            call ray_sphere_intersect(intersection, current_ray, scene_spheres(1))
            call ray_sphere_intersect(intersection, current_ray, scene_spheres(2))
            call ray_plane_intersect(intersection, current_ray, scene_plane)
            if (intersection%hit /= 0_int32) then
              call ambient_occlusion(col, intersection, scene_spheres, scene_plane, rng_state)
              s0 = s0 + col%x
              s1 = s1 + col%y
              s2 = s2 + col%z
            end if
          end do
        end do
        img(3 * (y_index * w + x_index) + 0) = my_clamp(s0 / real(samples * samples, real32))
        img(3 * (y_index * w + x_index) + 1) = my_clamp(s1 / real(samples * samples, real32))
        img(3 * (y_index * w + x_index) + 2) = my_clamp(s2 / real(samples * samples, real32))
      end do
    end do
    !$omp end target teams distribute parallel do
    end_time = omp_get_wtime()
    !$omp end target data
    render = end_time - start_time
  end function render

  subroutine saveppm(filename, w, h, img)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: w, h
    integer(int8), intent(in) :: img(0:)
    integer :: file_unit, io_status
    character(len=64) :: header
    write(header, '("P6",A,I0,1X,I0,A,"255",A)') achar(10), w, h, achar(10), achar(10)
    open(newunit=file_unit, file=filename, access='stream', form='unformatted', status='replace', action='write', iostat=io_status)
    if (io_status /= 0) then
      write(*, '(A,A)') 'Failed to open the file ', filename
      return
    end if
    write(file_unit) trim(header)
    write(file_unit) img(0:w*h*3-1)
    close(file_unit)
  end subroutine saveppm
end program aobench
