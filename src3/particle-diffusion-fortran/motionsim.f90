module motionsim_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine motion_device(particle_x, particle_y, random_x, random_y, grid, grid_size, n_particles, n_iterations, radius, map, n_repeat)
    integer, intent(in) :: grid_size, n_particles, n_iterations, n_repeat
    real(real32), intent(inout) :: particle_x(0:*), particle_y(0:*)
    real(real32), intent(in) :: random_x(0:*), random_y(0:*), radius
    integer(int64), intent(inout) :: map(0:*)
    integer(int32), intent(inout) :: grid(0:*)
    integer :: rep, ii, iter, ix, iy, x, y, map_count
    integer(int64) :: map_base
    real(real32) :: px, py, randnum_x, randnum_y, displacement_x, displacement_y, dx, dy
    real(real64) :: time_total, t0, t1
    map_count = n_particles * grid_size * grid_size
    !$omp target data map(to:random_x(0:n_particles*n_iterations-1),random_y(0:n_particles*n_iterations-1)) &
    !$omp& map(alloc:particle_x(0:n_particles-1),particle_y(0:n_particles-1),map(0:map_count-1))
    print '(a,i0)', ' The number of kernel execution is ', n_repeat
    print '(a,i0)', ' The number of particles is ', n_particles
    time_total = 0.0_real64
    do rep = 1, n_repeat
      !$omp target update to(particle_x(0:n_particles-1))
      !$omp target update to(particle_y(0:n_particles-1))
      !$omp target update to(map(0:map_count-1))
      t0 = seconds()
      !$omp target teams distribute parallel do simd thread_limit(256) &
      !$omp& private(iter,px,py,map_base,randnum_x,randnum_y,displacement_x,displacement_y,dx,dy,ix,iy) &
      !$omp& map(to:random_x(0:n_particles*n_iterations-1),random_y(0:n_particles*n_iterations-1)) &
      !$omp& map(tofrom:particle_x(0:n_particles-1),particle_y(0:n_particles-1),map(0:map_count-1))
      do ii = 0, n_particles-1
        iter = 0
        px = particle_x(ii); py = particle_y(ii)
        map_base = int(ii,int64) * grid_size * grid_size
        do while (iter < n_iterations)
          randnum_x = random_x(iter*n_particles + ii)
          randnum_y = random_y(iter*n_particles + ii)
          displacement_x = randnum_x / 1000.0_real32 - 0.0495_real32
          displacement_y = randnum_y / 1000.0_real32 - 0.0495_real32
          px = px + displacement_x; py = py + displacement_y
          dx = px - real(int(px), real32)
          dy = py - real(int(py), real32)
          ix = floor(px); iy = floor(py)
          if (px < grid_size .and. py < grid_size .and. px >= 0.0_real32 .and. py >= 0.0_real32) then
            if (dx*dx + dy*dy <= radius*radius) map(map_base + iy*grid_size + ix) = map(map_base + iy*grid_size + ix) + 1_int64
          end if
          iter = iter + 1
        end do
        particle_x(ii) = px; particle_y(ii) = py
      end do
      !$omp end target teams distribute parallel do simd
      t1 = seconds(); time_total = time_total + (t1-t0)*1.0e9_real64
    end do
    print *
    print '(a,f10.6,a)', 'Average kernel execution time: ', (time_total*1.0e-9_real64)/real(n_repeat,real64), ' (s)'
    !$omp target update from(map(0:map_count-1))
    !$omp end target data
    do ii = 0, n_particles-1
      do y = 0, grid_size-1
        do x = 0, grid_size-1
          if (map(int(ii,int64)*grid_size*grid_size + y*grid_size + x) > 0) &
            grid(y*grid_size+x) = grid(y*grid_size+x) + int(map(int(ii,int64)*grid_size*grid_size + y*grid_size + x), int32)
        end do
      end do
    end do
  end subroutine

  subroutine motion_host(particle_x, particle_y, random_x, random_y, grid, grid_size, n_particles, n_iterations, radius, map, n_repeat)
    integer, intent(in) :: grid_size, n_particles, n_iterations, n_repeat
    real(real32), intent(inout) :: particle_x(0:*), particle_y(0:*)
    real(real32), intent(in) :: random_x(0:*), random_y(0:*), radius
    integer(int64), intent(inout) :: map(0:*)
    integer(int32), intent(inout) :: grid(0:*)
    integer :: rep, ii, iter, ix, iy
    integer(int64) :: map_base
    real(real32) :: px, py, dx, dy
    do rep = 1, n_repeat
      do ii = 0, n_particles-1
        px = particle_x(ii); py = particle_y(ii); map_base = int(ii,int64)*grid_size*grid_size
        do iter = 0, n_iterations-1
          px = px + random_x(iter*n_particles+ii)/1000.0_real32 - 0.0495_real32
          py = py + random_y(iter*n_particles+ii)/1000.0_real32 - 0.0495_real32
          dx = px - real(int(px),real32); dy = py - real(int(py),real32)
          ix = floor(px); iy = floor(py)
          if (px < grid_size .and. py < grid_size .and. px >= 0.0_real32 .and. py >= 0.0_real32) then
            if (dx*dx + dy*dy <= radius*radius) map(map_base + iy*grid_size + ix) = map(map_base + iy*grid_size + ix) + 1_int64
          end if
        end do
        particle_x(ii) = px; particle_y(ii) = py
      end do
    end do
  end subroutine
end module

program main
  use motionsim_mod
  implicit none
  integer, parameter :: grid_size=21, n_particles=147456
  real(real32), parameter :: radius=0.5_real32
  integer :: n_iterations, n_repeat, i, x, y, ios
  integer(int64) :: map_size, count
  character(len=64) :: arg
  integer(int32), allocatable :: grid(:)
  real(real32), allocatable :: random_x(:), random_y(:), particle_x(:), particle_y(:)
  integer(int64), allocatable :: map(:), map_ref(:)
  real(real64) :: t0, t1
  if (command_argument_count() /= 2) then
    print '(a)', ' Incorrect number of parameters '
    print '(a)', ' Usage: main <Number of iterations within the kernel> <Kernel execution count>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) n_iterations
  call get_command_argument(2,arg); read(arg,*,iostat=ios) n_repeat
  map_size = int(n_particles,int64) * grid_size * grid_size
  allocate(grid(0:grid_size*grid_size-1), random_x(0:n_particles*n_iterations-1), random_y(0:n_particles*n_iterations-1), &
    particle_x(0:n_particles-1), particle_y(0:n_particles-1), map(0:map_size-1), map_ref(0:map_size-1))
  particle_x = 10.0_real32; particle_y = 10.0_real32; map = 0_int64; map_ref = 0_int64; grid = 0_int32
  call c_srand(17_int32)
  do i = 0, n_particles*n_iterations-1
    random_x(i) = real(mod(c_rand(), 100), real32)
    random_y(i) = real(mod(c_rand(), 100), real32)
  end do
  t0 = seconds()
  call motion_device(particle_x, particle_y, random_x, random_y, grid, grid_size, n_particles, n_iterations, radius, map, n_repeat)
  t1 = seconds()
  print *
  print '(a,f10.6,a)', 'Simulation time: ', t1-t0, ' (s) '
  call motion_host(particle_x, particle_y, random_x, random_y, grid, grid_size, n_particles, n_iterations, radius, map_ref, n_repeat)
  count = 0
  do i = 0, map_size-1
    if (map(i) /= map_ref(i)) count = count + 1
  end do
  print '(a)', merge('PASS', 'FAIL', count <= 2)
end program
