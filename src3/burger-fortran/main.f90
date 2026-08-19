program burger
  use iso_fortran_env, only: int32, real64
  use omp_lib, only: omp_get_wtime
  implicit none

  integer(int32) :: x_points, y_points, num_itrs, grid_elements
  integer(int32) :: i, j, itr
  real(real64), parameter :: x_len = 2.0_real64, y_len = 2.0_real64
  real(real64), parameter :: nu = 0.01_real64, sigma = 0.0009_real64
  real(real64) :: del_x, del_y, del_t, start_time, end_time, kernel_time
  logical :: ok
  character(len=128) :: argument
  real(real64), allocatable :: x(:), y(:), u(:), v(:), u_new(:), v_new(:), d_u(:), d_v(:)

  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: ./main <dim_x> <dim_y> <nt>'
    print '(a)', 'dim_x: number of grid points in the x axis'
    print '(a)', 'dim_y: number of grid points in the y axis'
    print '(a)', 'nt: number of time steps'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument, *) x_points
  call get_command_argument(2, argument)
  read(argument, *) y_points
  call get_command_argument(3, argument)
  read(argument, *) num_itrs

  del_x = x_len / real(x_points - 1, real64)
  del_y = y_len / real(y_points - 1, real64)
  del_t = sigma * del_x * del_y / nu
  grid_elements = x_points * y_points

  allocate(x(0:x_points-1), y(0:y_points-1))
  allocate(u(0:grid_elements-1), v(0:grid_elements-1), u_new(0:grid_elements-1), v_new(0:grid_elements-1))
  allocate(d_u(0:grid_elements-1), d_v(0:grid_elements-1))

  print '(a)', "2D Burger's equation"
  print '(a,i0,a,i0)', 'Grid dimension: x = ', x_points, ' y = ', y_points
  print '(a,i0)', 'Number of time steps: ', num_itrs

  do i = 0, x_points - 1
    x(i) = real(i, real64) * del_x
  end do
  do i = 0, y_points - 1
    y(i) = real(i, real64) * del_y
  end do
  call initialize_velocity(x_points, y_points, x, y, u, v, u_new, v_new)

!$omp target data map(to: u_new(0:grid_elements-1), v_new(0:grid_elements-1)) &
!$omp& map(tofrom: u(0:grid_elements-1), v(0:grid_elements-1))
  start_time = omp_get_wtime()
  do itr = 0, num_itrs - 1
!$omp target teams distribute parallel do collapse(2) thread_limit(256) nowait
    do i = 1, y_points - 2
      do j = 1, x_points - 2
        u_new(i*x_points+j) = u(i*x_points+j) + (nu*del_t/(del_x*del_x)) * &
          (u(i*x_points+j+1) + u(i*x_points+j-1) - 2.0_real64*u(i*x_points+j)) + &
          (nu*del_t/(del_y*del_y)) * (u((i+1)*x_points+j) + u((i-1)*x_points+j) - 2.0_real64*u(i*x_points+j)) - &
          (del_t/del_x)*u(i*x_points+j)*(u(i*x_points+j)-u(i*x_points+j-1)) - &
          (del_t/del_y)*v(i*x_points+j)*(u(i*x_points+j)-u((i-1)*x_points+j))
        v_new(i*x_points+j) = v(i*x_points+j) + (nu*del_t/(del_x*del_x)) * &
          (v(i*x_points+j+1) + v(i*x_points+j-1) - 2.0_real64*v(i*x_points+j)) + &
          (nu*del_t/(del_y*del_y)) * (v((i+1)*x_points+j) + v((i-1)*x_points+j) - 2.0_real64*v(i*x_points+j)) - &
          (del_t/del_x)*u(i*x_points+j)*(v(i*x_points+j)-v(i*x_points+j-1)) - &
          (del_t/del_y)*v(i*x_points+j)*(v(i*x_points+j)-v((i-1)*x_points+j))
      end do
    end do
!$omp end target teams distribute parallel do

!$omp target teams distribute parallel do thread_limit(256) nowait
    do i = 0, x_points - 1
      u_new(i) = 1.0_real64
      v_new(i) = 1.0_real64
      u_new((y_points-1)*x_points+i) = 1.0_real64
      v_new((y_points-1)*x_points+i) = 1.0_real64
    end do
!$omp end target teams distribute parallel do

!$omp target teams distribute parallel do thread_limit(256) nowait
    do j = 0, y_points - 1
      u_new(j*x_points) = 1.0_real64
      v_new(j*x_points) = 1.0_real64
      u_new(j*x_points+x_points-1) = 1.0_real64
      v_new(j*x_points+x_points-1) = 1.0_real64
    end do
!$omp end target teams distribute parallel do

!$omp target teams distribute parallel do collapse(2) thread_limit(256)
    do i = 0, y_points - 1
      do j = 0, x_points - 1
        u(i*x_points+j) = u_new(i*x_points+j)
        v(i*x_points+j) = v_new(i*x_points+j)
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  end_time = omp_get_wtime()
  kernel_time = end_time - start_time
  print '(a,f0.6,a)', 'Total kernel execution time ', kernel_time, ' (s)'
!$omp end target data

  d_u = u
  d_v = v
  print '(a)', 'Serial computing for verification...'
  call initialize_velocity(x_points, y_points, x, y, u, v, u_new, v_new)

  do itr = 0, num_itrs - 1
    do i = 1, y_points - 2
      do j = 1, x_points - 2
        u_new(i*x_points+j) = u(i*x_points+j) + (nu*del_t/(del_x*del_x)) * &
          (u(i*x_points+j+1) + u(i*x_points+j-1) - 2.0_real64*u(i*x_points+j)) + &
          (nu*del_t/(del_y*del_y)) * (u((i+1)*x_points+j) + u((i-1)*x_points+j) - 2.0_real64*u(i*x_points+j)) - &
          (del_t/del_x)*u(i*x_points+j)*(u(i*x_points+j)-u(i*x_points+j-1)) - &
          (del_t/del_y)*v(i*x_points+j)*(u(i*x_points+j)-u((i-1)*x_points+j))
        v_new(i*x_points+j) = v(i*x_points+j) + (nu*del_t/(del_x*del_x)) * &
          (v(i*x_points+j+1) + v(i*x_points+j-1) - 2.0_real64*v(i*x_points+j)) + &
          (nu*del_t/(del_y*del_y)) * (v((i+1)*x_points+j) + v((i-1)*x_points+j) - 2.0_real64*v(i*x_points+j)) - &
          (del_t/del_x)*u(i*x_points+j)*(v(i*x_points+j)-v(i*x_points+j-1)) - &
          (del_t/del_y)*v(i*x_points+j)*(v(i*x_points+j)-v((i-1)*x_points+j))
      end do
    end do

    do i = 0, x_points - 1
      u_new(i) = 1.0_real64
      v_new(i) = 1.0_real64
      u_new((y_points-1)*x_points+i) = 1.0_real64
      v_new((y_points-1)*x_points+i) = 1.0_real64
    end do
    do j = 0, y_points - 1
      u_new(j*x_points) = 1.0_real64
      v_new(j*x_points) = 1.0_real64
      u_new(j*x_points+x_points-1) = 1.0_real64
      v_new(j*x_points+x_points-1) = 1.0_real64
    end do
    do i = 0, y_points - 1
      do j = 0, x_points - 1
        u(i*x_points+j) = u_new(i*x_points+j)
        v(i*x_points+j) = v_new(i*x_points+j)
      end do
    end do
  end do

  ok = .true.
  do i = 0, y_points - 1
    do j = 0, x_points - 1
      if (abs(d_u(i*x_points+j)-u(i*x_points+j)) > 1.0e-6_real64 .or. &
          abs(d_v(i*x_points+j)-v(i*x_points+j)) > 1.0e-6_real64) ok = .false.
    end do
  end do
  if (ok) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if

  deallocate(x, y, u, v, u_new, v_new, d_u, d_v)

contains

  subroutine initialize_velocity(nx, ny, x_coord, y_coord, u_field, v_field, u_next, v_next)
    integer(int32), intent(in) :: nx, ny
    real(real64), intent(in) :: x_coord(0:), y_coord(0:)
    real(real64), intent(out) :: u_field(0:), v_field(0:), u_next(0:), v_next(0:)
    integer(int32) :: row, col

    do row = 0, ny - 1
      do col = 0, nx - 1
        u_field(row*nx+col) = 1.0_real64
        v_field(row*nx+col) = 1.0_real64
        u_next(row*nx+col) = 1.0_real64
        v_next(row*nx+col) = 1.0_real64
        if (x_coord(col) > 0.5_real64 .and. x_coord(col) < 1.0_real64 .and. &
            y_coord(row) > 0.5_real64 .and. y_coord(row) < 1.0_real64) then
          u_field(row*nx+col) = 2.0_real64
          v_field(row*nx+col) = 2.0_real64
          u_next(row*nx+col) = 2.0_real64
          v_next(row*nx+col) = 2.0_real64
        end if
      end do
    end do
  end subroutine initialize_velocity

end program burger
