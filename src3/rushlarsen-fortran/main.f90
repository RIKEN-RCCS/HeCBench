program rushlarsen
  use iso_fortran_env, only: real64
  use omp_lib
  use rushlarsen_utils
  use rushlarsen_kernels
  use rushlarsen_reference
  implicit none
  integer :: argc, num_timesteps, num_nodes, it
  integer :: total_num_states, total_num_parameters, i
  real(real64), allocatable :: states(:), states2(:), parameters(:)
  real(real64) :: t_start, dt, t, start_time, end_time, time_elapsed, rmse
  character(len=64) :: arg
  t_start = 0.0_real64
  dt = 0.02e-3_real64
  num_timesteps = 1000000
  num_nodes = 1
  argc = command_argument_count()
  if (argc > 0) then
    call get_command_argument(1,arg); read(arg,*) num_timesteps
    print '(a,i0)', 'num_timesteps set to ', num_timesteps
    call get_command_argument(2,arg); read(arg,*) num_nodes
    print '(a,i0)', 'num_nodes set to ', num_nodes
    if (num_timesteps <= 0 .or. num_nodes <= 0) stop 1
  end if
  total_num_states = num_nodes * num_states
  allocate(states(0:total_num_states-1), states2(0:total_num_states-1))
  call init_state_values(states, num_nodes)
  states2 = states
  total_num_parameters = num_nodes * num_params
  allocate(parameters(0:total_num_parameters-1))
  call init_parameters_values(parameters, num_nodes)
  t = t_start
  print '(a)', 'Host: Rush Larsen (exp integrator on all gates)'
  do it = 0, num_timesteps - 1
    call forward_rush_larsen(states, t, dt, parameters, num_nodes)
    t = t + dt
  end do
  print '(a)', 'Device: Rush Larsen (exp integrator on all gates)'
  !$omp target data map(tofrom: states2(0:total_num_states-1)) map(to: parameters(0:total_num_parameters-1))
    t = t_start
    start_time = omp_get_wtime()
    do it = 0, num_timesteps - 1
      call k_forward_rush_larsen(states2, t, dt, parameters, num_nodes)
      t = t + dt
    end do
    end_time = omp_get_wtime()
    time_elapsed = end_time - start_time
    print '(a,i0,a,g0,a,g0)', 'Device: computed ', num_timesteps, ' time steps in ', &
      time_elapsed, ' s. Time steps per second: ', num_timesteps/time_elapsed
    print '(a)', ''
  !$omp end target data
  rmse = 0.0_real64
  do i = 0, total_num_states - 1
    rmse = rmse + (states2(i)-states(i))*(states2(i)-states(i))
  end do
  print '(a,f12.6)', 'RMSE = ', sqrt(rmse / total_num_states)
end program rushlarsen
