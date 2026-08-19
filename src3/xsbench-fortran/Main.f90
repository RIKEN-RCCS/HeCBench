program XSBench
  use iso_fortran_env, only: real64, int64
  use xsbench_simulation
  use xsbench_gridinit
  implicit none
  type(inputs_t) :: in
  type(simulation_data_t) :: sd
  integer(int64) :: verification
  real(real64) :: kernel_time, runtime
  call read_cli(in)
  call print_inputs(in)
  call grid_init_do_not_profile(in, sd)
  runtime = wall_seconds()
  verification = run_event_based_simulation(in, sd, kernel_time)
  runtime = wall_seconds() - runtime
  print '(a,i0)', 'Verification checksum: ', verification
  print '(a,f12.6,a)', 'Runtime: ', runtime, ' seconds'
  print '(a,f12.6,a)', 'Kernel time: ', kernel_time, ' seconds'
contains
  subroutine read_cli(in)
    type(inputs_t), intent(inout) :: in
    integer :: i, n, stat
    character(len=128) :: arg, nxt
    n = command_argument_count()
    i = 1
    do while (i <= n)
      call get_command_argument(i, arg)
      if (trim(arg) == '-s' .and. i < n) then
        call get_command_argument(i+1, nxt)
        if (trim(nxt) == 'large') then
          in%n_isotopes = 355_int64
          in%n_gridpoints = 11303_int64
          in%lookups = 150000
        end if
        i = i + 1
      else if (trim(arg) == '-m' .and. i < n) then
        i = i + 1
      else if (trim(arg) == '-r' .and. i < n) then
        call get_command_argument(i+1, nxt)
        read(nxt, *, iostat=stat) in%kernel_repeat
        if (stat /= 0) in%kernel_repeat = 10
        i = i + 1
      end if
      i = i + 1
    end do
  end subroutine read_cli

  subroutine print_inputs(in)
    type(inputs_t), intent(in) :: in
    print '(a)', '================================================================================'
    print '(a)', '                                  XSBench'
    print '(a)', '================================================================================'
    print '(a,i0)', 'Number of isotopes: ', in%n_isotopes
    print '(a,i0)', 'Gridpoints per isotope: ', in%n_gridpoints
    print '(a,i0)', 'Lookups: ', in%lookups
    print '(a,i0)', 'Kernel repetitions: ', in%kernel_repeat
  end subroutine print_inputs
end program XSBench
