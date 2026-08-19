program main
  use gsimulation_mod
  implicit none
  integer :: n, nsteps, ios
  character(len=64) :: arg
  n = 16000
  nsteps = 10
  if (command_argument_count() > 0) then
    call get_command_argument(1,arg); read(arg,*,iostat=ios) n
    if (command_argument_count() == 2) then
      call get_command_argument(2,arg); read(arg,*,iostat=ios) nsteps
    end if
  end if
  call start_simulation(n, nsteps)
end program
