program easywave
  use easywave_solver, only: run_easywave
  implicit none
  character(len=1024) :: grid, source, arg
  integer :: n, i, time_minutes, ierr
  grid=''; source=''; time_minutes=-1
  n=command_argument_count(); i=1
  do while(i<=n)
    call get_command_argument(i,arg)
    select case(trim(arg))
    case('-grid'); i=i+1; call get_command_argument(i,grid)
    case('-source'); i=i+1; call get_command_argument(i,source)
    case('-time'); i=i+1; call get_command_argument(i,arg); read(arg,*) time_minutes
    end select
    i=i+1
  end do
  if(len_trim(grid)==0 .or. len_trim(source)==0 .or. time_minutes<0) then
    write(*,'(A)') 'Usage: ./main -grid <bathymetry.grd> -source <faults.flt> -time <minutes>'
    error stop 1
  end if
  call run_easywave(trim(grid),trim(source),time_minutes,ierr)
  if(ierr/=0) error stop ierr
end program easywave
