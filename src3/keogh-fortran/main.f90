program keogh
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real32, real64, int64
  implicit none
  integer :: m,n,repeat,count,grids,blocks,idx,i,r,clk0,clk1,rate
  real(real32),allocatable :: subject(:),lower_bound(:),upper_bound(:),lb(:),lb_h(:),avgs(:),stds(:)
  real(real32) :: residues,avg,stddev,value,lower_diff,upper_diff
  character(len=32)::arg
  logical::ok
  interface
    subroutine c_srand(x) bind(C,name='srand')
      import c_int
      integer(c_int),value::x
    end subroutine c_srand
    function c_rand() bind(C,name='rand') result(v)
      import c_int
      integer(c_int)::v
    end function c_rand
  end interface

  if(command_argument_count()/=3) then
    print '(a)','Usage: ./main <query length> <subject length> <repeat>'; stop 1
  end if
  call get_command_argument(1,arg);read(arg,*)m
  call get_command_argument(2,arg);read(arg,*)n
  call get_command_argument(3,arg);read(arg,*)repeat
  if(m<=0 .or. n<m .or. repeat<=0) error stop 'invalid arguments'
  count=n-m+1; blocks=256; grids=(count+blocks-1)/blocks
  print '(a,i0)','Query length = ',m
  print '(a,i0)','Subject length = ',n

  allocate(subject(0:n-1),lower_bound(0:n-1),upper_bound(0:n-1))
  allocate(lb(0:count-1),lb_h(0:count-1),avgs(0:count-1),stds(0:count-1))
  call c_srand(123_c_int)
  do i=0,n-1; subject(i)=real(c_rand(),real32)/real(huge(0_c_int),real32); end do
  do i=0,count-1; avgs(i)=real(c_rand(),real32)/real(huge(0_c_int),real32); end do
  do i=0,count-1; stds(i)=real(c_rand(),real32)/real(huge(0_c_int),real32); end do
  do i=0,m-1; upper_bound(i)=real(c_rand(),real32)/real(huge(0_c_int),real32); end do
  do i=0,m-1; lower_bound(i)=real(c_rand(),real32)/real(huge(0_c_int),real32); end do

  !$omp target data map(to:subject(0:n-1),avgs(0:count-1),stds(0:count-1), &
  !$omp& lower_bound(0:n-1),upper_bound(0:n-1)) map(from:lb(0:count-1))
  call system_clock(clk0,rate)
  do r=1,repeat
    !$omp target teams distribute num_teams(grids) thread_limit(blocks) &
    !$omp& private(residues,avg,stddev,i,value,lower_diff,upper_diff)
    do idx=0,count-1
      residues=0.0_real32
      avg=avgs(idx)
      stddev=stds(idx)
      !$omp parallel do reduction(+:residues) private(value,lower_diff,upper_diff)
      do i=0,m-1
        value=(subject(idx+i)-avg)/stddev
        lower_diff=value-lower_bound(i)
        upper_diff=value-upper_bound(i)
        residues=residues+upper_diff*upper_diff*merge(1.0_real32,0.0_real32,upper_diff>0.0_real32) &
          +lower_diff*lower_diff*merge(1.0_real32,0.0_real32,lower_diff<0.0_real32)
      end do
      !$omp end parallel do
      lb(idx)=residues
    end do
    !$omp end target teams distribute
  end do
  call system_clock(clk1)
  !$omp end target data
  write(*,'(a,f12.6,a)')'Average kernel execution time: ', &
    real(clk1-clk0,real64)/real(int(rate,int64)*int(repeat,int64),real64),' (s)'

  do idx=0,count-1
    lb_h(idx)=0.0_real32
    avg=avgs(idx); stddev=stds(idx)
    do i=0,m-1
      value=(subject(idx+i)-avg)/stddev
      lower_diff=value-lower_bound(i)
      upper_diff=value-upper_bound(i)
      lb_h(idx)=lb_h(idx)+upper_diff*upper_diff*merge(1.0_real32,0.0_real32,upper_diff>0.0_real32) &
        +lower_diff*lower_diff*merge(1.0_real32,0.0_real32,lower_diff<0.0_real32)
    end do
  end do
  ok=.true.
  do i=0,count-1
    if(abs(lb(i)-lb_h(i))>1.0e-2_real32) then
      write(*,'(i0,1x,f0.6,1x,f0.6)')i,lb(i),lb_h(i)
      ok=.false.;exit
    end if
  end do
  if(ok)then
    print '(a)','PASS'
  else
    print '(a)','FAIL';error stop 2
  end if
end program keogh
