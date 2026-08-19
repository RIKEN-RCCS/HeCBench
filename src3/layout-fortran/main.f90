program layout
  use iso_fortran_env,only:int32,int64,real64
  implicit none
  integer,parameter::tree_number=4096,tree_size=4096,group_size=256
  integer::iterations,i,j,n,c0,c1,rate
  integer(int64)::elapsed_ticks
  integer(int32),allocatable::data(:),output(:),reference(:)
  integer(int32)::res
  character(len=32)::arg

  if(command_argument_count()/=1) then
    print '(a)','Usage: ./main <repeat>';stop 1
  end if
  call get_command_argument(1,arg);read(arg,*)iterations
  if(iterations<1) error stop 'Iterations cannot be 0 or negative.'
  allocate(data(0:tree_number*tree_size-1),output(0:tree_number-1),reference(0:tree_number-1))

  reference=0_int32
  do i=0,tree_number-1
    do j=0,tree_size-1
      reference(i)=reference(i)+int(i*tree_size+j,int32)
    end do
  end do

  !$omp target data map(alloc:data(0:tree_number*tree_size-1),output(0:tree_number-1))
  do i=0,tree_number-1
    do j=0,tree_size-1
      data(j+i*tree_size)=int(j+i*tree_size,int32)
    end do
  end do
  !$omp target update to(data(0:tree_number*tree_size-1))
  call system_clock(c0,rate)
  do n=1,iterations
    !$omp target teams distribute parallel do thread_limit(group_size) private(j,res)
    do i=0,tree_number-1
      res=0_int32
      do j=0,tree_size-1
        res=res+data(j+i*tree_size)
      end do
      output(i)=res
    end do
    !$omp end target teams distribute parallel do
  end do
  call system_clock(c1)
  elapsed_ticks=int(c1-c0,int64)
  write(*,'(a,f0.3,a)')'Average kernel execution time (AoS): ', &
    1.0e6_real64*real(elapsed_ticks,real64)/(real(rate,real64)*real(iterations,real64)),' (us)'
  !$omp target update from(output(0:tree_number-1))
  if(all(output==reference))then
    print '(a)','PASS'
  else
    print '(a)','FAIL';error stop 2
  end if

  do i=0,tree_number-1
    do j=0,tree_size-1
      data(i+j*tree_number)=int(j+i*tree_size,int32)
    end do
  end do
  !$omp target update to(data(0:tree_number*tree_size-1))
  call system_clock(c0,rate)
  do n=1,iterations
    !$omp target teams distribute parallel do thread_limit(group_size) private(j,res)
    do i=0,tree_number-1
      res=0_int32
      do j=0,tree_size-1
        res=res+data(i+j*tree_number)
      end do
      output(i)=res
    end do
    !$omp end target teams distribute parallel do
  end do
  call system_clock(c1)
  elapsed_ticks=int(c1-c0,int64)
  write(*,'(a,f0.3,a)')'Average kernel execution time (SoA): ', &
    1.0e6_real64*real(elapsed_ticks,real64)/(real(rate,real64)*real(iterations,real64)),' (us)'
  !$omp target update from(output(0:tree_number-1))
  if(all(output==reference))then
    print '(a)','PASS'
  else
    print '(a)','FAIL';error stop 2
  end if
  !$omp end target data
end program layout
