program memcpy_benchmark
  use iso_fortran_env, only: int32, int64, real64
  implicit none
  integer, parameter :: num_size=16
  integer :: argc,repeat,i,j
  integer(int64) :: sizes(0:num_size-1),bytes,len,t0,t1,rate,h2d,d2h
  integer(int32),allocatable :: a(:)
  character(len=64)::arg
  argc=command_argument_count()
  if(argc/=1) then; print '(a)','Usage: ./main <repeat>'; stop 1; end if
  call get_command_argument(1,arg); read(arg,*)repeat
  if(repeat<1) stop 1
  do i=0,num_size-1; sizes(i)=shiftl(1_int64,i+6); end do
  do i=0,num_size-1
    bytes=sizes(i); len=bytes/int(storage_size(0_int32)/8,int64)
    allocate(a(0:len-1)); a=1_int32
!$omp target data map(alloc:a(0:len-1))
    do j=1,repeat
!$omp target update to(a(0:len-1))
    end do
    call system_clock(t0,rate)
    do j=1,repeat
!$omp target update to(a(0:len-1))
    end do
    call system_clock(t1); h2d=t1-t0
    print '(a,i0,a,f0.6,a)', 'Copy ',bytes,' bytes from host to device takes ',real(h2d,real64)*1.0e6_real64/real(rate,real64)/repeat,' us'
    do j=1,repeat
!$omp target update from(a(0:len-1))
    end do
    call system_clock(t0,rate)
    do j=1,repeat
!$omp target update from(a(0:len-1))
    end do
    call system_clock(t1); d2h=t1-t0
    print '(a,i0,a,f0.6,a)', 'Copy ',bytes,' bytes from device to host takes ',real(d2h,real64)*1.0e6_real64/real(rate,real64)/repeat,' us'
    print '(a,f0.9)', 'Timing gap in nanoseconds per byte: ',abs(real(h2d-d2h,real64))*1.0e9_real64/real(rate,real64)/repeat/bytes
!$omp end target data
    print *; print *
    deallocate(a)
  end do
end program
