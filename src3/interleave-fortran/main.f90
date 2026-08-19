program interleave
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int32, int64, real64
  implicit none
  integer, parameter :: num_elements=4096, count=4096, members=16
  interface
    function c_rand() bind(C,name='rand') result(value)
      import c_int
      integer(c_int) :: value
    end function
  end interface
  integer :: argc, repeat, tid, field, i, k
  integer(int32), allocatable :: inter_src(:,:), inter_dst(:,:), non_src(:,:), non_dst(:,:)
  integer(int64) :: t0,t1,rate
  character(len=64) :: arg
  argc=command_argument_count()
  if(argc/=1) then; print '(a)', 'Usage: ./main <repeat>'; stop 1; end if
  call get_command_argument(1,arg); read(arg,*) repeat
  allocate(inter_src(0:members-1,0:num_elements-1),inter_dst(0:members-1,0:num_elements-1))
  allocate(non_src(0:num_elements-1,0:members-1),non_dst(0:num_elements-1,0:members-1))
  do tid=0,num_elements-1
    do field=0,members-1
      inter_src(field,tid)=mod(c_rand(),16_c_int); non_src(tid,field)=inter_src(field,tid)
      inter_dst(field,tid)=0_int32; non_dst(tid,field)=0_int32
    end do
  end do

!$omp target data map(to:non_src(0:num_elements-1,0:members-1)) map(tofrom:non_dst(0:num_elements-1,0:members-1))
  call system_clock(t0,rate)
  do i=1,repeat
!$omp target teams distribute parallel do thread_limit(256) private(field,k)
    do tid=0,num_elements-1
      do field=0,members-1
        do k=1,count
          non_dst(tid,field)=non_dst(tid,field)+non_src(tid,field)
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  call system_clock(t1)
  print '(a,f0.6,a)', 'Average kernel (non-interleaved) execution time ',real(t1-t0,real64)/real(rate,real64)/repeat,' (s)'
!$omp end target data

!$omp target data map(to:inter_src(0:members-1,0:num_elements-1)) map(tofrom:inter_dst(0:members-1,0:num_elements-1))
  call system_clock(t0,rate)
  do i=1,repeat
!$omp target teams distribute parallel do thread_limit(256) private(field,k)
    do tid=0,num_elements-1
      do field=0,members-1
        do k=1,count
          inter_dst(field,tid)=inter_dst(field,tid)+inter_src(field,tid)
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  call system_clock(t1)
  print '(a,f0.6,a)', 'Average kernel (interleaved) execution time ',real(t1-t0,real64)/real(rate,real64)/repeat,' (s)'
!$omp end target data
  if(all(transpose(inter_dst)==non_dst)) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
    stop 2
  end if
end program
