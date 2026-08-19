module floyd_reference
  use iso_fortran_env, only: int32, int64
  implicit none
contains
  subroutine floyd_cpu(distance, path, nodes)
    integer(int64), intent(in) :: nodes
    integer(int32), intent(inout) :: distance(0:nodes*nodes-1), path(0:nodes*nodes-1)
    integer(int64) :: k, y, x, yx, indirect
    integer(int32) :: dyx, dyk, dkx
    do k=0,nodes-1
      do y=0,nodes-1
        yx=y*nodes
        do x=0,nodes-1
          dyx=distance(yx+x); dyk=distance(yx+k); dkx=distance(k*nodes+x)
          indirect=int(dyk,int64)+int(dkx,int64)
          if (indirect < int(dyx,int64)) then
            distance(yx+x)=int(indirect,int32); path(yx+x)=int(k,int32)
          end if
        end do
      end do
    end do
  end subroutine
end module

program floydwarshall
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int32, int64, real64
  use floyd_reference
  implicit none
  interface
    subroutine c_srand(seed) bind(C,name='srand')
      import c_int
      integer(c_int), value :: seed
    end subroutine
    function c_rand() bind(C,name='rand') result(value)
      import c_int
      integer(c_int) :: value
    end function
  end interface
  integer :: argc, ios, block_size, iterations, i
  integer(int64) :: nodes, matrix_size, k, y, x, n, t0, t1, rate
  integer(int32), allocatable :: distance(:), path(:), verify_distance(:), verify_path(:)
  integer(int32) :: dyx, dyk, dkx
  integer(int64) :: indirect
  real(real64) :: total_time
  character(len=64) :: arg

  argc=command_argument_count()
  if(argc/=3) then
    print '(a)', 'Usage: ./main <number of nodes> <iterations> <block size>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) nodes
  call get_command_argument(2,arg); read(arg,*,iostat=ios) iterations
  call get_command_argument(3,arg); read(arg,*,iostat=ios) block_size
  if (nodes <= 0 .or. iterations <= 0 .or. block_size <= 0) stop 1
  if (mod(nodes,int(block_size,int64))/=0) nodes=(nodes/int(block_size,int64)+1_int64)*block_size
  matrix_size=nodes*nodes
  allocate(distance(0:matrix_size-1),path(0:matrix_size-1),verify_distance(0:matrix_size-1),verify_path(0:matrix_size-1))
  call c_srand(2_c_int)
  do y=0,nodes-1
    do x=0,nodes-1
      distance(y*nodes+x)=mod(c_rand(),201_c_int)
    end do
    distance(y*nodes+y)=0_int32
  end do
  do y=0,nodes-1
    do x=0,y-1
      path(y*nodes+x)=int(y,int32); path(x*nodes+y)=int(x,int32)
    end do
    path(y*nodes+y)=int(y,int32)
  end do
  verify_distance=distance; verify_path=path

!$omp target data map(alloc:distance(0:matrix_size-1),path(0:matrix_size-1))
  total_time=0.0_real64
  do n=1,iterations
!$omp target update to(distance(0:matrix_size-1))
    call system_clock(t0,rate)
    do k=0,nodes-1
!$omp target teams distribute parallel do collapse(2) thread_limit(block_size*block_size) nowait private(dyx,dyk,dkx,indirect)
      do y=0,nodes-1
        do x=0,nodes-1
          dyx=distance(y*nodes+x); dyk=distance(y*nodes+k); dkx=distance(k*nodes+x)
          indirect=int(dyk,int64)+int(dkx,int64)
          if(indirect < int(dyx,int64)) then
            distance(y*nodes+x)=int(indirect,int32)
            path(y*nodes+x)=int(k,int32)
          end if
        end do
      end do
!$omp end target teams distribute parallel do
    end do
!$omp taskwait
    call system_clock(t1)
    total_time=total_time+real(t1-t0,real64)/real(rate,real64)
  end do
  print '(a,f0.6,a)', 'Average kernel execution time ', total_time/iterations, ' (s)'
!$omp target update from(distance(0:matrix_size-1))
!$omp end target data

  call floyd_cpu(verify_distance,verify_path,nodes)
  if(all(distance==verify_distance)) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
end program
