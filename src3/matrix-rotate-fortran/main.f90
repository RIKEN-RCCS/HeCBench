module matrix_rotation
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
contains
  subroutine rotate_parallel(matrix,n,repeats)
    integer, intent(in) :: n,repeats
    real(real32), intent(inout) :: matrix(0:n*n-1)
    integer :: iteration
    integer(int64) :: t0,t1,rate
    integer :: layer, first, last, i, offset
    real(real32) :: top
!$omp target data map(tofrom:matrix(0:n*n-1))
    call system_clock(t0,rate)
    do iteration=1,repeats
!$omp target teams distribute parallel do thread_limit(256) private(first,last,i,offset,top)
      do layer=0,n/2-1
        first=layer; last=n-1-layer
        do i=first,last-1
          offset=i-first
          top=matrix(first*n+i)
          matrix(first*n+i)=matrix((last-offset)*n+first)
          matrix((last-offset)*n+first)=matrix(last*n+(last-offset))
          matrix(last*n+(last-offset))=matrix(i*n+last)
          matrix(i*n+last)=top
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(t1)
    print '(a,f0.6,a)', 'Average kernel execution time: ',real(t1-t0,real64)/real(rate,real64)/repeats,' (s)'
!$omp end target data
  end subroutine
  subroutine rotate_serial(matrix,n)
    integer, intent(in)::n
    real(real32), intent(inout)::matrix(0:n*n-1)
    integer::layer,first,last,i,offset
    real(real32)::top
    do layer=0,n/2-1
      first=layer; last=n-1-layer
      do i=first,last-1
        offset=i-first; top=matrix(first*n+i)
        matrix(first*n+i)=matrix((last-offset)*n+first)
        matrix((last-offset)*n+first)=matrix(last*n+(last-offset))
        matrix(last*n+(last-offset))=matrix(i*n+last)
        matrix(i*n+last)=top
      end do
    end do
  end subroutine
end module

program matrix_rotate
  use iso_fortran_env, only: real32
  use matrix_rotation
  implicit none
  integer::argc,n,repeats,i,j,iteration
  real(real32),allocatable::serial_res(:),parallel_res(:)
  character(len=64)::arg
  argc=command_argument_count()
  if(argc/=2) then; print '(a)','Usage: ./main <matrix size> <repeat>'; stop 1; end if
  call get_command_argument(1,arg);read(arg,*)n
  call get_command_argument(2,arg);read(arg,*)repeats
  if(n<1 .or. repeats<1) stop 1
  allocate(serial_res(0:n*n-1),parallel_res(0:n*n-1))
  do i=0,n-1
    do j=0,n-1
      serial_res(i*n+j)=real(i*n+j,real32); parallel_res(i*n+j)=serial_res(i*n+j)
    end do
  end do
  do iteration=1,repeats; call rotate_serial(serial_res,n); end do
  call rotate_parallel(parallel_res,n,repeats)
  if(all(serial_res==parallel_res)) then; print '(a)','PASS'; else; print '(a)','FAIL'; stop 2; end if
end program
