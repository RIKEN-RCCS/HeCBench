program dp
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use omp_lib, only : omp_get_wtime
  implicit none
  integer(int64) :: elements
  integer(int32) :: repeats,ios
  character(len=64) :: argument
  if(command_argument_count()/=2) then
    write(*,'(a)') 'Usage: ./main <number of elements> <repeat>';stop 1
  end if
  call get_command_argument(1,argument);read(argument,*,iostat=ios)elements
  if(ios/=0 .or. elements<=0) error stop 'number of elements must be positive'
  call get_command_argument(2,argument);read(argument,*,iostat=ios)repeats
  if(ios/=0 .or. repeats<=0) error stop 'repeat must be positive'
  call dot_float(elements,repeats)
  call dot_double(elements,repeats)
contains
  subroutine dot_float(number,iterations)
    integer(int64),intent(in)::number
    integer(int32),intent(in)::iterations
    integer(int64)::global_size,gid,offset,index
    integer(int32)::iteration
    real(real32),allocatable::source_a(:),source_b(:)
    real(real32)::destination
    real(real64)::start,elapsed
    global_size=((number+255_int64)/256_int64)*256_int64
    write(*,'(a,i0)') 'Global Work Size 		= ',global_size
    write(*,'(a,i0)') 'Local Work Size 		= ',256
    allocate(source_a(0:global_size-1),source_b(0:global_size-1))
    do index=0,number-1
      if(index<number/2_int64)then;source_a(index)=-1.0_real32;else;source_a(index)=1.0_real32;end if
      source_b(index)=-1.0_real32
    end do
    do index=number,global_size-1;source_a(index)=0.0_real32;source_b(index)=0.0_real32;end do
!$omp target data map(to:source_a(0:global_size-1),source_b(0:global_size-1))
    start=omp_get_wtime()
    do iteration=1,iterations
      destination=0.0_real32
!$omp target teams distribute parallel do map(tofrom:destination) reduction(+:destination) thread_limit(256)
      do gid=0,global_size/4_int64-1
        offset=gid*4_int64
        destination=destination+source_a(offset)*source_b(offset)+source_a(offset+1)*source_b(offset+1) &
          +source_a(offset+2)*source_b(offset+2)+source_a(offset+3)*source_b(offset+3)
      end do
!$omp end target teams distribute parallel do
    end do
    elapsed=(omp_get_wtime()-start)*1.0e3_real64/real(iterations,real64)
!$omp end target data
    write(*,'(a,f0.6,a)') 'Average kernel execution time ',elapsed,' (ms)'
    if(destination==0.0_real32)then;write(*,'(a)') 'PASS';else;write(*,'(a)') 'FAIL';end if
    write(*,'(a)') ''
    deallocate(source_a,source_b)
  end subroutine dot_float
  subroutine dot_double(number,iterations)
    integer(int64),intent(in)::number
    integer(int32),intent(in)::iterations
    integer(int64)::global_size,gid,offset,index
    integer(int32)::iteration
    real(real64),allocatable::source_a(:),source_b(:)
    real(real64)::destination,start,elapsed
    global_size=((number+255_int64)/256_int64)*256_int64
    write(*,'(a,i0)') 'Global Work Size 		= ',global_size
    write(*,'(a,i0)') 'Local Work Size 		= ',256
    allocate(source_a(0:global_size-1),source_b(0:global_size-1))
    do index=0,number-1
      if(index<number/2_int64)then;source_a(index)=-1.0_real64;else;source_a(index)=1.0_real64;end if
      source_b(index)=-1.0_real64
    end do
    do index=number,global_size-1;source_a(index)=0.0_real64;source_b(index)=0.0_real64;end do
!$omp target data map(to:source_a(0:global_size-1),source_b(0:global_size-1))
    start=omp_get_wtime()
    do iteration=1,iterations
      destination=0.0_real64
!$omp target teams distribute parallel do map(tofrom:destination) reduction(+:destination) thread_limit(256)
      do gid=0,global_size/4_int64-1
        offset=gid*4_int64
        destination=destination+source_a(offset)*source_b(offset)+source_a(offset+1)*source_b(offset+1) &
          +source_a(offset+2)*source_b(offset+2)+source_a(offset+3)*source_b(offset+3)
      end do
!$omp end target teams distribute parallel do
    end do
    elapsed=(omp_get_wtime()-start)*1.0e3_real64/real(iterations,real64)
!$omp end target data
    write(*,'(a,f0.6,a)') 'Average kernel execution time ',elapsed,' (ms)'
    if(destination==0.0_real64)then;write(*,'(a)') 'PASS';else;write(*,'(a)') 'FAIL';end if
    write(*,'(a)') ''
    deallocate(source_a,source_b)
  end subroutine dot_double
end program dp
