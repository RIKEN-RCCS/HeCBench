program hellinger
  use, intrinsic :: iso_fortran_env, only : int32, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib, only : omp_get_wtime
  implicit none
  integer(int32),parameter::m=768_int32,n=1536_int32,p=3072_int32
  integer(int32)::repeat,i,j,k,iteration,ios,printed
  real(real32),allocatable::a(:),b(:),c(:),host(:)
  real(real64)::start,elapsed
  real(real32)::column_sum,value
  logical::failed
  character(len=64)::arg
  interface
    subroutine c_srand(seed) bind(C,name='srand');import c_int;integer(c_int),value::seed;end subroutine
    function c_rand() bind(C,name='rand') result(value);import c_int;integer(c_int)::value;end function
  end interface
  if(command_argument_count()/=1)then;write(*,'(a)')'Usage: ./main <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*,iostat=ios)repeat;if(ios/=0.or.repeat<=0)error stop 'repeat must be positive'
  allocate(a(0:m*n-1),b(0:n*p-1),c(0:m*p-1),host(0:m*p-1))
  a=1.0_real32/real(n,real32);call c_srand(123_c_int)
  do i=0,n-1;do j=0,p-1;b(i*p+j)=real(mod(c_rand(),256_c_int),real32);end do;end do
  do j=0,p-1
    column_sum=0.0_real32;do i=1,n;column_sum=column_sum+b((i-1)*p+j);end do
    do i=0,n-1;b(i*p+j)=b(i*p+j)/column_sum;end do
  end do
  write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0,a)')'Problem size: c(',m,',',p,') = a(',m,',',n,') * b(',n,',',p,')'
!$omp target data map(to:a(0:m*n-1),b(0:n*p-1)) map(from:c(0:m*p-1))
  start=omp_get_wtime()
  do iteration=1,repeat
!$omp target teams distribute parallel do collapse(2) thread_limit(256) private(k,value)
    do i=0,m-1
      do j=0,p-1
        c(i*p+j)=0.0_real32
        do k=0,n-1;c(i*p+j)=c(i*p+j)+sqrt(a(i*n+k)*b(k*p+j));end do
        value=1.0_real32-c(i*p+j);c(i*p+j)=sqrt(merge(value,0.0_real32,value>=0.0_real32))
      end do
    end do
!$omp end target teams distribute parallel do
  end do
  elapsed=(omp_get_wtime()-start)/real(repeat,real64)
!$omp end target data
  write(*,'(a,f0.6,a)')'Average kernel execution time ',elapsed,' (s)'
  host=0.0_real32
  do i=0,m-1;do k=0,n-1;do j=0,p-1;host(i*p+j)=host(i*p+j)+sqrt(a(i*n+k)*b(k*p+j));end do;end do;end do
  do i=0,m*p-1;value=1.0_real32-host(i);host(i)=sqrt(merge(value,0.0_real32,value>=0.0_real32));end do
  failed=.false.;printed=0
  do i=0,m-1;do j=0,p-1;if(abs(c(i*p+j)-host(i*p+j))>=1.0e-5_real32)then
    write(*,'(a,i0,a,i0,a,f0.6,a,f0.6)')'Fail - The result is incorrect for element: [',i,', ',j,'], expected: ',host(i*p+j),', but found: ',c(i*p+j)
    failed=.true.;printed=printed+1;if(printed==5)exit
  end if;end do;if(printed==5)exit;end do
  if(failed)then;write(*,'(a)')'FAIL';else;write(*,'(a)')'PASS';end if
  deallocate(a,b,c,host)
end program hellinger
