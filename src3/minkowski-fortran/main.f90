program minkowski
  use iso_fortran_env,only:real32,real64,int64
  use iso_c_binding,only:c_int
  implicit none
  integer::matrix_size,m,n,k,repeats,i,j,order,iteration,kk,c0,c1,rate
  real(real32),allocatable::a(:),b(:),c_gpu(:),c_host(:)
  real(real32)::p,one_over_p,sumv
  character(len=32)::arg
  logical::ok
  interface
    subroutine c_srand(seed) bind(C,name='srand')
      import c_int
      integer(c_int),value::seed
    end subroutine c_srand
    function c_rand() bind(C,name='rand') result(v)
      import c_int
      integer(c_int)::v
    end function c_rand
  end interface

  if(command_argument_count()<1.or.command_argument_count()>2) then
    print '(a)','Usage: ./main <repeat>';stop 1
  end if
  call get_command_argument(1,arg);read(arg,*)repeats
  matrix_size=4096
  if(command_argument_count()==2)then
    call get_command_argument(2,arg);read(arg,*)matrix_size
  end if
  if(repeats<1.or.mod(matrix_size,8)/=0) error stop 'invalid arguments'
  m=matrix_size/8;n=matrix_size/4;k=matrix_size/2
  allocate(a(0:m*n-1),b(0:n*k-1),c_gpu(0:m*k-1),c_host(0:m*k-1))
  a=1.0_real32/real(n,real32)
  call c_srand(123_c_int)
  do i=0,n*k-1; b(i)=real(mod(c_rand(),256_c_int),real32); end do
  do j=0,k-1
    sumv=0.0_real32
    do i=0,n-1;sumv=sumv+b(i*k+j);end do
    do i=0,n-1;b(i*k+j)=b(i*k+j)/sumv;end do
  end do
  write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0,a)') &
    'Problem size: c(',m,',',k,') = a(',m,',',n,') * b(',n,',',k,')'

  !$omp target data map(to:a(0:m*n-1),b(0:n*k-1)) map(alloc:c_gpu(0:m*k-1))
  do order=1,4
    p=real(order,real32);one_over_p=1.0_real32/p
    print '(a,i0)','Minkowski distance with p = ',order
    call system_clock(c0,rate)
    do iteration=1,repeats
      !$omp target teams distribute parallel do collapse(2) thread_limit(256) private(sumv,kk)
      do i=0,m-1
        do j=0,k-1
          sumv=0.0_real32
          do kk=0,n-1
            sumv=sumv+abs(a(i*n+kk)-b(kk*k+j))**p
          end do
          c_gpu(i*k+j)=sumv**one_over_p
        end do
      end do
      !$omp end target teams distribute parallel do
    end do
    call system_clock(c1)
    write(*,'(a,f0.6,a)')'Average kernel execution time: ', &
      real(c1-c0,real64)/(real(rate,real64)*real(repeats,real64)),' (s)'
    !$omp target update from(c_gpu(0:m*k-1))

    c_host=0.0_real32
    do i=0,m-1
      do kk=0,n-1
        do j=0,k-1
          c_host(i*k+j)=c_host(i*k+j)+abs(a(i*n+kk)-b(kk*k+j))**p
        end do
      end do
    end do
    do i=0,m*k-1;c_host(i)=c_host(i)**one_over_p;end do
    ok=.true.
    do i=0,m*k-1
      if(abs(c_gpu(i)-c_host(i))>1.0e-5_real32)then
        write(*,'(a,i0,a,f0.8,a,f0.8)')'Fail - element ',i,', expected ',c_host(i),', found ',c_gpu(i)
        ok=.false.;exit
      end if
    end do
    if(ok)then;print '(a)','PASS';else;print '(a)','FAIL';error stop 2;end if
  end do
  !$omp end target data
end program minkowski
