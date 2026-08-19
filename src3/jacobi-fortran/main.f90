program jacobi
  use iso_fortran_env,only:int64,real32,real64
  implicit none
  integer,parameter::n=2048,max_iters=10000
  integer::i,j,num_iters
  integer(int64)::t0,t1,rate,all0,all1
  real(real32)::error,tolerance,t,df
  real(real32),allocatable::f(:),fold(:)
  allocate(f(0:n*n-1),fold(0:n*n-1));tolerance=1.0e-5_real32;error=huge(1.0_real32)
  call system_clock(all0,rate)
  do j=0,n-1;do i=0,n-1
    if(i==0.or.i==n-1)then;f(i+j*n)=sin(real(j,real32)*2.0_real32*acos(-1.0_real32)/(n-1));else if(j==0.or.j==n-1)then;f(i+j*n)=sin(real(i,real32)*2.0_real32*acos(-1.0_real32)/(n-1));else;f(i+j*n)=0.0_real32;end if
  end do;end do
  fold=f;num_iters=0
!$omp target data map(to:f(0:n*n-1),fold(0:n*n-1))
  call system_clock(t0,rate)
  do while(error>tolerance.and.num_iters<max_iters)
    error=0.0_real32
!$omp target teams distribute parallel do collapse(2) reduction(+:error) num_teams(n*n/256) thread_limit(256) map(tofrom:error) private(t,df)
    do i=1,n-2
      do j=1,n-2
        t=0.25_real32*(fold(i-1+j*n)+fold(i+1+j*n)+fold(i+(j-1)*n)+fold(i+(j+1)*n));df=t-fold(i+j*n);f(i+j*n)=t;error=error+df*df
      end do
    end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(2) thread_limit(256)
    do j=0,n-1
      do i=0,n-1
        if(j>=1.and.j<=n-2.and.i>=1.and.i<=n-2)fold(i+j*n)=f(i+j*n)
      end do
    end do
!$omp end target teams distribute parallel do
    error=sqrt(error/real(n*n,real32));if(mod(num_iters,1000)==0)print '(a,i0,a,es12.5)','Error after iteration ',num_iters,' = ',error;num_iters=num_iters+1
  end do
  call system_clock(t1)
  print '(a,f0.6,a)','Average execution time per iteration: ',real(t1-t0,real64)/rate/num_iters,' (s)'
!$omp end target data
  if(error<=tolerance.and.num_iters<max_iters)then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
  call system_clock(all1);print '(a,f0.4,a)','Total elapsed time: ',real(all1-all0,real64)/rate,' seconds'
end program
