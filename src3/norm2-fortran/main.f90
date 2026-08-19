program norm2
 use iso_fortran_env,only:real32,real64;implicit none
 integer::repeat,n,nmax,i,j,c0,c1,rate;real(real32),allocatable::a(:),result(:);real(real64)::gold,sumv;character(len=32)::arg;logical::ok
 if(command_argument_count()<1.or.command_argument_count()>2)stop 1;call get_command_argument(1,arg);read(arg,*)repeat;repeat=max(1,repeat);nmax=512*1024*1024;if(command_argument_count()==2)then;call get_command_argument(2,arg);read(arg,*)nmax;end if
 allocate(result(repeat));ok=.true.;n=512*1024
 do while(n<=nmax)
  allocate(a(n));gold=0;do i=1,n;a(i)=modulo(i,7);gold=gold+a(i)*a(i);end do;gold=sqrt(gold)
  !$omp target data map(to:a)
  call system_clock(c0,rate);do j=1,repeat;sumv=0.0_real64
  !$omp target teams distribute parallel do thread_limit(256) map(tofrom:sumv) reduction(+:sumv)
  do i=1,n;sumv=sumv+a(i)*a(i);end do
  !$omp end target teams distribute parallel do
  result(j)=sqrt(sumv);end do;call system_clock(c1)
  !$omp end target data
  write(*,'(a,f8.2,a,f12.3,a)')'#elements = ',real(n,real32)/(1024*1024),' M: average omp nrm2 execution time = ',1.e6_real64*(c1-c0)/(rate*repeat),' (us)'
  do j=1,repeat;if(abs(real(gold,real32)-result(j))>1.e-3_real32)then;ok=.false.;end if;end do;deallocate(a);n=n*2
 end do
 if(ok)then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
