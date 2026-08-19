module contract_mod
 use iso_fortran_env,only:real32,real64
 use omp_lib,only:omp_get_wtime
 implicit none
 integer,parameter::nc=18
contains
 subroutine kernel4(tensor,adj,value,out,nv,ncchan)
  real(real32),intent(in)::tensor(0:),adj(0:);real(real32),intent(out)::value(0:);integer,intent(in)::out,nv,ncchan
  integer::tid,channels,b0,a0,c0,ystep,y,x,f,case_id,a,b,c,d,e;real(real32)::q,sum
  !$omp target teams distribute parallel do thread_limit(256) private(channels,b0,a0,c0,ystep,y,x,f,case_id,a,b,c,d,e,q,sum)
  do tid=0,out-1
   channels=ncchan;b0=nv*channels;a0=nv*b0;c0=channels;ystep=ncchan*nc;f=mod(mod(tid,ystep),ncchan);case_id=mod(tid,ystep)/ncchan+1;y=mod(tid/ystep,nv);x=(tid/ystep)/nv;sum=0.0_real32
   include 'contract_cases.inc'
   value(tid)=sum
  end do
  !$omp end target teams distribute parallel do
 end subroutine
 subroutine kernel8(tensor,adj,value,out,nv,ncchan)
  real(real64),intent(in)::tensor(0:),adj(0:);real(real64),intent(out)::value(0:);integer,intent(in)::out,nv,ncchan
  integer::tid,channels,b0,a0,c0,ystep,y,x,f,case_id,a,b,c,d,e;real(real64)::q,sum
  !$omp target teams distribute parallel do thread_limit(256) private(channels,b0,a0,c0,ystep,y,x,f,case_id,a,b,c,d,e,q,sum)
  do tid=0,out-1
   channels=ncchan;b0=nv*channels;a0=nv*b0;c0=channels;ystep=ncchan*nc;f=mod(mod(tid,ystep),ncchan);case_id=mod(tid,ystep)/ncchan+1;y=mod(tid/ystep,nv);x=(tid/ystep)/nv;sum=0.0_real64
   ! The included source uses C's variable name c; alias it here through c.
   include 'contract_cases.inc'
   value(tid)=sum
  end do
  !$omp end target teams distribute parallel do
 end subroutine
 subroutine run4(nv,rep)
  integer,intent(in)::nv,rep;integer::ts,as,os,i;real(real32),allocatable::t(:),a(:),v(:);real(real64)::s,z,check
  ts=nv*nv*nv*nc;as=nv*nv;os=nv*nv*nc*nc;allocate(t(0:ts-1),a(0:as-1),v(0:os-1));t=1.0_real32;a=1.0_real32
  !$omp target data map(to:t(0:ts-1),a(0:as-1)) map(from:v(0:os-1))
  s=omp_get_wtime();do i=1,rep;call kernel4(t,a,v,os,nv,nc);end do;z=omp_get_wtime();print '(a,f0.6,a)','Average kernel execution time ',(z-s)/rep,' (s)'
  !$omp end target data
  check=sum(real(v,real64));print '(a,f0.6,a,f0.6,a,f0.6)','Checksum: ',check,' min:',real(minval(v),real64),' max:',real(maxval(v),real64);deallocate(t,a,v)
 end subroutine
 subroutine run8(nv,rep)
  integer,intent(in)::nv,rep;integer::ts,as,os,i;real(real64),allocatable::t(:),a(:),v(:);real(real64)::s,z,check
  ts=nv*nv*nv*nc;as=nv*nv;os=nv*nv*nc*nc;allocate(t(0:ts-1),a(0:as-1),v(0:os-1));t=1.0_real64;a=1.0_real64
  !$omp target data map(to:t(0:ts-1),a(0:as-1)) map(from:v(0:os-1))
  s=omp_get_wtime();do i=1,rep;call kernel8(t,a,v,os,nv,nc);end do;z=omp_get_wtime();print '(a,f0.6,a)','Average kernel execution time ',(z-s)/rep,' (s)'
  !$omp end target data
  check=sum(v);print '(a,f0.6,a,f0.6,a,f0.6)','Checksum: ',check,' min:',minval(v),' max:',maxval(v);deallocate(t,a,v)
 end subroutine
end module
program main
 use contract_mod
 implicit none
 integer::argc,nv,rep,ios;character(len=64)::arg
 argc=command_argument_count();if(argc/=2)then;call get_command_argument(0,arg);print '(a,a,a)','Usage: ',trim(arg),' <dimension> <repeat>';error stop 1;end if
 call get_command_argument(1,arg);read(arg,*,iostat=ios)nv;if(ios/=0)error stop 1;call get_command_argument(2,arg);read(arg,*,iostat=ios)rep;if(ios/=0)error stop 1
 call run4(nv,rep);call run8(nv,rep)
end program
