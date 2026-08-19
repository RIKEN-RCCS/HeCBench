program glu
 use iso_fortran_env,only:int64,real32,real64
 implicit none
 integer::nd,ds,rep,i,d,split,m,n,j,k,ix,iy,input_dim
 integer(int64)::nelems,state,s,e,r
 real(real64)::us
 real(real32),allocatable::x(:),y(:),yr(:)
 character(64)::arg
 if(command_argument_count()/=3)then;write(*,'(a)')'Usage: ./main <number of dimensions> <size of each dimension> <repeat>';stop 1;end if
 call get_command_argument(1,arg);read(arg,*)nd;call get_command_argument(2,arg);read(arg,*)ds;call get_command_argument(3,arg);read(arg,*)rep
 nelems=int(ds,int64)**nd;allocate(x(0:nelems-1),y(0:nelems-1),yr(0:nelems-1));write(*,'(a)',advance='no')'Shape of input tensor: ( ';do i=1,nd;write(*,'(i0,1x)',advance='no')ds;end do;write(*,'(a)')')'
 state=123;do i=0,nelems-1;state=modulo(16807_int64*state,2147483647_int64);x(i)=real(-6d0+12d0*real(state,real64)/2147483647d0,real32);end do
!$omp target data map(to:x(0:nelems-1)) map(alloc:y(0:nelems-1))
 do input_dim=-1,3*(nd-1)-1
  if(input_dim==-1)then;split=nd-1;else;split=modulo(input_dim,nd);end if
  if(modulo(ds,2)/=0)then;write(*,'(a,i0,a)')'Split dimension ',ds,' should be divided by two. Skip';cycle;end if
  m=ds**split;n=ds**(nd-split-1);d=ds/2;call host_glu(m,d,n,x,yr);call system_clock(s,r);do i=1,rep
!$omp target teams distribute parallel do thread_limit(256) private(ix,iy)
   do j=0,m*d*n-1;ix=(j/(d*n))*2*d*n+modulo(j/n,d)*n+modulo(j,n);iy=j;y(iy)=x(ix)/(1._real32+exp(-x(ix+d*n)));end do
!$omp end target teams distribute parallel do
  end do;call system_clock(e);us=real(e-s,real64)*1d6/real(r,real64)/rep;write(*,'(a,i0,a,f0.6,a)')'Average execution time of GLU kernel (split dimension = ',split,'): ',us,' (us)'
!$omp target update from(y(0:nelems-1))
  if(all(abs(y(0:nelems/2-1)-yr(0:nelems/2-1))<=1e-3_real32))then;write(*,'(a)')'PASS';else;write(*,'(a)')'FAIL';end if
 end do
!$omp end target data
contains
 subroutine host_glu(mm,dd,nn,a,b);integer,intent(in)::mm,dd,nn;real(real32),intent(in)::a(0:);real(real32),intent(out)::b(0:);integer::ii,jj,kk,base;real(real32)::z
  do ii=0,mm-1;do jj=0,dd-1;do kk=0,nn-1;base=ii*2*dd*nn+jj*nn+kk;z=a(base+dd*nn);if(z>=0.)then;b(ii*dd*nn+jj*nn+kk)=a(base)/(1.+exp(-z));else;z=exp(z);b(ii*dd*nn+jj*nn+kk)=a(base)*z/(1.+z);end if;end do;end do;end do
 end subroutine
end program
