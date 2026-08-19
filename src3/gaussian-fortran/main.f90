program gaussian
 use iso_fortran_env,only:real32,real64,int64
 implicit none
 integer::n,i,ios;logical::quiet,timing;character(256)::file,arg
 real(real32),allocatable::a(:),b(:),m(:),ar(:),br(:),mr(:),x(:),xr(:)
 integer(int64)::s,e,r
 quiet=.false.;timing=.false.;n=-1;file=''
 i=1;do while(i<=command_argument_count());call get_command_argument(i,arg);select case(trim(arg));case('-q');quiet=.true.;case('-t');timing=.true.;case('-s');i=i+1;call get_command_argument(i,arg);read(arg,*)n;write(*,'(a,i0,a)')'Create a square matrix (',n,' x internally';case('-f');i=i+1;call get_command_argument(i,file);end select;i=i+1;end do
 if(n<1)then;open(10,file=trim(file),status='old',iostat=ios);if(ios/=0)error stop 'cannot open matrix file';read(10,*)n;allocate(a(0:n*n-1),b(0:n-1));do i=0,n*n-1;read(10,*)a(i);end do;do i=0,n-1;read(10,*)b(i);end do;close(10)
 else;allocate(a(0:n*n-1),b(0:n-1));call initmat(a,n);b=1.0_real32;end if
 allocate(m(0:n*n-1),ar(0:n*n-1),br(0:n-1),mr(0:n*n-1),x(0:n-1),xr(0:n-1));m=0.;ar=a;br=b;mr=m;call forward_host(ar,br,mr,n);call backsub(ar,br,xr,n)
 call system_clock(s,r);call forward_gpu(a,b,m,n);call system_clock(e);if(timing)write(*,'(a,i0,a)')'Device offloading time ',int(real(e-s,real64)*1d6/real(r,real64)),' (us)'
 call backsub(a,b,x,n);if(.not.quiet)then;write(*,'(a)')'The solution is:';write(*,*)x;end if
 write(*,'(a)')'Checking the results..';if(all(abs(x-xr)<=1e-3_real32))then;write(*,'(a)')'PASS';else;write(*,'(a)')'FAIL';end if
contains
 subroutine initmat(z,q);real(real32),intent(out)::z(0:);integer,intent(in)::q;real(real32)::c(0:2*q-2);integer::ii,jj;c=0.;do ii=0,q-1;c(q-1+ii)=10.*exp(-.01_real32*ii);c(q-1-ii)=c(q-1+ii);end do;do ii=0,q-1;do jj=0,q-1;z(ii*q+jj)=c(q-1-ii+jj);end do;end do;end subroutine
 subroutine forward_host(z,v,mm,q);real(real32),intent(inout)::z(0:),v(0:),mm(0:);integer,intent(in)::q;integer::t,ii,xx,yy;do t=0,q-2;do ii=0,q-2-t;mm(q*(ii+t+1)+t)=z(q*(ii+t+1)+t)/z(q*t+t);end do;do xx=0,q-2-t;do yy=0,q-1-t;z(q*(xx+t+1)+yy+t)=z(q*(xx+t+1)+yy+t)-mm(q*(xx+t+1)+t)*z(q*t+yy+t);if(yy==0)v(xx+1+t)=v(xx+1+t)-mm(q*(xx+1+t)+yy+t)*v(t);end do;end do;end do;end subroutine
 subroutine forward_gpu(z,v,mm,q);real(real32),intent(inout)::z(0:),v(0:),mm(0:);integer,intent(in)::q;integer::t,ii,xx,yy
!$omp target data map(tofrom:z(0:q*q-1),v(0:q-1),mm(0:q*q-1))
 do t=0,q-2
!$omp target teams distribute parallel do thread_limit(256)
 do ii=0,q-2-t;mm(q*(ii+t+1)+t)=z(q*(ii+t+1)+t)/z(q*t+t);end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(2) thread_limit(256)
 do xx=0,q-2-t;do yy=0,q-1-t;z(q*(xx+t+1)+yy+t)=z(q*(xx+t+1)+yy+t)-mm(q*(xx+t+1)+t)*z(q*t+yy+t);if(yy==0)v(xx+1+t)=v(xx+1+t)-mm(q*(xx+1+t)+yy+t)*v(t);end do;end do
!$omp end target teams distribute parallel do
 end do
!$omp end target data
 end subroutine
 subroutine backsub(z,v,out,q);real(real32),intent(in)::z(0:),v(0:);real(real32),intent(out)::out(0:);integer,intent(in)::q;integer::ii,jj;do ii=0,q-1;out(q-ii-1)=v(q-ii-1);do jj=0,ii-1;out(q-ii-1)=out(q-ii-1)-z(q*(q-ii-1)+(q-jj-1))*out(q-jj-1);end do;out(q-ii-1)=out(q-ii-1)/z(q*(q-ii-1)+(q-ii-1));end do;end subroutine
end program
