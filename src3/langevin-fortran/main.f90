module langevin_kernels
  use iso_fortran_env,only:real32
  implicit none
contains
  subroutine k0(a,o,n);integer,intent(in)::n;real(real32),intent(in)::a(:);real(real32),intent(out)::o(:);integer::t
    !$omp target teams distribute parallel do thread_limit(256)
    do t=1,n;o(t)=cosh(a(t))/sinh(a(t))-1.0_real32/a(t);end do
    !$omp end target teams distribute parallel do
  end subroutine
  subroutine k1(a,o,n);integer,intent(in)::n;real(real32),intent(in)::a(:);real(real32),intent(out)::o(:);integer::t
    !$omp target teams distribute parallel do thread_limit(256)
    do t=1,n;o(t)=1.0_real32/tanh(a(t))-1.0_real32/a(t);end do
    !$omp end target teams distribute parallel do
  end subroutine
  subroutine k2(a,o,n);integer,intent(in)::n;real(real32),intent(in)::a(:);real(real32),intent(out)::o(:);integer::t;real(real32)::x,s,r
    !$omp target teams distribute parallel do thread_limit(256) private(x,s,r)
    do t=1,n;x=a(t);s=x*x;r=7.70960469e-8_real32;r=r*s-1.65101926e-6_real32;r=r*s+2.03457112e-5_real32;r=r*s-2.10521728e-4_real32;r=r*s+2.11580913e-3_real32;r=r*s-2.22220998e-2_real32;r=r*s+8.33333284e-2_real32;r=r*x+.25_real32*x;o(t)=r;end do
    !$omp end target teams distribute parallel do
  end subroutine
end module
program main
 use iso_fortran_env,only:real32,real64;use langevin_kernels;implicit none
 integer::n,reps,i,r,c0,c1,rate;character(len=32)::arg
 real(real32),allocatable::a(:),o(:),o0(:),o1(:),o2(:);real(real32)::e(3),x,x2,x4,x6
 if(command_argument_count()/=2)stop 1;call get_command_argument(1,arg);read(arg,*)n;call get_command_argument(2,arg);read(arg,*)reps
 allocate(a(n),o(n),o0(n),o1(n),o2(n));do i=1,n;a(i)=-1.8_real32+(i-1)*(1.79999_real32/n);end do
 !$omp target data map(to:a) map(from:o0,o1,o2)
 call system_clock(c0,rate);do r=1,reps;call k0(a,o0,n);end do;call system_clock(c1);write(*,'(a,f10.6,a)')'Average execution time of k0: ',real(c1-c0,real64)/(rate*reps),' (s)'
 call system_clock(c0);do r=1,reps;call k1(a,o1,n);end do;call system_clock(c1);write(*,'(a,f10.6,a)')'Average execution time of k1: ',real(c1-c0,real64)/(rate*reps),' (s)'
 call system_clock(c0);do r=1,reps;call k2(a,o2,n);end do;call system_clock(c1);write(*,'(a,f10.6,a)')'Average execution time of k2: ',real(c1-c0,real64)/(rate*reps),' (s)'
 !$omp end target data
 do i=1,n;x=a(i);x2=x*x;x4=x2*x2;x6=x4*x2;o(i)=x*(1.0_real32/3-1.0_real32/45*x2+2.0_real32/945*x4-1.0_real32/4725*x6);end do
 e=0;do i=1,n;e(1)=e(1)+(o(i)-o0(i))**2;e(2)=e(2)+(o(i)-o1(i))**2;e(3)=e(3)+(o(i)-o2(i))**2;end do
 print '(a)','Error statistics for the kernels:';print *,sqrt(e)
 print '(a)','PASS'
end program
