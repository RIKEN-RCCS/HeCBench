module projectile_kernel
 use iso_fortran_env,only:real32
 implicit none
contains
 subroutine gpu(angle,vel,ran,tim,height,n)
  integer,intent(in)::n;real(real32),intent(in)::angle(:),vel(:);real(real32),intent(out)::ran(:),tim(:),height(:);integer::i;real(real32)::s,c
  !$omp target teams distribute parallel do thread_limit(256) private(s,c)
  do i=1,n;s=sin(angle(i)*acos(-1.0_real32)/180.0_real32);c=cos(angle(i)*acos(-1.0_real32)/180.0_real32);tim(i)=abs(2*vel(i)*s)/9.81_real32;ran(i)=abs(vel(i)*tim(i)*c);height(i)=vel(i)*vel(i)*s*s/2.0_real32*9.81_real32;end do
  !$omp end target teams distribute parallel do
 end subroutine
end module
program projectile
 use iso_fortran_env,only:real32,real64;use projectile_kernel;implicit none
 integer::reps,n,i,r,c0,c1,rate;real(real32),allocatable::angle(:),vel(:),ran(:),tim(:),height(:),rr(:),rt(:),rh(:);character(len=32)::arg;logical::ok
 if(command_argument_count()<1.or.command_argument_count()>2)stop 1;call get_command_argument(1,arg);read(arg,*)reps;n=10000000;if(command_argument_count()==2)then;call get_command_argument(2,arg);read(arg,*)n;end if
 allocate(angle(n),vel(n),ran(n),tim(n),height(n),rr(n),rt(n),rh(n));call srand(2);do i=1,n;angle(i)=modulo(rand(),90)+10;vel(i)=modulo(rand(),400)+10;end do
 !$omp target data map(to:angle,vel) map(from:ran,tim,height)
 call system_clock(c0,rate);do r=1,reps;call gpu(angle,vel,ran,tim,height,n);end do;call system_clock(c1)
 !$omp end target data
 call cpu();ok=all(abs(ran-rr)<=1 .and. abs(tim-rt)<=1 .and. abs(height-rh)<=1);write(*,'(a,f10.6,a)')'Average kernel execution time: ',real(c1-c0,real64)/(rate*reps),' (s)';if(ok)then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
contains
 subroutine cpu();integer::q;real(real32)::s,c;do q=1,n;s=sin(angle(q)*acos(-1.0_real32)/180);c=cos(angle(q)*acos(-1.0_real32)/180);rt(q)=abs(2*vel(q)*s)/9.81;rr(q)=abs(vel(q)*rt(q)*c);rh(q)=vel(q)*vel(q)*s*s/2*9.81;end do;end subroutine
 subroutine srand(seed);use iso_c_binding,only:c_int;integer,intent(in)::seed;interface;subroutine cs(x)bind(C,name='srand');import c_int;integer(c_int),value::x;end subroutine;end interface;call cs(seed);end subroutine
 integer function rand();use iso_c_binding,only:c_int;interface;function cr()bind(C,name='rand');import c_int;integer(c_int)::cr;end function;end interface;rand=cr();end function
end program
