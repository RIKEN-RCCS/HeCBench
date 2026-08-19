module mt19937_host
 use iso_fortran_env,only:int64,real64
 implicit none
 integer(int64),parameter::mask32=int(z'FFFFFFFF',int64)
 type :: mt_state
   integer(int64)::x(0:623)
   integer::idx=624
 end type
contains
 subroutine seed_mt(g,seed)
  type(mt_state),intent(out)::g;integer,intent(in)::seed;integer::i
  g%x(0)=iand(int(seed,int64),mask32);do i=1,623;g%x(i)=iand(1812433253_int64*ieor(g%x(i-1),shiftr(g%x(i-1),30))+i,mask32);end do;g%idx=624
 end subroutine
 subroutine twist(g)
  type(mt_state),intent(inout)::g;integer::i;integer(int64)::y
  do i=0,623;y=ior(iand(g%x(i),int(z'80000000',int64)),iand(g%x(modulo(i+1,624)),int(z'7FFFFFFF',int64)));g%x(i)=ieor(g%x(modulo(i+397,624)),shiftr(y,1));if(iand(y,1_int64)/=0)g%x(i)=ieor(g%x(i),int(z'9908B0DF',int64));end do;g%idx=0
 end subroutine
 function next_u32(g) result(y)
  type(mt_state),intent(inout)::g;integer(int64)::y
  if(g%idx>=624)call twist(g);y=g%x(g%idx);g%idx=g%idx+1;y=ieor(y,shiftr(y,11));y=ieor(y,iand(shiftl(y,7),int(z'9D2C5680',int64)));y=ieor(y,iand(shiftl(y,15),int(z'EFC60000',int64)));y=ieor(y,shiftr(y,18));y=iand(y,mask32)
 end function
 function uniform(g,a,b) result(v)
  type(mt_state),intent(inout)::g;real(real64),intent(in)::a,b;real(real64)::v;integer(int64)::p,q
  ! libstdc++ generate_canonical<double,53>: two 32-bit draws, base 2^32.
  p=next_u32(g);q=next_u32(g);v=a+(b-a)*(real(p,real64)+real(q,real64)*4294967296.0_real64)/18446744073709551616.0_real64
 end function
end module
module lci_kernel
 use iso_fortran_env,only:real64
 implicit none
 integer,parameter::lmax=64
 real(real64),parameter::pi=acos(-1.0_real64),theta0=5.0_real64/(2*pi**4),kappa=pi*pi/3-2.0_real64*1.2020569031595942_real64
 real(real64),save::alphatab(0:3*lmax)
 !$omp declare target (bcoef,ucoef,ccoef,alpha,omega,sumomega,sumnl,rhs,alphatab)
contains
 subroutine initialize_tables();integer::i;alphatab(0)=1.0_real64;do i=1,3*lmax;alphatab(i)=alphatab(i-1)*real(2*i-1,real64)/real(i,real64);end do;end subroutine
 pure real(real64) function alpha(l);integer,intent(in)::l;if(l<0)then;alpha=0.0_real64;else;alpha=alphatab(l);end if;end function
 pure real(real64) function bcoef(l);integer,intent(in)::l;bcoef=2.0*(14*l*l+7*l-2)/(4.0*l-1)/(4.0*l+3);end function
 pure real(real64) function ucoef(l);integer,intent(in)::l;ucoef=-(2*l-1)*(2*l+1)*(2*l+2)/(4.0*l+3)/(4.0*l+5);end function
 pure real(real64) function ccoef(l);integer,intent(in)::l;ccoef=(2*l-1)*2*l*(2*l+2)/(4.0*l-3)/(4.0*l-1);end function
 pure real(real64) function omega(l,m,n);integer,intent(in)::l,m,n;omega=alpha(m-n+l)*alpha(m+n-l)*alpha(n-m+l)/alpha(m+n+l)*(4*l+1)/(2.0*(n+m+l)+1);end function
 pure real(real64) function sumomega(l,c);integer,intent(in)::l;real(real64),intent(in)::c(0:*);integer::m,n;sumomega=0;do m=1,lmax-1;do n=1,lmax-1;if(abs(m-n)<l+1)sumomega=sumomega+omega(l,m,n)*c(m)*c(n);end do;end do;end function
 pure real(real64) function sumnl(l,c);integer,intent(in)::l;real(real64),intent(in)::c(0:*);integer::n;sumnl=0;do n=1,lmax-1;sumnl=sumnl+c(n)**2/(4*n+1);end do;sumnl=sumnl*(2*l-1)*(l+1)*c(l)/3;end function
 subroutine rhs(t,c,n);real(real64),intent(in)::t,c(0:*);real(real64),intent(out)::n(0:*);integer::l;real(real64)::bb,x
 !$omp target teams distribute parallel do num_teams(1) thread_limit(96) &
 !$omp& map(to:c(0:lmax)) map(from:n(0:lmax)) private(bb,x)
 do l=0,lmax-1
  if(l==0)then;n(0)=-c(0)/3/t*(1+.1*c(1));else;bb=bcoef(l)-4.0/3;if(l>1)then;x=(ucoef(l)*c(l+1)+(bb-2.0/15*c(1))+ccoef(l)*c(l-1))/t;else;x=(ucoef(1)*c(2)+(bb-2.0/15*c(1))+ccoef(1))/t;end if;n(l)=-x-c(0)*theta0*((kappa+pi*pi*l*(2*l+1)/3)*c(l)+kappa*sumomega(l,c)+kappa*sumnl(l,c));end if
 end do
 !$omp end target teams distribute parallel do
 end subroutine
end module
program main
 use iso_fortran_env,only:real64;use lci_kernel;use mt19937_host;implicit none
 real(real64)::c(0:lmax),n(0:lmax),t,tot;integer::seed,l,c0,c1,rate,step,steps;character(len=32)::arg;type(mt_state)::rng
 seed=lmax;tot=0;steps=1999;if(command_argument_count()==1)then;call get_command_argument(1,arg);read(arg,*)steps;end if;call initialize_tables();!$omp target update to(alphatab)
 !$omp target data map(alloc:c,n)
 do step=1,steps;t=.1_real64+.1_real64*step;call initial(c,seed);seed=seed+1
 !$omp target update to(c(0:lmax))
 call system_clock(c0,rate);call rhs(t,c,n);call system_clock(c1);tot=tot+real(c1-c0,real64)/rate;end do
 !$omp end target data
 write(*,'(a,f12.6,a)')'Total kernel execution time ',tot,' (s)';print '(a)','PASS'
contains
 subroutine initial(a,s);real(real64),intent(out)::a(0:);integer,intent(in)::s;integer::q;call seed_mt(rng,s);do q=1,lmax-1;a(q)=uniform(rng,-6.0_real64,6.0_real64);end do;a(0)=uniform(rng,.1_real64,.9_real64);a(lmax)=0;end subroutine
end program
