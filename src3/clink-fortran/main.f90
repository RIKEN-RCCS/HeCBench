module clink_lstm
  use iso_fortran_env, only: real32,real64,int64
  use omp_lib, only: omp_get_wtime
  implicit none
  integer, parameter :: N=8192,WGS=256,L=20000
contains
  subroutine init(dir,input,weight,x,iw,rw,ib,ow,ob)
    character(*),intent(in)::dir,input,weight
    real(real32),intent(out)::x(0:N*L-1),iw(0:19),rw(0:99),ib(0:19),ow(0:4),ob
    integer::u,s,i,j,k,g; character(len=512)::f
    f=trim(dir)//'/'//trim(input); open(newunit=u,file=trim(f),status='old',iostat=s)
    if(s/=0)then;print '(a,a,a)','File ',trim(input),' cannot be opened for read.';error stop 1;end if
    do i=0,L-1;read(u,*,iostat=s)x(i);if(s/=0)error stop 1;end do;close(u)
    do g=1,N-1;x(g*L:(g+1)*L-1)=x(0:L-1);end do
    f=trim(dir)//'/'//trim(weight);open(newunit=u,file=trim(f),status='old',iostat=s)
    if(s/=0)then;print '(a,a,a)','File ',trim(weight),' cannot be opened for read.';error stop 1;end if
    do j=0,3;do i=0,4;read(u,*,iostat=s)iw(j*5+i);if(s/=0)error stop 1;end do;end do
    do k=0,3;do j=0,4;do i=0,4;read(u,*,iostat=s)rw(k*25+j*5+i);if(s/=0)error stop 1;end do;end do;end do
    do j=0,3;do i=0,4;read(u,*,iostat=s)ib(j*5+i);if(s/=0)error stop 1;end do;end do
    do i=0,4;read(u,*,iostat=s)ow(i);if(s/=0)error stop 1;end do;read(u,*,iostat=s)ob;if(s/=0)error stop 1;close(u)
  end subroutine init
  subroutine lstm(x,iw,rw,ib,ow,ob,y,ns)
    real(real32),intent(in)::x(0:N*L-1),iw(0:19),rw(0:99),ib(0:19),ow(0:4),ob
    real(real32),intent(out)::y(0:N*L-1)
    integer(int64),intent(out)::ns
    integer::g,t,i,j;real(real32)::h(0:4),c(0:4),ii(0:4),ff(0:4),oo(0:4),gg(0:4),v,b;real(real64)::a,z
    !$omp target data map(to:x(0:N*L-1),iw(0:19),rw(0:99),ib(0:19),ow(0:4),ob) map(from:y(0:N*L-1))
    a=omp_get_wtime()
    !$omp target teams distribute parallel do thread_limit(WGS) private(h,c,ii,ff,oo,gg,v,b,i,j,t)
    do g=0,N-1
      h=0.0_real32;c=0.0_real32;ii=0.0_real32;ff=0.0_real32;oo=0.0_real32;gg=0.0_real32
      do t=0,L-1
        v=x(g*L+t)
        do j=0,4;ii(j)=iw(j)*v;do i=0,4;ii(j)=ii(j)+h(i)*rw(j*5+i);end do;ii(j)=1.0_real32/(1.0_real32+exp(-(ii(j)+ib(j))));end do
        do j=0,4;ff(j)=iw(5+j)*v;do i=0,4;ff(j)=ff(j)+h(i)*rw(25+j*5+i);end do;ff(j)=1.0_real32/(1.0_real32+exp(-(ff(j)+ib(5+j))));end do
        do j=0,4;oo(j)=iw(10+j)*v;do i=0,4;oo(j)=oo(j)+h(i)*rw(50+j*5+i);end do;oo(j)=1.0_real32/(1.0_real32+exp(-(oo(j)+ib(10+j))));end do
        do j=0,4;gg(j)=iw(15+j)*v;do i=0,4;gg(j)=gg(j)+h(i)*rw(75+j*5+i);end do;gg(j)=tanh(gg(j)+ib(15+j));end do
        do j=0,4;c(j)=c(j)*ff(j)+gg(j)*ii(j);h(j)=tanh(c(j))*oo(j);end do
        b=ob;do j=0,4;b=b+h(j)*ow(j);end do;y(g*L+t)=b
      end do
    end do
    !$omp end target teams distribute parallel do
    z=omp_get_wtime();ns=int((z-a)*1e9_real64,int64)
    !$omp end target data
  end subroutine lstm
  subroutine reference(x,iw,rw,ib,ow,ob,y)
    real(real32),intent(in)::x(0:N*L-1),iw(0:19),rw(0:99),ib(0:19),ow(0:4),ob,y(0:N*L-1)
    real(real32),allocatable::r(:);integer::g,t,i,j,k;real(real32)::h(0:4),c(0:4),ii(0:4),ff(0:4),oo(0:4),gg(0:4),v,b;logical::ok
    allocate(r(0:N*L-1))
    !$omp parallel do private(h,c,ii,ff,oo,gg,v,b,i,j,t)
    do g=0,N-1
      h=0.0_real32;c=0.0_real32;ii=0.0_real32;ff=0.0_real32;oo=0.0_real32;gg=0.0_real32
      do t=0,L-1
        v=x(g*L+t)
        do j=0,4;ii(j)=iw(j)*v;do i=0,4;ii(j)=ii(j)+h(i)*rw(j*5+i);end do;ii(j)=1.0_real32/(1.0_real32+exp(-(ii(j)+ib(j))));end do
        do j=0,4;ff(j)=iw(5+j)*v;do i=0,4;ff(j)=ff(j)+h(i)*rw(25+j*5+i);end do;ff(j)=1.0_real32/(1.0_real32+exp(-(ff(j)+ib(5+j))));end do
        do j=0,4;oo(j)=iw(10+j)*v;do i=0,4;oo(j)=oo(j)+h(i)*rw(50+j*5+i);end do;oo(j)=1.0_real32/(1.0_real32+exp(-(oo(j)+ib(10+j))));end do
        do j=0,4;gg(j)=iw(15+j)*v;do i=0,4;gg(j)=gg(j)+h(i)*rw(75+j*5+i);end do;gg(j)=tanh(gg(j)+ib(15+j));end do
        do j=0,4;c(j)=c(j)*ff(j)+gg(j)*ii(j);h(j)=tanh(c(j))*oo(j);end do;b=ob;do j=0,4;b=b+h(j)*ow(j);end do;r(g*L+t)=b
      end do
    end do
    !$omp end parallel do
    ok=.true.;do k=0,N*L-1;if(abs(r(k)-y(k))>1.0e-3_real32)then;ok=.false.;exit;end if;end do
    if(ok)then;print '(a)','PASS';else;print '(a)','FAIL';end if;deallocate(r)
  end subroutine reference
end module clink_lstm
program main
  use iso_fortran_env,only:real32,real64,int64
  use omp_lib,only:omp_get_wtime
  use clink_lstm
  implicit none
  integer::argc,rep,q,ios;integer(int64)::kt,part;real(real64)::a,z
  real(real32),allocatable::x(:),y1(:),y2(:),iw(:),rw(:),ib(:),ow(:);real(real32)::ob;character(len=512)::dir,arg
  argc=command_argument_count();if(argc/=1)then;call get_command_argument(0,arg);print '(a,a,a)','Usage: ',trim(arg),' <repeat>';error stop 1;end if
  call get_command_argument(1,arg);read(arg,*,iostat=ios)rep;if(ios/=0)error stop 1
  dir='../clink-omp';call get_environment_variable('CLINK_DATA_DIR',dir,status=ios);if(ios/=0 .or. len_trim(dir)==0)dir='../clink-omp'
  allocate(x(0:N*L-1),y1(0:N*L-1),y2(0:N*L-1),iw(0:19),rw(0:99),ib(0:19),ow(0:4));kt=0_int64
  do q=0,rep-1
    call init(dir,'input.hpp','weight_1.hpp',x,iw,rw,ib,ow,ob);a=omp_get_wtime();call lstm(x,iw,rw,ib,ow,ob,y1,part);kt=kt+part;z=omp_get_wtime();print '(a,i0,a)','Device offload time: ',int((z-a)*1000.0_real64),' ms';if(q==0)call reference(x,iw,rw,ib,ow,ob,y1)
    call init(dir,'input.hpp','weight_2.hpp',x,iw,rw,ib,ow,ob);a=omp_get_wtime();call lstm(x,iw,rw,ib,ow,ob,y2,part);kt=kt+part;z=omp_get_wtime();print '(a,i0,a)','Device offload time: ',int((z-a)*1000.0_real64),' ms';if(q==0)call reference(x,iw,rw,ib,ow,ob,y2)
  end do
  print '(a,f0.6,a)','Average kernel time: ',real(kt,real64)*1.0e-6_real64/(2.0_real64*real(rep,real64)),' ms';deallocate(x,y1,y2,iw,rw,ib,ow);print '(a)','Processing complete.'
end program main
