module fastdiv_mod
  use iso_fortran_env, only: int32, int64, real64
  implicit none
!$omp declare target (fast_div,arshift64)
contains
  subroutine make_magic(d,m,s,add_sign)
    integer(int32),intent(in)::d
    integer(int32),intent(out)::m,s,add_sign
    integer(int64)::ad,anc,delta,q1,r1,q2,r2,t,two31,p,mm
    if(d==1_int32 .or. d==-1_int32) then
      m=0; s=-1; add_sign=merge(1_int32,-1_int32,d==1); return
    end if
    two31=2147483648_int64; ad=abs(int(d,int64)); if(ad==0) ad=1
    t=two31+merge(1_int64,0_int64,d<0); anc=t-1-mod(t,ad); p=31
    q1=two31/anc; r1=two31-q1*anc; q2=two31/ad; r2=two31-q2*ad
    do
      p=p+1; q1=2*q1; r1=2*r1; if(r1>=anc) then; q1=q1+1; r1=r1-anc; end if
      q2=2*q2; r2=2*r2; if(r2>=ad) then; q2=q2+1; r2=r2-ad; end if
      delta=ad-r2; if(.not.(q1<delta .or. (q1==delta .and. r1==0))) exit
    end do
    mm=q2+1; if(d<0) mm=-mm; m=int(mm,int32); s=int(p-32,int32)
    if(d>0 .and. m<0) then; add_sign=1
    else if(d<0 .and. m>0) then; add_sign=-1
    else; add_sign=0
    end if
  end subroutine
  integer(int32) function fast_div(n,m,s,add_sign)
    integer(int32),intent(in)::n,m,s,add_sign
    integer(int64)::q, bits
    bits=shiftr(int(m,int64)*int(n,int64),32)
    if(bits>=2147483648_int64) then
      q=bits-4294967296_int64
    else
      q=bits
    end if
    q=q+int(n,int64)*int(add_sign,int64)
    if(s>=0) then
      q=arshift64(q,int(s)); if(q<0) q=q+1
    end if
    fast_div=int(q,int32)
  end function
  pure integer(int64) function arshift64(value,shift)
    integer(int64),intent(in)::value
    integer,intent(in)::shift
    if(shift==0) then
      arshift64=value
    else if(value>=0_int64) then
      arshift64=shiftr(value,shift)
    else
      arshift64=-shiftr(-value+shiftl(1_int64,shift)-1_int64,shift)
    end if
  end function
end module
program idivide
  use iso_fortran_env, only:int32,real64
  use fastdiv_mod
  implicit none
  integer::argc,repeat,ndiv,n,grids,i,d,sign
  integer(int32)::m,s,a,divisor,x,q,fastq,errors(4)
  real(real64)::start,finish,slow,fast
  character(32)::arg
  argc=command_argument_count(); if(argc<1 .or. argc>3) error stop 'Usage: ./main <repeat> [divisor_count] [dividend_count]'
  call get_command_argument(1,arg);read(arg,*)repeat;ndiv=100000;n=1000000
  if(argc>=2)then;call get_command_argument(2,arg);read(arg,*)ndiv;end if
  if(argc==3)then;call get_command_argument(3,arg);read(arg,*)n;end if
  grids=(n+255)/256;n=grids*256
  write(*,'(A,I0,A,I0,A)')'Running functional test on ',ndiv,' divisors, with ',n,' dividends for each divisor'
  do d=1,ndiv-1
    do sign=1,-1,-2
      divisor=int(d*sign,int32);call make_magic(divisor,m,s,a);errors=0
!$omp target data map(tofrom:errors)
!$omp target teams distribute parallel do thread_limit(256) private(q,fastq,x)
      do i=0,n-1
        q=int(i,int32)/divisor;fastq=fast_div(int(i,int32),m,s,a)
        if(q/=fastq) then
!$omp atomic capture
          x=errors(1);errors(1)=errors(1)+1
!$omp end atomic
          if(x==0) then; errors(2)=i;errors(3)=q;errors(4)=fastq;end if
        end if
        q=int(-i,int32)/divisor;fastq=fast_div(int(-i,int32),m,s,a)
        if(q/=fastq) then
!$omp atomic capture
          x=errors(1);errors(1)=errors(1)+1
!$omp end atomic
          if(x==0) then; errors(2)=-i;errors(3)=q;errors(4)=fastq;end if
        end if
      end do
!$omp end target teams distribute parallel do
!$omp end target data
      if(errors(1)>0) then
        write(*,'(A,I0,A,I0,A,I0,A,I0,A,I0)') 'divisor ',divisor,': ',errors(1),' wrong results; dividend ',errors(2),', correct ',errors(3),', fast ',errors(4)
        error stop 'FAIL'
      end if
    end do
  end do
  n=32*1024*256;call make_magic(3_int32,m,s,a)
  call cpu_time(start)
!$omp target teams distribute parallel do thread_limit(256)
  do i=0,n-1; x=int(i,int32)/3_int32+int(i,int32)/5_int32+int(i,int32)/7_int32; end do
!$omp end target teams distribute parallel do
  call cpu_time(finish);slow=finish-start
  call cpu_time(start)
!$omp target teams distribute parallel do thread_limit(256)
  do i=0,n-1; x=fast_div(int(i,int32),m,s,a)+fast_div(int(i,int32),m,s,a)+fast_div(int(i,int32),m,s,a); end do
!$omp end target teams distribute parallel do
  call cpu_time(finish);fast=finish-start
  write(*,'(A)')'THROUGHPUT TEST';write(*,'(A,F10.6,A)')'Benchmarking plain division by constant... ',slow*repeat,' seconds';write(*,'(A,F10.6,A)')'Benchmarking fast division by constant... ',fast*repeat,' seconds'
  write(*,'(A)')'PASS'
end program
