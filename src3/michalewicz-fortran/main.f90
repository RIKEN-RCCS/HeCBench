module mt19937_rng
  use iso_fortran_env, only: int64, real32
  implicit none
  integer, parameter :: nmt=624,mmt=397
  integer(int64), parameter :: mask32=int(z'00000000FFFFFFFF',int64), upper=int(z'0000000080000000',int64), lower=int(z'000000007FFFFFFF',int64), mata=int(z'000000009908B0DF',int64)
  type :: mt_state
    integer(int64)::s(0:nmt-1)
    integer::idx=nmt
  end type
contains
  subroutine seed(g,seed_value)
    type(mt_state),intent(out)::g
    integer(int64),intent(in)::seed_value
    integer::i
    g%s(0)=iand(seed_value,mask32)
    do i=1,nmt-1; g%s(i)=iand(1812433253_int64*ieor(g%s(i-1),ishft(g%s(i-1),-30))+i,mask32);end do
    g%idx=nmt
  end subroutine
  subroutine twist(g)
    type(mt_state),intent(inout)::g
    integer::i
    integer(int64)::y
    do i=0,nmt-1
      y=ior(iand(g%s(i),upper),iand(g%s(mod(i+1,nmt)),lower)); g%s(i)=ieor(g%s(mod(i+mmt,nmt)),ishft(y,-1))
      if(iand(y,1_int64)/=0)g%s(i)=ieor(g%s(i),mata); g%s(i)=iand(g%s(i),mask32)
    end do
    g%idx=0
  end subroutine
  function next_u32(g) result(v)
    type(mt_state),intent(inout)::g
    integer(int64)::v
    if(g%idx>=nmt)call twist(g)
    v=g%s(g%idx);g%idx=g%idx+1;v=ieor(v,ishft(v,-11));v=ieor(v,iand(ishft(v,7),int(z'000000009D2C5680',int64)));v=ieor(v,iand(ishft(v,15),int(z'00000000EFC60000',int64)));v=ieor(v,ishft(v,-18));v=iand(v,mask32)
  end function
  function uniform_0_4(g) result(v)
    type(mt_state),intent(inout)::g
    real(real32)::v
    v=4.0_real32*real(next_u32(g),real32)/4294967296.0_real32
  end function
end module

module michalewicz_kernel
  use iso_fortran_env, only:int64,real32,real64
  implicit none
contains
  function value(x,dim) result(result)
    integer,intent(in)::dim
    real(real32),intent(in)::x(0:dim-1)
    real(real32)::result,a,b,c
    integer::i
    result=0.0_real32
    do i=0,dim-1
      a=sin(x(i)); b=sin(real(i+1,real32)*x(i)*x(i)/acos(-1.0_real32)); c=b**20
      result=result+a*c
    end do
    result=-result
  end function
  subroutine evaluate(values,n,dim,repeats,min_value)
    integer(int64),intent(in)::n
    integer,intent(in)::dim,repeats
    real(real32),intent(in)::values(0:n*dim-1)
    real(real32),intent(inout)::min_value
    integer::i
    integer(int64)::j,t0,t1,rate
!$omp target data map(to:values(0:n*dim-1)) map(tofrom:min_value)
    call system_clock(t0,rate)
    do i=1,repeats
!$omp target teams distribute parallel do thread_limit(256) reduction(min:min_value)
      do j=0,n-1
        min_value=min(min_value,value(values(j*dim:j*dim+dim-1),dim))
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(t1)
    print '(a,i0,a,f0.6,a)','Average execution time of kernel (dim = ',dim,'): ',real(t1-t0,real64)*1.0e6_real64/real(rate,real64)/repeats,' (us)'
!$omp end target data
  end subroutine
end module

program michalewicz
  use iso_fortran_env,only:int64,real32
  use mt19937_rng
  use michalewicz_kernel
  implicit none
  type(mt_state)::g
  integer::argc,repeats,d,dim,i
  integer,parameter::dims(0:2)=[2,5,10]
  integer(int64)::n,size
  real(real32)::min_value,true_min
  real(real32),allocatable::values(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <number of vectors> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)n;call get_command_argument(2,arg);read(arg,*)repeats
  call seed(g,19937_int64)
  do d=0,2
    dim=dims(d);size=n*dim;allocate(values(0:size-1))
    do i=0,int(size)-1;values(i)=uniform_0_4(g);end do
    min_value=0.0_real32;call evaluate(values,n,dim,repeats,min_value)
    true_min=0.0_real32;if(dim==2)true_min=-1.8013_real32;if(dim==5)true_min=-4.687658_real32;if(dim==10)true_min=-9.66015_real32
    print '(a,f0.6)','Global minima = ',min_value;print '(a,f0.6)','Error = ',abs(true_min-min_value)
    deallocate(values)
  end do
end program
