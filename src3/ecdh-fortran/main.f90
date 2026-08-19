module ecdh_math
  use, intrinsic :: iso_fortran_env, only : int32
  implicit none
!$omp declare target (ext_euclidian_alg, make_positive, find_inverse, point_addition, point_doubling, first_set_bit, make_pk_fast, make_pk_slow)
contains
  integer(int32) function ext_euclidian_alg(a, b, x, y)
    integer(int32), intent(in) :: a,b
    integer(int32), intent(out) :: x,y
    integer(int32) :: x1,y1,a1,b1,q,s,t
    x=1; y=0; x1=0; y1=1; a1=a; b1=b
    do while(b1 /= 0)
      q=a1/b1
      s=x1; t=x-q*x1; x=s; x1=t
      s=y1; t=y-q*y1; y=s; y1=t
      s=b1; t=a1-q*b1; a1=s; b1=t
    end do
    ext_euclidian_alg=a1
  end function ext_euclidian_alg

  integer(int32) function make_positive(a,m)
    integer(int32), intent(in) :: a,m
    integer(int32) :: work
    work=a
    do while(work < 0); work=work+m; end do
    make_positive=modulo(work,m)
  end function make_positive

  integer(int32) function find_inverse(a,m)
    integer(int32), intent(in) :: a,m
    integer(int32) :: t,s,ignored
    ignored=ext_euclidian_alg(a,m,t,s)
    find_inverse=make_positive(t,m)
  end function find_inverse

  subroutine point_addition(m,x1,y1,x2,y2,x3,y3)
    integer(int32), intent(in) :: m,x1,y1,x2,y2
    integer(int32), intent(out) :: x3,y3
    integer(int32) :: temp,slope
    temp=make_positive(x2-x1,m)
    slope=make_positive((y2-y1)*find_inverse(temp,m),m)
    x3=make_positive(slope*slope-x1-x2,m)
    y3=make_positive(slope*(x1-x3)-y1,m)
  end subroutine point_addition

  subroutine point_doubling(m,a,x1,y1,x3,y3)
    integer(int32), intent(in) :: m,a,x1,y1
    integer(int32), intent(out) :: x3,y3
    integer(int32) :: slope
    slope=(3*x1*x1+a)*find_inverse(2*y1,m)
    x3=make_positive(slope*slope-2*x1,m)
    y3=make_positive(slope*(x1-x3)-y1,m)
  end subroutine point_doubling

  integer(int32) function first_set_bit(n)
    integer(int32), intent(in) :: n
    integer :: bit
    do bit=bit_size(n)-1,0,-1
      if(btest(n,bit)) then; first_set_bit=bit; return; end if
    end do
    first_set_bit=0
  end function first_set_bit

  subroutine make_pk_fast(sk,px,py,tx,ty,m,a)
    integer(int32), intent(in) :: sk,px,py,m,a
    integer(int32), intent(out) :: tx,ty
    integer :: bit
    integer(int32) :: next_x,next_y
    tx=px; ty=py
    do bit=first_set_bit(sk)-1,0,-1
      call point_doubling(m,a,tx,ty,next_x,next_y)
      tx=next_x; ty=next_y
      if(btest(sk,bit)) then
        call point_addition(m,tx,ty,px,py,next_x,next_y)
        tx=next_x; ty=next_y
      end if
    end do
  end subroutine make_pk_fast

  subroutine make_pk_slow(sk,px,py,tx,ty,m,a)
    integer(int32), intent(in) :: sk,px,py,m,a
    integer(int32), intent(out) :: tx,ty
    integer(int32) :: work,next_x,next_y
    call point_doubling(m,a,px,py,tx,ty)
    work=sk-2
    do while(work > 0)
      call point_addition(m,tx,ty,px,py,next_x,next_y)
      tx=next_x; ty=next_y
      work=work-1
    end do
  end subroutine make_pk_slow
end module ecdh_math

program ecdh_benchmark
  use, intrinsic :: iso_fortran_env, only : int32, int64, real64
  use ecdh_math
  implicit none
  integer :: number_keys, repeat, iteration
  integer(int64) :: start_count,end_count,clock_rate
  real(real64) :: elapsed_seconds
  integer(int32), allocatable :: slow_x(:),slow_y(:),fast_x(:),fast_y(:)
  character(len=64) :: arg

  if(command_argument_count()/=2)then
    write(*,'(a)') 'Usage: ./main <positive number of keys> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) number_keys
  call get_command_argument(2,arg); read(arg,*) repeat
  if(number_keys<=0 .or. repeat<=0) error stop 'arguments must be positive'
  allocate(slow_x(0:number_keys-1),slow_y(0:number_keys-1),fast_x(0:number_keys-1),fast_y(0:number_keys-1))

!$omp target data map(from:slow_x(0:number_keys-1),slow_y(0:number_keys-1),fast_x(0:number_keys-1),fast_y(0:number_keys-1))
  call system_clock(start_count,clock_rate)
  do iteration=1,repeat
    call k_slow(18_int32,5_int32,1_int32,slow_x,slow_y,17_int32,2_int32,number_keys)
  end do
  call system_clock(end_count)
  elapsed_seconds=real(end_count-start_count,real64)/real(clock_rate,real64)/real(repeat,real64)
  write(*,'(a,f0.6,a)') 'Average time (slow kernel): ',elapsed_seconds,' s'

  call system_clock(start_count,clock_rate)
  do iteration=1,repeat
    call k_fast(18_int32,5_int32,1_int32,fast_x,fast_y,17_int32,2_int32,number_keys)
  end do
  call system_clock(end_count)
  elapsed_seconds=real(end_count-start_count,real64)/real(clock_rate,real64)/real(repeat,real64)
  write(*,'(a,f0.6,a)') 'Average time (fast kernel): ',elapsed_seconds,' s'
!$omp end target data

  if(any(slow_x/=fast_x) .or. any(slow_y/=fast_y))then;write(*,'(a)')'FAIL';else;write(*,'(a)')'PASS';end if
  deallocate(slow_x,slow_y,fast_x,fast_y)
contains
  subroutine k_slow(sk,px,py,tx,ty,m,a,n)
    integer(int32), intent(in) :: sk,px,py,m,a
    integer, intent(in) :: n
    integer(int32), intent(out) :: tx(0:),ty(0:)
    integer :: i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,n-1
      call make_pk_slow(sk,px,py,tx(i),ty(i),m,a)
    end do
!$omp end target teams distribute parallel do
  end subroutine k_slow
  subroutine k_fast(sk,px,py,tx,ty,m,a,n)
    integer(int32), intent(in) :: sk,px,py,m,a
    integer, intent(in) :: n
    integer(int32), intent(out) :: tx(0:),ty(0:)
    integer :: i
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,n-1
      call make_pk_fast(sk,px,py,tx(i),ty(i),m,a)
    end do
!$omp end target teams distribute parallel do
  end subroutine k_fast
end program ecdh_benchmark
