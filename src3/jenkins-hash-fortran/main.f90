module jenkins_hash
  use iso_fortran_env, only: int32
  implicit none
  !$omp declare target (rot32,mix,final_mix,remainder_hash,hash_words)
contains
  pure integer(int32) function rot32(x, k)
    integer(int32), intent(in) :: x
    integer, intent(in) :: k
    rot32 = ior(shiftl(x, k), shiftr(x, 32-k))
  end function rot32

  subroutine mix(a, b, c)
    integer(int32), intent(inout) :: a, b, c
    a=a-c; a=ieor(a,rot32(c,4));  c=c+b
    b=b-a; b=ieor(b,rot32(a,6));  a=a+c
    c=c-b; c=ieor(c,rot32(b,8));  b=b+a
    a=a-c; a=ieor(a,rot32(c,16)); c=c+b
    b=b-a; b=ieor(b,rot32(a,19)); a=a+c
    c=c-b; c=ieor(c,rot32(b,4));  b=b+a
  end subroutine mix

  subroutine final_mix(a, b, c)
    integer(int32), intent(inout) :: a, b, c
    c=ieor(c,b); c=c-rot32(b,14)
    a=ieor(a,c); a=a-rot32(c,11)
    b=ieor(b,a); b=b-rot32(a,25)
    c=ieor(c,b); c=c-rot32(b,16)
    a=ieor(a,c); a=a-rot32(c,4)
    b=ieor(b,a); b=b-rot32(a,14)
    c=ieor(c,b); c=c-rot32(b,24)
  end subroutine final_mix

  function remainder_hash(a0,b0,c0,k0,k1,k2,length) result(c)
    integer(int32), intent(in) :: a0,b0,c0,k0,k1,k2,length
    integer(int32) :: a,b,c
    a=a0; b=b0; c=c0
    select case(length)
    case(12); c=c+k2; b=b+k1; a=a+k0
    case(11); c=c+iand(k2,int(z'00FFFFFF',int32)); b=b+k1; a=a+k0
    case(10); c=c+iand(k2,int(z'0000FFFF',int32)); b=b+k1; a=a+k0
    case(9);  c=c+iand(k2,int(z'000000FF',int32)); b=b+k1; a=a+k0
    case(8);  b=b+k1; a=a+k0
    case(7);  b=b+iand(k1,int(z'00FFFFFF',int32)); a=a+k0
    case(6);  b=b+iand(k1,int(z'0000FFFF',int32)); a=a+k0
    case(5);  b=b+iand(k1,int(z'000000FF',int32)); a=a+k0
    case(4);  a=a+k0
    case(3);  a=a+iand(k0,int(z'00FFFFFF',int32))
    case(2);  a=a+iand(k0,int(z'0000FFFF',int32))
    case(1);  a=a+iand(k0,int(z'000000FF',int32))
    case(0);  return
    end select
    call final_mix(a,b,c)
  end function remainder_hash

  function hash_words(keys, first, length0, initval) result(c)
    integer(int32), intent(in) :: keys(0:*), first, length0, initval
    integer(int32) :: c,a,b,k0,k1,k2,pos,left
    a=int(z'DEADBEEF',int32)+length0+initval
    b=a; c=a; pos=first; left=length0
    do while(left > 12)
      a=a+keys(pos); b=b+keys(pos+1); c=c+keys(pos+2)
      call mix(a,b,c)
      pos=pos+3; left=left-12
    end do
    k0=keys(pos); k1=keys(pos+1); k2=keys(pos+2)
    c=remainder_hash(a,b,c,k0,k1,k2,left)
  end function hash_words
end module jenkins_hash

program main
  use iso_fortran_env, only: int32, int64, real64
  use iso_c_binding, only: c_int
  use jenkins_hash
  implicit none
  integer(int32) :: block_size, repeat, n, id, rep, i, clk0, clk1, rate, c
  integer(int32) :: sample(0:15), keyword(0:15)
  integer(int32), allocatable :: keys(:), lens(:), initvals(:), out(:)
  character(len=64) :: arg, str
  logical :: error

  interface
    subroutine c_srand(seed) bind(C,name='srand')
      import c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C,name='rand') result(value)
      import c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: ./main <block size> <number of strings> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) block_size
  call get_command_argument(2,arg); read(arg,*) n
  call get_command_argument(3,arg); read(arg,*) repeat

  str='Four score and seven years ago'
  call pack_string(str,sample)
  c=hash_words(sample,0_int32,30_int32,1_int32)
  write(*,'(a,z8.8)') 'input string: Four score and seven years ago hash is ',c

  allocate(keys(0:16*n-1),lens(0:n-1),initvals(0:n-1),out(0:n-1))
  call pack_string(str,keyword)
  call c_srand(2_c_int)
  do i=0,n-1
    keys(16*i:16*i+15)=keyword
    lens(i)=mod(c_rand(),61_c_int)
    initvals(i)=mod(i,2)
  end do

  call system_clock(clk0,rate)
  !$omp target data map(to:keys(0:16*n-1),lens(0:n-1),initvals(0:n-1)) map(from:out(0:n-1))
  do rep=1,repeat
    !$omp target teams distribute parallel do thread_limit(block_size) private(c)
    do id=0,n-1
      out(id)=hash_words(keys,16*id,lens(id),initvals(id))
    end do
    !$omp end target teams distribute parallel do
  end do
  !$omp end target data
  call system_clock(clk1)
  write(*,'(a,f12.6,a)') 'Average kernel execution time : ', &
    real(clk1-clk0,real64)/real(int(rate,int64)*int(repeat,int64),real64),' (s)'

  print '(a)','Verify the results computed on the device..'
  error=.false.
  do i=0,n-1
    c=hash_words(keys,16*i,lens(i),initvals(i))
    if(out(i)/=c) then
      write(*,'(a,i0,a,z8.8,a,z8.8)') 'Error: at ',i,' gpu hash is ',out(i),' cpu hash is ',c
      error=.true.; exit
    end if
  end do
  if(error) then
    print '(a)','FAIL'; error stop 2
  else
    print '(a)','PASS'
  end if
contains
  subroutine pack_string(text,words)
    character(len=*),intent(in)::text
    integer(int32),intent(out)::words(0:15)
    integer::j,p,l
    words=0_int32; l=len_trim(text)
    do p=1,min(l,64)
      j=(p-1)/4
      words(j)=ior(words(j),shiftl(int(iachar(text(p:p)),int32),8*mod(p-1,4)))
    end do
  end subroutine pack_string
end program main
