module permute_kernel
  use iso_fortran_env,only:real32,real64
  implicit none
contains
  subroutine permute(out,inp,b,t,c,nh,block_size)
    integer,intent(in)::b,t,c,nh,block_size
    real(real32),intent(out)::out(0:b*t*c*3-1)
    real(real32),intent(in)::inp(0:b*t*c*3-1)
    integer::idx,bb,rest,nhh,nn,dd,d,total_threads,num_blocks
    d=c/nh;total_threads=b*t*c;num_blocks=(total_threads+block_size-1)/block_size
!$omp target teams distribute parallel do num_teams(num_blocks) thread_limit(block_size) private(bb,rest,nhh,nn,dd)
    do idx=0,total_threads-1
      bb=idx/(c*t);rest=mod(idx,c*t);nhh=rest/(t*d);rest=mod(rest,t*d);nn=rest/d;dd=mod(rest,d)
      out(idx)=inp(bb*t*3*c+nn*3*c+nhh*d+dd)
      out(idx+b*t*c)=inp(bb*t*3*c+nn*3*c+c+nhh*d+dd)
      out(idx+2*b*t*c)=inp(bb*t*3*c+nn*3*c+2*c+nhh*d+dd)
    end do
!$omp end target teams distribute parallel do
  end subroutine
end module
program permute_main
  use iso_c_binding,only:c_int
  use iso_fortran_env,only:int64,real32,real64
  use permute_kernel
  implicit none
  interface
    function c_rand() bind(C,name='rand') result(v)
      import c_int
      integer(c_int)::v
    end function
  end interface
  integer,parameter::t=1024,c=768,nh=12
  integer::argc,b,repeats,i,j,block_sizes(1:5),block_size,bb,rest,head,token,dd,d
  integer(int64)::s,t0,t1,rate
  real(real32),allocatable::inp(:),out(:),q(:),k(:),v(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <batch size> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)b;call get_command_argument(2,arg);read(arg,*)repeats
  s=int(b,int64)*t*c;allocate(inp(0:3*s-1),out(0:3*s-1),q(0:s-1),k(0:s-1),v(0:s-1))
  do i=0,int(3*s)-1;inp(i)=2.0_real32*real(c_rand(),real32)/2147483647.0_real32-1.0_real32;out(i)=inp(i);end do
  d=c/nh
  do i=0,int(s)-1
    bb=i/(c*t);rest=mod(i,c*t);head=rest/(t*d);rest=mod(rest,t*d);token=rest/d;dd=mod(rest,d)
    q(i)=inp(bb*t*3*c+token*3*c+head*d+dd)
    k(i)=inp(bb*t*3*c+token*3*c+c+head*d+dd)
    v(i)=inp(bb*t*3*c+token*3*c+2*c+head*d+dd)
  end do
  block_sizes=[32,64,128,256,512]
!$omp target data map(to:inp(0:3*s-1)) map(alloc:out(0:3*s-1))
  do j=1,5
    block_size=block_sizes(j);print '(a,i0,a)','Checking block size ',block_size,'.';call permute(out,inp,b,t,c,nh,block_size)
!$omp target update from(out(0:3*s-1))
    if(any(abs(out(0:s-1)-q)>1.0e-6_real32).or.any(abs(out(s:2*s-1)-k)>1.0e-6_real32).or.any(abs(out(2*s:3*s-1)-v)>1.0e-6_real32))then;print '(a)','FAIL';stop 2;end if
  end do
  print '(a)','All results match. Starting benchmarks.';print *
  do j=1,5
    block_size=block_sizes(j);call system_clock(t0,rate);do i=1,repeats;call permute(out,inp,b,t,c,nh,block_size);end do;call system_clock(t1)
    print '(a,i4,a,f0.6,a)','block_size ',block_size,' | time ',real(t1-t0,real64)*1000.0_real64/real(rate,real64)/repeats,' ms'
  end do
!$omp end target data
end program
