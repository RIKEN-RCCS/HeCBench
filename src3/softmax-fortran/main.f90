module softmax_kernels
  use iso_fortran_env,only:real32
  implicit none
contains
  subroutine softmax_cpu(nslices,ssize,input,output)
    integer,intent(in)::nslices,ssize
    real(real32),intent(in)::input(0:nslices*ssize-1)
    real(real32),intent(out)::output(0:nslices*ssize-1)
    integer::i,j
    real(real32)::maximum,sum,element
    do i=0,nslices-1
      maximum=input(i*ssize)
      do j=1,ssize-1;maximum=max(maximum,input(i*ssize+j));end do
      sum=0.0_real32
      do j=0,ssize-1;element=exp(input(i*ssize+j)-maximum);sum=sum+element;output(i*ssize+j)=element;end do
      do j=0,ssize-1;output(i*ssize+j)=output(i*ssize+j)/sum;end do
    end do
  end subroutine
  subroutine softmax_gpu(nslices,ssize,input,output)
    integer,intent(in)::nslices,ssize
    real(real32),intent(in)::input(0:nslices*ssize-1)
    real(real32),intent(out)::output(0:nslices*ssize-1)
    integer::i,j
    real(real32)::maximum,sum
!$omp target teams distribute parallel do simd thread_limit(256) private(maximum,sum,j)
    do i=0,nslices-1
      maximum=input(i*ssize)
      do j=1,ssize-1;maximum=max(maximum,input(i*ssize+j));end do
      sum=0.0_real32
      do j=0,ssize-1;sum=sum+exp(input(i*ssize+j)-maximum);end do
      do j=0,ssize-1;output(i*ssize+j)=exp(input(i*ssize+j)-maximum)/sum;end do
    end do
!$omp end target teams distribute parallel do simd
  end subroutine
end module
program softmax
  use iso_c_binding,only:c_int
  use iso_fortran_env,only:int64,real32,real64
  use softmax_kernels
  implicit none
  interface
    subroutine c_srand(seed) bind(C,name='srand');import c_int;integer(c_int),value::seed;end subroutine
    function c_rand() bind(C,name='rand') result(v);import c_int;integer(c_int)::v;end function
  end interface
  integer::argc,nslices,ssize,repeats,i,j,n
  integer(int64)::nelems,t0,t1,rate
  real(real32),allocatable::input(:),gpu(:),cpu(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=3)then;print '(a)','Usage: ./main <number of slices> <slice size> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)nslices;call get_command_argument(2,arg);read(arg,*)ssize;call get_command_argument(3,arg);read(arg,*)repeats
  nelems=int(nslices,int64)*ssize;allocate(input(0:nelems-1),gpu(0:nelems-1),cpu(0:nelems-1))
  call c_srand(2_c_int);do i=0,nslices-1;do j=0,ssize-1;input(i*ssize+j)=mod(c_rand(),13_c_int);end do;end do
!$omp target data map(to:input(0:nelems-1)) map(from:gpu(0:nelems-1))
  call system_clock(t0,rate);do n=1,repeats;call softmax_gpu(nslices,ssize,input,gpu);end do;call system_clock(t1)
  print '(a,f0.6,a)','Average kernel execution time: ',real(t1-t0,real64)*1000.0_real64/real(rate,real64)/repeats,' (ms)'
!$omp end target data
  call softmax_cpu(nslices,ssize,input,cpu)
  if(all(abs(cpu-gpu)<=1.0e-3_real32))then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
