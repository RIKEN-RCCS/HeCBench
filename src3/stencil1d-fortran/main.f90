program stencil_1d
 use iso_fortran_env,only:int32,real64;implicit none
 integer,parameter::radius=7,block=256;integer::length,reps,pad,i,j,base,r,c0,c1,rate,s;integer(int32),allocatable::a(:),b(:);integer(int32)::temp(0:block+2*radius-1);character(len=32)::arg;logical::ok
 if(command_argument_count()/=2)stop 1;call get_command_argument(1,arg);read(arg,*)length;call get_command_argument(2,arg);read(arg,*)reps;if(mod(length,block)/=0.or.reps<1)stop 1
 pad=length+radius;allocate(a(0:pad-1),b(0:length-1));do i=0,pad-1;a(i)=i;end do
 call system_clock(c0,rate);do r=1,reps
 !$omp target teams distribute map(to:a(0:pad-1)) map(from:b(0:length-1)) private(temp,j,i,s)
 do base=0,length-1,block
  !$omp parallel do schedule(static,1)
  do j=0,block-1
   i=base+j
   temp(j+radius)=a(i)
   if(j<radius)then
    if(i<radius)then
     temp(j)=0_int32
    else
     temp(j)=a(i-radius)
    end if
    temp(j+radius+block)=a(i+block)
   end if
  end do
  !$omp end parallel do
  !$omp parallel do schedule(static,1)
  do j=0,block-1;s=0;do i=-radius,radius;s=s+temp(j+radius+i);end do;b(base+j)=s;end do
  !$omp end parallel do
 end do
 !$omp end target teams distribute
 end do;call system_clock(c1)
 write(*,'(a,f10.6,a)')'Average kernel execution time: ', &
  real(c1-c0,real64)/(real(rate,real64)*real(reps,real64)),' (s)'
 ok=.true.
 do i=0,length-1
  s=0
  do j=i-radius,i+radius
   if(j>=0) s=s+a(j)
  end do
  if(s/=b(i))then;ok=.false.;exit;end if
 end do
 if(ok)then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
end program
