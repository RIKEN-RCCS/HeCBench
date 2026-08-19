module mandel_kernel
 use iso_fortran_env,only:int32,real32
 implicit none
contains
 subroutine evaluate_gpu(data,rows,cols,maxit,elapsed)
  integer(int32),intent(out)::data(:);integer,intent(in)::rows,cols,maxit;real,intent(out)::elapsed
  integer::i,j,k;real(real32)::cr,ci,zr,zi,r,im;integer::c0,c1,rate
  call system_clock(c0,rate)
  !$omp target teams distribute parallel do simd collapse(2) thread_limit(256) private(k,cr,ci,zr,zi,r,im)
  do i=0,rows-1;do j=0,cols-1
   cr=-1.5_real32+i*(2.0_real32/rows);ci=-1.0_real32+j*(2.0_real32/cols);zr=0;zi=0;data(i*cols+j+1)=0
   do k=1,maxit;r=zr;im=zi;if(r*r+im*im>=4.0_real32)exit;zr=r*r-im*im+cr;zi=2*r*im+ci;data(i*cols+j+1)=data(i*cols+j+1)+1;end do
  end do;end do
  !$omp end target teams distribute parallel do simd
  call system_clock(c1);elapsed=real(c1-c0)/rate
 end subroutine
 subroutine evaluate_cpu(data,rows,cols,maxit)
  integer(int32),intent(out)::data(:);integer,intent(in)::rows,cols,maxit;integer::i,j,k;real(real32)::cr,ci,zr,zi,r,im
  do i=0,rows-1;do j=0,cols-1;cr=-1.5_real32+i*(2.0_real32/rows);ci=-1.0_real32+j*(2.0_real32/cols);zr=0;zi=0;data(i*cols+j+1)=0;do k=1,maxit;r=zr;im=zi;if(r*r+im*im>=4)exit;zr=r*r-im*im+cr;zi=2*r*im+ci;data(i*cols+j+1)=data(i*cols+j+1)+1;end do;end do;end do
 end subroutine
end module
program main
 use iso_fortran_env,only:int32,real64;use mandel_kernel;implicit none
 integer,parameter::rows=1080,cols=1920,maxit=100;integer::reps,r,c0,c1,rate,diff;real::kt;real(real64)::sumt;integer(int32),allocatable::gpu(:),cpu(:);character(len=32)::arg
 if(command_argument_count()/=1)stop 1;call get_command_argument(1,arg);read(arg,*)reps;if(reps<1)stop 1;allocate(gpu(rows*cols),cpu(rows*cols))
 call evaluate_gpu(gpu,rows,cols,maxit,kt);sumt=0;call system_clock(c0,rate);do r=1,reps;call evaluate_gpu(gpu,rows,cols,maxit,kt);sumt=sumt+kt;end do;call system_clock(c1)
 call evaluate_cpu(cpu,rows,cols,maxit);diff=count(cpu/=gpu)
 write(*,'(a,f10.3,a)')'Average parallel time: ',1000.0_real64*(c1-c0)/(rate*reps),' ms';write(*,'(a,f10.3,a)')'Average kernel execution time: ',1000.0_real64*sumt/reps,' ms'
 if(real(diff)/size(cpu)>.05)then;print '(a)','Failure';stop 2;end if;print '(a)','Success'
end program
