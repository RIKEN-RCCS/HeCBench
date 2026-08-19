module iso2dfd_kernels
  use iso_fortran_env,only:real32
  implicit none
contains
  subroutine initialize(prev,next,vel,nrows,ncols)
    integer,intent(in)::nrows,ncols
    real(real32),intent(out)::prev(0:nrows*ncols-1),next(0:nrows*ncols-1),vel(0:nrows*ncols-1)
    integer::i,k,s
    real(real32),parameter::wavelet(0:11)=[0.016387336_real32,-0.041464937_real32,-0.067372555_real32,0.386110067_real32,0.812723635_real32,0.416998396_real32,0.076488599_real32,-0.059434419_real32,0.023680172_real32,0.005611435_real32,0.001823209_real32,-0.000720549_real32]
    prev=0.0_real32;next=0.0_real32;vel=2250000.0_real32
    do s=11,0,-1
      do i=nrows/2-s,nrows/2+s-1
        do k=ncols/2-s,ncols/2+s-1;prev(i*ncols+k)=wavelet(s);end do
      end do
    end do
  end subroutine
  subroutine kernel(next,prev,vel,scale,nrows,ncols)
    integer,intent(in)::nrows,ncols
    real(real32),intent(in)::prev(0:nrows*ncols-1),vel(0:nrows*ncols-1),scale
    real(real32),intent(inout)::next(0:nrows*ncols-1)
    integer::r,c,gid
    real(real32)::value
!$omp target teams distribute parallel do simd collapse(2) thread_limit(256) private(gid,value)
    do r=0,nrows-1
      do c=0,ncols-1
        gid=r*ncols+c
        if(c>=1.and.c<ncols-1.and.r>=1.and.r<nrows-1)then
          value=prev(gid+1)-2.0_real32*prev(gid)+prev(gid-1)+prev(gid+ncols)-2.0_real32*prev(gid)+prev(gid-ncols)
          next(gid)=2.0_real32*prev(gid)-next(gid)+value*scale*vel(gid)
        end if
      end do
    end do
!$omp end target teams distribute parallel do simd
  end subroutine
  subroutine cpu_iteration(next,prev,vel,scale,nrows,ncols,niter)
    integer,intent(in)::nrows,ncols,niter
    real(real32),target,intent(inout)::next(0:nrows*ncols-1),prev(0:nrows*ncols-1)
    real(real32),intent(in)::vel(0:nrows*ncols-1),scale
    real(real32),pointer::a(:),b(:),swap(:)
    integer::k,r,c,gid
    a=>next;b=>prev
    do k=1,niter
      do r=1,nrows-2;do c=1,ncols-2;gid=r*ncols+c;a(gid)=2.0_real32*b(gid)-a(gid)+(b(gid+1)-2.0_real32*b(gid)+b(gid-1)+b(gid+ncols)-2.0_real32*b(gid)+b(gid-ncols))*scale*vel(gid);end do;end do
      swap=>a;a=>b;b=>swap
    end do
  end subroutine
end module
program iso2dfd
  use iso_fortran_env,only:int64,real32,real64
  use iso2dfd_kernels
  implicit none
  integer::argc,nrows,ncols,niter,k
  integer(int64)::nsize,t0,t1,rate
  real(real32)::scale
  real(real32),allocatable,target::prev(:),next(:),nextcpu(:),vel(:)
  character(len=64)::arg
  argc=command_argument_count();if(argc/=3)then;print '(a)','Usage: ./main n1 n2 Iterations';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)nrows;call get_command_argument(2,arg);read(arg,*)ncols;call get_command_argument(3,arg);read(arg,*)niter
  nsize=int(nrows,int64)*ncols;scale=(0.002_real32*0.002_real32)/(20.0_real32*20.0_real32);allocate(prev(0:nsize-1),next(0:nsize-1),nextcpu(0:nsize-1),vel(0:nsize-1));call initialize(prev,next,vel,nrows,ncols);print '(a,2(i0,1x))','Grid Sizes: ',nrows,ncols;print '(a,i0)','Iterations: ',niter;print '(a)','Computing wavefield in device ..'
!$omp target data map(next(0:nsize-1),prev(0:nsize-1)) map(to:vel(0:nsize-1))
  call system_clock(t0,rate);do k=0,niter-1;if(mod(k,2)==0)then;call kernel(next,prev,vel,scale,nrows,ncols);else;call kernel(prev,next,vel,scale,nrows,ncols);end if;end do;call system_clock(t1)
  print '(a,f0.6,a)','Total kernel execution time ',real(t1-t0,real64)*1000.0_real64/rate,' (ms)';print '(a,f0.6,a)','Average kernel execution time ',real(t1-t0,real64)*1.0e6_real64/rate/niter,' (us)'
!$omp end target data
  open(10,file='wavefield_snapshot.bin',form='unformatted',access='stream',status='replace');write(10)next;close(10)
  print '(a)','Computing wavefield in CPU ..';call initialize(prev,nextcpu,vel,nrows,ncols);call cpu_iteration(nextcpu,prev,vel,scale,nrows,ncols,niter)
  if(maxval(abs(next-nextcpu))<=0.1_real32)then;print '(a)','PASS';else;print '(a)','FAIL';stop 2;end if
  open(10,file='wavefield_snapshot_cpu.bin',form='unformatted',access='stream',status='replace');write(10)nextcpu;close(10)
end program
