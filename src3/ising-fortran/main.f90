module ising_kernels
  use iso_fortran_env,only:int8,int64,real32
  implicit none
contains
  subroutine init_spins(lattice,randvals,nx,half_ny)
    integer(int64),intent(in)::nx,half_ny
    integer(int8),intent(out)::lattice(0:nx*half_ny-1)
    real(real32),intent(in)::randvals(0:nx*half_ny-1)
    integer(int64)::tid
!$omp target teams distribute parallel do simd thread_limit(128)
    do tid=0,nx*half_ny-1;lattice(tid)=merge(-1_int8,1_int8,randvals(tid)<0.5_real32);end do
!$omp end target teams distribute parallel do simd
  end subroutine
  subroutine update_one(lattice,other,randvals,invtemp,nx,halfny,is_black)
    integer(int64),intent(in)::nx,halfny
    integer(int8),intent(inout)::lattice(0:nx*halfny-1)
    integer(int8),intent(in)::other(0:nx*halfny-1)
    real(real32),intent(in)::randvals(0:nx*halfny-1),invtemp
    logical,intent(in)::is_black
    integer::i,j,ipp,inn,jpp,jnn,joff
    integer(int8)::nn_sum,lij
    real(real32)::ratio
!$omp target teams distribute parallel do collapse(2) thread_limit(128) private(ipp,inn,jpp,jnn,joff,nn_sum,lij,ratio)
    do i=0,int(nx)-1
      do j=0,int(halfny)-1
        ipp=merge(i+1,0,i+1<int(nx));inn=merge(i-1,int(nx)-1,i-1>=0);jpp=merge(j+1,0,j+1<int(halfny));jnn=merge(j-1,int(halfny)-1,j-1>=0)
        if(is_black)then;joff=merge(jpp,jnn,mod(i,2)/=0);else;joff=merge(jnn,jpp,mod(i,2)/=0);end if
        nn_sum=other(inn*halfny+j)+other(i*halfny+j)+other(ipp*halfny+j)+other(i*halfny+joff);lij=lattice(i*halfny+j);ratio=exp(-2.0_real32*invtemp*real(nn_sum,real32)*real(lij,real32))
        if(randvals(i*halfny+j)<ratio)lattice(i*halfny+j)=-lij
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine
  subroutine update(lattice_b,lattice_w,randvals,invtemp,nx,ny)
    integer(int64),intent(in)::nx,ny
    integer(int8),intent(inout)::lattice_b(0:nx*ny/2-1),lattice_w(0:nx*ny/2-1)
    real(real32),intent(in)::randvals(0:nx*ny/2-1),invtemp
    call update_one(lattice_b,lattice_w,randvals,invtemp,nx,ny/2,.true.)
    call update_one(lattice_w,lattice_b,randvals,invtemp,nx,ny/2,.false.)
  end subroutine
  subroutine update_ref(lattice_b,lattice_w,randvals,invtemp,nx,ny)
    integer(int64),intent(in)::nx,ny
    integer(int8),intent(inout)::lattice_b(0:nx*ny/2-1),lattice_w(0:nx*ny/2-1)
    real(real32),intent(in)::randvals(0:nx*ny/2-1),invtemp
    integer::i,j,ipp,inn,jpp,jnn,joff
    integer(int8)::nn_sum,lij
    real(real32)::ratio
    do i=0,int(nx)-1;do j=0,int(ny/2)-1
      ipp=merge(i+1,0,i+1<int(nx));inn=merge(i-1,int(nx)-1,i-1>=0);jpp=merge(j+1,0,j+1<int(ny/2));jnn=merge(j-1,int(ny/2)-1,j-1>=0);joff=merge(jpp,jnn,mod(i,2)/=0)
      nn_sum=lattice_w(inn*ny/2+j)+lattice_w(i*ny/2+j)+lattice_w(ipp*ny/2+j)+lattice_w(i*ny/2+joff);lij=lattice_b(i*ny/2+j);ratio=exp(-2.0_real32*invtemp*real(nn_sum,real32)*real(lij,real32));if(randvals(i*ny/2+j)<ratio)lattice_b(i*ny/2+j)=-lij
    end do;end do
    do i=0,int(nx)-1;do j=0,int(ny/2)-1
      ipp=merge(i+1,0,i+1<int(nx));inn=merge(i-1,int(nx)-1,i-1>=0);jpp=merge(j+1,0,j+1<int(ny/2));jnn=merge(j-1,int(ny/2)-1,j-1>=0);joff=merge(jnn,jpp,mod(i,2)/=0)
      nn_sum=lattice_b(inn*ny/2+j)+lattice_b(i*ny/2+j)+lattice_b(ipp*ny/2+j)+lattice_b(i*ny/2+joff);lij=lattice_w(i*ny/2+j);ratio=exp(-2.0_real32*invtemp*real(nn_sum,real32)*real(lij,real32));if(randvals(i*ny/2+j)<ratio)lattice_w(i*ny/2+j)=-lij
    end do;end do
  end subroutine
end module
program ising
  use iso_c_binding,only:c_int
  use iso_fortran_env,only:int8,int64,real32,real64
  use ising_kernels
  implicit none
  interface
    subroutine c_srand(seed) bind(C,name='srand');import c_int;integer(c_int),value::seed;end subroutine
    function c_rand() bind(C,name='rand') result(v);import c_int;integer(c_int)::v;end function
  end interface
  integer(int64)::nx=5120,ny=5120,cells,t0,t1,rate
  integer::nwarmup=100,niters=1000,i,argc
  integer(c_int)::seedv=1234_c_int
  real(real32)::alpha=0.1_real32,invtemp
  real(real64)::duration
  real(real32),allocatable::randvals(:)
  integer(int8),allocatable::lb(:),lw(:),lbr(:),lwr(:)
  character(len=64)::arg;logical::ok
  argc=command_argument_count();i=1
  do while(i<=argc)
    call get_command_argument(i,arg)
    select case(trim(arg));case('-x','--lattice-n');i=i+1;call get_command_argument(i,arg);read(arg,*)nx;case('-y','--lattice-m');i=i+1;call get_command_argument(i,arg);read(arg,*)ny;case('-w','--nwarmup');i=i+1;call get_command_argument(i,arg);read(arg,*)nwarmup;case('-n','--niters');i=i+1;call get_command_argument(i,arg);read(arg,*)niters;case('-a','--alpha');i=i+1;call get_command_argument(i,arg);read(arg,*)alpha;case('-s','--seed');i=i+1;call get_command_argument(i,arg);read(arg,*)seedv;end select;i=i+1
  end do
  if(mod(nx,2_int64)/=0.or.mod(ny,2_int64)/=0)stop 1;invtemp=1.0_real32/(alpha*2.26918531421_real32);cells=nx*ny/2
  allocate(randvals(0:cells-1),lb(0:cells-1),lw(0:cells-1),lbr(0:cells-1),lwr(0:cells-1));call c_srand(seedv);do i=0,int(cells)-1;randvals(i)=real(c_rand(),real32)/2147483647.0_real32;end do
!$omp target enter data map(to:randvals(0:cells-1)) map(alloc:lb(0:cells-1),lw(0:cells-1))
  call init_spins(lb,randvals,nx,ny/2);call init_spins(lw,randvals,nx,ny/2);print '(a)','Starting warmup...';do i=1,nwarmup;call update(lb,lw,randvals,invtemp,nx,ny);end do;print '(a)','Starting trial iterations...';call system_clock(t0,rate);do i=1,niters;call update(lb,lw,randvals,invtemp,nx,ny);end do;call system_clock(t1);duration=real(t1-t0,real64)/rate
!$omp target exit data map(from:lb(0:cells-1),lw(0:cells-1)) map(delete:randvals(0:cells-1))
  print '(a)','REPORT:';print '(a,f0.6,a,f0.6)',' temperature: ',alpha,' * ',2.26918531421_real32;print '(a,f0.6,a)',' elapsed time: ',duration,' sec'
  print '(a)','Starting verification iterations ...';do i=0,int(cells)-1;lbr(i)=merge(-1_int8,1_int8,randvals(i)<0.5_real32);lwr(i)=lbr(i);end do;do i=1,nwarmup+niters;call update_ref(lbr,lwr,randvals,invtemp,nx,ny);end do;ok=all(lb==lbr).and.all(lw==lwr);print '(a)',merge('PASS','FAIL',ok);if(.not.ok)stop 2
end program
