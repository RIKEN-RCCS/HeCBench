program bezier_surface
  use omp_lib
  implicit none
  type :: xyz
    real :: x,y,z
  end type
  integer :: ni,nj,ri,rj,work_group,argc,i,ios
  character(len=1024)::filename,arg
  type(xyz),allocatable::input(:),cpu_out(:),gpu_out(:)
  real(8)::t0,t1
  ni=3; nj=3; ri=300; rj=300; work_group=256; filename='../../src/bezier-surface-omp/input/control.txt'; argc=command_argument_count(); i=1
  do while(i<=argc)
    call get_command_argument(i,arg)
    select case(trim(arg))
    case('-h'); print *,'Usage: ./main [-g G] [-f file] [-m N] [-n R]'; stop
    case('-g'); i=i+1; call get_command_argument(i,arg); read(arg,*)work_group
    case('-f'); i=i+1; call get_command_argument(i,filename)
    case('-m'); i=i+1; call get_command_argument(i,arg); read(arg,*)ni; nj=ni
    case('-n'); i=i+1; call get_command_argument(i,arg); read(arg,*)ri; rj=ri
    case default; print *,'Unrecognized option!'; stop 1
    end select
    i=i+1
  end do
  allocate(input(0:(ni+1)*(nj+1)-1),cpu_out(0:ri*rj-1),gpu_out(0:ri*rj-1))
  call read_input(input,ni,nj,trim(filename),ios)
  if(ios/=0) then; print *,'Error opening file'; stop 1; end if
  print *,'Read data from file ',trim(filename)
  t0=omp_get_wtime(); call bezier_cpu(input,cpu_out,ni,nj,ri,rj); t1=omp_get_wtime()
  print '(a,f12.3,a)','host execution time: ',(t1-t0)*1.d3,' ms'
  !$omp target data map(to:input(0:(ni+1)*(nj+1)-1)) map(from:gpu_out(0:ri*rj-1))
  t0=omp_get_wtime()
  !$omp target teams distribute parallel do thread_limit(256) private(i)
  do i=0,ri-1
    call bezier_row(input,gpu_out,ni,nj,ri,rj,i)
  end do
  !$omp end target teams distribute parallel do
  t1=omp_get_wtime()
  print '(a,f12.3,a)','kernel execution time: ',(t1-t0)*1.d3,' ms'
  !$omp end target data
  if(compare_output(gpu_out,cpu_out)) then; print *,'PASS'; else; print *,'FAIL'; end if
contains
  subroutine read_input(a,ni,nj,name,ios)
    type(xyz),intent(out)::a(0:); integer,intent(in)::ni,nj; character(*),intent(in)::name; integer,intent(out)::ios
    type(xyz)::v(0:9999); integer::u,k,ic,i,j
    open(newunit=u,file=name,status='old',action='read',iostat=ios); if(ios/=0)return
    ic=0; do; read(u,*,iostat=ios)v(ic)%x,v(ic)%y,v(ic)%z; if(ios/=0)exit; ic=ic+1; end do; close(u); ios=0; k=0
    do i=0,ni; do j=0,nj; a(i*(nj+1)+j)=v(k); k=modulo(k+1,16); end do; end do
  end subroutine
  pure real function blend(k,mu,n)
    integer,intent(in)::k,n; real,intent(in)::mu; integer::nn,kn,nkn
    blend=1.; nn=n; kn=k; nkn=n-k
    do while(nn>=1); blend=blend*real(nn); nn=nn-1; if(kn>1)then;blend=blend/real(kn);kn=kn-1;end if; if(nkn>1)then;blend=blend/real(nkn);nkn=nkn-1;end if; end do
    if(k>0)blend=blend*mu**k; if(n-k>0)blend=blend*(1.-mu)**(n-k)
  end function
  subroutine bezier_row(inp,out,ni,nj,ri,rj,row)
    type(xyz),intent(in)::inp(0:); type(xyz),intent(inout)::out(0:); integer,intent(in)::ni,nj,ri,rj,row
    integer::j,ki,kj; real::mui,muj,bi,bj; type(xyz)::value
    mui=real(row)/real(ri-1)
    do j=0,rj-1
      muj=real(j)/real(rj-1); value%x=0.;value%y=0.;value%z=0.
      do ki=0,ni; bi=blend(ki,mui,ni); do kj=0,nj; bj=blend(kj,muj,nj); value%x=value%x+inp(ki*(nj+1)+kj)%x*bi*bj; value%y=value%y+inp(ki*(nj+1)+kj)%y*bi*bj; value%z=value%z+inp(ki*(nj+1)+kj)%z*bi*bj; end do; end do
      out(row*rj+j)=value
    end do
  end subroutine
  subroutine bezier_cpu(inp,out,ni,nj,ri,rj)
    type(xyz),intent(in)::inp(0:);type(xyz),intent(out)::out(0:);integer,intent(in)::ni,nj,ri,rj;integer::row
    do row=0,ri-1; call bezier_row(inp,out,ni,nj,ri,rj,row); end do
  end subroutine
  logical function compare_output(a,b)
    type(xyz),intent(in)::a(0:),b(0:); integer::q; real(8)::delta,ref
    delta=0.d0;ref=0.d0;do q=0,size(a)-1;delta=delta+abs(real(a(q)%x-b(q)%x,8))+abs(real(a(q)%y-b(q)%y,8))+abs(real(a(q)%z-b(q)%z,8));ref=ref+abs(real(b(q)%x,8))+abs(real(b(q)%y,8))+abs(real(b(q)%z,8));end do
    compare_output=(delta/ref<1.d-6)
  end function
end program
