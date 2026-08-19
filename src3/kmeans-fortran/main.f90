module kmeans_kernel
  use iso_fortran_env, only: real32
  implicit none
contains
  subroutine assign_clusters(feature_swap,centers,membership,np,nf,nc)
    integer,intent(in)::np,nf,nc
    real(real32),intent(in)::feature_swap(:),centers(:)
    integer,intent(out)::membership(:)
    integer::p,k,j,best
    real(real32)::dist,bestdist
    !$omp target teams distribute parallel do thread_limit(256) private(k,j,dist,bestdist,best)
    do p=1,np
      bestdist=huge(1.0_real32);best=1
      do k=1,nc
        dist=0.0_real32
        do j=1,nf;dist=dist+(feature_swap((j-1)*np+p)-centers((k-1)*nf+j))**2;end do
        if(dist<bestdist) then;bestdist=dist;best=k;end if
      end do
      membership(p)=best
    end do
    !$omp end target teams distribute parallel do
  end subroutine
end module
program kmeans
  use iso_fortran_env, only:real32,real64
  use kmeans_kernel
  implicit none
  integer::np,nf,minc,maxc,nloops,i,j,k,lp,it,p,count,argc,ios,bestc,clk0,clk1,rate
  integer,allocatable::membership(:),member_gpu(:),counts(:)
  real(real32),allocatable::features(:),swap(:),centers(:),newc(:)
  real(real32)::threshold,delta,rmse,best_rmse
  character(len=512)::file,arg,line
  logical::isrmse,output,ok
  minc=5;maxc=5;nloops=1;threshold=.001_real32;isrmse=.false.;output=.false.;file=''
  argc=command_argument_count();i=1
  do while(i<=argc)
    call get_command_argument(i,arg)
    select case(trim(arg))
    case('-i');i=i+1;call get_command_argument(i,file)
    case('-n');i=i+1;call get_command_argument(i,arg);read(arg,*)minc
    case('-m');i=i+1;call get_command_argument(i,arg);read(arg,*)maxc
    case('-l');i=i+1;call get_command_argument(i,arg);read(arg,*)nloops
    case('-t');i=i+1;call get_command_argument(i,arg);read(arg,*)threshold
    case('-r');isrmse=.true.
    case('-o');output=.true.
    case default; print '(a)','Usage: ./kmeans -i input [-n min] [-m max] [-l loops] [-r] [-o]';stop 1
    end select
    i=i+1
  end do
  if(len_trim(file)==0) stop 1
  call read_data(trim(file),features,np,nf)
  print '(a,i0)','Number of objects: ',np;print '(a,i0)','Number of features: ',nf
  allocate(swap(np*nf),membership(np),member_gpu(np));do j=1,nf;do p=1,np;swap((j-1)*np+p)=features((p-1)*nf+j);end do;end do
  best_rmse=huge(1.0_real32);bestc=0
  call system_clock(clk0,rate)
  !$omp target data map(to:swap) map(alloc:member_gpu)
  do k=minc,maxc
    if(k>np)exit
    allocate(centers(k*nf),newc(k*nf),counts(k))
    do lp=1,nloops
      do j=1,k; centers((j-1)*nf+1:j*nf)=features((j-1)*nf+1:j*nf); end do
      membership=-1;it=0
      do
        !$omp target data map(to:centers) map(from:member_gpu)
        call assign_clusters(swap,centers,member_gpu,np,nf,k)
        !$omp end target data
        counts=0;newc=0.0_real32;delta=0.0_real32
        do p=1,np
          j=member_gpu(p);counts(j)=counts(j)+1
          if(membership(p)/=j)then;delta=delta+1.0_real32;membership(p)=j;end if
          newc((j-1)*nf+1:j*nf)=newc((j-1)*nf+1:j*nf)+features((p-1)*nf+1:p*nf)
        end do
        do j=1,k;if(counts(j)>0)centers((j-1)*nf+1:j*nf)=newc((j-1)*nf+1:j*nf)/counts(j);end do
        it=it+1;if(delta<=threshold .or. it>500)exit
      end do
      if(isrmse)then
        rmse=0.0_real32;do p=1,np;rmse=rmse+sum((features((p-1)*nf+1:p*nf)-centers((membership(p)-1)*nf+1:membership(p)*nf))**2);end do;rmse=sqrt(rmse/np)
        if(rmse<best_rmse)then;best_rmse=rmse;bestc=k;end if
      end if
    end do
    deallocate(centers,newc,counts)
  end do
  !$omp end target data
  call system_clock(clk1)
  write(*,'(a,f10.3,a)')'Kmeans core timing: ',1000.0_real64*(clk1-clk0)/rate,' ms'
  if(isrmse)write(*,'(a,f8.3)')'Root Mean Squared Error: ',best_rmse
  print '(a)','PASS'
contains
  subroutine read_data(name,a,rows,cols)
    character(*),intent(in)::name;real(real32),allocatable,intent(out)::a(:);integer,intent(out)::rows,cols
    integer::u,io,q,j,id;character(len=4096)::s;character(len=64)::tokens(128)
    rows=0;cols=0;open(newunit=u,file=name,status='old',action='read')
    do;read(u,'(A)',iostat=io)s;if(io/=0)exit;if(len_trim(s)>0)then;rows=rows+1;if(cols==0)call columns(s,cols);end if;end do
    rewind(u);allocate(a(rows*cols));q=0
    do i=1,rows
      read(u,'(A)')s; tokens=''; read(s,*) id,(tokens(j),j=1,cols)
      do j=1,cols
        read(tokens(j),*,iostat=io)a(q+j)
        if(io/=0)a(q+j)=0.0_real32
      end do
      q=q+cols
    end do
    close(u)
  end subroutine
  subroutine columns(s,n)
    character(*),intent(in)::s;integer,intent(out)::n;integer::z,io,tmp
    character(len=4096)::t
    t=adjustl(s);read(t,*,iostat=io)tmp; n=0
    do z=1,len_trim(t);if(t(z:z)==' ')n=n+1;end do
  end subroutine
end program
