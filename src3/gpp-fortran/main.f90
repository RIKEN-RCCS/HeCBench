program gpp
  use, intrinsic :: iso_fortran_env, only : int32, real64
  use omp_lib, only : omp_get_wtime
  implicit none
  integer(int32),parameter::nstart=0_int32,nend=3_int32
  integer(int32)::bands,nvband,ncouls,nodes,ngpown,i,n1,ig,iw,my_igp,indigp,igp,iteration
  integer(int32),allocatable::inv_index(:),indinv(:)
  real(real64),allocatable::vcoul(:),wx(:),ach_re(:),ach_im(:)
  complex(real64),allocatable::aqsm(:),aqsn(:),ieps(:),wtilde(:),achtemp(:)
  complex(real64)::expr,store,wdiff,delw,sch
  real(real64)::ar0,ar1,ar2,ai0,ai1,ai2,start,total,elapsed
  real(real64)::local_re(0:2),local_im(0:2)
  character(len=64)::arg
  call parse_arguments(bands,nvband,ncouls,nodes)
  ngpown=ncouls/nodes
  write(*,'(a,i0,a)')'Sizeof(CustomComplex<dataType> = ',16,' bytes'
  write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0)')'number_bands = ',bands,' nvband = ',nvband,' ncouls = ',ncouls,' nodes_per_group  = ',nodes,' ngpown = ',ngpown,' nend = ',nend
  allocate(aqsm(0:bands*ncouls-1),aqsn(0:bands*ncouls-1),ieps(0:ngpown*ncouls-1),wtilde(0:ngpown*ncouls-1),vcoul(0:ncouls-1),inv_index(0:ngpown-1),indinv(0:ncouls),ach_re(0:2),ach_im(0:2),wx(0:2),achtemp(0:2))
  expr=cmplx(0.025_real64,0.025_real64,real64)
  aqsm=expr;aqsn=expr;ieps=expr;wtilde=expr
  do i=0,ncouls-1;vcoul(i)=real(i,real64)*0.025_real64;indinv(i)=i;end do
  indinv(ncouls)=ncouls-1;do i=0,ngpown-1;inv_index(i)=(i+1)*ncouls/ngpown;end do
  do iw=0,2;ach_re(iw)=0.0_real64;ach_im(iw)=0.0_real64;wx(iw)=10.0_real64-6.0_real64+real((iw+1)-2,real64);if(wx(iw)<1.0e-6_real64)wx(iw)=1.0e-6_real64;end do
!$omp target data map(to:aqsm(0:bands*ncouls-1),vcoul(0:ncouls-1),inv_index(0:ngpown-1),indinv(0:ncouls),aqsn(0:bands*ncouls-1),ieps(0:ngpown*ncouls-1),wx(0:2),wtilde(0:ngpown*ncouls-1))
  total=0.0_real64
  do iteration=1,10
    ar0=0.0_real64;ar1=0.0_real64;ar2=0.0_real64;ai0=0.0_real64;ai1=0.0_real64;ai2=0.0_real64;start=omp_get_wtime()
!$omp target teams distribute parallel do collapse(2) reduction(+:ar0,ar1,ar2,ai0,ai1,ai2) private(indigp,igp,local_re,local_im,store,wdiff,delw,sch,ig,iw)
    do my_igp=0,ngpown-1
      do n1=0,bands-1
        indigp=inv_index(my_igp);igp=indinv(indigp);local_re=0.0_real64;local_im=0.0_real64
        store=conjg(aqsm(n1*ncouls+igp))*aqsn(n1*ncouls+igp)*0.5_real64*vcoul(igp)
        do ig=0,ncouls-1
          do iw=0,2
            wdiff=cmplx(wx(iw),0.0_real64,real64)-wtilde(my_igp*ncouls+ig)
            delw=wtilde(my_igp*ncouls+ig)*conjg(wdiff)*(1.0_real64/real(wdiff*conjg(wdiff),real64))
            sch=delw*ieps(my_igp*ncouls+ig)*store;local_re(iw)=local_re(iw)+real(sch,real64);local_im(iw)=local_im(iw)+aimag(sch)
          end do
        end do
        ar0=ar0+local_re(0);ar1=ar1+local_re(1);ar2=ar2+local_re(2);ai0=ai0+local_im(0);ai1=ai1+local_im(1);ai2=ai2+local_im(2)
      end do
    end do
!$omp end target teams distribute parallel do
    total=total+omp_get_wtime()-start
  end do
  write(*,'(a,f0.6,a)')'Average kernel execution time ',total/10.0_real64,' (s)'
  ach_re=(/ar0,ar1,ar2/);ach_im=(/ai0,ai1,ai2/);do iw=0,2;achtemp(iw)=cmplx(ach_re(iw),ach_im(iw),real64);end do
  call correctness(bands,ncouls,achtemp(0))
  write(*,'(a)')' Final achtemp';write(*,'(a,f0.6,a,f0.6,a)')'( ',real(achtemp(0),real64),', ',aimag(achtemp(0)),' )'
!$omp end target data
  deallocate(aqsm,aqsn,ieps,wtilde,vcoul,inv_index,indinv,ach_re,ach_im,wx,achtemp)
contains
  subroutine parse_arguments(nb,nv,nc,np)
    integer(int32),intent(out)::nb,nv,nc,np;character(len=64)::a1,a2,a3,a4
    if(command_argument_count()==0)then;nb=512;nv=2;nc=512;np=20
    else if(command_argument_count()==1)then;call get_command_argument(1,a1);if(trim(a1)=='benchmark')then;nb=512;nv=2;nc=32768;np=20;else if(trim(a1)=='test')then;nb=512;nv=2;nc=512;np=20;else;write(*,'(a)')'Usage: ./main <test or benchmark>';stop 1;end if
    else if(command_argument_count()==4)then;call get_command_argument(1,a1);call get_command_argument(2,a2);call get_command_argument(3,a3);call get_command_argument(4,a4);read(a1,*)nb;read(a2,*)nv;read(a3,*)nc;read(a4,*)np
    else;write(*,'(a)')' ./main <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group>';stop 1;end if
  end subroutine parse_arguments
  subroutine correctness(nb,nc,result)
    integer(int32),intent(in)::nb,nc;complex(real64),intent(in)::result;real(real64)::rd,id
    if(nb==512.and.nc==32768)then;rd=real(result,real64)-(-24852.551547_real64);id=aimag(result)-2957453.638101_real64;if(rd<1.0e-5_real64.and.id<1.0e-5_real64)then;write(*,'(a)')'Benchmark result: SUCCESS';else;write(*,'(a)')'Benchmark result: FAILURE';end if
    else;rd=real(result,real64)-(-0.096066_real64);id=aimag(result)-11.431852_real64;if(rd<1.0e-5_real64.and.id<1.0e-5_real64)then;write(*,'(a)')'Test result: SUCCESS';else;write(*,'(a)')'Test result: FAILURE';end if
    end if
  end subroutine correctness
end program gpp
