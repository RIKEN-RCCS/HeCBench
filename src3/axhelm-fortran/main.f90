module axhelm_impl
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  use omp_lib
  implicit none
  integer, parameter :: nq=8, np=512, nggeo=7, g00=1, g01=2, g02=3, g11=4, g12=5, g22=6, gwj=0
  integer(int64), parameter :: base24=16777216_int64, mult_hi=1502_int64, mult_lo=15525485_int64
  integer(int64) :: rng_hi=1193131_int64, rng_lo=13447950_int64
contains
  real(real64) function drand48_value() result(v)
    integer(int64) :: p, old_lo
    old_lo=rng_lo
    p=mult_lo*old_lo+11_int64
    rng_lo=modulo(p,base24)
    rng_hi=modulo(mult_hi*old_lo+mult_lo*rng_hi+p/base24,base24)
    v=real(rng_hi*base24+rng_lo,real64)/real(base24*base24,real64)
  end function drand48_value

  subroutine drand_alloc(a,n)
    integer,intent(in)::n
    real(real32),allocatable,intent(out)::a(:)
    integer::i
    allocate(a(0:n-1))
    do i=0,n-1; a(i)=real(drand48_value(),real32); end do
  end subroutine drand_alloc

  subroutine reference_element(e,lambda,ggeo,d,q,aq)
    integer,intent(in)::e
    real(real32),intent(in)::lambda,ggeo(0:),d(0:63),q(0:)
    real(real32),intent(inout)::aq(0:)
    real(real32)::gqr(0:7,0:7,0:7),gqs(0:7,0:7,0:7),gqt(0:7,0:7,0:7)
    real(real32)::qr,qs,qt,lap
    integer::i,j,k,n,id,gbase
    do k=0,7; do j=0,7; do i=0,7
      qr=0.0_real32; qs=0.0_real32; qt=0.0_real32
      do n=0,7
        qr=qr+d(n+i*8)*q(n+j*8+k*64)
        qs=qs+d(n+j*8)*q(i+n*8+k*64)
        qt=qt+d(n+k*8)*q(i+j*8+n*64)
      end do
      gbase=e*nggeo*np+i+j*8+k*64
      gqr(i,j,k)=ggeo(gbase+g00*np)*qr+ggeo(gbase+g01*np)*qs+ggeo(gbase+g02*np)*qt
      gqs(i,j,k)=ggeo(gbase+g01*np)*qr+ggeo(gbase+g11*np)*qs+ggeo(gbase+g12*np)*qt
      gqt(i,j,k)=ggeo(gbase+g02*np)*qr+ggeo(gbase+g12*np)*qs+ggeo(gbase+g22*np)*qt
    end do; end do; end do
    do k=0,7; do j=0,7; do i=0,7
      id=i+j*8+k*64; gbase=e*nggeo*np+id
      lap=lambda*ggeo(gbase+gwj*np)*q(id)
      do n=0,7
        lap=lap+d(i+n*8)*gqr(n,j,k)+d(j+n*8)*gqs(i,n,k)+d(k+n*8)*gqt(i,j,n)
      end do
      aq(id)=lap
    end do; end do; end do
  end subroutine reference_element

  subroutine reference_all(ndim,nelements,ggeo,d,q,aq)
    integer,intent(in)::ndim,nelements
    real(real32),intent(in)::ggeo(0:),d(0:63),q(0:)
    real(real32),intent(out)::aq(0:)
    integer::e,n,offset
    offset=nelements*np
    do n=0,ndim-1
      do e=0,nelements-1
        call reference_element(e,1.1_real32,ggeo,d,q(n*offset:),aq(n*offset:))
      end do
    end do
  end subroutine reference_all

  subroutine gpu_1d(nelements,offset,ggeo,d,lambda,q,aqd)
    integer,intent(in)::nelements,offset
    real(real32),intent(in)::ggeo(0:),d(0:63),lambda(0:),q(0:)
    real(real32),intent(inout)::aqd(0:)
    real(real32)::sd(0:63),sq(0:63),sgqr(0:63),sgqs(0:63)
    real(real32)::rqt,rgqt,rauk,rq(0:7),raq(0:7),rg00,rg01,rg02,rg11,rg12,rg22,rgwj,rl0,rl1,qr,qs
    integer::e,i,j,k,m,base,id,gbase
!$omp target teams num_teams(nelements) thread_limit(64) private(sd,sq,sgqr,sgqs)
!$omp parallel private(e,i,j,k,m,base,id,gbase,rqt,rgqt,rauk,rq,raq,rg00,rg01,rg02,rg11,rg12,rg22,rgwj,rl0,rl1,qr,qs) shared(sd,sq,sgqr,sgqs)
    e=omp_get_team_num(); j=omp_get_thread_num()/8; i=mod(omp_get_thread_num(),8)
    sd(j*8+i)=d(j*8+i); base=i+j*8+e*512
    do k=0,7; rq(k)=q(base+k*64); raq(k)=0.0_real32; end do
    do k=0,7
      id=e*512+k*64+j*8+i; gbase=e*nggeo*512+k*64+j*8+i
      rg00=ggeo(gbase+g00*512); rg01=ggeo(gbase+g01*512); rg02=ggeo(gbase+g02*512)
      rg11=ggeo(gbase+g11*512); rg12=ggeo(gbase+g12*512); rg22=ggeo(gbase+g22*512); rgwj=ggeo(gbase+gwj*512)
      rl0=lambda(id); rl1=lambda(id+offset)
!$omp barrier
      sq(j*8+i)=rq(k); rqt=0.0_real32
      do m=0,7; rqt=rqt+sd(k*8+m)*rq(m); end do
!$omp barrier
      qr=0.0_real32; qs=0.0_real32
      do m=0,7
        qr=qr+sd(i*8+m)*sq(j*8+m); qs=qs+sd(j*8+m)*sq(m*8+i)
      end do
      sgqs(j*8+i)=rl0*(rg01*qr+rg11*qs+rg12*rqt); sgqr(j*8+i)=rl0*(rg00*qr+rg01*qs+rg02*rqt)
      rgqt=rl0*(rg02*qr+rg12*qs+rg22*rqt); rauk=rgwj*rl1*rq(k)
!$omp barrier
      do m=0,7
        rauk=rauk+sd(m*8+j)*sgqs(m*8+i); raq(m)=raq(m)+sd(k*8+m)*rgqt; rauk=rauk+sd(m*8+i)*sgqr(j*8+m)
      end do
      raq(k)=raq(k)+rauk
!$omp barrier
    end do
    do k=0,7; id=e*512+k*64+j*8+i; aqd(id)=raq(k); end do
!$omp end parallel
!$omp end target teams
  end subroutine gpu_1d

  subroutine gpu_3d(nelements,offset,ggeo,d,lambda,q,aqd)
    integer,intent(in)::nelements,offset
    real(real32),intent(in)::ggeo(0:),d(0:63),lambda(0:),q(0:)
    real(real32),intent(inout)::aqd(0:)
    real(real32)::sd(0:63),su(0:63),sv(0:63),sw(0:63),sgur(0:63),sgus(0:63),sgvr(0:63),sgvs(0:63),sgwr(0:63),sgws(0:63)
    real(real32)::rut,rvt,rwt,ru(0:7),rv(0:7),rw(0:7),rau(0:7),rav(0:7),raw(0:7),rg00,rg01,rg02,rg11,rg12,rg22,rgwj,rl0,rl1,ur,us,vr,vs,wr,ws,aut,avt,awt
    integer::e,i,j,k,m,base,id,gbase
!$omp target teams num_teams(nelements) thread_limit(64) private(sd,su,sv,sw,sgur,sgus,sgvr,sgvs,sgwr,sgws)
!$omp parallel private(e,i,j,k,m,base,id,gbase,rut,rvt,rwt,ru,rv,rw,rau,rav,raw,rg00,rg01,rg02,rg11,rg12,rg22,rgwj,rl0,rl1,ur,us,vr,vs,wr,ws,aut,avt,awt) shared(sd,su,sv,sw,sgur,sgus,sgvr,sgvs,sgwr,sgws)
    e=omp_get_team_num(); j=omp_get_thread_num()/8; i=mod(omp_get_thread_num(),8); sd(j*8+i)=d(j*8+i); base=i+j*8+e*512
    do k=0,7
      ru(k)=q(base+k*64); rv(k)=q(base+k*64+offset); rw(k)=q(base+k*64+2*offset)
      rau(k)=0.0_real32; rav(k)=0.0_real32; raw(k)=0.0_real32
    end do
    do k=0,7
      id=e*512+k*64+j*8+i; gbase=e*nggeo*512+k*64+j*8+i
      rg00=ggeo(gbase+g00*512); rg01=ggeo(gbase+g01*512); rg02=ggeo(gbase+g02*512); rg11=ggeo(gbase+g11*512); rg12=ggeo(gbase+g12*512); rg22=ggeo(gbase+g22*512); rgwj=ggeo(gbase+gwj*512); rl0=lambda(id); rl1=lambda(id+offset)
!$omp barrier
      su(j*8+i)=ru(k); sv(j*8+i)=rv(k); sw(j*8+i)=rw(k); rut=0.0_real32; rvt=0.0_real32; rwt=0.0_real32
      do m=0,7; rut=rut+sd(k*8+m)*ru(m); rvt=rvt+sd(k*8+m)*rv(m); rwt=rwt+sd(k*8+m)*rw(m); end do
!$omp barrier
      ur=0.0_real32; us=0.0_real32; vr=0.0_real32; vs=0.0_real32; wr=0.0_real32; ws=0.0_real32
      do m=0,7
        ur=ur+sd(i*8+m)*su(j*8+m); us=us+sd(j*8+m)*su(m*8+i); vr=vr+sd(i*8+m)*sv(j*8+m); vs=vs+sd(j*8+m)*sv(m*8+i); wr=wr+sd(i*8+m)*sw(j*8+m); ws=ws+sd(j*8+m)*sw(m*8+i)
      end do
      sgur(j*8+i)=rl0*(rg00*ur+rg01*us+rg02*rut); sgvr(j*8+i)=rl0*(rg00*vr+rg01*vs+rg02*rvt); sgwr(j*8+i)=rl0*(rg00*wr+rg01*ws+rg02*rwt)
      sgus(j*8+i)=rl0*(rg01*ur+rg11*us+rg12*rut); sgvs(j*8+i)=rl0*(rg01*vr+rg11*vs+rg12*rvt); sgws(j*8+i)=rl0*(rg01*wr+rg11*ws+rg12*rwt)
      rut=rl0*(rg02*ur+rg12*us+rg22*rut); rvt=rl0*(rg02*vr+rg12*vs+rg22*rvt); rwt=rl0*(rg02*wr+rg12*ws+rg22*rwt)
      rau(k)=rau(k)+rgwj*rl1*ru(k); rav(k)=rav(k)+rgwj*rl1*rv(k); raw(k)=raw(k)+rgwj*rl1*rw(k)
!$omp barrier
      aut=0.0_real32; avt=0.0_real32; awt=0.0_real32
      do m=0,7
        aut=aut+sd(m*8+i)*sgur(j*8+m)+sd(m*8+j)*sgus(m*8+i); avt=avt+sd(m*8+i)*sgvr(j*8+m)+sd(m*8+j)*sgvs(m*8+i); awt=awt+sd(m*8+i)*sgwr(j*8+m)+sd(m*8+j)*sgws(m*8+i)
        rau(m)=rau(m)+sd(k*8+m)*rut; rav(m)=rav(m)+sd(k*8+m)*rvt; raw(m)=raw(m)+sd(k*8+m)*rwt
      end do
      rau(k)=rau(k)+aut; rav(k)=rav(k)+avt; raw(k)=raw(k)+awt
    end do
    do k=0,7; id=e*512+k*64+j*8+i; aqd(id)=rau(k); aqd(id+offset)=rav(k); aqd(id+2*offset)=raw(k); end do
!$omp end parallel
!$omp end target teams
  end subroutine gpu_3d
end module axhelm_impl

program axhelm
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  use mesh_basis
  use axhelm_impl
  implicit none
  character(len=64)::arg
  integer::ndim,nelements,ntests,ios,offset,n,rep
  integer(int64)::tick,rate,stop_tick
  real(real64)::elapsed,bw,gflops,gdof,maxdiff,diff,flops
  real(real32)::rv(0:7),wv(0:7),dmat(0:7,0:7),dflat(0:63)
  real(real32),allocatable::ggeo(:),q(:),aq(:),aqd(:),lambda(:)
  if(command_argument_count()<3) then; print '(A)','Usage: ./axhelm Ndim numElements [nRepetitions]'; stop 1; end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) ndim
  call get_command_argument(2,arg); read(arg,*,iostat=ios) nelements
  call get_command_argument(3,arg); read(arg,*,iostat=ios) ntests
  if(ios/=0 .or. (ndim/=1 .and. ndim<=1)) error stop 'invalid arguments'
  offset=nelements*np
  call mesh_jacobi_gq(7,rv,wv); call mesh_dmatrix_1d(7,rv,dmat)
  do n=0,7; do rep=0,7; dflat(n*8+rep)=dmat(n,rep); end do; end do
  print '(A,I0,A)','word size: ',storage_size(0.0_real32)/8,' bytes'
  call drand_alloc(ggeo,np*nelements*nggeo); call drand_alloc(q,ndim*np*nelements); call drand_alloc(aq,ndim*np*nelements); call drand_alloc(aqd,ndim*np*nelements)
  allocate(lambda(0:2*offset-1)); lambda(0:offset-1)=1.0_real32; lambda(offset:2*offset-1)=1.1_real32
  call reference_all(ndim,nelements,ggeo,dflat,q,aq)
  call system_clock(tick,rate)
!$omp target data map(to:ggeo(0:np*nelements*nggeo-1),q(0:ndim*np*nelements-1),dflat(0:63),lambda(0:2*offset-1)) map(from:aqd(0:ndim*np*nelements-1))
  do rep=1,ntests
    if(ndim>1) then
      call gpu_3d(nelements,offset,ggeo,dflat,lambda,q,aqd)
    else
      call gpu_1d(nelements,offset,ggeo,dflat,lambda,q,aqd)
    end if
  end do
!$omp end target data
  call system_clock(stop_tick); elapsed=real(stop_tick-tick,real64)*1.0e9_real64/real(rate,real64)/real(ntests,real64)
  maxdiff=0.0_real64
  do n=0,ndim*np*nelements-1; diff=abs(real(aqd(n),real64)-real(aq(n),real64)); if(diff>maxdiff) maxdiff=diff; end do
  print '(A,ES14.6)','Correctness check: maxError = ',maxdiff
  gdof=real(ndim*7*7*7*nelements,real64)/elapsed; bw=real((ndim*2*np+7*np+2*np)*4*nelements,real64)/elapsed
  flops=real(ndim*np*12*nq,real64); if(ndim==1) flops=flops+real(22*np,real64); if(ndim==3) flops=flops+real(69*np,real64)
  gflops=flops*real(nelements,real64)/elapsed
  print '(A,I0,A,I0,A,I0,A,I0,A,ES14.6,A,ES14.6,A,ES14.6,A,ES14.6)', ' NRepetitions=',ntests,' Ndim=',ndim,' N=',7,' Nelements=',nelements,' elapsed time=',elapsed,' GDOF/s=',gdof,' GB/s=',bw,' GFLOPS/s=',gflops
  deallocate(ggeo,q,aq,aqd,lambda)
end program axhelm
