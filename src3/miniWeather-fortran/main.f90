module miniweather_mod
  use iso_fortran_env, only: real64
  implicit none
  real(real64), parameter :: pi=3.14159265358979323846264338327_real64, grav=9.8_real64, cp=1004.0_real64, cv=717.0_real64
  real(real64), parameter :: rd=287.0_real64, p0=1.0e5_real64, c0=27.5629410929725921310572974482_real64
  real(real64), parameter :: gamm=1.40027894002789400278940027894_real64, xlen=2.0e4_real64, zlen=1.0e4_real64
  real(real64), parameter :: hv_beta=0.25_real64, cfl=1.50_real64, max_speed=450.0_real64
  integer, parameter :: hs=2, sten_size=4, NUM_VARS=4, ID_DENS=0, ID_UMOM=1, ID_WMOM=2, ID_RHOT=3, DIR_X=1, DIR_Z=2
  integer, parameter :: DATA_SPEC_THERMAL=2
  integer, parameter :: NX=400, NZ=200, SIM_TIME=600
  integer :: nx, nz, nx_glob, nz_glob, nranks, myrank, left_rank, right_rank, masterproc, direction_switch=1
  real(real64) :: sim_time, dt, dx, dz, etime, mass0, te0, mass, te
  real(real64), allocatable :: hy_dens_cell(:), hy_dens_theta_cell(:), hy_dens_int(:), hy_dens_theta_int(:), hy_pressure_int(:)
contains
  pure integer function state_idx(i,k,ll)
    integer,intent(in)::i,k,ll
    state_idx = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs
  end function state_idx
  pure integer function flux_idx(i,k,ll)
    integer,intent(in)::i,k,ll
    flux_idx = ll*(nz+1)*(nx+1) + k*(nx+1) + i
  end function flux_idx
  pure integer function tend_idx(i,k,ll)
    integer,intent(in)::i,k,ll
    tend_idx = ll*nz*nx + k*nx + i
  end function tend_idx
  pure real(real64) function dmin(a,b)
    real(real64),intent(in)::a,b
    dmin = min(a,b)
  end function dmin
  subroutine hydro_const_theta(z,r,t)
    real(real64),intent(in)::z
    real(real64),intent(out)::r,t
    real(real64)::theta0, exner, p
    theta0 = 300.0_real64
    exner = 1.0_real64 - grav*z/(cp*theta0)
    p = p0 * exner**(cp/rd)
    r = p / (rd*theta0*exner)
    t = theta0
  end subroutine hydro_const_theta
  pure real(real64) function sample_ellipse_cosine(x,z,amp,x0,z0,xrad,zrad)
    real(real64),intent(in)::x,z,amp,x0,z0,xrad,zrad
    real(real64)::dist
    dist = sqrt(((x-x0)/xrad)**2 + ((z-z0)/zrad)**2) * pi / 2.0_real64
    if (dist <= pi/2.0_real64) then
      sample_ellipse_cosine = amp * cos(dist)**2
    else
      sample_ellipse_cosine = 0.0_real64
    end if
  end function sample_ellipse_cosine
  subroutine thermal(x,z,r,u,w,t,hr,ht)
    real(real64),intent(in)::x,z
    real(real64),intent(out)::r,u,w,t,hr,ht
    call hydro_const_theta(z,hr,ht)
    r = 0.0_real64; u = 0.0_real64; w = 0.0_real64
    t = sample_ellipse_cosine(x,z,3.0_real64,xlen/2.0_real64,2000.0_real64,2000.0_real64,2000.0_real64)
  end subroutine thermal
  subroutine init_state(state,state_tmp,flux,tend)
    real(real64),allocatable,intent(out)::state(:),state_tmp(:),flux(:),tend(:)
    integer :: i,k,ll
    real(real64)::x,z,r,u,w,t,hr,ht
    nx_glob=NX; nz_glob=NZ; sim_time=real(SIM_TIME,real64); nx=nx_glob; nz=nz_glob
    nranks=1; myrank=0; left_rank=0; right_rank=0; masterproc=1
    dx=xlen/nx_glob; dz=zlen/nz_glob; dt=dmin(dx,dz)*cfl/max_speed; etime=0.0_real64
    allocate(state(0:(nz+2*hs)*(nx+2*hs)*NUM_VARS-1),state_tmp(0:(nz+2*hs)*(nx+2*hs)*NUM_VARS-1))
    allocate(flux(0:(nz+1)*(nx+1)*NUM_VARS-1),tend(0:nz*nx*NUM_VARS-1))
    allocate(hy_dens_cell(0:nz+2*hs-1),hy_dens_theta_cell(0:nz+2*hs-1),hy_dens_int(0:nz),hy_dens_theta_int(0:nz),hy_pressure_int(0:nz))
    state=0.0_real64; state_tmp=0.0_real64; flux=0.0_real64; tend=0.0_real64
    do k=0,nz+2*hs-1
      z = (real(k-hs,real64)+0.5_real64)*dz
      call hydro_const_theta(z,hy_dens_cell(k),hy_dens_theta_cell(k))
      hy_dens_theta_cell(k)=hy_dens_cell(k)*hy_dens_theta_cell(k)
    end do
    do k=0,nz
      z=real(k,real64)*dz
      call hydro_const_theta(z,hy_dens_int(k),hy_dens_theta_int(k))
      hy_dens_theta_int(k)=hy_dens_int(k)*hy_dens_theta_int(k)
      hy_pressure_int(k)=c0*(hy_dens_theta_int(k))**gamm
    end do
    do k=0,nz-1; do i=0,nx-1
      x=(real(i,real64)+0.5_real64)*dx; z=(real(k,real64)+0.5_real64)*dz
      call thermal(x,z,r,u,w,t,hr,ht)
      state(state_idx(i,k,ID_DENS))=r+hr
      state(state_idx(i,k,ID_UMOM))=(r+hr)*u
      state(state_idx(i,k,ID_WMOM))=(r+hr)*w
      state(state_idx(i,k,ID_RHOT))=(r+hr)*(t+ht)
    end do; end do
    state_tmp=state
    print '(a,i0,1x,i0)', 'nx_glob, nz_glob: ', nx_glob, nz_glob
    print '(a,f0.6,1x,f0.6)', 'dx,dz: ', dx, dz
    print '(a,f0.6)', 'dt: ', dt
  end subroutine init_state
end module miniweather_mod

program main
  use omp_lib
  use miniweather_mod
  implicit none
  real(real64), allocatable :: state(:), state_tmp(:), flux(:), tend(:), sendbuf_l(:), sendbuf_r(:), recvbuf_l(:), recvbuf_r(:)
  real(8) :: c_start, c_time, d_mass, d_te
  call init_state(state,state_tmp,flux,tend)
  allocate(sendbuf_l(0:hs*nz*NUM_VARS-1),sendbuf_r(0:hs*nz*NUM_VARS-1),recvbuf_l(0:hs*nz*NUM_VARS-1),recvbuf_r(0:hs*nz*NUM_VARS-1))
!$omp target data map(to:state_tmp(0:(nz+2*hs)*(nx+2*hs)*NUM_VARS-1),state(0:(nz+2*hs)*(nx+2*hs)*NUM_VARS-1),hy_dens_cell(0:nz+2*hs-1),hy_dens_theta_cell(0:nz+2*hs-1),hy_dens_int(0:nz),hy_dens_theta_int(0:nz),hy_pressure_int(0:nz)) map(alloc:flux(0:(nz+1)*(nx+1)*NUM_VARS-1),tend(0:nz*nx*NUM_VARS-1),sendbuf_l(0:hs*nz*NUM_VARS-1),sendbuf_r(0:hs*nz*NUM_VARS-1),recvbuf_l(0:hs*nz*NUM_VARS-1),recvbuf_r(0:hs*nz*NUM_VARS-1))
  call reductions(state,mass0,te0)
  c_start=omp_get_wtime()
  do while (etime < sim_time)
    if (etime + dt > sim_time) dt = sim_time - etime
    call perform_timestep(state,state_tmp,flux,tend,dt)
    etime = etime + dt
  end do
  c_time=omp_get_wtime()-c_start
  if (masterproc /= 0) print '(a,f0.6,a)', 'Total main time step loop: ', c_time, ' sec'
  call reductions(state,mass,te)
!$omp end target data
  d_mass=(mass-mass0)/mass0; d_te=(te-te0)/te0
  print '(a,es12.5)', 'd_mass: ', d_mass
  print '(a,es12.5)', 'd_te:   ', d_te
  print '(a)', merge('PASS','FAIL',abs(d_mass)<1.0e-8_real64 .and. abs(d_te)<1.0e-7_real64)
contains
  subroutine perform_timestep(state,state_tmp,flux,tend,dtloc)
    real(real64),intent(inout)::state(0:),state_tmp(0:),flux(0:),tend(0:)
    real(real64),intent(in)::dtloc
    if (direction_switch /= 0) then
      call semi_discrete_step(state,state,state_tmp,dtloc/3.0_real64,DIR_X,flux,tend)
      call semi_discrete_step(state,state_tmp,state_tmp,dtloc/2.0_real64,DIR_X,flux,tend)
      call semi_discrete_step(state,state_tmp,state,dtloc,DIR_X,flux,tend)
      call semi_discrete_step(state,state,state_tmp,dtloc/3.0_real64,DIR_Z,flux,tend)
      call semi_discrete_step(state,state_tmp,state_tmp,dtloc/2.0_real64,DIR_Z,flux,tend)
      call semi_discrete_step(state,state_tmp,state,dtloc,DIR_Z,flux,tend)
      direction_switch=0
    else
      call semi_discrete_step(state,state,state_tmp,dtloc/3.0_real64,DIR_Z,flux,tend)
      call semi_discrete_step(state,state_tmp,state_tmp,dtloc/2.0_real64,DIR_Z,flux,tend)
      call semi_discrete_step(state,state_tmp,state,dtloc,DIR_Z,flux,tend)
      call semi_discrete_step(state,state,state_tmp,dtloc/3.0_real64,DIR_X,flux,tend)
      call semi_discrete_step(state,state_tmp,state_tmp,dtloc/2.0_real64,DIR_X,flux,tend)
      call semi_discrete_step(state,state_tmp,state,dtloc,DIR_X,flux,tend)
      direction_switch=1
    end if
  end subroutine perform_timestep
  subroutine semi_discrete_step(state_init,state_forcing,state_out,dtloc,dir,flux,tend)
    real(real64),intent(in)::state_init(0:),state_forcing(0:)
    real(real64),intent(inout)::state_out(0:),flux(0:),tend(0:)
    real(real64),intent(in)::dtloc
    integer,intent(in)::dir
    integer::i,k,ll,inds,indt
    if (dir==DIR_X) then
      call set_halo_values_x(state_forcing)
      call compute_tendencies_x(state_forcing,flux,tend)
    else
      call set_halo_values_z(state_forcing)
      call compute_tendencies_z(state_forcing,flux,tend)
    end if
!$omp target teams distribute parallel do collapse(3) private(inds,indt)
    do ll=0,NUM_VARS-1; do k=0,nz-1; do i=0,nx-1
      inds=state_idx(i,k,ll); indt=tend_idx(i,k,ll)
      state_out(inds)=state_init(inds)+dtloc*tend(indt)
    end do; end do; end do
!$omp end target teams distribute parallel do
  end subroutine semi_discrete_step
  subroutine compute_tendencies_x(state,flux,tend)
    real(real64),intent(in)::state(0:)
    real(real64),intent(inout)::flux(0:),tend(0:)
    integer::i,k,ll,s,inds,indf1,indf2,indt
    real(real64)::stencil(0:3),vals(0:NUM_VARS-1),d3(0:NUM_VARS-1),r,u,w,t,p,hv_coef
    hv_coef=-hv_beta*dx/(16.0_real64*dt)
!$omp target teams distribute parallel do collapse(2) private(ll,s,inds,stencil,vals,d3,r,u,w,t,p)
    do k=0,nz-1; do i=0,nx
      do ll=0,NUM_VARS-1
        do s=0,sten_size-1; inds=ll*(nz+2*hs)*(nx+2*hs)+(k+hs)*(nx+2*hs)+i+s; stencil(s)=state(inds); end do
        vals(ll)=(-stencil(0)+7.0_real64*stencil(1)+7.0_real64*stencil(2)-stencil(3))/12.0_real64
        d3(ll)=(-stencil(0)+3.0_real64*stencil(1)-3.0_real64*stencil(2)+stencil(3))
      end do
      r=vals(ID_DENS); u=vals(ID_UMOM)/r; w=vals(ID_WMOM)/r; t=vals(ID_RHOT)/r; p=c0*(r*t)**gamm
      flux(flux_idx(i,k,ID_DENS))=r*u-hv_coef*d3(ID_DENS)
      flux(flux_idx(i,k,ID_UMOM))=r*u*u+p-hv_coef*d3(ID_UMOM)
      flux(flux_idx(i,k,ID_WMOM))=r*u*w-hv_coef*d3(ID_WMOM)
      flux(flux_idx(i,k,ID_RHOT))=r*u*t-hv_coef*d3(ID_RHOT)
    end do; end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(3) private(indt,indf1,indf2)
    do ll=0,NUM_VARS-1; do k=0,nz-1; do i=0,nx-1
      indt=tend_idx(i,k,ll); indf1=flux_idx(i,k,ll); indf2=flux_idx(i+1,k,ll)
      tend(indt)=-(flux(indf2)-flux(indf1))/dx
    end do; end do; end do
!$omp end target teams distribute parallel do
  end subroutine compute_tendencies_x
  subroutine compute_tendencies_z(state,flux,tend)
    real(real64),intent(in)::state(0:)
    real(real64),intent(inout)::flux(0:),tend(0:)
    integer::i,k,ll,s,inds,indf1,indf2,indt
    real(real64)::stencil(0:3),vals(0:NUM_VARS-1),d3(0:NUM_VARS-1),r,u,w,t,p,hv_coef
    hv_coef=-hv_beta*dz/(16.0_real64*dt)
!$omp target teams distribute parallel do collapse(2) private(ll,s,inds,stencil,vals,d3,r,u,w,t,p)
    do k=0,nz; do i=0,nx-1
      do ll=0,NUM_VARS-1
        do s=0,sten_size-1; inds=ll*(nz+2*hs)*(nx+2*hs)+(k+s)*(nx+2*hs)+i+hs; stencil(s)=state(inds); end do
        vals(ll)=(-stencil(0)+7.0_real64*stencil(1)+7.0_real64*stencil(2)-stencil(3))/12.0_real64
        d3(ll)=(-stencil(0)+3.0_real64*stencil(1)-3.0_real64*stencil(2)+stencil(3))
      end do
      r=vals(ID_DENS)+hy_dens_int(k); u=vals(ID_UMOM)/r; w=vals(ID_WMOM)/r; t=(vals(ID_RHOT)+hy_dens_theta_int(k))/r; p=c0*(r*t)**gamm-hy_pressure_int(k)
      flux(flux_idx(i,k,ID_DENS))=r*w-hv_coef*d3(ID_DENS)
      flux(flux_idx(i,k,ID_UMOM))=r*w*u-hv_coef*d3(ID_UMOM)
      flux(flux_idx(i,k,ID_WMOM))=r*w*w+p-hv_coef*d3(ID_WMOM)
      flux(flux_idx(i,k,ID_RHOT))=r*w*t-hv_coef*d3(ID_RHOT)
    end do; end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(3) private(indt,indf1,indf2)
    do ll=0,NUM_VARS-1; do k=0,nz-1; do i=0,nx-1
      indt=tend_idx(i,k,ll); indf1=flux_idx(i,k,ll); indf2=flux_idx(i,k+1,ll)
      tend(indt)=-(flux(indf2)-flux(indf1))/dz
      if (ll==ID_WMOM) tend(indt)=tend(indt)-state(state_idx(i,k,ID_DENS))*grav
    end do; end do; end do
!$omp end target teams distribute parallel do
  end subroutine compute_tendencies_z
  subroutine set_halo_values_x(state)
    real(real64),intent(inout)::state(0:)
    integer::k,ll
!$omp target teams distribute parallel do collapse(2)
    do ll=0,NUM_VARS-1; do k=0,nz-1
      state(state_idx(-2,k,ll))=state(state_idx(nx-2,k,ll)); state(state_idx(-1,k,ll))=state(state_idx(nx-1,k,ll))
      state(state_idx(nx,k,ll))=state(state_idx(0,k,ll)); state(state_idx(nx+1,k,ll))=state(state_idx(1,k,ll))
    end do; end do
!$omp end target teams distribute parallel do
  end subroutine set_halo_values_x
  subroutine set_halo_values_z(state)
    real(real64),intent(inout)::state(0:)
    integer::i,ll
!$omp target teams distribute parallel do collapse(2)
    do ll=0,NUM_VARS-1; do i=-hs,nx+hs-1
      state(state_idx(i,-2,ll))=state(state_idx(i,0,ll)); state(state_idx(i,-1,ll))=state(state_idx(i,0,ll))
      state(state_idx(i,nz,ll))=state(state_idx(i,nz-1,ll)); state(state_idx(i,nz+1,ll))=state(state_idx(i,nz-1,ll))
    end do; end do
!$omp end target teams distribute parallel do
  end subroutine set_halo_values_z
  subroutine reductions(state,rmass,rte)
    real(real64),intent(in)::state(0:)
    real(real64),intent(out)::rmass,rte
    integer::i,k
    real(real64)::r,u,w,t,p
    rmass=0.0_real64; rte=0.0_real64
!$omp target teams distribute parallel do collapse(2) reduction(+:rmass,rte) private(r,u,w,t,p)
    do k=0,nz-1; do i=0,nx-1
      r=state(state_idx(i,k,ID_DENS)); u=state(state_idx(i,k,ID_UMOM))/r; w=state(state_idx(i,k,ID_WMOM))/r; t=state(state_idx(i,k,ID_RHOT))/r
      p=c0*(r*t)**gamm
      rmass=rmass+r*dx*dz
      rte=rte+(cv*p+0.5_real64*r*(u*u+w*w))*dx*dz
    end do; end do
!$omp end target teams distribute parallel do
  end subroutine reductions
end program main
