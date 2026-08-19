program main
  use omp_lib
  use utils_mod
  use kernels_mod
  implicit none
  type(dimensions_t) :: dims_g, dims, dims_b
  type(stepscheduler_t) :: scheduler
  type(stepinfo_t) :: si(0:NOCTANT-1)
  integer :: niterations, argc, i, im, ia, octant, iu, ie, ix, iy, iz, step, nstep, wavefront, ixwav, iywav
  integer :: a_size, v_size, facexy_size, facexz_size, faceyz_size, vslocal_size, noctant_per_block, num_wavefronts
  integer :: sf, dirx, diry, dirz, iz0, ix_g, iy_g, iz_g
  character(len=128) :: arg
  real(P), allocatable :: a_from_m(:), m_from_a(:), vi(:), vo(:), facexy(:), facexz(:), faceyz(:), vslocal(:), tmp(:)
  real(8) :: t1,t2,k_start,k_end,time,ktime,flops,floprate_h,floprate_d
  real(P) :: normsq, normsqdiff

  dims_g%ncell_x = arg_int('--ncell_x',5)
  dims_g%ncell_y = arg_int('--ncell_y',5)
  dims_g%ncell_z = arg_int('--ncell_z',5)
  dims_g%ne = arg_int('--ne',30)
  dims_g%na = arg_int('--na',33)
  niterations = arg_int('--niterations',1)
  dims_g%nm = NM
  dims = dims_g
  a_size = dims%nm * dims%na * NOCTANT
  allocate(a_from_m(0:a_size-1), m_from_a(0:a_size-1))
  a_from_m = 0.0_P; m_from_a = 0.0_P
  do octant=0,NOCTANT-1; do i=0,dims%na-1
    a_from_m(a_from_m_addr(dims%na,dims%nm-1,i,octant)) = a_from_m(a_from_m_addr(dims%na,dims%nm-1,i,octant)) + (i+1)/dims%nm
    if (mod(i+1,dims%nm) /= 0) then
      a_from_m(a_from_m_addr(dims%na,0,i,octant)) = a_from_m(a_from_m_addr(dims%na,0,i,octant)) - 1.0_P
      a_from_m(a_from_m_addr(dims%na,mod(i+1,dims%nm),i,octant)) = a_from_m(a_from_m_addr(dims%na,mod(i+1,dims%nm),i,octant)) + 1.0_P
    end if
  end do; end do
  do octant=0,NOCTANT-1; do im=0,dims%nm-3; do ia=0,dims%na-1
    i = 21 + mod(im + dims%nm*ia, 17)
    a_from_m(a_from_m_addr(dims%na,im,ia,octant)) = a_from_m(a_from_m_addr(dims%na,im,ia,octant)) - i
    a_from_m(a_from_m_addr(dims%na,im+1,ia,octant)) = a_from_m(a_from_m_addr(dims%na,im+1,ia,octant)) + 2*i
    a_from_m(a_from_m_addr(dims%na,im+2,ia,octant)) = a_from_m(a_from_m_addr(dims%na,im+2,ia,octant)) - i
  end do; end do; end do
  do octant=0,NOCTANT-1; do i=0,dims%nm-1
    m_from_a(m_from_a_addr(dims%na,i,dims%na-1,octant)) = m_from_a(m_from_a_addr(dims%na,i,dims%na-1,octant)) + (i+1)/dims%na
    if (mod(i+1,dims%na) /= 0) then
      m_from_a(m_from_a_addr(dims%na,i,0,octant)) = m_from_a(m_from_a_addr(dims%na,i,0,octant)) - 1.0_P
      m_from_a(m_from_a_addr(dims%na,i,mod(i+1,dims%na),octant)) = m_from_a(m_from_a_addr(dims%na,i,mod(i+1,dims%na),octant)) + 1.0_P
    end if
  end do; end do
  do octant=0,NOCTANT-1; do im=0,dims%nm-1; do ia=0,dims%na-3
    i = 37 + mod(im + dims%nm*ia, 19)
    m_from_a(m_from_a_addr(dims%na,im,ia,octant)) = m_from_a(m_from_a_addr(dims%na,im,ia,octant)) - i
    m_from_a(m_from_a_addr(dims%na,im,ia+1,octant)) = m_from_a(m_from_a_addr(dims%na,im,ia+1,octant)) + 2*i
    m_from_a(m_from_a_addr(dims%na,im,ia+2,octant)) = m_from_a(m_from_a_addr(dims%na,im,ia+2,octant)) - i
  end do; end do; end do
  do octant=0,NOCTANT-1; do im=0,dims%nm-1; do ia=0,dims%na-1
    m_from_a(m_from_a_addr(dims%na,im,ia,octant)) = m_from_a(m_from_a_addr(dims%na,im,ia,octant)) / NOCTANT / real(ishft(1,iand(ia,ishft(1,3)-1)),P)
  end do; end do; end do
  v_size = dimensions_size_state(dims,NU)
  allocate(vi(0:v_size-1), vo(0:v_size-1), tmp(0:v_size-1))
  call initialize_input_state(vi,dims,NU)
  vo = 0.0_P
  scheduler%nblock_z_ = arg_int('--nblock_z',1)
  scheduler%nblock_octant_ = 1
  scheduler%noctant_per_block_ = NOCTANT
  noctant_per_block = scheduler%noctant_per_block_
  dims_b = dims
  dims_b%ncell_z = dims%ncell_z / scheduler%nblock_z_
  facexy_size = dimensions_size_facexy(dims_b,NU,noctant_per_block)
  facexz_size = dimensions_size_facexz(dims_b,NU,noctant_per_block)
  faceyz_size = dimensions_size_faceyz(dims_b,NU,noctant_per_block)
  vslocal_size = dims%na*NU*dims%ne*NOCTANT*dims%ncell_x*dims%ncell_y
  allocate(facexy(0:facexy_size-1),facexz(0:facexz_size-1),faceyz(0:faceyz_size-1),vslocal(0:vslocal_size-1))
  ktime=0.0d0
!$omp target data map(to:a_from_m(0:a_size-1),m_from_a(0:a_size-1),vi(0:v_size-1)) map(alloc:facexy(0:facexy_size-1),facexz(0:facexz_size-1),faceyz(0:faceyz_size-1),vslocal(0:vslocal_size-1),vo(0:v_size-1))
  t1=omp_get_wtime()
  do i=1,niterations
    nstep = step_scheduler_nstep(scheduler)
    do step=0,nstep-1
      do octant=0,NOCTANT-1
        si(octant)=step_scheduler_stepinfo(scheduler,step,octant)
      end do
      num_wavefronts = dims_b%ncell_z + dims_b%ncell_y + dims_b%ncell_x - 2
      if (step == 0) then
        k_start=omp_get_wtime(); vo=0.0_P
!$omp target update to(vo(0:v_size-1))
!$omp target teams distribute parallel do collapse(3) private(ie,iu,ia,dirz,iz0,ix_g,iy_g,iz_g,sf)
        do octant=0,NOCTANT-1; do iy=0,dims_b%ncell_y-1; do ix=0,dims_b%ncell_x-1
          do ie=0,dims_b%ne-1; do iu=0,NU-1; do ia=0,dims_b%na-1
            dirz=dir_z(octant); iz0=merge(-1,dims_b%ncell_z,dirz==DIR_UP); ix_g=ix; iy_g=iy; iz_g=iz0+merge(0,dims%ncell_z-dims_b%ncell_z,dirz==DIR_UP)
            sf=quantities_scalefactor_space(ix_g,iy_g,iz_g)
            facexy(facexy_addr(dims_b,ia,iu,ie,ix,iy,octant))=quantities_init_face(ia,ie,iu,sf,octant,dims_b)
          end do; end do; end do
        end do; end do; end do
!$omp end target teams distribute parallel do
      end if
!$omp target teams distribute parallel do collapse(3) private(ie,iu,ia,diry,ix_g,iy_g,iz_g,sf)
      do octant=0,NOCTANT-1; do iz=0,dims_b%ncell_z-1; do ix=0,dims_b%ncell_x-1
        do ie=0,dims_b%ne-1; do iu=0,NU-1; do ia=0,dims_b%na-1
          diry=dir_y(octant); ix_g=ix; iy_g=merge(-1,dims_b%ncell_y,diry==DIR_UP); iz_g=iz+si(octant)%block_z*dims_b%ncell_z
          sf=quantities_scalefactor_space(ix_g,iy_g,iz_g)
          facexz(facexz_addr(dims_b,ia,iu,ie,ix,iz,octant))=quantities_init_face(ia,ie,iu,sf,octant,dims_b)
        end do; end do; end do
      end do; end do; end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(3) private(ie,iu,ia,dirx,ix_g,iy_g,iz_g,sf)
      do octant=0,NOCTANT-1; do iz=0,dims_b%ncell_z-1; do iy=0,dims_b%ncell_y-1
        do ie=0,dims_b%ne-1; do iu=0,NU-1; do ia=0,dims_b%na-1
          dirx=dir_x(octant); ix_g=merge(-1,dims_b%ncell_x,dirx==DIR_UP); iy_g=iy; iz_g=iz+si(octant)%block_z*dims_b%ncell_z
          sf=quantities_scalefactor_space(ix_g,iy_g,iz_g)
          faceyz(faceyz_addr(dims_b,ia,iu,ie,iy,iz,octant))=quantities_init_face(ia,ie,iu,sf,octant,dims_b)
        end do; end do; end do
      end do; end do; end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(2) private(wavefront,iywav,ixwav,dirx,diry,dirz,ix,iy,ix_g,iy_g,iz_g)
      do ie=0,dims_b%ne-1; do octant=0,NOCTANT-1
        do wavefront=0,num_wavefronts-1; do iywav=0,dims_b%ncell_y-1; do ixwav=0,dims_b%ncell_x-1
          if (si(octant)%is_active /= 0) then
            dirx=dir_x(octant); diry=dir_y(octant); dirz=dir_z(octant)
            ix=merge(ixwav,dims_b%ncell_x-1-ixwav,dirx==DIR_UP)
            iy=merge(iywav,dims_b%ncell_y-1-iywav,diry==DIR_UP)
            ix_g=ix; iy_g=iy; iz_g=wavefront-ixwav-iywav+si(octant)%block_z*dims_b%ncell_z
            call sweep_cell(dims_b,wavefront,octant,ix,iy,ix_g,iy_g,iz_g,dirx,diry,dirz,facexy,facexz,faceyz,a_from_m,m_from_a,vi,vo,vslocal,ie)
          end if
        end do; end do; end do
      end do; end do
!$omp end target teams distribute parallel do
      if (step == nstep-1) then
        k_end=omp_get_wtime(); ktime=ktime+k_end-k_start
!$omp target update from(vo(0:v_size-1))
      end if
    end do
    tmp=vo; vo=vi; vi=tmp
  end do
  t2=omp_get_wtime(); time=t2-t1
!$omp end target data
  normsq=sum(vo*vo); normsqdiff=sum((vi-vo)*(vi-vo))
  flops=niterations*(dimensions_size_state(dims,NU)*NOCTANT*2.0d0*dims%na + dimensions_size_state_angles(dims,NU)*(2.0d0*dims%na*dims%nm+6.0d0*NU*dims%na) + dimensions_size_state(dims,NU)*NOCTANT*2.0d0*dims%na)
  floprate_h=merge(0.0d0, flops/(time*1.0d-6)/1.0d9, time<=0.0d0)
  floprate_d=merge(0.0d0, flops/(ktime*1.0d-6)/1.0d9, ktime<=0.0d0)
  print '(a,es14.8,a,es10.3,a,a,a,f0.3,a,f0.3,a)', 'Normsq result: ', normsq, '  diff: ', normsqdiff, '  verify: ', merge('PASS','FAIL',normsqdiff==0.0_P), '  host time: ', time, ' (s) kernel time: ', ktime, ' (s)'
  print '(a,f0.3)', 'GF/s (host): ', floprate_h
  print '(a,f0.3)', 'GF/s (device): ', floprate_d
contains
  integer function arg_int(name, default_value)
    character(len=*),intent(in)::name
    integer,intent(in)::default_value
    integer :: j, argc_local
    character(len=128)::a,v
    arg_int=default_value; argc_local=command_argument_count()
    do j=1,argc_local-1
      call get_command_argument(j,a)
      if (trim(a)==name) then
        call get_command_argument(j+1,v); read(v,*) arg_int; return
      end if
    end do
  end function arg_int
end program main
