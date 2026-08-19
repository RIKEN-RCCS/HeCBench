module utils_mod
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: P = real64
  integer, parameter :: NDIM=3, DIM_X=0, DIM_Y=1, DIM_Z=2
  integer, parameter :: NOCTANT=8, DIR_UP=1, DIR_DN=-1, DIR_HI=1, DIR_LO=-1
  integer, parameter :: NU=4, NM=4
  type :: dimensions_t
    integer :: ncell_x=0, ncell_y=0, ncell_z=0, ne=0, nm=0, na=0
  end type dimensions_t
  type :: stepscheduler_t
    integer :: nblock_z_=1, nproc_x_=1, nproc_y_=1, nblock_octant_=1, noctant_per_block_=8
  end type stepscheduler_t
  type :: stepinfo_t
    integer :: block_z=0, octant=0, is_active=0
  end type stepinfo_t
contains
  pure integer function dir_x(octant)
    integer,intent(in)::octant
    dir_x = merge(DIR_DN, DIR_UP, iand(octant,1) /= 0)
  end function dir_x
  pure integer function dir_y(octant)
    integer,intent(in)::octant
    dir_y = merge(DIR_DN, DIR_UP, iand(octant,2) /= 0)
  end function dir_y
  pure integer function dir_z(octant)
    integer,intent(in)::octant
    dir_z = merge(DIR_DN, DIR_UP, iand(octant,4) /= 0)
  end function dir_z
  pure integer function a_from_m_addr(dims_na, im, ia, octant)
    integer,intent(in)::dims_na,im,ia,octant
    a_from_m_addr = ia + dims_na * (im + NM * octant)
  end function a_from_m_addr
  pure integer function m_from_a_addr(dims_na, im, ia, octant)
    integer,intent(in)::dims_na,im,ia,octant
    m_from_a_addr = im + NM * (ia + dims_na * octant)
  end function m_from_a_addr
  pure integer function state_addr(dims, im, iu, ix, iy, ie, iz)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::im,iu,ix,iy,ie,iz
    state_addr = im + dims%nm * (iu + NU * (ix + dims%ncell_x * (iy + dims%ncell_y * (ie + dims%ne * iz))))
  end function state_addr
  pure integer function vslocal_addr(dims, ia, iu, ie, ix, iy, octant)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::ia,iu,ie,ix,iy,octant
    vslocal_addr = ia + dims%na * (iu + NU * (ie + dims%ne * (ix + dims%ncell_x * (iy + dims%ncell_y * octant))))
  end function vslocal_addr
  pure integer function facexy_addr(dims, ia, iu, ie, ix, iy, octant)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::ia,iu,ie,ix,iy,octant
    facexy_addr = ia + dims%na * (iu + NU * (ie + dims%ne * (ix + dims%ncell_x * (iy + dims%ncell_y * octant))))
  end function facexy_addr
  pure integer function facexz_addr(dims, ia, iu, ie, ix, iz, octant)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::ia,iu,ie,ix,iz,octant
    facexz_addr = ia + dims%na * (iu + NU * (ie + dims%ne * (ix + dims%ncell_x * (iz + dims%ncell_z * octant))))
  end function facexz_addr
  pure integer function faceyz_addr(dims, ia, iu, ie, iy, iz, octant)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::ia,iu,ie,iy,iz,octant
    faceyz_addr = ia + dims%na * (iu + NU * (ie + dims%ne * (iy + dims%ncell_y * (iz + dims%ncell_z * octant))))
  end function faceyz_addr
  pure integer function quantities_scalefactor_space(ix_g, iy_g, iz_g)
    integer,intent(in)::ix_g,iy_g,iz_g
    quantities_scalefactor_space = ishft(1, iand(ix_g + 3*iy_g + 7*iz_g + 2, ishft(1,2)-1))
  end function quantities_scalefactor_space
  pure integer function quantities_scalefactor_energy(ie, dims)
    integer,intent(in)::ie
    type(dimensions_t),intent(in)::dims
    quantities_scalefactor_energy = ishft(1, iand(mod(ie*1366 + 150889, 714025), ishft(1,2)-1))
  end function quantities_scalefactor_energy
  pure integer function quantities_scalefactor_unknown(iu)
    integer,intent(in)::iu
    quantities_scalefactor_unknown = ishft(1, iand(mod(iu*741 + 66037, 312500), ishft(1,2)-1))
  end function quantities_scalefactor_unknown
  pure real(P) function quantities_init_face(ia, ie, iu, scalefactor_space, octant, dims)
    integer,intent(in)::ia,ie,iu,scalefactor_space,octant
    type(dimensions_t),intent(in)::dims
    quantities_init_face = real(1+ia,P) * real(ishft(1, iand(ia, ishft(1,3)-1)),P) * real(scalefactor_space,P) * &
      real(quantities_scalefactor_energy(ie,dims),P) * real(quantities_scalefactor_unknown(iu),P) * real(1+octant,P)
  end function quantities_init_face
  pure integer function dimensions_size_state(dims, nu)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::nu
    dimensions_size_state = dims%ncell_x*dims%ncell_y*dims%ncell_z*dims%ne*dims%nm*nu
  end function dimensions_size_state
  pure integer function dimensions_size_facexy(dims, nu, no)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::nu,no
    dimensions_size_facexy = dims%ncell_x*dims%ncell_y*dims%ne*dims%na*nu*no
  end function dimensions_size_facexy
  pure integer function dimensions_size_facexz(dims, nu, no)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::nu,no
    dimensions_size_facexz = dims%ncell_x*dims%ncell_z*dims%ne*dims%na*nu*no
  end function dimensions_size_facexz
  pure integer function dimensions_size_faceyz(dims, nu, no)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::nu,no
    dimensions_size_faceyz = dims%ncell_y*dims%ncell_z*dims%ne*dims%na*nu*no
  end function dimensions_size_faceyz
  pure integer function dimensions_size_state_angles(dims, nu)
    type(dimensions_t),intent(in)::dims
    integer,intent(in)::nu
    dimensions_size_state_angles = dims%ncell_x*dims%ncell_y*dims%ncell_z*dims%ne*dims%na*nu*NOCTANT
  end function dimensions_size_state_angles
  pure integer function step_scheduler_nstep(s)
    type(stepscheduler_t),intent(in)::s
    select case (s%nblock_octant_)
    case (8); step_scheduler_nstep = 8*s%nblock_z_ + 2*(s%nproc_x_-1) + 3*(s%nproc_y_-1)
    case (4); step_scheduler_nstep = 4*s%nblock_z_ + (s%nproc_x_-1) + 2*(s%nproc_y_-1)
    case (2); step_scheduler_nstep = 2*s%nblock_z_ + (s%nproc_x_-1) + (s%nproc_y_-1)
    case default; step_scheduler_nstep = s%nblock_z_
    end select
  end function step_scheduler_nstep
  pure type(stepinfo_t) function step_scheduler_stepinfo(s, step, octant_in_block)
    type(stepscheduler_t),intent(in)::s
    integer,intent(in)::step,octant_in_block
    step_scheduler_stepinfo%block_z = min(max(step,0), s%nblock_z_-1)
    step_scheduler_stepinfo%octant = octant_in_block
    step_scheduler_stepinfo%is_active = 1
  end function step_scheduler_stepinfo
  subroutine initialize_input_state(v, dims, nu)
    real(P), intent(out) :: v(0:)
    type(dimensions_t), intent(in) :: dims
    integer, intent(in) :: nu
    integer :: iz,iy,ix,ie,im,iu
    do iz=0,dims%ncell_z-1; do iy=0,dims%ncell_y-1; do ix=0,dims%ncell_x-1
      do ie=0,dims%ne-1; do im=0,dims%nm-1; do iu=0,nu-1
        v(state_addr(dims,im,iu,ix,iy,ie,iz)) = real(1+im,P) * real(quantities_scalefactor_space(ix,iy,iz),P) * &
          real(quantities_scalefactor_energy(ie,dims),P) * real(quantities_scalefactor_unknown(iu),P)
      end do; end do; end do
    end do; end do; end do
  end subroutine initialize_input_state
end module utils_mod
