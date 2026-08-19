module kernels_mod
  use utils_mod
  implicit none
contains
  subroutine quantities_solve(vs_local, dims, facexy, facexz, faceyz, ix, iy, iz, ix_g, iy_g, iz_g, ie, ia, octant)
    real(P), intent(inout) :: vs_local(0:), facexy(0:), facexz(0:), faceyz(0:)
    type(dimensions_t), intent(in) :: dims
    integer, intent(in) :: ix,iy,iz,ix_g,iy_g,iz_g,ie,ia,octant
    integer :: iu, dx, dy, dz, sf
    real(P) :: sf_o_r, sf_r, sfx_r, sfy_r, sfz_r, result, result_scaled
    dx=dir_x(octant); dy=dir_y(octant); dz=dir_z(octant)
    sf_o_r = 1.0_P / real(1+octant,P)
    sf = quantities_scalefactor_space(ix_g,iy_g,iz_g)
    sf_r = 1.0_P / real(sf,P)
    sfx_r = 1.0_P / real(quantities_scalefactor_space(ix_g-dx,iy_g,iz_g),P)
    sfy_r = 1.0_P / real(quantities_scalefactor_space(ix_g,iy_g-dy,iz_g),P)
    sfz_r = 1.0_P / real(quantities_scalefactor_space(ix_g,iy_g,iz_g-dz),P)
    do iu=0,NU-1
      result = (vs_local(vslocal_addr(dims,ia,iu,ie,ix,iy,octant))*sf_r + &
        (facexy(facexy_addr(dims,ia,iu,ie,ix,iy,octant))*0.5_P*sfz_r + &
         facexz(facexz_addr(dims,ia,iu,ie,ix,iz,octant))*0.25_P*sfy_r + &
         faceyz(faceyz_addr(dims,ia,iu,ie,iy,iz,octant))*(0.25_P - 1.0_P/real(ishft(1,iand(ia,ishft(1,3)-1)),P))*sfx_r) * sf_o_r) * real(sf,P)
      vs_local(vslocal_addr(dims,ia,iu,ie,ix,iy,octant)) = result
      result_scaled = result * real(1+octant,P)
      facexy(facexy_addr(dims,ia,iu,ie,ix,iy,octant)) = result_scaled
      facexz(facexz_addr(dims,ia,iu,ie,ix,iz,octant)) = result_scaled
      faceyz(faceyz_addr(dims,ia,iu,ie,iy,iz,octant)) = result_scaled
    end do
  end subroutine quantities_solve

  subroutine sweep_cell(dims, wavefront, octant, ix, iy, ix_g, iy_g, iz_g, dx, dy, dz, facexy, facexz, faceyz, a_from_m, m_from_a, vi, vo, vs_local, ie)
    type(dimensions_t), intent(in) :: dims
    integer,intent(in)::wavefront,octant,ix,iy,ix_g,iy_g,iz_g,dx,dy,dz,ie
    real(P),intent(inout)::facexy(0:),facexz(0:),faceyz(0:),vo(0:),vs_local(0:)
    real(P),intent(in)::a_from_m(0:),m_from_a(0:),vi(0:)
    integer :: izwav, iz, im, ia, iu
    real(P) :: result
    izwav = wavefront - merge(ix, dims%ncell_x-1-ix, dx==DIR_UP) - merge(iy, dims%ncell_y-1-iy, dy==DIR_UP)
    iz = merge(izwav, dims%ncell_z-1-izwav, dz==DIR_UP)
    if (iz < 0 .or. iz >= dims%ncell_z) return
    do iu=0,NU-1; do ia=0,dims%na-1
      result = 0.0_P
      do im=0,dims%nm-1
        result = result + a_from_m(a_from_m_addr(dims%na,im,ia,octant)) * vi(state_addr(dims,im,iu,ix,iy,ie,iz))
      end do
      vs_local(vslocal_addr(dims,ia,iu,ie,ix,iy,octant)) = result
    end do; end do
    do ia=0,dims%na-1
      call quantities_solve(vs_local,dims,facexy,facexz,faceyz,ix,iy,iz,ix_g,iy_g,iz_g,ie,ia,octant)
    end do
    do iu=0,NU-1; do im=0,dims%nm-1
      result = 0.0_P
      do ia=0,dims%na-1
        result = result + m_from_a(m_from_a_addr(dims%na,im,ia,octant)) * vs_local(vslocal_addr(dims,ia,iu,ie,ix,iy,octant))
      end do
!$omp atomic update
      vo(state_addr(dims,im,iu,ix,iy,ie,iz)) = vo(state_addr(dims,im,iu,ix,iy,ie,iz)) + result
    end do; end do
  end subroutine sweep_cell
end module kernels_mod
