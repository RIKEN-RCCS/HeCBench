module multimat_types
  use iso_fortran_env, only: int32, int64, real64
  implicit none
  integer, parameter :: nmats_default = 50
  type :: full_data
    integer :: sizex, sizey, nmats
    real(real64), allocatable :: rho(:), rho_mat_ave(:), p(:), vf(:), t(:)
    real(real64), allocatable :: v(:), x(:), y(:), n(:), rho_ave(:)
  end type
  type :: compact_data
    integer :: sizex, sizey, nmats, mm_len, mmc_cells
    real(real64), allocatable :: rho_compact(:), rho_compact_list(:)
    real(real64), allocatable :: rho_mat_ave_compact(:), rho_mat_ave_compact_list(:)
    real(real64), allocatable :: p_compact(:), p_compact_list(:), vf_compact_list(:)
    real(real64), allocatable :: t_compact(:), t_compact_list(:)
    real(real64), allocatable :: v(:), x(:), y(:), n(:), rho_ave_compact(:)
    integer(int32), allocatable :: imaterial(:), matids(:), nextfrac(:), mmc_index(:), mmc_i(:), mmc_j(:)
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  pure integer function cidx(i, j, sizex) result(idx)
    integer, intent(in) :: i, j, sizex
    idx = i + sizex*j
  end function

  pure integer function fmidx(i, j, mat, sizex, nmats) result(idx)
    integer, intent(in) :: i, j, mat, sizex, nmats
    idx = (i + sizex*j) * nmats + mat
  end function

  pure integer function mcidx(i, j, mat, sizex, sizey) result(idx)
    integer, intent(in) :: i, j, mat, sizex, sizey
    idx = sizex*sizey*mat + i + sizex*j
  end function
end module
