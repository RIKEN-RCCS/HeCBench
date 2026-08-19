program multimat
  use multimat_types
  use full_matrix_mod
  use compact_mod
  implicit none
  type(full_data) :: cc, mc
  type(compact_data) :: ccc
  integer :: sizex, sizey, ncells, nmats, i, j, mat, count, matindex, list_size, list_pos, ios
  integer :: mat_indices(0:3), cell_counts(0:3)
  character(len=64) :: arg
  real(real64) :: dx, dy, prob2, prob3, prob4
  logical :: ok
  sizex = 1000; sizey = 1000
  if (command_argument_count() >= 1) then; call get_command_argument(1,arg); read(arg,*,iostat=ios) sizex; end if
  if (command_argument_count() >= 2) then; call get_command_argument(2,arg); read(arg,*,iostat=ios) sizey; end if
  ncells = sizex * sizey; nmats = nmats_default
  call allocate_full(cc, sizex, sizey, nmats)
  call allocate_full(mc, sizex, sizey, nmats)
  call allocate_compact_base(ccc, sizex, sizey, nmats, ncells*4)
  dx = 1.0_real64 / sizex; dy = 1.0_real64 / sizey
  do j = 0, sizey-1
    do i = 0, sizex-1
      cc%v(cidx(i,j,sizex)) = dx*dy; cc%x(cidx(i,j,sizex)) = dx*i; cc%y(cidx(i,j,sizex)) = dy*j
    end do
  end do
  cc%n = 1.0_real64
  if (command_argument_count() >= 5) then
    call get_command_argument(3,arg); read(arg,*) prob2
    call get_command_argument(4,arg); read(arg,*) prob3
    call get_command_argument(5,arg); read(arg,*) prob4
    call initialise_field_rand(cc, prob2, prob3, prob4)
  else
    call initialise_field_static(cc)
  end if
  do i = 0, ncells-1
    count = count_materials(cc, i)
    if (count == 0) then
      cc%rho(i*nmats+1)=1.0_real64; cc%t(i*nmats+1)=1.0_real64; cc%p(i*nmats+1)=1.0_real64; cc%vf(i*nmats+1)=1.0_real64
      count = 1
    end if
    cell_counts(count-1) = cell_counts(count-1) + 1
    do mat = 0, nmats-1
      if (cc%rho(i*nmats+mat) /= 0.0_real64) cc%vf(i*nmats+mat) = 1.0_real64 / real(count,real64)
      mc%rho(ncells*mat+i) = cc%rho(i*nmats+mat)
      mc%p(ncells*mat+i) = cc%p(i*nmats+mat)
      mc%vf(ncells*mat+i) = cc%vf(i*nmats+mat)
      mc%t(ncells*mat+i) = cc%t(i*nmats+mat)
    end do
  end do
  mc%v = cc%v; mc%x = cc%x; mc%y = cc%y; mc%n = cc%n
  ccc%v = cc%v; ccc%x = cc%x; ccc%y = cc%y; ccc%n = cc%n
  list_pos = 0; ccc%mmc_cells = 0
  do j = 0, sizey-1
    do i = 0, sizex-1
      mat_indices = -1; matindex = 0
      do mat = 0, nmats-1
        if (cc%rho(fmidx(i,j,mat,sizex,nmats)) /= 0.0_real64) then
          mat_indices(matindex) = mat; matindex = matindex + 1
        end if
      end do
      if (matindex == 1) then
        mat = mat_indices(0)
        ccc%rho_compact(cidx(i,j,sizex)) = cc%rho(fmidx(i,j,mat,sizex,nmats))
        ccc%p_compact(cidx(i,j,sizex)) = cc%p(fmidx(i,j,mat,sizex,nmats))
        ccc%t_compact(cidx(i,j,sizex)) = cc%t(fmidx(i,j,mat,sizex,nmats))
        ccc%imaterial(cidx(i,j,sizex)) = mat + 1
      else
        ccc%imaterial(cidx(i,j,sizex)) = -ccc%mmc_cells
        ccc%mmc_index(ccc%mmc_cells) = list_pos; ccc%mmc_i(ccc%mmc_cells) = i; ccc%mmc_j(ccc%mmc_cells) = j
        ccc%mmc_cells = ccc%mmc_cells + 1
        do matindex = 0, matindex-1
          mat = mat_indices(matindex)
          ccc%nextfrac(list_pos) = merge(-1, list_pos+1, matindex == count_materials(cc,cidx(i,j,sizex))-1)
          ccc%matids(list_pos) = mat
          ccc%vf_compact_list(list_pos) = cc%vf(fmidx(i,j,mat,sizex,nmats))
          ccc%rho_compact_list(list_pos) = cc%rho(fmidx(i,j,mat,sizex,nmats))
          ccc%p_compact_list(list_pos) = cc%p(fmidx(i,j,mat,sizex,nmats))
          ccc%t_compact_list(list_pos) = cc%t(fmidx(i,j,mat,sizex,nmats))
          list_pos = list_pos + 1
        end do
      end if
    end do
  end do
  ccc%mmc_index(ccc%mmc_cells) = list_pos; ccc%mm_len = list_pos
  call full_matrix_cell_centric(cc)
  call full_matrix_material_centric(cc, mc)
  call compact_cell_centric(cc, ccc)
  ok = check_full(cc, mc) .and. check_compact(cc, ccc)
  print '(a)', merge('All tests passed!', 'FAIL', ok)
contains
  subroutine allocate_full(a, sx, sy, nm)
    type(full_data), intent(inout) :: a
    integer, intent(in) :: sx, sy, nm
    integer :: nc
    nc = sx*sy; a%sizex=sx; a%sizey=sy; a%nmats=nm
    allocate(a%rho(0:nc*nm-1),a%rho_mat_ave(0:nc*nm-1),a%p(0:nc*nm-1),a%vf(0:nc*nm-1),a%t(0:nc*nm-1))
    allocate(a%v(0:nc-1),a%x(0:nc-1),a%y(0:nc-1),a%n(0:nm-1),a%rho_ave(0:nc-1))
    a%rho=0; a%rho_mat_ave=0; a%p=0; a%vf=0; a%t=0; a%v=0; a%x=0; a%y=0; a%n=0; a%rho_ave=0
  end subroutine
  subroutine allocate_compact_base(a, sx, sy, nm, lsize)
    type(compact_data), intent(inout) :: a
    integer, intent(in) :: sx, sy, nm, lsize
    integer :: nc
    nc = sx*sy; a%sizex=sx; a%sizey=sy; a%nmats=nm
    allocate(a%rho_compact(0:nc-1),a%rho_mat_ave_compact(0:nc-1),a%p_compact(0:nc-1),a%t_compact(0:nc-1),a%rho_ave_compact(0:nc-1))
    allocate(a%rho_compact_list(0:lsize-1),a%rho_mat_ave_compact_list(0:lsize-1),a%p_compact_list(0:lsize-1),a%vf_compact_list(0:lsize-1),a%t_compact_list(0:lsize-1))
    allocate(a%v(0:nc-1),a%x(0:nc-1),a%y(0:nc-1),a%n(0:nm-1),a%imaterial(0:nc-1),a%matids(0:lsize-1),a%nextfrac(0:lsize-1),a%mmc_index(0:nc),a%mmc_i(0:nc),a%mmc_j(0:nc))
    a%rho_compact=0; a%rho_mat_ave_compact=0; a%p_compact=0; a%t_compact=0; a%rho_ave_compact=0
    a%rho_compact_list=0; a%rho_mat_ave_compact_list=0; a%p_compact_list=0; a%vf_compact_list=0; a%t_compact_list=0
  end subroutine
  subroutine initialise_field_static(a)
    type(full_data), intent(inout) :: a
    integer :: cell, mat, count
    do cell = 0, a%sizex*a%sizey-1
      count = 1 + mod(cell,4)
      do mat = 0, count-1
        a%rho(cell*a%nmats + mod(cell+mat,a%nmats)) = 1.0_real64
        a%t(cell*a%nmats + mod(cell+mat,a%nmats)) = 1.0_real64
        a%p(cell*a%nmats + mod(cell+mat,a%nmats)) = 1.0_real64
      end do
    end do
  end subroutine
  subroutine initialise_field_rand(a, p2, p3, p4)
    type(full_data), intent(inout) :: a
    real(real64), intent(in) :: p2, p3, p4
    integer :: cell, mat, extra
    real(real64) :: r
    call random_seed()
    do cell = 0, a%sizex*a%sizey-1
      call random_number(r); extra = merge(3, merge(2, merge(1,0,r >= 1.0_real64-p2-p3-p4), r >= 1.0_real64-p3-p4), r >= 1.0_real64-p4)
      do mat = 0, extra
        a%rho(cell*a%nmats + mod(cell+mat,a%nmats)) = 1.0_real64
        a%t(cell*a%nmats + mod(cell+mat,a%nmats)) = 1.0_real64
        a%p(cell*a%nmats + mod(cell+mat,a%nmats)) = 1.0_real64
      end do
    end do
  end subroutine
  integer function count_materials(a, cell) result(count)
    type(full_data), intent(in) :: a
    integer, intent(in) :: cell
    integer :: mat
    count = 0
    do mat = 0, a%nmats-1
      if (a%rho(cell*a%nmats+mat) /= 0.0_real64) count = count + 1
    end do
  end function
  logical function check_full(cc, mc) result(ok)
    type(full_data), intent(in) :: cc, mc
    ok = maxval(abs(cc%rho_ave - mc%rho_ave)) < 1.0e-4_real64
  end function
  logical function check_compact(cc, ccc) result(ok)
    type(full_data), intent(in) :: cc
    type(compact_data), intent(in) :: ccc
    ok = maxval(abs(cc%rho_ave - ccc%rho_ave_compact)) < 1.0e-4_real64
  end function
end program
