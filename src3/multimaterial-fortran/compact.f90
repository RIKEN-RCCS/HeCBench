module compact_mod
  use multimat_types
  implicit none
contains
  subroutine compact_cell_centric(cc, ccc)
    type(full_data), intent(in) :: cc
    type(compact_data), intent(inout) :: ccc
    integer :: i, j, c, idx, mat, ix, jx, ni, nj, nn, sizex, sizey, mmc_cells, mm_len
    real(real64) :: ave, nm, xo, yo, xi, yi, dsqr(0:8), rho_sum
    real(real64) :: t0, t1
    sizex = ccc%sizex; sizey = ccc%sizey; mmc_cells = ccc%mmc_cells; mm_len = ccc%mm_len
    !$omp target data map(to:ccc%imaterial,ccc%matids,ccc%nextfrac,ccc%mmc_index,ccc%mmc_i,ccc%mmc_j) &
    !$omp& map(tofrom:ccc%x,ccc%y,ccc%rho_compact,ccc%rho_compact_list,ccc%rho_mat_ave_compact,ccc%rho_mat_ave_compact_list, &
    !$omp& ccc%p_compact,ccc%p_compact_list,ccc%t_compact,ccc%t_compact_list,ccc%v,ccc%vf_compact_list,ccc%n,ccc%rho_ave_compact)
    t0 = seconds()
    !$omp target teams distribute parallel do collapse(2) private(ix,ave) thread_limit(128)
    do j = 0, sizey-1
      do i = 0, sizex-1
        ix = ccc%imaterial(cidx(i,j,sizex))
        if (ix > 0) ccc%rho_ave_compact(cidx(i,j,sizex)) = ccc%rho_compact(cidx(i,j,sizex)) / ccc%v(cidx(i,j,sizex))
      end do
    end do
    !$omp end target teams distribute parallel do
    !$omp target teams distribute parallel do private(idx,ave) thread_limit(128)
    do c = 0, mmc_cells-1
      ave = 0.0_real64
      do idx = ccc%mmc_index(c), ccc%mmc_index(c+1)-1
        ave = ave + ccc%rho_compact_list(idx) * ccc%vf_compact_list(idx)
      end do
      ccc%rho_ave_compact(cidx(ccc%mmc_i(c),ccc%mmc_j(c),sizex)) = ave / ccc%v(cidx(ccc%mmc_i(c),ccc%mmc_j(c),sizex))
    end do
    !$omp end target teams distribute parallel do
    t1 = seconds()
    print '(a,f10.6,a)', 'Compact matrix, cell centric, alg 1: ', (t1-t0)*1000.0_real64, ' msec'

    t0 = seconds()
    !$omp target teams distribute parallel do collapse(2) private(ix,mat) thread_limit(128)
    do j = 0, sizey-1
      do i = 0, sizex-1
        ix = ccc%imaterial(cidx(i,j,sizex))
        if (ix > 0) then
          mat = ix - 1
          ccc%p_compact(cidx(i,j,sizex)) = ccc%n(mat) * ccc%rho_compact(cidx(i,j,sizex)) * ccc%t_compact(cidx(i,j,sizex))
        end if
      end do
    end do
    !$omp end target teams distribute parallel do
    !$omp target teams distribute parallel do private(mat,nm) thread_limit(128)
    do idx = 0, mm_len-1
      mat = ccc%matids(idx); nm = ccc%n(mat)
      ccc%p_compact_list(idx) = (nm * ccc%rho_compact_list(idx) * ccc%t_compact_list(idx)) / ccc%vf_compact_list(idx)
    end do
    !$omp end target teams distribute parallel do
    t1 = seconds()
    print '(a,f10.6,a)', 'Compact matrix, cell centric, alg 2: ', (t1-t0)*1000.0_real64, ' msec'

    t0 = seconds()
    !$omp target teams distribute parallel do collapse(2) private(ix,jx,idx,mat,ni,nj,nn,xo,yo,xi,yi,dsqr,rho_sum) thread_limit(128)
    do j = 1, sizey-2
      do i = 1, sizex-2
        xo = ccc%x(cidx(i,j,sizex)); yo = ccc%y(cidx(i,j,sizex))
        do nj = -1, 1
          do ni = -1, 1
            xi = ccc%x(cidx(i+ni,j+nj,sizex)); yi = ccc%y(cidx(i+ni,j+nj,sizex))
            dsqr((nj+1)*3+(ni+1)) = (xo-xi)*(xo-xi) + (yo-yi)*(yo-yi)
          end do
        end do
        ix = ccc%imaterial(cidx(i,j,sizex))
        if (ix <= 0) then
          do idx = ccc%mmc_index(-ix), ccc%mmc_index(-ix+1)-1
            mat = ccc%matids(idx); rho_sum = 0.0_real64; nn = 0
            call compact_neighbor_sum(ccc, i, j, mat, dsqr, rho_sum, nn)
            ccc%rho_mat_ave_compact_list(idx) = rho_sum / real(max(1,nn),real64)
          end do
        else
          mat = ix - 1; rho_sum = 0.0_real64; nn = 0
          call compact_neighbor_sum(ccc, i, j, mat, dsqr, rho_sum, nn)
          ccc%rho_mat_ave_compact(cidx(i,j,sizex)) = rho_sum / real(max(1,nn),real64)
        end if
      end do
    end do
    !$omp end target teams distribute parallel do
    t1 = seconds()
    print '(a,f10.6,a)', 'Compact matrix, cell centric, alg 3: ', (t1-t0)*1000.0_real64, ' msec'
    !$omp end target data
  end subroutine

  subroutine compact_neighbor_sum(ccc, i, j, mat, dsqr, rho_sum, nn)
    type(compact_data), intent(in) :: ccc
    integer, intent(in) :: i, j, mat
    real(real64), intent(in) :: dsqr(0:8)
    real(real64), intent(inout) :: rho_sum
    integer, intent(inout) :: nn
    integer :: ni, nj, ci, cj, jx, k
    do nj = -1, 1
      do ni = -1, 1
        ci = i + ni; cj = j + nj
        jx = ccc%imaterial(cidx(ci,cj,ccc%sizex))
        if (jx <= 0) then
          do k = ccc%mmc_index(-jx), ccc%mmc_index(-jx+1)-1
            if (ccc%matids(k) == mat .and. dsqr((nj+1)*3+(ni+1)) > 0.0_real64) then
              rho_sum = rho_sum + ccc%rho_compact_list(k) / dsqr((nj+1)*3+(ni+1)); nn = nn + 1; exit
            end if
          end do
        else if (jx - 1 == mat .and. dsqr((nj+1)*3+(ni+1)) > 0.0_real64) then
          rho_sum = rho_sum + ccc%rho_compact(cidx(ci,cj,ccc%sizex)) / dsqr((nj+1)*3+(ni+1)); nn = nn + 1
        end if
      end do
    end do
  end subroutine
end module
