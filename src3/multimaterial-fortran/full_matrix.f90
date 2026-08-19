module full_matrix_mod
  use multimat_types
  implicit none
contains
  subroutine full_matrix_cell_centric(cc)
    type(full_data), intent(inout) :: cc
    integer :: i, j, mat, ni, nj, nn, sizex, sizey, nmats
    real(real64) :: ave, nm, xo, yo, xi, yi, dsqr(0:8), rho_sum
    sizex = cc%sizex; sizey = cc%sizey; nmats = cc%nmats
    !$omp target data map(tofrom:cc%rho,cc%p,cc%t,cc%vf,cc%v,cc%x,cc%y,cc%n,cc%rho_ave,cc%rho_mat_ave)
    !$omp target teams distribute parallel do collapse(2) private(mat,ave) thread_limit(256)
    do j = 0, sizey-1
      do i = 0, sizex-1
        ave = 0.0_real64
        do mat = 0, nmats-1
          if (cc%vf(fmidx(i,j,mat,sizex,nmats)) > 0.0_real64) &
            ave = ave + cc%rho(fmidx(i,j,mat,sizex,nmats)) * cc%vf(fmidx(i,j,mat,sizex,nmats))
        end do
        cc%rho_ave(cidx(i,j,sizex)) = ave / cc%v(cidx(i,j,sizex))
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do collapse(2) private(mat,nm) thread_limit(256)
    do j = 0, sizey-1
      do i = 0, sizex-1
        do mat = 0, nmats-1
          if (cc%vf(fmidx(i,j,mat,sizex,nmats)) > 0.0_real64) then
            nm = cc%n(mat)
            cc%p(fmidx(i,j,mat,sizex,nmats)) = (nm * cc%rho(fmidx(i,j,mat,sizex,nmats)) * &
              cc%t(fmidx(i,j,mat,sizex,nmats))) / cc%vf(fmidx(i,j,mat,sizex,nmats))
          else
            cc%p(fmidx(i,j,mat,sizex,nmats)) = 0.0_real64
          end if
        end do
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do collapse(2) private(mat,ni,nj,nn,xo,yo,xi,yi,dsqr,rho_sum) thread_limit(256)
    do j = 1, sizey-2
      do i = 1, sizex-2
        xo = cc%x(cidx(i,j,sizex)); yo = cc%y(cidx(i,j,sizex))
        do nj = -1, 1
          do ni = -1, 1
            xi = cc%x(cidx(i+ni,j+nj,sizex)); yi = cc%y(cidx(i+ni,j+nj,sizex))
            dsqr((nj+1)*3 + (ni+1)) = (xo-xi)*(xo-xi) + (yo-yi)*(yo-yi)
          end do
        end do
        do mat = 0, nmats-1
          if (cc%vf(fmidx(i,j,mat,sizex,nmats)) > 0.0_real64) then
            rho_sum = 0.0_real64; nn = 0
            do nj = -1, 1
              do ni = -1, 1
                if (cc%vf(fmidx(i+ni,j+nj,mat,sizex,nmats)) > 0.0_real64 .and. dsqr((nj+1)*3+(ni+1)) > 0.0_real64) then
                  rho_sum = rho_sum + cc%rho(fmidx(i+ni,j+nj,mat,sizex,nmats)) / dsqr((nj+1)*3+(ni+1))
                  nn = nn + 1
                end if
              end do
            end do
            cc%rho_mat_ave(fmidx(i,j,mat,sizex,nmats)) = rho_sum / real(max(1,nn),real64)
          else
            cc%rho_mat_ave(fmidx(i,j,mat,sizex,nmats)) = 0.0_real64
          end if
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
    !$omp end target data
  end subroutine

  subroutine full_matrix_material_centric(cc, mc)
    type(full_data), intent(in) :: cc
    type(full_data), intent(inout) :: mc
    integer :: i, j, mat, ni, nj, nn, sizex, sizey, nmats
    real(real64) :: nm, xo, yo, xi, yi, dsqr, rho_sum
    sizex = mc%sizex; sizey = mc%sizey; nmats = mc%nmats
    !$omp target data map(tofrom:mc%rho,mc%p,mc%t,mc%vf,mc%v,mc%x,mc%y,mc%n,mc%rho_ave,mc%rho_mat_ave)
    !$omp target teams distribute parallel do thread_limit(256)
    do i = 0, sizex*sizey-1
      mc%rho_ave(i) = 0.0_real64
    end do
    !$omp end target teams distribute parallel do
    do mat = 0, nmats-1
      !$omp target teams distribute parallel do collapse(2) thread_limit(256)
      do j = 0, sizey-1
        do i = 0, sizex-1
          if (mc%vf(mcidx(i,j,mat,sizex,sizey)) > 0.0_real64) &
            mc%rho_ave(cidx(i,j,sizex)) = mc%rho_ave(cidx(i,j,sizex)) + &
              mc%rho(mcidx(i,j,mat,sizex,sizey)) * mc%vf(mcidx(i,j,mat,sizex,sizey))
        end do
      end do
      !$omp end target teams distribute parallel do
    end do
    !$omp target teams distribute parallel do thread_limit(256)
    do i = 0, sizex*sizey-1
      mc%rho_ave(i) = mc%rho_ave(i) / mc%v(i)
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do collapse(2) private(i,j,nm) thread_limit(256)
    do mat = 0, nmats-1
      do j = 0, sizey-1
        do i = 0, sizex-1
          nm = mc%n(mat)
          if (mc%vf(mcidx(i,j,mat,sizex,sizey)) > 0.0_real64) then
            mc%p(mcidx(i,j,mat,sizex,sizey)) = (nm * mc%rho(mcidx(i,j,mat,sizex,sizey)) * &
              mc%t(mcidx(i,j,mat,sizex,sizey))) / mc%vf(mcidx(i,j,mat,sizex,sizey))
          else
            mc%p(mcidx(i,j,mat,sizex,sizey)) = 0.0_real64
          end if
        end do
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do collapse(2) private(i,j,ni,nj,nn,xo,yo,xi,yi,dsqr,rho_sum) thread_limit(256)
    do mat = 0, nmats-1
      do j = 1, sizey-2
        do i = 1, sizex-2
          if (mc%vf(mcidx(i,j,mat,sizex,sizey)) > 0.0_real64) then
            xo = mc%x(cidx(i,j,sizex)); yo = mc%y(cidx(i,j,sizex)); rho_sum = 0.0_real64; nn = 0
            do nj = -1, 1
              do ni = -1, 1
                if (mc%vf(mcidx(i+ni,j+nj,mat,sizex,sizey)) > 0.0_real64) then
                  xi = mc%x(cidx(i+ni,j+nj,sizex)); yi = mc%y(cidx(i+ni,j+nj,sizex))
                  dsqr = (xo-xi)*(xo-xi) + (yo-yi)*(yo-yi)
                  if (dsqr > 0.0_real64) then
                    rho_sum = rho_sum + mc%rho(mcidx(i+ni,j+nj,mat,sizex,sizey)) / dsqr; nn = nn + 1
                  end if
                end if
              end do
            end do
            mc%rho_mat_ave(mcidx(i,j,mat,sizex,sizey)) = rho_sum / real(max(1,nn),real64)
          else
            mc%rho_mat_ave(mcidx(i,j,mat,sizex,sizey)) = 0.0_real64
          end if
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
    !$omp end target data
  end subroutine
end module
