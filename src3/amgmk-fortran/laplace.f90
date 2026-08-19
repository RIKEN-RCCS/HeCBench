module amgmk_laplace
  use amgmk_types
  use amgmk_csr_matrix
  use amgmk_vector
  implicit none
contains
  subroutine generate_seq_laplacian(a, rhs, x, sol)
    type(csr_matrix), intent(out) :: a
    type(seq_vector), intent(out) :: rhs, x, sol
    integer :: ix, iy, iz, row, p, nnz
    real(real64), parameter :: diag = 6.0_real64, offdiag = -1.0_real64

    nnz = 0
    do iz = 0, nz-1; do iy = 0, ny-1; do ix = 0, nx-1
      nnz = nnz + 1
      if (iz > 0) nnz = nnz + 1
      if (iy > 0) nnz = nnz + 1
      if (ix > 0) nnz = nnz + 1
      if (ix < nx-1) nnz = nnz + 1
      if (iy < ny-1) nnz = nnz + 1
      if (iz < nz-1) nnz = nnz + 1
    end do; end do; end do
    call csr_create(a, nrows, nrows, nnz)
    call vector_create(rhs, nrows); call vector_create(x, nrows); call vector_create(sol, nrows)
    rhs%data = 1.0_real64
    p = 1; row = 1; a%row_ptr(1) = 1
    do iz = 0, nz-1; do iy = 0, ny-1; do ix = 0, nx-1
      a%col_ind(p) = row; a%data(p) = diag; p = p + 1
      if (iz > 0) then; a%col_ind(p)=row-nx*ny; a%data(p)=offdiag; p=p+1; end if
      if (iy > 0) then; a%col_ind(p)=row-nx; a%data(p)=offdiag; p=p+1; end if
      if (ix > 0) then; a%col_ind(p)=row-1; a%data(p)=offdiag; p=p+1; end if
      if (ix < nx-1) then; a%col_ind(p)=row+1; a%data(p)=offdiag; p=p+1; end if
      if (iy < ny-1) then; a%col_ind(p)=row+nx; a%data(p)=offdiag; p=p+1; end if
      if (iz < nz-1) then; a%col_ind(p)=row+nx*ny; a%data(p)=offdiag; p=p+1; end if
      a%row_ptr(row+1) = p; row = row + 1
    end do; end do; end do
    do row = 1, nrows
      sol%data(row) = sum(a%data(a%row_ptr(row):a%row_ptr(row+1)-1))
    end do
  end subroutine generate_seq_laplacian
end module amgmk_laplace
