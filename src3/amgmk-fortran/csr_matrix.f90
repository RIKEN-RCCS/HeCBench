module amgmk_csr_matrix
  use amgmk_types
  implicit none
contains
  subroutine csr_create(a, rows, cols, nnz)
    type(csr_matrix), intent(out) :: a
    integer, intent(in) :: rows, cols, nnz
    a%num_rows = rows; a%num_cols = cols; a%num_nonzeros = nnz
    allocate(a%row_ptr(1:rows+1), a%col_ind(1:nnz), a%data(1:nnz))
    a%row_ptr = 1; a%col_ind = 1; a%data = 0.0_real64
  end subroutine csr_create

  subroutine csr_destroy(a)
    type(csr_matrix), intent(inout) :: a
    if (allocated(a%row_ptr)) deallocate(a%row_ptr)
    if (allocated(a%col_ind)) deallocate(a%col_ind)
    if (allocated(a%data)) deallocate(a%data)
    a%num_rows = 0; a%num_cols = 0; a%num_nonzeros = 0
  end subroutine csr_destroy
end module amgmk_csr_matrix
