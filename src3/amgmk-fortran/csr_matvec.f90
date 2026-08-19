module amgmk_csr_matvec
  use amgmk_types
  implicit none
contains
  subroutine csr_matvec(alpha, a, x, beta, y)
    real(real64), intent(in) :: alpha, beta
    type(csr_matrix), intent(in) :: a
    type(seq_vector), intent(in) :: x
    type(seq_vector), intent(inout) :: y
    integer :: row, entry
    real(real64) :: value
    do row = 1, a%num_rows
      value = 0.0_real64
      do entry = a%row_ptr(row), a%row_ptr(row+1)-1
        value = value + a%data(entry) * x%data(a%col_ind(entry))
      end do
      y%data(row) = alpha * value + beta * y%data(row)
    end do
  end subroutine csr_matvec
end module amgmk_csr_matvec
