module amgmk_relax
  use amgmk_types
  implicit none
contains
  subroutine relax_gpu(values, row_ptr, col_ind, f, u, nnz)
    integer, intent(in) :: nnz
    real(real64), intent(in) :: values(1:nnz), f(1:nrows)
    integer, intent(in) :: row_ptr(1:nrows+1), col_ind(1:nnz)
    real(real64), intent(inout) :: u(1:nrows)
    integer :: row, entry
    real(real64) :: residual
    !$omp target teams distribute parallel do thread_limit(block_size) &
    !$omp& map(to: values(1:nnz), row_ptr(1:nrows+1), col_ind(1:nnz), f(1:nrows)) &
    !$omp& map(tofrom: u(1:nrows)) &
    !$omp& private(entry,residual)
    do row = 1, nrows
      residual = f(row)
      do entry = row_ptr(row)+1, row_ptr(row+1)-1
        residual = residual - values(entry) * u(col_ind(entry))
      end do
      u(row) = residual / values(row_ptr(row))
    end do
    !$omp end target teams distribute parallel do
  end subroutine relax_gpu
end module amgmk_relax
