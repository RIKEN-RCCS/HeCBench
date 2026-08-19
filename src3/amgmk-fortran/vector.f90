module amgmk_vector
  use amgmk_types
  implicit none
contains
  subroutine vector_create(v, n)
    type(seq_vector), intent(out) :: v
    integer, intent(in) :: n
    v%size = n
    allocate(v%data(1:n))
    v%data = 0.0_real64
  end subroutine vector_create

  subroutine vector_destroy(v)
    type(seq_vector), intent(inout) :: v
    if (allocated(v%data)) deallocate(v%data)
    v%size = 0
  end subroutine vector_destroy

  subroutine vector_set_constant(v, value)
    type(seq_vector), intent(inout) :: v
    real(real64), intent(in) :: value
    v%data = value
  end subroutine vector_set_constant
end module amgmk_vector
