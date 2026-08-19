module scatteradd_kernels
  use iso_fortran_env, only: real32
  implicit none
  integer, parameter :: max_threads_per_block = 512

contains

  subroutine scatter_add_reference(batch_size, vector_dim, out, idx, src)
    integer, intent(in) :: batch_size, vector_dim
    real(real32), intent(inout) :: out(0:)
    integer, intent(in) :: idx(0:)
    real(real32), intent(in) :: src(0:)
    integer :: d, i, index

    do d = 0, vector_dim - 1
      do i = 0, batch_size - 1
        index = idx(i)
        out(index * vector_dim + d) = out(index * vector_dim + d) + src(i * vector_dim + d)
      end do
    end do
  end subroutine scatter_add_reference

  subroutine scatterAdd_kernel(idx, src, out, batch_size, output_size, vector_dim)
    integer, intent(in) :: batch_size, output_size, vector_dim
    integer, intent(in) :: idx(0:)
    real(real32), intent(in) :: src(0:)
    real(real32), intent(inout) :: out(0:)
    integer :: d, i

    !$omp target teams distribute parallel do collapse(2) num_threads(max_threads_per_block)
    do d = 0, vector_dim - 1
      do i = 0, batch_size - 1
        !$omp atomic update
        out(idx(i) * vector_dim + d) = out(idx(i) * vector_dim + d) + src(i * vector_dim + d)
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine scatterAdd_kernel

  subroutine scatterAdd2_kernel(idx, src, out, batch_size, output_size, vector_dim)
    integer, intent(in) :: batch_size, output_size, vector_dim
    integer, intent(in) :: idx(0:)
    real(real32), intent(in) :: src(0:)
    real(real32), intent(inout) :: out(0:)
    integer :: i, d

    !$omp target teams distribute parallel do collapse(2) num_threads(max_threads_per_block)
    do i = 0, batch_size - 1
      do d = 0, vector_dim - 1
        !$omp atomic update
        out(idx(i) * vector_dim + d) = out(idx(i) * vector_dim + d) + src(i * vector_dim + d)
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine scatterAdd2_kernel

end module scatteradd_kernels
