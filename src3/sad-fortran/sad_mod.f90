module sad_mod
  use iso_fortran_env, only: int8, real64
  use omp_lib
  use bmp_mod, only: u8
  implicit none
  integer, parameter :: block_size = 16 * 16
  integer, parameter :: threshold = 20

contains

  subroutine compute_sad_array(sad_array, image, kernel, sad_array_size, min_mse, num_occurrences, &
                               image_width, image_height, kernel_width, kernel_height, kernel_size, kernel_time)
    integer, intent(out) :: sad_array(0:)
    integer(int8), intent(in) :: image(0:), kernel(0:)
    integer, intent(in) :: sad_array_size, image_width, image_height, kernel_width, kernel_height, kernel_size
    integer, intent(out) :: min_mse, num_occurrences
    real(real64), intent(inout) :: kernel_time
    integer :: row, col, kr, kc, image_addr, kernel_addr, overlap_width, overlap_height
    integer :: m_r, m_g, m_b, t_r, t_g, t_b, sad_result, norm_sad, idx
    integer :: m, n, i
    real(real64) :: start_time, end_time

    start_time = omp_get_wtime()

    !$omp target teams distribute parallel do collapse(2) thread_limit(block_size) &
    !$omp& private(kr,kc,image_addr,kernel_addr,overlap_width,overlap_height,m_r,m_g,m_b,t_r,t_g,t_b,sad_result,norm_sad,idx)
    do row = 0, image_height - 1
      do col = 0, image_width - 1
        sad_result = 0
        overlap_width = min(image_width - col, kernel_width)
        overlap_height = min(image_height - row, kernel_height)
        do kr = 0, overlap_height - 1
          do kc = 0, overlap_width - 1
            image_addr = ((row + kr) * image_width + (col + kc)) * 3
            kernel_addr = (kr * kernel_width + kc) * 3
            m_r = u8(image(image_addr)); m_g = u8(image(image_addr + 1)); m_b = u8(image(image_addr + 2))
            t_r = u8(kernel(kernel_addr)); t_g = u8(kernel(kernel_addr + 1)); t_b = u8(kernel(kernel_addr + 2))
            sad_result = sad_result + abs(m_r - t_r) + abs(m_g - t_g) + abs(m_b - t_b)
          end do
        end do
        norm_sad = int(real(sad_result) / real(kernel_size))
        idx = row * image_width + col
        if (idx < sad_array_size) sad_array(idx) = norm_sad
      end do
    end do
    !$omp end target teams distribute parallel do

    m = threshold
    !$omp target teams distribute parallel do thread_limit(256) map(tofrom:m) reduction(min:m)
    do i = 0, sad_array_size - 1
      m = min(m, sad_array(i))
    end do
    !$omp end target teams distribute parallel do

    n = 0
    !$omp target teams distribute parallel do thread_limit(256) map(tofrom:n) reduction(+:n)
    do i = 0, sad_array_size - 1
      if (sad_array(i) == m) n = n + 1
    end do
    !$omp end target teams distribute parallel do

    end_time = omp_get_wtime()
    kernel_time = kernel_time + (end_time - start_time) * 1000.0_real64
    min_mse = m
    num_occurrences = n
  end subroutine compute_sad_array

end module sad_mod
