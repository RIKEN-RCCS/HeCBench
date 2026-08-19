module rtm8_mod
  use iso_fortran_env, only: real32, real64
  implicit none
  integer, parameter :: nx = 680
  integer, parameter :: ny = 134
  integer, parameter :: nz = 450

contains

  pure integer function index_to_1d(x, y, z) result(idx)
    integer, intent(in) :: x, y, z
    idx = x + y * nx + z * nx * ny
  end function index_to_1d

  subroutine rtm8_cpu(vsq, current_s, current_r, next_s, next_r, image, a, n_total)
    real(real32), intent(in) :: vsq(0:), current_s(0:), current_r(0:), a(0:)
    real(real32), intent(inout) :: next_s(0:), next_r(0:), image(0:)
    integer, intent(in) :: n_total
    integer :: x, y, z, idx
    real(real32) :: div

    !$omp parallel do collapse(3) private(idx,div)
    do z = 4, nz - 5
      do y = 4, ny - 5
        do x = 4, nx - 5
          idx = index_to_1d(x, y, z)
          div = a(0) * current_s(idx) + &
            a(1) * (current_s(index_to_1d(x+1,y,z)) + current_s(index_to_1d(x-1,y,z)) + current_s(index_to_1d(x,y+1,z)) + current_s(index_to_1d(x,y-1,z)) + current_s(index_to_1d(x,y,z+1)) + current_s(index_to_1d(x,y,z-1))) + &
            a(2) * (current_s(index_to_1d(x+2,y,z)) + current_s(index_to_1d(x-2,y,z)) + current_s(index_to_1d(x,y+2,z)) + current_s(index_to_1d(x,y-2,z)) + current_s(index_to_1d(x,y,z+2)) + current_s(index_to_1d(x,y,z-2))) + &
            a(3) * (current_s(index_to_1d(x+3,y,z)) + current_s(index_to_1d(x-3,y,z)) + current_s(index_to_1d(x,y+3,z)) + current_s(index_to_1d(x,y-3,z)) + current_s(index_to_1d(x,y,z+3)) + current_s(index_to_1d(x,y,z-3))) + &
            a(4) * (current_s(index_to_1d(x+4,y,z)) + current_s(index_to_1d(x-4,y,z)) + current_s(index_to_1d(x,y+4,z)) + current_s(index_to_1d(x,y-4,z)) + current_s(index_to_1d(x,y,z+4)) + current_s(index_to_1d(x,y,z-4)))
          next_s(idx) = 2.0_real32 * current_s(idx) - next_s(idx) + vsq(idx) * div
          div = a(0) * current_r(idx) + &
            a(1) * (current_r(index_to_1d(x+1,y,z)) + current_r(index_to_1d(x-1,y,z)) + current_r(index_to_1d(x,y+1,z)) + current_r(index_to_1d(x,y-1,z)) + current_r(index_to_1d(x,y,z+1)) + current_r(index_to_1d(x,y,z-1))) + &
            a(2) * (current_r(index_to_1d(x+2,y,z)) + current_r(index_to_1d(x-2,y,z)) + current_r(index_to_1d(x,y+2,z)) + current_r(index_to_1d(x,y-2,z)) + current_r(index_to_1d(x,y,z+2)) + current_r(index_to_1d(x,y,z-2))) + &
            a(3) * (current_r(index_to_1d(x+3,y,z)) + current_r(index_to_1d(x-3,y,z)) + current_r(index_to_1d(x,y+3,z)) + current_r(index_to_1d(x,y-3,z)) + current_r(index_to_1d(x,y,z+3)) + current_r(index_to_1d(x,y,z-3))) + &
            a(4) * (current_r(index_to_1d(x+4,y,z)) + current_r(index_to_1d(x-4,y,z)) + current_r(index_to_1d(x,y+4,z)) + current_r(index_to_1d(x,y-4,z)) + current_r(index_to_1d(x,y,z+4)) + current_r(index_to_1d(x,y,z-4)))
          next_r(idx) = 2.0_real32 * current_r(idx) - next_r(idx) + vsq(idx) * div
          image(idx) = next_s(idx) * next_r(idx)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine rtm8_cpu

  subroutine rtm8_kernel(vsq, current_s, current_r, next_s, next_r, image, a)
    real(real32), intent(in) :: vsq(0:), current_s(0:), current_r(0:), a(0:)
    real(real32), intent(inout) :: next_s(0:), next_r(0:), image(0:)
    integer :: x, y, z, idx
    real(real32) :: div

    !$omp target teams distribute parallel do collapse(3) thread_limit(256) private(idx,div)
    do z = 4, nz - 5
      do y = 4, ny - 5
        do x = 4, nx - 5
          idx = index_to_1d(x, y, z)
          div = a(0) * current_s(idx) + &
            a(1) * (current_s(index_to_1d(x+1,y,z)) + current_s(index_to_1d(x-1,y,z)) + current_s(index_to_1d(x,y+1,z)) + current_s(index_to_1d(x,y-1,z)) + current_s(index_to_1d(x,y,z+1)) + current_s(index_to_1d(x,y,z-1))) + &
            a(2) * (current_s(index_to_1d(x+2,y,z)) + current_s(index_to_1d(x-2,y,z)) + current_s(index_to_1d(x,y+2,z)) + current_s(index_to_1d(x,y-2,z)) + current_s(index_to_1d(x,y,z+2)) + current_s(index_to_1d(x,y,z-2))) + &
            a(3) * (current_s(index_to_1d(x+3,y,z)) + current_s(index_to_1d(x-3,y,z)) + current_s(index_to_1d(x,y+3,z)) + current_s(index_to_1d(x,y-3,z)) + current_s(index_to_1d(x,y,z+3)) + current_s(index_to_1d(x,y,z-3))) + &
            a(4) * (current_s(index_to_1d(x+4,y,z)) + current_s(index_to_1d(x-4,y,z)) + current_s(index_to_1d(x,y+4,z)) + current_s(index_to_1d(x,y-4,z)) + current_s(index_to_1d(x,y,z+4)) + current_s(index_to_1d(x,y,z-4)))
          next_s(idx) = 2.0_real32 * current_s(idx) - next_s(idx) + vsq(idx) * div
          div = a(0) * current_r(idx) + &
            a(1) * (current_r(index_to_1d(x+1,y,z)) + current_r(index_to_1d(x-1,y,z)) + current_r(index_to_1d(x,y+1,z)) + current_r(index_to_1d(x,y-1,z)) + current_r(index_to_1d(x,y,z+1)) + current_r(index_to_1d(x,y,z-1))) + &
            a(2) * (current_r(index_to_1d(x+2,y,z)) + current_r(index_to_1d(x-2,y,z)) + current_r(index_to_1d(x,y+2,z)) + current_r(index_to_1d(x,y-2,z)) + current_r(index_to_1d(x,y,z+2)) + current_r(index_to_1d(x,y,z-2))) + &
            a(3) * (current_r(index_to_1d(x+3,y,z)) + current_r(index_to_1d(x-3,y,z)) + current_r(index_to_1d(x,y+3,z)) + current_r(index_to_1d(x,y-3,z)) + current_r(index_to_1d(x,y,z+3)) + current_r(index_to_1d(x,y,z-3))) + &
            a(4) * (current_r(index_to_1d(x+4,y,z)) + current_r(index_to_1d(x-4,y,z)) + current_r(index_to_1d(x,y+4,z)) + current_r(index_to_1d(x,y-4,z)) + current_r(index_to_1d(x,y,z+4)) + current_r(index_to_1d(x,y,z-4)))
          next_r(idx) = 2.0_real32 * current_r(idx) - next_r(idx) + vsq(idx) * div
          image(idx) = next_s(idx) * next_r(idx)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine rtm8_kernel

end module rtm8_mod
