module rodrigues_mod
  use iso_fortran_env, only: real32
  implicit none

  type :: float3
    real(real32) :: x, y, z
  end type float3

  type :: float4
    real(real32) :: x, y, z, w
  end type float4

contains

  subroutine rotate(n, angle, w, d)
    integer, intent(in) :: n
    real(real32), intent(in) :: angle
    type(float3), intent(in) :: w
    type(float3), intent(inout) :: d(0:)
    integer :: i
    real(real32) :: s, c, mc, m1, m2, m3, m4, m5, m6, m7, m8, m9
    real(real32) :: ox, oy, oz
    type(float3) :: p

    !$omp target teams distribute parallel do thread_limit(256) private(s,c,mc,m1,m2,m3,m4,m5,m6,m7,m8,m9,ox,oy,oz,p)
    do i = 0, n - 1
      s = sin(angle)
      c = cos(angle)
      p = d(i)
      mc = 1.0_real32 - c
      m1 = c + w%x * w%x * mc
      m2 = w%z * s + w%x * w%y * mc
      m3 = -w%y * s + w%x * w%z * mc
      m4 = -w%z * s + w%x * w%y * mc
      m5 = c + w%y * w%y * mc
      m6 = w%x * s + w%y * w%z * mc
      m7 = w%y * s + w%x * w%z * mc
      m8 = -w%x * s + w%y * w%z * mc
      m9 = c + w%z * w%z * mc
      ox = p%x * m1 + p%y * m2 + p%z * m3
      oy = p%x * m4 + p%y * m5 + p%z * m6
      oz = p%x * m7 + p%y * m8 + p%z * m9
      d(i) = float3(ox, oy, oz)
    end do
    !$omp end target teams distribute parallel do
  end subroutine rotate

  subroutine rotate2(n, angle, w, d)
    integer, intent(in) :: n
    real(real32), intent(in) :: angle
    type(float3), intent(in) :: w
    type(float4), intent(inout) :: d(0:)
    integer :: i
    real(real32) :: s, c, mc, m1, m2, m3, m4, m5, m6, m7, m8, m9
    real(real32) :: ox, oy, oz
    type(float4) :: p

    !$omp target teams distribute parallel do thread_limit(256) private(s,c,mc,m1,m2,m3,m4,m5,m6,m7,m8,m9,ox,oy,oz,p)
    do i = 0, n - 1
      s = sin(angle)
      c = cos(angle)
      p = d(i)
      mc = 1.0_real32 - c
      m1 = c + w%x * w%x * mc
      m2 = w%z * s + w%x * w%y * mc
      m3 = -w%y * s + w%x * w%z * mc
      m4 = -w%z * s + w%x * w%y * mc
      m5 = c + w%y * w%y * mc
      m6 = w%x * s + w%y * w%z * mc
      m7 = w%y * s + w%x * w%z * mc
      m8 = -w%x * s + w%y * w%z * mc
      m9 = c + w%z * w%z * mc
      ox = p%x * m1 + p%y * m2 + p%z * m3
      oy = p%x * m4 + p%y * m5 + p%z * m6
      oz = p%x * m7 + p%y * m8 + p%z * m9
      d(i) = float4(ox, oy, oz, 0.0_real32)
    end do
    !$omp end target teams distribute parallel do
  end subroutine rotate2

end module rodrigues_mod
