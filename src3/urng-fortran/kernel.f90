module urng_kernel_mod
  use iso_fortran_env, only: real32, int32
  implicit none
  integer, parameter :: group_size = 256, factor = 25
  integer, parameter :: ia = 16807, im = 2147483647, iq = 127773, ir = 2836, ntab = 16
  integer, parameter :: ndiv = 1 + (im - 1) / ntab
  real(real32), parameter :: am = 1.0_real32 / real(im, real32)
contains
  pure real(real32) function ran1(idum_in, tid) result(r)
    integer, intent(in) :: idum_in, tid
    integer :: idum, iv(0:ntab-1), j, k, iy
    idum = idum_in
    do j = ntab, 0, -1
      k = idum / iq
      idum = ia * (idum - k * iq) - ir * k
      if (idum < 0) idum = idum + im
      if (j < ntab) iv(j) = idum
    end do
    iy = iv(mod(tid, ntab))
    k = idum / iq
    idum = ia * (idum - k * iq) - ir * k
    if (idum < 0) idum = idum + im
    j = iy / ndiv
    iy = iv(mod(j, ntab))
    r = am * real(iy, real32)
  end function ran1

  pure integer function clamp_uchar(v)
    real(real32), intent(in) :: v
    if (v > 255.0_real32) then
      clamp_uchar = 255
    else if (v < 0.0_real32) then
      clamp_uchar = 0
    else
      clamp_uchar = int(v)
    end if
  end function clamp_uchar

  subroutine urng_kernel(input_image, output_image, size)
    integer, intent(in) :: input_image(0:)
    integer, intent(out) :: output_image(0:)
    integer, intent(in) :: size
    integer :: pos, base
    real(real32) :: tx, ty, tz, tw, avg, dev
!$omp target teams distribute parallel do thread_limit(256) &
!$omp& map(to:input_image) map(from:output_image) private(pos,base,tx,ty,tz,tw,avg,dev)
    do pos = 0, size - 1
      base = pos * 4
      tx = real(input_image(base + 0), real32)
      ty = real(input_image(base + 1), real32)
      tz = real(input_image(base + 2), real32)
      tw = real(input_image(base + 3), real32)
      avg = (tx + ty + tz + tw) / 4.0_real32
      dev = ran1(-int(avg), mod(pos, group_size))
      dev = (dev - 0.55_real32) * real(factor, real32)
      output_image(base + 0) = clamp_uchar(tx + dev)
      output_image(base + 1) = clamp_uchar(ty + dev)
      output_image(base + 2) = clamp_uchar(tz + dev)
      output_image(base + 3) = clamp_uchar(tw + dev)
    end do
!$omp end target teams distribute parallel do
  end subroutine urng_kernel
end module urng_kernel_mod
