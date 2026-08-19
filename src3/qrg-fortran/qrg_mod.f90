module qrg_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  integer, parameter :: qrng_dimensions = 3, qrng_resolution = 31
  integer, parameter :: n = 1048576
  real(real32), parameter :: int_scale = 1.0_real32 / real(int(z'80000001', int64), real32)
  integer(int64), save :: cjn(0:62,0:qrng_dimensions-1)
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  integer function generate_polynomials(buffer, primitive) result(total_degree)
    integer, intent(out) :: buffer(0:qrng_dimensions-1)
    logical, intent(in) :: primitive
    integer :: i, j, n0, p1, p2, l, e_p1, e_p2, e_b
    buffer(0) = 2; p2 = 0; l = 0
    do n0 = 1, qrng_dimensions-1
      p1 = buffer(n0-1) + 1
      do
        e_p1 = 30
        do while (iand(p1, ishft(1,e_p1)) == 0); e_p1 = e_p1 - 1; end do
        do i = 0, n0-1
          e_b = e_p1
          do while (iand(buffer(i), ishft(1,e_b)) == 0); e_b = e_b - 1; end do
          e_p2 = e_p1
          p2 = ieor(ishft(buffer(i), e_p2-e_b), p1)
          do while (p2 >= buffer(i))
            do while (iand(p2, ishft(1,e_p2)) == 0); e_p2 = e_p2 - 1; end do
            p2 = ieor(ishft(buffer(i), e_p2-e_b), p2)
          end do
          if (p2 == 0) exit
        end do
        if (p2 /= 0) then
          e_p2 = 0
          if (primitive) then
            j = not(ishft(int(z'FFFFFFFF'), e_p1+1))
            e_b = ior(ishft(1,e_p1), 1)
            p2 = e_b
            do e_p2 = ishft(1,e_p1)-2, 1, -1
              p2 = ishft(p2,1)
              i = iand(p2,p1)
              i = iand(i,int(z'55555555')) + iand(ishft(i,-1),int(z'55555555'))
              i = iand(i,int(z'33333333')) + iand(ishft(i,-2),int(z'33333333'))
              i = iand(i,int(z'07070707')) + iand(ishft(i,-4),int(z'07070707'))
              p2 = ior(p2, iand(mod(i,255),1))
              if (iand(p2,j) == e_b) exit
            end do
          end if
          if (e_p2 == 0) then
            buffer(n0) = p1
            l = l + e_p1
            exit
          end if
        end if
        p1 = p1 + 1
      end do
    end do
    total_degree = l + 1
  end function

  subroutine generate_cj()
    integer :: buffer(0:qrng_dimensions-1), polynomials(0:2047)
    integer :: n0, p1, l, e_p1, ppos, e, d, b_arr(0:1023), v_arr(0:1023), t_arr(0:1023)
    integer :: b0, v0, t0, m, m1, i, j, u, ip, it
    l = generate_polynomials(buffer, .false.)
    l = 0
    do n0 = 0, qrng_dimensions-1
      p1 = buffer(n0); e_p1 = 30
      do while (iand(p1, ishft(1,e_p1)) == 0); e_p1 = e_p1 - 1; end do
      polynomials(l) = 1; l = l + 1
      e_p1 = e_p1 - 1
      do while (e_p1 >= 0)
        polynomials(l) = iand(ishft(p1,-e_p1),1); l = l + 1; e_p1 = e_p1 - 1
      end do
      polynomials(l) = -1; l = l + 1
    end do
    polynomials(l) = -1
    ppos = 0; d = 0
    do while (polynomials(ppos) /= -1)
      cjn(:,d) = 0_int64
      e = 0
      do while (polynomials(ppos+e+1) /= -1); e = e + 1; end do
      b_arr = 0; v_arr = 0; t_arr = 0
      b0 = 1023; m = 0; b_arr(b0) = 1
      v0 = 1023 - (63 + e - 2)
      j = 62; u = e
      do while (j >= 0)
        if (u == e) then
          u = 0
          m1 = m; t0 = 1023 - m1
          do i = 0, m
            t_arr(t0+i) = b_arr(b0+i)
          end do
          m = m + e; b0 = 1023 - m
          do i = 0, m
            b_arr(b0+i) = 0
            ip = e - (m - i); it = m1
            do while (ip <= e .and. it >= 0)
              if (ip >= 0) b_arr(b0+i) = ieor(b_arr(b0+i), iand(polynomials(ppos+ip), t_arr(t0+it)))
              ip = ip + 1; it = it - 1
            end do
          end do
          do i = 0, m1-1; v_arr(v0+i) = 0; end do
          do i = m1, m-1; v_arr(v0+i) = 1; end do
          do i = m, 63+e-2
            v_arr(v0+i) = 0
            do it = 1, m
              v_arr(v0+i) = ieor(v_arr(v0+i), iand(v_arr(v0+i-it), b_arr(b0+it)))
            end do
          end do
        end if
        do i = 0, 62
          cjn(i,d) = ior(cjn(i,d), ishft(int(v_arr(v0+i+u),int64), j))
        end do
        j = j - 1; u = u + 1
      end do
      d = d + 1; ppos = ppos + e + 2
    end do
  end subroutine

  subroutine init_quasirandom_generator(table)
    integer(int32), intent(out) :: table(0:qrng_dimensions*qrng_resolution-1)
    integer :: dim, bit
    call generate_cj()
    do dim = 0, qrng_dimensions-1
      do bit = 0, qrng_resolution-1
        table(dim*qrng_resolution+bit) = int(iand(shiftr(cjn(bit,dim),32), int(z'7FFFFFFF',int64)), int32)
      end do
    end do
  end subroutine
end module
