module secp256k1_mod
  use iso_fortran_env, only: int64
  implicit none

  integer(int64), parameter :: MASK26 = int(z'3FFFFFF', int64)
  integer(int64), parameter :: MASK22 = int(z'3FFFFF', int64)
  integer(int64), parameter :: MASK32 = int(z'FFFFFFFF', int64)
  integer(int64), parameter :: R0 = int(z'3D10', int64)
  integer(int64), parameter :: R1 = int(z'400', int64)

  type :: secp256k1_fe
    integer(int64) :: n(0:9)
  end type secp256k1_fe

  type :: secp256k1_fe_storage
    integer(int64) :: n(0:7)
  end type secp256k1_fe_storage

  type :: secp256k1_ge
    type(secp256k1_fe) :: x
    type(secp256k1_fe) :: y
  end type secp256k1_ge

  type :: secp256k1_gej
    type(secp256k1_fe) :: x
    type(secp256k1_fe) :: y
    type(secp256k1_fe) :: z
  end type secp256k1_gej

  type :: secp256k1_ge_storage
    type(secp256k1_fe_storage) :: x
    type(secp256k1_fe_storage) :: y
  end type secp256k1_ge_storage

contains

  !$omp declare target

  subroutine secp256k1_fe_from_storage(r, a)
    type(secp256k1_fe), intent(out) :: r
    type(secp256k1_fe_storage), intent(in) :: a
    integer(int64) :: an(0:7)
    integer :: i

    do i = 0, 7
      an(i) = iand(a%n(i), MASK32)
    end do

    r%n(0) = iand(an(0), MASK26)
    r%n(1) = ior(shiftr(an(0), 26), iand(shiftl(an(1), 6), MASK26))
    r%n(2) = ior(shiftr(an(1), 20), iand(shiftl(an(2), 12), MASK26))
    r%n(3) = ior(shiftr(an(2), 14), iand(shiftl(an(3), 18), MASK26))
    r%n(4) = ior(shiftr(an(3), 8), iand(shiftl(an(4), 24), MASK26))
    r%n(5) = iand(shiftr(an(4), 2), MASK26)
    r%n(6) = ior(shiftr(an(4), 28), iand(shiftl(an(5), 4), MASK26))
    r%n(7) = ior(shiftr(an(5), 22), iand(shiftl(an(6), 10), MASK26))
    r%n(8) = ior(shiftr(an(6), 16), iand(shiftl(an(7), 16), MASK26))
    r%n(9) = shiftr(an(7), 10)
  end subroutine secp256k1_fe_from_storage

  subroutine secp256k1_fe_sqr_inner(r, a)
    integer(int64), intent(out) :: r(0:9)
    integer(int64), intent(in) :: a(0:9)
    integer(int64) :: c, d
    integer(int64) :: u0, u1, u2, u3, u4, u5, u6, u7, u8
    integer(int64) :: t0, t1, t2, t3, t4, t5, t6, t7, t9

    d = 2_int64*a(0)*a(9) + 2_int64*a(1)*a(8) + 2_int64*a(2)*a(7) + &
        2_int64*a(3)*a(6) + 2_int64*a(4)*a(5)
    t9 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = a(0)*a(0)

    d = d + 2_int64*a(1)*a(9) + 2_int64*a(2)*a(8) + 2_int64*a(3)*a(7) + &
        2_int64*a(4)*a(6) + a(5)*a(5)
    u0 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u0*R0
    t0 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u0*R1

    c = c + 2_int64*a(0)*a(1)
    d = d + 2_int64*a(2)*a(9) + 2_int64*a(3)*a(8) + 2_int64*a(4)*a(7) + &
        2_int64*a(5)*a(6)
    u1 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u1*R0
    t1 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u1*R1

    c = c + 2_int64*a(0)*a(2) + a(1)*a(1)
    d = d + 2_int64*a(3)*a(9) + 2_int64*a(4)*a(8) + 2_int64*a(5)*a(7) + &
        a(6)*a(6)
    u2 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u2*R0
    t2 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u2*R1

    c = c + 2_int64*a(0)*a(3) + 2_int64*a(1)*a(2)
    d = d + 2_int64*a(4)*a(9) + 2_int64*a(5)*a(8) + 2_int64*a(6)*a(7)
    u3 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u3*R0
    t3 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u3*R1

    c = c + 2_int64*a(0)*a(4) + 2_int64*a(1)*a(3) + a(2)*a(2)
    d = d + 2_int64*a(5)*a(9) + 2_int64*a(6)*a(8) + a(7)*a(7)
    u4 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u4*R0
    t4 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u4*R1

    c = c + 2_int64*a(0)*a(5) + 2_int64*a(1)*a(4) + 2_int64*a(2)*a(3)
    d = d + 2_int64*a(6)*a(9) + 2_int64*a(7)*a(8)
    u5 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u5*R0
    t5 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u5*R1

    c = c + 2_int64*a(0)*a(6) + 2_int64*a(1)*a(5) + 2_int64*a(2)*a(4) + &
        a(3)*a(3)
    d = d + 2_int64*a(7)*a(9) + a(8)*a(8)
    u6 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u6*R0
    t6 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u6*R1

    c = c + 2_int64*a(0)*a(7) + 2_int64*a(1)*a(6) + 2_int64*a(2)*a(5) + &
        2_int64*a(3)*a(4)
    d = d + 2_int64*a(8)*a(9)
    u7 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u7*R0
    t7 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u7*R1

    c = c + 2_int64*a(0)*a(8) + 2_int64*a(1)*a(7) + 2_int64*a(2)*a(6) + &
        2_int64*a(3)*a(5) + a(4)*a(4)
    d = d + a(9)*a(9)
    u8 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u8*R0

    r(3) = t3
    r(4) = t4
    r(5) = t5
    r(6) = t6
    r(7) = t7
    r(8) = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u8*R1
    c = c + d*R0 + t9
    r(9) = iand(c, shiftr(MASK26, 4))
    c = shiftr(c, 22)
    c = c + d*shiftl(R1, 4)
    d = c*shiftr(R0, 4) + t0
    r(0) = iand(d, MASK26)
    d = shiftr(d, 26)
    d = d + c*shiftr(R1, 4) + t1
    r(1) = iand(d, MASK26)
    d = shiftr(d, 26)
    d = d + t2
    r(2) = d
  end subroutine secp256k1_fe_sqr_inner

  subroutine secp256k1_fe_sqr(r, a)
    type(secp256k1_fe), intent(out) :: r
    type(secp256k1_fe), intent(in) :: a
    call secp256k1_fe_sqr_inner(r%n, a%n)
  end subroutine secp256k1_fe_sqr

  subroutine secp256k1_fe_normalize_weak(r)
    type(secp256k1_fe), intent(inout) :: r
    integer(int64) :: t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, x

    t0 = r%n(0); t1 = r%n(1); t2 = r%n(2); t3 = r%n(3); t4 = r%n(4)
    t5 = r%n(5); t6 = r%n(6); t7 = r%n(7); t8 = r%n(8); t9 = r%n(9)

    x = shiftr(t9, 22)
    t9 = iand(t9, MASK22)

    t0 = t0 + x*int(z'3D1', int64)
    t1 = t1 + shiftl(x, 6)
    t1 = t1 + shiftr(t0, 26); t0 = iand(t0, MASK26)
    t2 = t2 + shiftr(t1, 26); t1 = iand(t1, MASK26)
    t3 = t3 + shiftr(t2, 26); t2 = iand(t2, MASK26)
    t4 = t4 + shiftr(t3, 26); t3 = iand(t3, MASK26)
    t5 = t5 + shiftr(t4, 26); t4 = iand(t4, MASK26)
    t6 = t6 + shiftr(t5, 26); t5 = iand(t5, MASK26)
    t7 = t7 + shiftr(t6, 26); t6 = iand(t6, MASK26)
    t8 = t8 + shiftr(t7, 26); t7 = iand(t7, MASK26)
    t9 = t9 + shiftr(t8, 26); t8 = iand(t8, MASK26)

    r%n(0) = t0; r%n(1) = t1; r%n(2) = t2; r%n(3) = t3; r%n(4) = t4
    r%n(5) = t5; r%n(6) = t6; r%n(7) = t7; r%n(8) = t8; r%n(9) = t9
  end subroutine secp256k1_fe_normalize_weak

  subroutine secp256k1_fe_mul_inner(r, a, b)
    integer(int64), intent(out) :: r(0:9)
    integer(int64), intent(in) :: a(0:9), b(0:9)
    integer(int64) :: c, d
    integer(int64) :: u0, u1, u2, u3, u4, u5, u6, u7, u8
    integer(int64) :: t0, t1, t2, t3, t4, t5, t6, t7, t9

    d = a(0)*b(9) + a(1)*b(8) + a(2)*b(7) + a(3)*b(6) + a(4)*b(5) + &
        a(5)*b(4) + a(6)*b(3) + a(7)*b(2) + a(8)*b(1) + a(9)*b(0)
    t9 = iand(d, MASK26)
    d = shiftr(d, 26)

    c = a(0)*b(0)

    d = d + a(1)*b(9) + a(2)*b(8) + a(3)*b(7) + a(4)*b(6) + a(5)*b(5) + &
        a(6)*b(4) + a(7)*b(3) + a(8)*b(2) + a(9)*b(1)
    u0 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u0*R0
    t0 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u0*R1

    c = c + a(0)*b(1) + a(1)*b(0)
    d = d + a(2)*b(9) + a(3)*b(8) + a(4)*b(7) + a(5)*b(6) + a(6)*b(5) + &
        a(7)*b(4) + a(8)*b(3) + a(9)*b(2)
    u1 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u1*R0
    t1 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u1*R1

    c = c + a(0)*b(2) + a(1)*b(1) + a(2)*b(0)
    d = d + a(3)*b(9) + a(4)*b(8) + a(5)*b(7) + a(6)*b(6) + a(7)*b(5) + &
        a(8)*b(4) + a(9)*b(3)
    u2 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u2*R0
    t2 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u2*R1

    c = c + a(0)*b(3) + a(1)*b(2) + a(2)*b(1) + a(3)*b(0)
    d = d + a(4)*b(9) + a(5)*b(8) + a(6)*b(7) + a(7)*b(6) + a(8)*b(5) + &
        a(9)*b(4)
    u3 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u3*R0
    t3 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u3*R1

    c = c + a(0)*b(4) + a(1)*b(3) + a(2)*b(2) + a(3)*b(1) + a(4)*b(0)
    d = d + a(5)*b(9) + a(6)*b(8) + a(7)*b(7) + a(8)*b(6) + a(9)*b(5)
    u4 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u4*R0
    t4 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u4*R1

    c = c + a(0)*b(5) + a(1)*b(4) + a(2)*b(3) + a(3)*b(2) + a(4)*b(1) + &
        a(5)*b(0)
    d = d + a(6)*b(9) + a(7)*b(8) + a(8)*b(7) + a(9)*b(6)
    u5 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u5*R0
    t5 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u5*R1

    c = c + a(0)*b(6) + a(1)*b(5) + a(2)*b(4) + a(3)*b(3) + a(4)*b(2) + &
        a(5)*b(1) + a(6)*b(0)
    d = d + a(7)*b(9) + a(8)*b(8) + a(9)*b(7)
    u6 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u6*R0
    t6 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u6*R1

    c = c + a(0)*b(7) + a(1)*b(6) + a(2)*b(5) + a(3)*b(4) + a(4)*b(3) + &
        a(5)*b(2) + a(6)*b(1) + a(7)*b(0)
    d = d + a(8)*b(9) + a(9)*b(8)
    u7 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u7*R0
    t7 = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u7*R1

    c = c + a(0)*b(8) + a(1)*b(7) + a(2)*b(6) + a(3)*b(5) + a(4)*b(4) + &
        a(5)*b(3) + a(6)*b(2) + a(7)*b(1) + a(8)*b(0)
    d = d + a(9)*b(9)
    u8 = iand(d, MASK26)
    d = shiftr(d, 26)
    c = c + u8*R0

    r(3) = t3
    r(4) = t4
    r(5) = t5
    r(6) = t6
    r(7) = t7
    r(8) = iand(c, MASK26)
    c = shiftr(c, 26)
    c = c + u8*R1
    c = c + d*R0 + t9
    r(9) = iand(c, shiftr(MASK26, 4))
    c = shiftr(c, 22)
    c = c + d*shiftl(R1, 4)
    d = c*shiftr(R0, 4) + t0
    r(0) = iand(d, MASK26)
    d = shiftr(d, 26)
    d = d + c*shiftr(R1, 4) + t1
    r(1) = iand(d, MASK26)
    d = shiftr(d, 26)
    d = d + t2
    r(2) = d
  end subroutine secp256k1_fe_mul_inner

  subroutine secp256k1_fe_mul(r, a, b)
    type(secp256k1_fe), intent(out) :: r
    type(secp256k1_fe), intent(in) :: a, b
    call secp256k1_fe_mul_inner(r%n, a%n, b%n)
  end subroutine secp256k1_fe_mul

  subroutine secp256k1_fe_sqr_self(a)
    type(secp256k1_fe), intent(inout) :: a
    type(secp256k1_fe) :: t
    call secp256k1_fe_sqr(t, a)
    a = t
  end subroutine secp256k1_fe_sqr_self

  subroutine secp256k1_fe_mul_self(a, b)
    type(secp256k1_fe), intent(inout) :: a
    type(secp256k1_fe), intent(in) :: b
    type(secp256k1_fe) :: t
    call secp256k1_fe_mul(t, a, b)
    a = t
  end subroutine secp256k1_fe_mul_self

  subroutine secp256k1_fe_add(r, a)
    type(secp256k1_fe), intent(inout) :: r
    type(secp256k1_fe), intent(in) :: a
    integer :: i
    do i = 0, 9
      r%n(i) = r%n(i) + a%n(i)
    end do
  end subroutine secp256k1_fe_add

  subroutine secp256k1_fe_negate(r, a, m)
    type(secp256k1_fe), intent(out) :: r
    type(secp256k1_fe), intent(in) :: a
    integer, intent(in) :: m
    integer(int64) :: f

    f = 2_int64 * int(m + 1, int64)
    r%n(0) = int(z'3FFFC2F', int64)*f - a%n(0)
    r%n(1) = int(z'3FFFFBF', int64)*f - a%n(1)
    r%n(2) = int(z'3FFFFFF', int64)*f - a%n(2)
    r%n(3) = int(z'3FFFFFF', int64)*f - a%n(3)
    r%n(4) = int(z'3FFFFFF', int64)*f - a%n(4)
    r%n(5) = int(z'3FFFFFF', int64)*f - a%n(5)
    r%n(6) = int(z'3FFFFFF', int64)*f - a%n(6)
    r%n(7) = int(z'3FFFFFF', int64)*f - a%n(7)
    r%n(8) = int(z'3FFFFFF', int64)*f - a%n(8)
    r%n(9) = int(z'03FFFFF', int64)*f - a%n(9)
  end subroutine secp256k1_fe_negate

  subroutine secp256k1_fe_negate_self(a, m)
    type(secp256k1_fe), intent(inout) :: a
    integer, intent(in) :: m
    type(secp256k1_fe) :: t
    call secp256k1_fe_negate(t, a, m)
    a = t
  end subroutine secp256k1_fe_negate_self

  subroutine secp256k1_fe_mul_int(r, a)
    type(secp256k1_fe), intent(inout) :: r
    integer, intent(in) :: a
    r%n = r%n * int(a, int64)
  end subroutine secp256k1_fe_mul_int

  subroutine secp256k1_fe_set_int(r, a)
    type(secp256k1_fe), intent(out) :: r
    integer, intent(in) :: a
    r%n = 0_int64
    r%n(0) = int(a, int64)
  end subroutine secp256k1_fe_set_int

  integer function secp256k1_fe_is_odd(a) result(r)
    type(secp256k1_fe), intent(in) :: a
    r = int(iand(a%n(0), 1_int64))
  end function secp256k1_fe_is_odd

  subroutine secp256k1_fe_normalize_var(r)
    type(secp256k1_fe), intent(inout) :: r
    integer(int64) :: t0, t1, t2, t3, t4, t5, t6, t7, t8, t9
    integer(int64) :: m, x

    t0 = r%n(0); t1 = r%n(1); t2 = r%n(2); t3 = r%n(3); t4 = r%n(4)
    t5 = r%n(5); t6 = r%n(6); t7 = r%n(7); t8 = r%n(8); t9 = r%n(9)

    x = shiftr(t9, 22)
    t9 = iand(t9, MASK22)

    t0 = t0 + x*int(z'3D1', int64)
    t1 = t1 + shiftl(x, 6)
    t1 = t1 + shiftr(t0, 26); t0 = iand(t0, MASK26)
    t2 = t2 + shiftr(t1, 26); t1 = iand(t1, MASK26)
    t3 = t3 + shiftr(t2, 26); t2 = iand(t2, MASK26); m = t2
    t4 = t4 + shiftr(t3, 26); t3 = iand(t3, MASK26); m = iand(m, t3)
    t5 = t5 + shiftr(t4, 26); t4 = iand(t4, MASK26); m = iand(m, t4)
    t6 = t6 + shiftr(t5, 26); t5 = iand(t5, MASK26); m = iand(m, t5)
    t7 = t7 + shiftr(t6, 26); t6 = iand(t6, MASK26); m = iand(m, t6)
    t8 = t8 + shiftr(t7, 26); t7 = iand(t7, MASK26); m = iand(m, t7)
    t9 = t9 + shiftr(t8, 26); t8 = iand(t8, MASK26); m = iand(m, t8)

    x = shiftr(t9, 22)
    if (t9 == MASK22 .and. m == MASK26 .and. &
        (t1 + int(z'40', int64) + shiftr(t0 + int(z'3D1', int64), 26)) > MASK26) then
      x = ior(x, 1_int64)
    end if

    if (x /= 0_int64) then
      t0 = t0 + int(z'3D1', int64)
      t1 = t1 + shiftl(x, 6)
      t1 = t1 + shiftr(t0, 26); t0 = iand(t0, MASK26)
      t2 = t2 + shiftr(t1, 26); t1 = iand(t1, MASK26)
      t3 = t3 + shiftr(t2, 26); t2 = iand(t2, MASK26)
      t4 = t4 + shiftr(t3, 26); t3 = iand(t3, MASK26)
      t5 = t5 + shiftr(t4, 26); t4 = iand(t4, MASK26)
      t6 = t6 + shiftr(t5, 26); t5 = iand(t5, MASK26)
      t7 = t7 + shiftr(t6, 26); t6 = iand(t6, MASK26)
      t8 = t8 + shiftr(t7, 26); t7 = iand(t7, MASK26)
      t9 = t9 + shiftr(t8, 26); t8 = iand(t8, MASK26)
      t9 = iand(t9, MASK22)
    end if

    r%n(0) = t0; r%n(1) = t1; r%n(2) = t2; r%n(3) = t3; r%n(4) = t4
    r%n(5) = t5; r%n(6) = t6; r%n(7) = t7; r%n(8) = t8; r%n(9) = t9
  end subroutine secp256k1_fe_normalize_var

  subroutine secp256k1_fe_clear(a)
    type(secp256k1_fe), intent(out) :: a
    a%n = 0_int64
  end subroutine secp256k1_fe_clear

  subroutine secp256k1_fe_inv(r, a)
    type(secp256k1_fe), intent(out) :: r
    type(secp256k1_fe), intent(in) :: a
    type(secp256k1_fe) :: aa
    type(secp256k1_fe) :: x2, x3, x6, x9, x11, x22, x44, x88
    type(secp256k1_fe) :: x176, x220, x223, t1
    integer :: j

    aa = a
    call secp256k1_fe_sqr(x2, aa)
    call secp256k1_fe_mul_self(x2, aa)

    call secp256k1_fe_sqr(x3, x2)
    call secp256k1_fe_mul_self(x3, aa)

    x6 = x3
    do j = 0, 2
      call secp256k1_fe_sqr_self(x6)
    end do
    call secp256k1_fe_mul_self(x6, x3)

    x9 = x6
    do j = 0, 2
      call secp256k1_fe_sqr_self(x9)
    end do
    call secp256k1_fe_mul_self(x9, x3)

    x11 = x9
    do j = 0, 1
      call secp256k1_fe_sqr_self(x11)
    end do
    call secp256k1_fe_mul_self(x11, x2)

    x22 = x11
    do j = 0, 10
      call secp256k1_fe_sqr_self(x22)
    end do
    call secp256k1_fe_mul_self(x22, x11)

    x44 = x22
    do j = 0, 21
      call secp256k1_fe_sqr_self(x44)
    end do
    call secp256k1_fe_mul_self(x44, x22)

    x88 = x44
    do j = 0, 43
      call secp256k1_fe_sqr_self(x88)
    end do
    call secp256k1_fe_mul_self(x88, x44)

    x176 = x88
    do j = 0, 87
      call secp256k1_fe_sqr_self(x176)
    end do
    call secp256k1_fe_mul_self(x176, x88)

    x220 = x176
    do j = 0, 43
      call secp256k1_fe_sqr_self(x220)
    end do
    call secp256k1_fe_mul_self(x220, x44)

    x223 = x220
    do j = 0, 2
      call secp256k1_fe_sqr_self(x223)
    end do
    call secp256k1_fe_mul_self(x223, x3)

    t1 = x223
    do j = 0, 22
      call secp256k1_fe_sqr_self(t1)
    end do
    call secp256k1_fe_mul_self(t1, x22)
    do j = 0, 4
      call secp256k1_fe_sqr_self(t1)
    end do
    call secp256k1_fe_mul_self(t1, aa)
    do j = 0, 2
      call secp256k1_fe_sqr_self(t1)
    end do
    call secp256k1_fe_mul_self(t1, x2)
    do j = 0, 1
      call secp256k1_fe_sqr_self(t1)
    end do
    call secp256k1_fe_mul(r, aa, t1)
  end subroutine secp256k1_fe_inv

  subroutine secp256k1_fe_get_b32(r, a)
    integer, intent(out) :: r(0:31)
    type(secp256k1_fe), intent(in) :: a

    r(0) = int(iand(shiftr(a%n(9), 14), int(z'FF', int64)))
    r(1) = int(iand(shiftr(a%n(9), 6), int(z'FF', int64)))
    r(2) = int(ior(shiftl(iand(a%n(9), int(z'3F', int64)), 2), &
        iand(shiftr(a%n(8), 24), int(z'3', int64))))
    r(3) = int(iand(shiftr(a%n(8), 16), int(z'FF', int64)))
    r(4) = int(iand(shiftr(a%n(8), 8), int(z'FF', int64)))
    r(5) = int(iand(a%n(8), int(z'FF', int64)))
    r(6) = int(iand(shiftr(a%n(7), 18), int(z'FF', int64)))
    r(7) = int(iand(shiftr(a%n(7), 10), int(z'FF', int64)))
    r(8) = int(iand(shiftr(a%n(7), 2), int(z'FF', int64)))
    r(9) = int(ior(shiftl(iand(a%n(7), int(z'3', int64)), 6), &
        iand(shiftr(a%n(6), 20), int(z'3F', int64))))
    r(10) = int(iand(shiftr(a%n(6), 12), int(z'FF', int64)))
    r(11) = int(iand(shiftr(a%n(6), 4), int(z'FF', int64)))
    r(12) = int(ior(shiftl(iand(a%n(6), int(z'F', int64)), 4), &
        iand(shiftr(a%n(5), 22), int(z'F', int64))))
    r(13) = int(iand(shiftr(a%n(5), 14), int(z'FF', int64)))
    r(14) = int(iand(shiftr(a%n(5), 6), int(z'FF', int64)))
    r(15) = int(ior(shiftl(iand(a%n(5), int(z'3F', int64)), 2), &
        iand(shiftr(a%n(4), 24), int(z'3', int64))))
    r(16) = int(iand(shiftr(a%n(4), 16), int(z'FF', int64)))
    r(17) = int(iand(shiftr(a%n(4), 8), int(z'FF', int64)))
    r(18) = int(iand(a%n(4), int(z'FF', int64)))
    r(19) = int(iand(shiftr(a%n(3), 18), int(z'FF', int64)))
    r(20) = int(iand(shiftr(a%n(3), 10), int(z'FF', int64)))
    r(21) = int(iand(shiftr(a%n(3), 2), int(z'FF', int64)))
    r(22) = int(ior(shiftl(iand(a%n(3), int(z'3', int64)), 6), &
        iand(shiftr(a%n(2), 20), int(z'3F', int64))))
    r(23) = int(iand(shiftr(a%n(2), 12), int(z'FF', int64)))
    r(24) = int(iand(shiftr(a%n(2), 4), int(z'FF', int64)))
    r(25) = int(ior(shiftl(iand(a%n(2), int(z'F', int64)), 4), &
        iand(shiftr(a%n(1), 22), int(z'F', int64))))
    r(26) = int(iand(shiftr(a%n(1), 14), int(z'FF', int64)))
    r(27) = int(iand(shiftr(a%n(1), 6), int(z'FF', int64)))
    r(28) = int(ior(shiftl(iand(a%n(1), int(z'3F', int64)), 2), &
        iand(shiftr(a%n(0), 24), int(z'3', int64))))
    r(29) = int(iand(shiftr(a%n(0), 16), int(z'FF', int64)))
    r(30) = int(iand(shiftr(a%n(0), 8), int(z'FF', int64)))
    r(31) = int(iand(a%n(0), int(z'FF', int64)))
  end subroutine secp256k1_fe_get_b32

  subroutine secp256k1_ge_from_storage(r, a)
    type(secp256k1_ge), intent(out) :: r
    type(secp256k1_ge_storage), intent(in) :: a
    call secp256k1_fe_from_storage(r%x, a%x)
    call secp256k1_fe_from_storage(r%y, a%y)
  end subroutine secp256k1_ge_from_storage

  subroutine secp256k1_gej_set_ge(r, a)
    type(secp256k1_gej), intent(out) :: r
    type(secp256k1_ge), intent(in) :: a
    r%x = a%x
    r%y = a%y
    call secp256k1_fe_set_int(r%z, 1)
  end subroutine secp256k1_gej_set_ge

  subroutine secp256k1_gej_add_ge_var(r, a, b)
    type(secp256k1_gej), intent(out) :: r
    type(secp256k1_gej), intent(in) :: a
    type(secp256k1_ge), intent(in) :: b
    type(secp256k1_fe) :: z12, u1, u2, s1, s2, h, iv, i2, h2, h3, t

    call secp256k1_fe_sqr(z12, a%z)
    u1 = a%x
    call secp256k1_fe_normalize_weak(u1)
    call secp256k1_fe_mul(u2, b%x, z12)
    s1 = a%y
    call secp256k1_fe_normalize_weak(s1)
    call secp256k1_fe_mul(s2, b%y, z12)
    call secp256k1_fe_mul_self(s2, a%z)
    call secp256k1_fe_negate(h, u1, 1)
    call secp256k1_fe_add(h, u2)
    call secp256k1_fe_negate(iv, s1, 1)
    call secp256k1_fe_add(iv, s2)
    call secp256k1_fe_sqr(i2, iv)
    call secp256k1_fe_sqr(h2, h)
    call secp256k1_fe_mul(h3, h, h2)
    call secp256k1_fe_mul(r%z, a%z, h)
    call secp256k1_fe_mul(t, u1, h2)
    r%x = t
    call secp256k1_fe_mul_int(r%x, 2)
    call secp256k1_fe_add(r%x, h3)
    call secp256k1_fe_negate_self(r%x, 3)
    call secp256k1_fe_add(r%x, i2)
    call secp256k1_fe_negate(r%y, r%x, 5)
    call secp256k1_fe_add(r%y, t)
    call secp256k1_fe_mul_self(r%y, iv)
    call secp256k1_fe_mul_self(h3, s1)
    call secp256k1_fe_negate_self(h3, 1)
    call secp256k1_fe_add(r%y, h3)
  end subroutine secp256k1_gej_add_ge_var

  subroutine secp256k1_ge_set_gej(r, a)
    type(secp256k1_ge), intent(out) :: r
    type(secp256k1_gej), intent(inout) :: a
    type(secp256k1_fe) :: z2, z3

    call secp256k1_fe_inv(z2, a%z)
    a%z = z2
    call secp256k1_fe_sqr(z2, a%z)
    call secp256k1_fe_mul(z3, a%z, z2)
    call secp256k1_fe_mul_self(a%x, z2)
    call secp256k1_fe_mul_self(a%y, z3)
    call secp256k1_fe_set_int(a%z, 1)
    r%x = a%x
    r%y = a%y
  end subroutine secp256k1_ge_set_gej

  subroutine secp256k1_compute(prec, output)
    type(secp256k1_ge_storage), intent(in) :: prec(0:511)
    integer, intent(out) :: output(0:31)
    type(secp256k1_ge) :: ge(0:511)
    type(secp256k1_gej) :: sum, next_sum
    type(secp256k1_fe) :: z_all, z_inv
    integer :: i

    call secp256k1_ge_from_storage(ge(0), prec(0))
    call secp256k1_gej_set_ge(sum, ge(0))
    z_all = sum%z

    do i = 1, 511
      call secp256k1_ge_from_storage(ge(i), prec(i))
      call secp256k1_gej_add_ge_var(next_sum, sum, ge(i))
      sum = next_sum
      call secp256k1_fe_mul_self(z_all, sum%z)
    end do

    call secp256k1_fe_inv(z_inv, z_all)
    z_all = z_inv
    call secp256k1_fe_get_b32(output, z_all)
  end subroutine secp256k1_compute

  !$omp end declare target

  subroutine secp256k1_kernel(prec, output)
    type(secp256k1_ge_storage), intent(in) :: prec(0:511)
    integer, intent(out) :: output(0:31)
    integer :: k

    !$omp target teams distribute parallel do num_teams(1) thread_limit(1)
    do k = 0, 0
      call secp256k1_compute(prec, output)
    end do
    !$omp end target teams distribute parallel do
  end subroutine secp256k1_kernel

end module secp256k1_mod
