module qrg_reference
  use qrg_mod
  implicit none
contains
  real(real64) function get_quasirandom_value63(i0, dim) result(v)
    integer(int64), intent(in) :: i0
    integer, intent(in) :: dim
    integer(int64) :: result, i
    integer :: bit
    real(real64), parameter :: int63_scale = 1.0_real64 / real(int(z'8000000000000001', int64), real64)
    result = 0_int64; i = i0
    do bit = 0, 62
      if (iand(i,1_int64) /= 0) result = ieor(result, cjn(bit,dim))
      i = shiftr(i,1)
    end do
    v = real(result + 1_int64, real64) * int63_scale
  end function

  real(real64) function moro_inv_cnd_cpu(x0) result(z)
    integer(int32), intent(in) :: x0
    real(real64), parameter :: a1=2.50662823884_real64,a2=-18.61500062529_real64,a3=41.39119773534_real64,a4=-25.44106049637_real64
    real(real64), parameter :: b1=-8.4735109309_real64,b2=23.08336743743_real64,b3=-21.06224101826_real64,b4=3.13082909833_real64
    real(real64), parameter :: c1=0.337475482272615_real64,c2=0.976169019091719_real64,c3=0.160797971491821_real64,c4=2.76438810333863e-2_real64
    real(real64), parameter :: c5=3.8405729373609e-3_real64,c6=3.951896511919e-4_real64,c7=3.21767881768e-5_real64,c8=2.888167364e-7_real64,c9=3.960315187e-7_real64
    integer(int32) :: x
    logical :: negate
    real(real64) :: x1, x2, p1, p2
    x = x0; negate = .false.
    if (int(x,int64) >= int(z'80000000',int64)) then
      x = int(int(z'FFFFFFFF',int64) - int(x,int64), int32); negate = .true.
    end if
    x1 = 1.0_real64 / real(int(z'FFFFFFFF',int64),real64)
    x2 = x1 / 2.0_real64
    p1 = real(iand(int(x,int64),int(z'FFFFFFFF',int64)),real64) * x1 + x2
    p2 = p1 - 0.5_real64
    if (p2 > -0.42_real64) then
      z = p2*p2
      z = p2 * (((a4*z+a3)*z+a2)*z+a1) / ((((b4*z+b3)*z+b2)*z+b1)*z+1.0_real64)
    else
      z = log(-log(p1))
      z = -(c1 + z*(c2 + z*(c3 + z*(c4 + z*(c5 + z*(c6 + z*(c7 + z*(c8 + z*c9))))))))
    end if
    if (negate) z = -z
  end function

  real(real32) function moro_inv_cnd_gpu(x0) result(z)
    integer(int32), intent(in) :: x0
    z = real(moro_inv_cnd_cpu(x0), real32)
  end function
end module
