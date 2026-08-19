module haccmk_module
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  use omp_lib
  implicit none
contains
  subroutine haccmk(repeat, n, ilp, fsrrmax, mp_rsm, fcoeff, xx, yy, zz, mass, vx2, vy2, vz2)
    integer, intent(in) :: repeat, n, ilp
    real(real32), intent(in) :: fsrrmax, mp_rsm, fcoeff
    real(real32), intent(in) :: xx(0:), yy(0:), zz(0:), mass(0:)
    real(real32), intent(inout) :: vx2(0:), vy2(0:), vz2(0:)
    integer :: iteration, i, j
    real(real32) :: total_time, ma0, ma1, ma2, ma3, ma4, ma5
    real(real32) :: dxc, dyc, dzc, m, r2, f, xi, yi, zi
    real(real64) :: start_time, end_time

!$omp target data map(to:xx(0:ilp),yy(0:ilp),zz(0:ilp),mass(0:ilp)) map(from:vx2(0:n),vy2(0:n),vz2(0:n))
    total_time = 0.0_real32
    do iteration = 1, repeat
!$omp target update to(vx2(0:n))
!$omp target update to(vy2(0:n))
!$omp target update to(vz2(0:n))
      start_time = omp_get_wtime()
!$omp target teams distribute parallel do private(ma0,ma1,ma2,ma3,ma4,ma5,dxc,dyc,dzc,m,r2,f,xi,yi,zi,j)
      do i = 0, n - 1
        ma0 = 0.269327_real32
        ma1 = -0.0750978_real32
        ma2 = 0.0114808_real32
        ma3 = -0.00109313_real32
        ma4 = 0.0000605491_real32
        ma5 = -0.00000147177_real32
        xi = 0.0_real32
        yi = 0.0_real32
        zi = 0.0_real32
        do j = 0, ilp - 1
          dxc = xx(j) - xx(i)
          dyc = yy(j) - yy(i)
          dzc = zz(j) - zz(i)
          r2 = dxc * dxc + dyc * dyc + dzc * dzc
          m = mass(j) * merge(1.0_real32, 0.0_real32, r2 < fsrrmax)
          f = r2 + mp_rsm
          f = m * (1.0_real32 / (f * sqrt(f)) - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))))
          xi = xi + f * dxc
          yi = yi + f * dyc
          zi = zi + f * dzc
        end do
        vx2(i) = vx2(i) + xi * fcoeff
        vy2(i) = vy2(i) + yi * fcoeff
        vz2(i) = vz2(i) + zi * fcoeff
      end do
!$omp end target teams distribute parallel do
      end_time = omp_get_wtime()
      total_time = total_time + real((end_time - start_time) * 1.0e9_real64, real32)
    end do
    print '(a,f0.6,a)', 'Average kernel execution time ', (total_time * 1.0e-9_real32) / real(repeat, real32), ' (s)'
!$omp end target data
  end subroutine haccmk

  subroutine haccmk_gold(count1, xxi, yyi, zzi, fsrrmax2, mp_rsm2, xx1, yy1, zz1, mass1, dxi, dyi, dzi)
    integer, intent(in) :: count1
    real(real32), intent(in) :: xxi, yyi, zzi, fsrrmax2, mp_rsm2
    real(real32), intent(in) :: xx1(0:), yy1(0:), zz1(0:), mass1(0:)
    real(real32), intent(out) :: dxi, dyi, dzi
    real(real32), parameter :: ma0=0.269327_real32, ma1=-0.0750978_real32, ma2=0.0114808_real32
    real(real32), parameter :: ma3=-0.00109313_real32, ma4=0.0000605491_real32, ma5=-0.00000147177_real32
    integer :: j
    real(real32) :: dxc, dyc, dzc, m, r2, f, xi, yi, zi
    xi = 0.0_real32; yi = 0.0_real32; zi = 0.0_real32
    do j = 0, count1 - 1
      dxc = xx1(j) - xxi; dyc = yy1(j) - yyi; dzc = zz1(j) - zzi
      r2 = dxc * dxc + dyc * dyc + dzc * dzc
      if (r2 < fsrrmax2) then
        m = mass1(j)
      else
        m = 0.0_real32
      end if
      f = r2 + mp_rsm2
      f = m * (1.0_real32 / (f * sqrt(f)) - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))))
      xi = xi + f * dxc; yi = yi + f * dyc; zi = zi + f * dzc
    end do
    dxi = xi; dyi = yi; dzi = zi
  end subroutine haccmk_gold
end module haccmk_module

program haccmk_main
  use, intrinsic :: iso_fortran_env, only : real32
  use haccmk_module
  implicit none
  integer, parameter :: n1 = 784, n2 = 15000
  integer :: argc, repeat, i, error
  real(real32) :: fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2
  real(real32), allocatable :: xx(:), yy(:), zz(:), mass(:), vx2(:), vy2(:), vz2(:), vx2_hw(:), vy2_hw(:), vz2_hw(:)
  character(len=64) :: argument
  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument); read(argument, *) repeat
  print '(a,i0)', 'Outer loop count is set ', n1
  print '(a,i0)', 'Inner loop count is set ', n2
  allocate(xx(0:n2-1), yy(0:n2-1), zz(0:n2-1), mass(0:n2-1), vx2(0:n2-1), vy2(0:n2-1), vz2(0:n2-1), &
           vx2_hw(0:n2-1), vy2_hw(0:n2-1), vz2_hw(0:n2-1))
  fcoeff = 0.23_real32; fsrrmax2 = 0.5_real32; mp_rsm2 = 0.03_real32
  dx1 = 1.0_real32 / real(n2, real32); dy1 = 2.0_real32 / real(n2, real32); dz1 = 3.0_real32 / real(n2, real32)
  xx(0) = 0.0_real32; yy(0) = 0.0_real32; zz(0) = 0.0_real32; mass(0) = 2.0_real32
  do i = 1, n2 - 1
    xx(i) = xx(i-1) + dx1; yy(i) = yy(i-1) + dy1; zz(i) = zz(i-1) + dz1
    mass(i) = real(i, real32) * 0.01_real32 + xx(i)
  end do
  vx2 = 0.0_real32; vy2 = 0.0_real32; vz2 = 0.0_real32
  vx2_hw = 0.0_real32; vy2_hw = 0.0_real32; vz2_hw = 0.0_real32
  do i = 0, n1 - 1
    call haccmk_gold(n2, xx(i), yy(i), zz(i), fsrrmax2, mp_rsm2, xx, yy, zz, mass, dx2, dy2, dz2)
    vx2(i) = vx2(i) + dx2 * fcoeff; vy2(i) = vy2(i) + dy2 * fcoeff; vz2(i) = vz2(i) + dz2 * fcoeff
  end do
  call haccmk(repeat, n1, n2, fsrrmax2, mp_rsm2, fcoeff, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw)
  error = 0
  do i = 0, n2 - 1
    if (abs(vx2(i) - vx2_hw(i)) > 1.0_real32) then
      print '(a,i0,1x,f0.6,1x,f0.6)', 'error at vx2[', i, vx2(i), vx2_hw(i); error = 1; exit
    end if
    if (abs(vy2(i) - vy2_hw(i)) > 1.0_real32) then
      print '(a,i0,a,1x,f0.6,1x,f0.6)', 'error at vy2[', i, ']:', vy2(i), vy2_hw(i); error = 1; exit
    end if
    if (abs(vz2(i) - vz2_hw(i)) > 1.0_real32) then
      print '(a,i0,a,1x,f0.6,1x,f0.6)', 'error at vz2[', i, ']:', vz2(i), vz2_hw(i); error = 1; exit
    end if
  end do
  if (error /= 0) then
    print '(a)', 'FAIL'
  else
    print '(a)', 'PASS'
  end if
end program haccmk_main
