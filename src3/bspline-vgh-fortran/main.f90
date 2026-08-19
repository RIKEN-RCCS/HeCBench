module bspline_vgh_mod
  use, intrinsic :: iso_fortran_env, only : real32, real64, int64
  implicit none
!$omp declare target (eval_ubspline_3d_s_vgh)
contains

  subroutine eval_ubspline_3d_s_vgh(coefs_init, xs, ys, zs, vals, grads, hess, &
                                    a, b, c, da, db, dc, d2a, d2b, d2c, dxinv, dyinv, dzinv)
    integer(int64), intent(in) :: xs, ys, zs
    real(real32), intent(in) :: coefs_init(0:), a(0:3), b(0:3), c(0:3)
    real(real32), intent(in) :: da(0:3), db(0:3), dc(0:3), d2a(0:3), d2b(0:3), d2c(0:3)
    real(real32), intent(out) :: vals, grads(0:2), hess(0:8)
    real(real32), intent(in) :: dxinv, dyinv, dzinv
    real(real32) :: h(0:8), v0, pre20, pre10, pre00, pre11, pre01, pre02
    real(real32) :: sum0, sum1, sum2
    integer :: i, j
    integer(int64) :: offset

    h = 0.0_real32
    v0 = 0.0_real32
    do i = 0, 3
      do j = 0, 3
        pre20 = d2a(i) * b(j)
        pre10 = da(i) * b(j)
        pre00 = a(i) * b(j)
        pre11 = da(i) * db(j)
        pre01 = a(i) * db(j)
        pre02 = a(i) * d2b(j)
        offset = int(i, int64) * xs + int(j, int64) * ys
        sum0 = c(0) * coefs_init(offset) + c(1) * coefs_init(offset + zs) + &
               c(2) * coefs_init(offset + 2_int64 * zs) + c(3) * coefs_init(offset + 3_int64 * zs)
        sum1 = dc(0) * coefs_init(offset) + dc(1) * coefs_init(offset + zs) + &
               dc(2) * coefs_init(offset + 2_int64 * zs) + dc(3) * coefs_init(offset + 3_int64 * zs)
        sum2 = d2c(0) * coefs_init(offset) + d2c(1) * coefs_init(offset + zs) + &
               d2c(2) * coefs_init(offset + 2_int64 * zs) + d2c(3) * coefs_init(offset + 3_int64 * zs)
        h(0) = h(0) + pre20 * sum0
        h(1) = h(1) + pre11 * sum0
        h(2) = h(2) + pre10 * sum1
        h(4) = h(4) + pre02 * sum0
        h(5) = h(5) + pre01 * sum1
        h(8) = h(8) + pre00 * sum2
        h(3) = h(3) + pre10 * sum0
        h(6) = h(6) + pre01 * sum0
        h(7) = h(7) + pre00 * sum1
        v0 = v0 + pre00 * sum0
      end do
    end do
    vals = v0
    grads(0) = h(3) * dxinv
    grads(1) = h(6) * dyinv
    grads(2) = h(7) * dzinv
    hess(0) = h(0) * dxinv * dxinv
    hess(1) = h(1) * dxinv * dyinv
    hess(2) = h(2) * dxinv * dzinv
    hess(3) = h(1) * dxinv * dyinv
    hess(4) = h(4) * dyinv * dyinv
    hess(5) = h(5) * dyinv * dzinv
    hess(6) = h(2) * dxinv * dzinv
    hess(7) = h(5) * dyinv * dzinv
    hess(8) = h(8) * dzinv * dzinv
  end subroutine eval_ubspline_3d_s_vgh

  subroutine eval_abc(af, tx, a)
    real(real32), intent(in) :: af(0:), tx
    real(real32), intent(out) :: a(0:3)
    a(0) = ((af(0) * tx + af(1)) * tx + af(2)) * tx + af(3)
    a(1) = ((af(4) * tx + af(5)) * tx + af(6)) * tx + af(7)
    a(2) = ((af(8) * tx + af(9)) * tx + af(10)) * tx + af(11)
    a(3) = ((af(12) * tx + af(13)) * tx + af(14)) * tx + af(15)
  end subroutine eval_abc

  subroutine bspline_ref(spline_coefs, xs, ys, zs, walkers_vals, walkers_grads, walkers_hess, &
                         a, b, c, da, db, dc, d2a, d2b, d2c, dxinv, dyinv, dzinv, nsplines, iw, ix, iy, iz)
    integer(int64), intent(in) :: xs, ys, zs
    integer, intent(in) :: nsplines, iw, ix, iy, iz
    real(real32), intent(in) :: spline_coefs(0:), a(0:3), b(0:3), c(0:3)
    real(real32), intent(in) :: da(0:3), db(0:3), dc(0:3), d2a(0:3), d2b(0:3), d2c(0:3)
    real(real32), intent(in) :: dxinv, dyinv, dzinv
    real(real32), intent(inout) :: walkers_vals(0:), walkers_grads(0:), walkers_hess(0:)
    integer :: n
    integer(int64) :: base
    base = int(ix, int64) * xs + int(iy, int64) * ys + int(iz, int64) * zs
    do n = 0, nsplines - 1
      call eval_ubspline_3d_s_vgh(spline_coefs(base+n:), xs, ys, zs, walkers_vals(iw*nsplines+n), &
           walkers_grads(iw*(nsplines*3+3)+n*3:), walkers_hess(iw*(nsplines*9+9)+n*9:), &
           a, b, c, da, db, dc, d2a, d2b, d2c, dxinv, dyinv, dzinv)
    end do
  end subroutine bspline_ref
end module bspline_vgh_mod

program bspline_vgh
  use, intrinsic :: iso_fortran_env, only : real32, real64, int64
  use omp_lib, only : omp_get_wtime
  use bspline_vgh_mod
  implicit none
  integer, parameter :: wsize = 12000, nsize = 2003, msize = nsize * 3 + 3, osize = nsize * 9 + 9
  integer, parameter :: nsize_round = ((nsize + 15) / 16) * 16
  integer(int64), parameter :: ssize = int(nsize_round, int64) * 48_int64 * 48_int64 * 48_int64
  real(real32), allocatable :: af(:), daf(:), d2af(:), walkers_vals(:), walkers_grads(:), walkers_hess(:)
  real(real32), allocatable :: walkers_vals_ref(:), walkers_grads_ref(:), walkers_hess_ref(:)
  real(real32), allocatable :: walkers_x(:), walkers_y(:), walkers_z(:), spline_coefs(:)
  real(real32) :: a(0:3), b(0:3), c(0:3), da(0:3), db(0:3), dc(0:3), d2a(0:3), d2b(0:3), d2c(0:3)
  real(real32) :: x, y, z, ux, uy, uz, tx, ty, tz
  real(real64) :: total_time, start_time, end_time
  integer :: i, n, ix, iy, iz, ipartx, iparty, ipartz
  integer(int64) :: ii, xs, ys, zs, base
  logical :: ok

  allocate(af(0:15), daf(0:15), d2af(0:15))
  af = [ -0.166667_real32, 0.500000_real32, -0.500000_real32, 0.166667_real32, &
         0.500000_real32, -1.000000_real32, 0.000000_real32, 0.666667_real32, &
        -0.500000_real32, 0.500000_real32, 0.500000_real32, 0.166667_real32, &
         0.166667_real32, 0.000000_real32, 0.000000_real32, 0.000000_real32 ]
  daf = [ 0.000000_real32, -0.500000_real32, 1.000000_real32, -0.500000_real32, &
          0.000000_real32, 1.500000_real32, -2.000000_real32, 0.000000_real32, &
          0.000000_real32, -1.500000_real32, 1.000000_real32, 0.500000_real32, &
          0.000000_real32, 0.500000_real32, 0.000000_real32, 0.000000_real32 ]
  d2af = [ 0.000000_real32, 0.000000_real32, -1.000000_real32, 1.000000_real32, &
           0.000000_real32, 0.000000_real32, 3.000000_real32, -2.000000_real32, &
           0.000000_real32, 0.000000_real32, -3.000000_real32, 1.000000_real32, &
           0.000000_real32, 0.000000_real32, 1.000000_real32, 0.000000_real32 ]

  x = 0.822387_real32; y = 0.989919_real32; z = 0.104573_real32
  allocate(walkers_vals(0:wsize*nsize-1), walkers_grads(0:wsize*msize-1), walkers_hess(0:wsize*osize-1))
  allocate(walkers_vals_ref(0:wsize*nsize-1), walkers_grads_ref(0:wsize*msize-1), walkers_hess_ref(0:wsize*osize-1))
  allocate(walkers_x(0:wsize-1), walkers_y(0:wsize-1), walkers_z(0:wsize-1))
  do i = 0, wsize - 1
    walkers_x(i) = x / real(wsize, real32)
    walkers_y(i) = y / real(wsize, real32)
    walkers_z(i) = z / real(wsize, real32)
  end do
  allocate(spline_coefs(0:ssize-1))
  do ii = 0_int64, ssize - 1_int64
    spline_coefs(ii) = real(sqrt(0.22_real64 + real(ii, real64)) * sin(real(ii, real64)), real32)
  end do

  xs = int(nsize_round, int64) * 48_int64 * 48_int64
  ys = int(nsize_round, int64) * 48_int64
  zs = int(nsize_round, int64)
  total_time = 0.0_real64

!$omp target data map(from:walkers_vals(0:wsize*nsize-1), walkers_grads(0:wsize*msize-1), walkers_hess(0:wsize*osize-1)) &
!$omp& map(to:spline_coefs(0:ssize-1)) map(alloc:a(0:3), b(0:3), c(0:3), da(0:3), db(0:3), dc(0:3), d2a(0:3), d2b(0:3), d2c(0:3))
  do i = 0, wsize - 1
    x = walkers_x(i); y = walkers_y(i); z = walkers_z(i)
    ux = x * 45.0_real32; uy = y * 45.0_real32; uz = z * 45.0_real32
    ipartx = int(ux); tx = ux - real(ipartx, real32); ix = min(max(0, ipartx), 44)
    iparty = int(uy); ty = uy - real(iparty, real32); iy = min(max(0, iparty), 44)
    ipartz = int(uz); tz = uz - real(ipartz, real32); iz = min(max(0, ipartz), 44)
    call eval_abc(af, tx, a); call eval_abc(af, ty, b); call eval_abc(af, tz, c)
    call eval_abc(daf, tx, da); call eval_abc(daf, ty, db); call eval_abc(daf, tz, dc)
    call eval_abc(d2af, tx, d2a); call eval_abc(d2af, ty, d2b); call eval_abc(d2af, tz, d2c)
    call bspline_ref(spline_coefs, xs, ys, zs, walkers_vals_ref, walkers_grads_ref, walkers_hess_ref, &
                     a, b, c, da, db, dc, d2a, d2b, d2c, 45.0_real32, 45.0_real32, 45.0_real32, nsize, i, ix, iy, iz)
!$omp target update to(a(0:3), b(0:3), c(0:3), da(0:3), db(0:3), dc(0:3), d2a(0:3), d2b(0:3), d2c(0:3))
    start_time = omp_get_wtime()
    base = int(ix, int64)*xs + int(iy, int64)*ys + int(iz, int64)*zs
!$omp target teams distribute parallel do thread_limit(256)
    do n = 0, nsize - 1
      call eval_ubspline_3d_s_vgh(spline_coefs(base+n:), xs, ys, zs, walkers_vals(i*nsize+n), &
        walkers_grads(i*msize+n*3:), walkers_hess(i*osize+n*9:), a, b, c, da, db, dc, d2a, d2b, d2c, &
        45.0_real32, 45.0_real32, 45.0_real32)
    end do
!$omp end target teams distribute parallel do
    end_time = omp_get_wtime()
    total_time = total_time + end_time - start_time
  end do
!$omp end target data

  write(*, '(A,F0.6,A)') 'Total kernel execution time ', total_time, ' (s)'
  ok = all(abs(walkers_vals - walkers_vals_ref) <= 2.0_real32) .and. &
       all(abs(walkers_grads - walkers_grads_ref) <= 2.0_real32) .and. &
       all(abs(walkers_hess - walkers_hess_ref) <= 2.0_real32)
  if (ok) then
    print '(A)', 'PASS'
  else
    print '(A)', 'FAIL'
  end if
  deallocate(af, daf, d2af, walkers_vals, walkers_grads, walkers_hess, walkers_vals_ref, walkers_grads_ref, walkers_hess_ref)
  deallocate(walkers_x, walkers_y, walkers_z, spline_coefs)
end program bspline_vgh
