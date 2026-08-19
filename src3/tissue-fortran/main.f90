program main
  use iso_fortran_env, only: real32, real64, int64
  use iso_c_binding, only: c_int
  use tissue_reference
  implicit none
  integer :: argc, dim, repeat, nnt, nntDev, nsp, step, i, rep, stat
  character(len=128) :: arg
  integer, allocatable :: tisspoints(:)
  real(real32), allocatable :: gtt(:), gbartt(:), ct(:), ctprev(:), qt(:), ct_gold(:)
  real(real64) :: t0, t1
  logical :: ok

  interface
    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <dimension of a 3D grid> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) dim
  if (dim > 32) then
    print '(a)', 'Maximum dimension is 32'
    stop 1
  end if
  call get_command_argument(2, arg); read(arg, *, iostat=stat) repeat
  nnt = dim * dim * dim
  nntDev = 32 * 32 * 32
  nsp = 2
  allocate(tisspoints(0:3*nntDev-1), gtt(0:nsp*nntDev-1), gbartt(0:nsp*nntDev-1))
  allocate(ct(0:nntDev-1), ctprev(0:nntDev-1), qt(0:nntDev-1), ct_gold(0:nntDev-1))
  do i = 0, 3 * nntDev - 1
    tisspoints(i) = modulo(c_rand(), nntDev / 3)
  end do
  do i = 0, nsp * nntDev - 1
    gtt(i) = real(c_rand(), real32) / real(huge(0_c_int), real32)
    gbartt(i) = real(c_rand(), real32) / real(huge(0_c_int), real32)
  end do
  do i = 0, nntDev - 1
    ct(i) = 0.0_real32
    ct_gold(i) = 0.0_real32
    ctprev(i) = real(c_rand(), real32) / real(huge(0_c_int), real32)
    qt(i) = real(c_rand(), real32) / real(huge(0_c_int), real32)
  end do
  step = 4
!$omp target data map(to:tisspoints(0:3*nntDev-1),gtt(0:nsp*nntDev-1),gbartt(0:nsp*nntDev-1), &
!$omp& ctprev(0:nntDev-1),qt(0:nntDev-1),ct(0:nntDev-1))
  do i = 1, 2
    call tissue(tisspoints, gtt, gbartt, ct, ctprev, qt, nnt, nntDev, step, 1)
    call tissue(tisspoints, gtt, gbartt, ct, ctprev, qt, nnt, nntDev, step, 2)
  end do
  do i = 1, 2
    call reference(tisspoints, gtt, gbartt, ct_gold, ctprev, qt, nnt, nntDev, step, 1)
    call reference(tisspoints, gtt, gbartt, ct_gold, ctprev, qt, nnt, nntDev, step, 2)
  end do
!$omp target update from(ct)
  ok = .true.
  do i = 0, nntDev - 1
    if (abs(ct(i) - ct_gold(i)) > 1.0e-1_real32) then
      print '(a,i0,a,f12.6,1x,f12.6)', '@', i, ': ', ct(i), ct_gold(i)
      ok = .false.
      exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
  t0 = wall_seconds()
  do rep = 1, repeat
    call tissue(tisspoints, gtt, gbartt, ct, ctprev, qt, nnt, nntDev, step, 1)
    call tissue(tisspoints, gtt, gbartt, ct, ctprev, qt, nnt, nntDev, step, 2)
  end do
  t1 = wall_seconds()
  print '(a,f12.6,a)', 'Average kernel execution time: ', (t1 - t0) / real(repeat, real64), ' (s)'
!$omp end target data
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine tissue(tisspoints, gtt, gbartt, ct, ctprev, qt, nnt, nntDev, step, isp)
    integer, intent(in) :: tisspoints(0:*)
    real(real32), intent(in) :: gtt(0:*), gbartt(0:*), ctprev(0:*), qt(0:*)
    real(real32), intent(inout) :: ct(0:*)
    integer, intent(in) :: nnt, nntDev, step, isp
    integer :: jtp, ixyz, ix, iy, iz, jx, jy, jz, nnt2, itp, itp1
    real(real32) :: p0, p1, p2, p3, p
!$omp target teams distribute parallel do thread_limit(256) &
!$omp& map(to:tisspoints(0:3*nntDev-1),gtt(0:2*nntDev-1),gbartt(0:2*nntDev-1), &
!$omp& ctprev(0:nntDev-1),qt(0:nntDev-1)) map(tofrom:ct(0:nntDev-1)) &
!$omp& private(jtp,ixyz,ix,iy,iz,jx,jy,jz,nnt2,itp1,p0,p1,p2,p3,p)
    do itp = 0, nnt - 1
      nnt2 = 2 * nnt
      ix = tisspoints(itp)
      iy = tisspoints(itp + nnt)
      iz = tisspoints(itp + nnt2)
      p0 = 0.0_real32
      p1 = 0.0_real32
      p2 = 0.0_real32
      p3 = 0.0_real32
      do itp1 = 0, step - 1
        p = 0.0_real32
        do jtp = itp1, nnt - 1, step
          jx = tisspoints(jtp)
          jy = tisspoints(jtp + nnt)
          jz = tisspoints(jtp + nnt2)
          ixyz = abs(jx - ix) + abs(jy - iy) + abs(jz - iz) + (isp - 1) * nntDev
          p = p + gtt(ixyz) * ctprev(jtp) + gbartt(ixyz) * qt(jtp)
        end do
        select case (itp1)
        case (0)
          p0 = p
        case (1)
          p1 = p
        case (2)
          p2 = p
        case (3)
          p3 = p
        end select
      end do
      ct(itp) = ((p0 + p1) + p2) + p3
    end do
!$omp end target teams distribute parallel do
  end subroutine tissue
end program main
