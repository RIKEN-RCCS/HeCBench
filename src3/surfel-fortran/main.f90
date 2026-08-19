module surfel_reference_mod
  use iso_fortran_env, only: real32, int64
  implicit none
  integer, parameter :: col_dim = 7
  integer, parameter :: col_p_x = 0, col_p_y = 1, col_p_z = 2
  integer, parameter :: col_n_x = 3, col_n_y = 4, col_n_z = 5
  integer, parameter :: col_rsq = 6
contains
  function wall_time() result(t)
    real(real32) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real32) / real(rate, real32)
  end function wall_time

  pure integer function idx_s(i, col) result(pos)
    integer, intent(in) :: i, col
    pos = i * col_dim + col
  end function idx_s

  pure integer function idx_d(x, y, w) result(pos)
    integer, intent(in) :: x, y, w
    pos = y * w + x
  end function idx_d

  subroutine fill_surfels(src, n)
    real(real32), intent(out) :: src(0:)
    integer, intent(in) :: n
    integer :: i
    real(real32) :: nx, ny, nz, s
    call random_seed()
    do i = 0, n - 1
      call random_number(src(idx_s(i, col_p_x))); src(idx_s(i, col_p_x)) = -5.0_real32 + 10.0_real32 * src(idx_s(i, col_p_x))
      call random_number(src(idx_s(i, col_p_y))); src(idx_s(i, col_p_y)) = -5.0_real32 + 10.0_real32 * src(idx_s(i, col_p_y))
      call random_number(src(idx_s(i, col_p_z))); src(idx_s(i, col_p_z)) = 0.3_real32 + 4.7_real32 * src(idx_s(i, col_p_z))
      call random_number(nx); call random_number(ny); call random_number(nz)
      nx = -1.0_real32 + 2.0_real32 * nx
      ny = -1.0_real32 + 2.0_real32 * ny
      nz = -1.0_real32 + 2.0_real32 * nz
      s = sqrt(nx * nx + ny * ny + nz * nz)
      src(idx_s(i, col_n_x)) = nx / s
      src(idx_s(i, col_n_y)) = ny / s
      src(idx_s(i, col_n_z)) = nz / s
      call random_number(src(idx_s(i, col_rsq)))
      src(idx_s(i, col_rsq)) = 4.0e-4_real32 + (2.5e-3_real32 - 4.0e-4_real32) * src(idx_s(i, col_rsq))
    end do
  end subroutine fill_surfels

  subroutine surfel_render(src, n, f, w, h, dst)
    real(real32), intent(in) :: src(0:)
    integer, intent(in) :: n, w, h
    real(real32), intent(in) :: f
    real(real32), intent(out) :: dst(0:)
    integer :: x, y, i
    real(real32) :: ray0, ray1, ray2, p0, p1, p2, n0, n1, n2
    real(real32) :: rsqmax, pdotn, dsdotray, alpha, pt0, pt1, pt2, t, rsq, dmin
!$omp target teams distribute parallel do collapse(2) thread_limit(256) &
!$omp& map(to:src) map(from:dst) private(x,y,i,ray0,ray1,ray2,p0,p1,p2,n0,n1,n2,rsqmax,pdotn,dsdotray,alpha,pt0,pt1,pt2,t,rsq,dmin)
    do y = 0, h - 1
      do x = 0, w - 1
        ray0 = real(x, real32) - real(w - 1, real32) * 0.5_real32
        ray1 = real(y, real32) - real(h - 1, real32) * 0.5_real32
        ray2 = f
        dmin = 1.0e20_real32
        do i = 0, n - 1
          p0 = src(idx_s(i, col_p_x)); p1 = src(idx_s(i, col_p_y)); p2 = src(idx_s(i, col_p_z))
          n0 = src(idx_s(i, col_n_x)); n1 = src(idx_s(i, col_n_y)); n2 = src(idx_s(i, col_n_z))
          rsqmax = src(idx_s(i, col_rsq))
          pdotn = p0 * n0 + p1 * n1 + p2 * n2
          dsdotray = ray0 * n0 + ray1 * n1 + ray2 * n2
          alpha = pdotn / dsdotray
          pt0 = ray0 * alpha - p0
          pt1 = ray1 * alpha - p1
          pt2 = ray2 * alpha - p2
          t = ray2 * alpha
          rsq = pt0 * pt0 + pt1 * pt1 + pt2 * pt2
          if (rsq < rsqmax .and. dmin > t) dmin = t
        end do
        if (dmin > 100.0_real32) then
          dst(idx_d(x, y, w)) = 0.0_real32
        else
          dst(idx_d(x, y, w)) = dmin
        end if
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine surfel_render
end module surfel_reference_mod

program main
  use iso_fortran_env, only: real32
  use surfel_reference_mod
  implicit none
  integer :: n, w, h, repeat, argc, fidx, i, rep, src_size, dst_size, stat
  character(len=128) :: arg
  real(real32), allocatable :: src(:), dst(:), ref(:)
  real(real32) :: invf(0:2), t0, t1
  logical :: ok

  argc = command_argument_count()
  if (argc /= 4) then
    print '(a)', 'Usage: ./main <number of surfels> <output width> <output height> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) n
  call get_command_argument(2, arg); read(arg, *, iostat=stat) w
  call get_command_argument(3, arg); read(arg, *, iostat=stat) h
  call get_command_argument(4, arg); read(arg, *, iostat=stat) repeat

  src_size = n * col_dim
  dst_size = w * h
  allocate(src(0:src_size-1), dst(0:dst_size-1), ref(0:dst_size-1))
  call fill_surfels(src, n)
  invf = [0.005_real32, 0.02_real32, 0.036_real32]

  print '(a)', '-------------------------------------'
  print '(a)', ' surfelRenderTest with type float32  '
  print '(a)', '-------------------------------------'
  ok = .true.
!$omp target data map(to:src) map(alloc:dst)
  do fidx = 0, 2
    print '(/a,i0)', 'f = ', fidx
    call surfel_render(src, n, invf(fidx), w, h, ref)
    t0 = wall_time()
    do rep = 1, repeat
      call surfel_render(src, n, invf(fidx), w, h, dst)
    end do
    t1 = wall_time()
    print '(a,f12.6,a)', 'Average kernel execution time: ', (t1 - t0) * 1000.0_real32 / real(repeat, real32), ' (ms)'
!$omp target update from(dst)
    do i = 0, dst_size - 1
      if (abs(dst(i) - ref(i)) > 1.0e-3_real32) then
        print '(2f12.6)', dst(i), ref(i)
        ok = .false.
        exit
      end if
    end do
    if (.not. ok) exit
  end do
!$omp end target data
  print '(a)', merge('PASS', 'FAIL', ok)
end program main
