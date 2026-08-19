program aidw
  use, intrinsic :: iso_fortran_env, only : real32, real64, int32, output_unit
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib
  implicit none

  integer, parameter :: block_size = 256
  real(real32), parameter :: a1 = 1.5_real32, a2 = 2.0_real32, a3 = 2.5_real32
  real(real32), parameter :: a4 = 3.0_real32, a5 = 3.5_real32
  real(real32), parameter :: r_min = 0.0_real32, r_max = 2.0_real32
  real(real32), parameter :: pi = 3.1415926_real32
  real(real32), parameter :: width = 2000.0_real32, height = 2000.0_real32
  real(real32), parameter :: area = width * height, eps = 1.0_real32

  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  integer :: argc, numk, check, iterations, dnum, inum
  integer :: i, r, tid, lid, m, e, part, num_threads, num_teams, ios
  character(len=64) :: argument
  real(real32), allocatable :: dx(:), dy(:), dz(:), avg_dist(:), ix(:), iy(:), iz(:), h_iz(:)
  real(real32) :: sdx(0:block_size-1), sdy(0:block_size-1), sdz(0:block_size-1)
  real(real32) :: dist, t, alpha, sum, z, r_obs, r_exp, r_s0, u_r
  real(real32) :: sum_up, sum_dn, six_s, siy_s, six_t, siy_t
  real(real64) :: start_time, end_time

  argc = command_argument_count()
  if (argc /= 3) then
    write(output_unit, '(a)') 'Usage: ./main <pts> <check> <iterations>'
    write(output_unit, '(a)') 'pts: number of points (unit: 1K)'
    write(output_unit, '(a)') 'check: enable verification when the value is 1'
    error stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *, iostat=ios) numk
  if (ios /= 0 .or. numk < 1) error stop 'invalid pts'
  call get_command_argument(2, argument)
  read(argument, *, iostat=ios) check
  if (ios /= 0) error stop 'invalid check'
  call get_command_argument(3, argument)
  read(argument, *, iostat=ios) iterations
  if (ios /= 0 .or. iterations < 1) error stop 'invalid iterations'

  if (numk > 2097151) error stop 'pts is too large'
  dnum = numk * 1024
  inum = dnum
  num_teams = (inum + block_size - 1) / block_size

  allocate(dx(0:dnum-1), dy(0:dnum-1), dz(0:dnum-1), avg_dist(0:dnum-1))
  allocate(ix(0:inum-1), iy(0:inum-1), iz(0:inum-1), h_iz(0:inum-1))

  call c_srand(123_c_int)
  do i = 0, dnum - 1
    dx(i) = real(c_rand(), real32) / real(huge(0_c_int), real32) * 1000.0_real32
    dy(i) = real(c_rand(), real32) / real(huge(0_c_int), real32) * 1000.0_real32
    dz(i) = real(c_rand(), real32) / real(huge(0_c_int), real32) * 1000.0_real32
  end do
  do i = 0, inum - 1
    ix(i) = real(c_rand(), real32) / real(huge(0_c_int), real32) * 1000.0_real32
    iy(i) = real(c_rand(), real32) / real(huge(0_c_int), real32) * 1000.0_real32
    iz(i) = 0.0_real32
  end do
  do i = 0, dnum - 1
    avg_dist(i) = real(c_rand(), real32) / real(huge(0_c_int), real32) * 3.0_real32
  end do

  write(output_unit, '(a,i0,a)') 'Size = : ', numk, ' K '
  write(output_unit, '(a,i0)') 'dnum = : ', dnum
  write(output_unit, '(a,i0)') 'inum = : ', inum

  if (check /= 0) then
    write(output_unit, '(a)') 'Verification enabled'
    call reference(dx, dy, dz, dnum, ix, iy, h_iz, inum, area, avg_dist)
  else
    write(output_unit, '(a)') 'Verification disabled'
  end if

  !$omp target data map(to: dx(0:dnum-1), dy(0:dnum-1), dz(0:dnum-1), &
  !$omp& ix(0:inum-1), iy(0:inum-1), avg_dist(0:dnum-1)) map(alloc: iz(0:inum-1))

  ! Untiled verification launch: one 256-thread target teams/distribute loop.
  !$omp target teams distribute parallel do thread_limit(block_size) private(sum, dist, t, z, alpha, r_obs, r_exp, r_s0, u_r, i)
  do tid = 0, inum - 1
    sum = 0.0_real32
    dist = 0.0_real32
    t = 0.0_real32
    z = 0.0_real32
    r_obs = avg_dist(tid)
    r_exp = 1.0_real32 / (2.0_real32 * sqrt(real(dnum, real32) / area))
    r_s0 = r_obs / r_exp
    u_r = 0.0_real32
    if (r_s0 >= r_min) u_r = 0.5_real32 - 0.5_real32 * cos(pi / r_max * (r_s0 - r_min))
    if (r_s0 >= r_max) u_r = 1.0_real32
    alpha = adaptive_alpha(u_r)
    do i = 0, dnum - 1
      dist = (ix(tid) - dx(i)) * (ix(tid) - dx(i)) + (iy(tid) - dy(i)) * (iy(tid) - dy(i))
      t = 1.0_real32 / (dist ** alpha)
      sum = sum + t
      z = z + dz(i) * t
    end do
    iz(tid) = z / sum
  end do

  !$omp target update from(iz(0:inum-1))
  if (check /= 0) then
    if (verify_results(iz, h_iz, inum, eps)) then
      write(output_unit, '(a)') 'PASS'
    else
      write(output_unit, '(a)') 'FAIL'
    end if
  end if

  ! Tiled verification launch: each team owns three shared 256-float tiles.
  !$omp target teams num_teams(num_teams) thread_limit(block_size) private(sdx, sdy, sdz)
  !$omp parallel private(lid, tid, dist, t, alpha, part, m, e, num_threads, sum_up, sum_dn, six_s, siy_s, six_t, siy_t, r_obs, r_exp, r_s0, u_r)
  lid = omp_get_thread_num()
  tid = omp_get_team_num() * block_size + lid
  if (tid < inum) then
    dist = 0.0_real32
    t = 0.0_real32
    part = (dnum - 1) / block_size
    sum_up = 0.0_real32
    sum_dn = 0.0_real32
    r_obs = avg_dist(tid)
    r_exp = 1.0_real32 / (2.0_real32 * sqrt(real(dnum, real32) / area))
    r_s0 = r_obs / r_exp
    u_r = 0.0_real32
    if (r_s0 >= r_min) u_r = 0.5_real32 - 0.5_real32 * cos(pi / r_max * (r_s0 - r_min))
    if (r_s0 >= r_max) u_r = 1.0_real32
    alpha = adaptive_alpha(u_r)
    six_t = ix(tid)
    siy_t = iy(tid)
    do m = 0, part
      num_threads = min(block_size, dnum - block_size * m)
      if (lid < num_threads) then
        sdx(lid) = dx(lid + block_size * m)
        sdy(lid) = dy(lid + block_size * m)
        sdz(lid) = dz(lid + block_size * m)
      end if
      !$omp barrier
      do e = 0, block_size - 1
        six_s = six_t - sdx(e)
        siy_s = siy_t - sdy(e)
        dist = six_s * six_s + siy_s * siy_s
        t = 1.0_real32 / (dist ** alpha)
        sum_dn = sum_dn + t
        sum_up = sum_up + t * sdz(e)
      end do
      !$omp barrier
    end do
    iz(tid) = sum_up / sum_dn
  end if
  !$omp end parallel
  !$omp end target teams

  !$omp target update from(iz(0:inum-1))
  if (check /= 0) then
    if (verify_results(iz, h_iz, inum, eps)) then
      write(output_unit, '(a)') 'PASS'
    else
      write(output_unit, '(a)') 'FAIL'
    end if
  end if

  start_time = omp_get_wtime()
  do r = 1, iterations
    !$omp target teams distribute parallel do thread_limit(block_size) private(sum, dist, t, z, alpha, r_obs, r_exp, r_s0, u_r, i)
    do tid = 0, inum - 1
      sum = 0.0_real32
      dist = 0.0_real32
      t = 0.0_real32
      z = 0.0_real32
      r_obs = avg_dist(tid)
      r_exp = 1.0_real32 / (2.0_real32 * sqrt(real(dnum, real32) / area))
      r_s0 = r_obs / r_exp
      u_r = 0.0_real32
      if (r_s0 >= r_min) u_r = 0.5_real32 - 0.5_real32 * cos(pi / r_max * (r_s0 - r_min))
      if (r_s0 >= r_max) u_r = 1.0_real32
      alpha = adaptive_alpha(u_r)
      do i = 0, dnum - 1
        dist = (ix(tid) - dx(i)) * (ix(tid) - dx(i)) + (iy(tid) - dy(i)) * (iy(tid) - dy(i))
        t = 1.0_real32 / (dist ** alpha)
        sum = sum + t
        z = z + dz(i) * t
      end do
      iz(tid) = z / sum
    end do
  end do
  end_time = omp_get_wtime()
  write(output_unit, '(a,f0.6,a)') 'Average execution time of AIDW_Kernel       ', &
    (end_time - start_time) / real(iterations, real64), ' (s)'

  start_time = omp_get_wtime()
  do r = 1, iterations
    !$omp target teams num_teams(num_teams) thread_limit(block_size) private(sdx, sdy, sdz)
    !$omp parallel private(lid, tid, dist, t, alpha, part, m, e, num_threads, sum_up, sum_dn, six_s, siy_s, six_t, siy_t, r_obs, r_exp, r_s0, u_r)
    lid = omp_get_thread_num()
    tid = omp_get_team_num() * block_size + lid
    if (tid < inum) then
      dist = 0.0_real32
      t = 0.0_real32
      part = (dnum - 1) / block_size
      sum_up = 0.0_real32
      sum_dn = 0.0_real32
      r_obs = avg_dist(tid)
      r_exp = 1.0_real32 / (2.0_real32 * sqrt(real(dnum, real32) / area))
      r_s0 = r_obs / r_exp
      u_r = 0.0_real32
      if (r_s0 >= r_min) u_r = 0.5_real32 - 0.5_real32 * cos(pi / r_max * (r_s0 - r_min))
      if (r_s0 >= r_max) u_r = 1.0_real32
      alpha = adaptive_alpha(u_r)
      six_t = ix(tid)
      siy_t = iy(tid)
      do m = 0, part
        num_threads = min(block_size, dnum - block_size * m)
        if (lid < num_threads) then
          sdx(lid) = dx(lid + block_size * m)
          sdy(lid) = dy(lid + block_size * m)
          sdz(lid) = dz(lid + block_size * m)
        end if
        !$omp barrier
        do e = 0, block_size - 1
          six_s = six_t - sdx(e)
          siy_s = siy_t - sdy(e)
          dist = six_s * six_s + siy_s * siy_s
          t = 1.0_real32 / (dist ** alpha)
          sum_dn = sum_dn + t
          sum_up = sum_up + t * sdz(e)
        end do
        !$omp barrier
      end do
      iz(tid) = sum_up / sum_dn
    end if
    !$omp end parallel
    !$omp end target teams
  end do
  end_time = omp_get_wtime()
  write(output_unit, '(a,f0.6,a)') 'Average execution time of AIDW_Kernel_Tiled ', &
    (end_time - start_time) / real(iterations, real64), ' (s)'

  !$omp end target data

contains

  pure function adaptive_alpha(u_r) result(alpha)
    real(real32), intent(in) :: u_r
    real(real32) :: alpha

    alpha = 0.0_real32
    if (u_r >= 0.0_real32 .and. u_r <= 0.1_real32) alpha = a1
    if (u_r > 0.1_real32 .and. u_r <= 0.3_real32) alpha = a1 * (1.0_real32 - 5.0_real32 * (u_r - 0.1_real32)) + a2 * 5.0_real32 * (u_r - 0.1_real32)
    if (u_r > 0.3_real32 .and. u_r <= 0.5_real32) alpha = a3 * 5.0_real32 * (u_r - 0.3_real32) + a1 * (1.0_real32 - 5.0_real32 * (u_r - 0.3_real32))
    if (u_r > 0.5_real32 .and. u_r <= 0.7_real32) alpha = a3 * (1.0_real32 - 5.0_real32 * (u_r - 0.5_real32)) + a4 * 5.0_real32 * (u_r - 0.5_real32)
    if (u_r > 0.7_real32 .and. u_r <= 0.9_real32) alpha = a5 * 5.0_real32 * (u_r - 0.7_real32) + a4 * (1.0_real32 - 5.0_real32 * (u_r - 0.7_real32))
    if (u_r > 0.9_real32 .and. u_r <= 1.0_real32) alpha = a5
    alpha = alpha * 0.5_real32
  end function adaptive_alpha
  subroutine reference(dx, dy, dz, dnum, ix, iy, iz, inum, area, avg_dist)
    integer, intent(in) :: dnum, inum
    real(real32), intent(in) :: dx(0:), dy(0:), dz(0:), ix(0:), iy(0:), area, avg_dist(0:)
    real(real32), intent(out) :: iz(0:)
    integer :: tid, j
    real(real32) :: sum, dist, t, z, alpha, r_obs, r_exp, r_s0, u_r

    !$omp parallel do private(sum, dist, t, z, alpha, r_obs, r_exp, r_s0, u_r, j)
    do tid = 0, inum - 1
      sum = 0.0_real32
      dist = 0.0_real32
      t = 0.0_real32
      z = 0.0_real32
      r_obs = avg_dist(tid)
      r_exp = 1.0_real32 / (2.0_real32 * sqrt(real(dnum, real32) / area))
      r_s0 = r_obs / r_exp
      u_r = 0.0_real32
      if (r_s0 >= r_min) u_r = 0.5_real32 - 0.5_real32 * cos(pi / r_max * (r_s0 - r_min))
      if (r_s0 >= r_max) u_r = 1.0_real32
      alpha = adaptive_alpha(u_r)
      do j = 0, dnum - 1
        dist = (ix(tid) - dx(j)) * (ix(tid) - dx(j)) + (iy(tid) - dy(j)) * (iy(tid) - dy(j))
        t = 1.0_real32 / (dist ** alpha)
        sum = sum + t
        z = z + dz(j) * t
      end do
      iz(tid) = z / sum
    end do
  end subroutine reference

  logical function verify_results(gold, test, len, tolerance) result(ok)
    real(real32), intent(in) :: gold(0:), test(0:), tolerance
    integer, intent(in) :: len
    integer :: i

    ok = .true.
    do i = 0, len - 1
      if (abs(gold(i) - test(i)) > tolerance) then
        write(output_unit, '(i0,1x,f0.6,1x,f0.6)') i, gold(i), test(i)
        ok = .false.
        return
      end if
    end do
  end function verify_results

end program aidw
