program adv
  use, intrinsic :: iso_fortran_env, only : int64, real64
  use omp_lib
  implicit none

  integer, parameter :: p_ijwid = 6, p_jwid = 5, p_np = 512, p_nvgeo = 12
  integer, parameter :: p_rxid = 0, p_ryid = 1, p_rzid = 7
  integer, parameter :: p_sxid = 2, p_syid = 3, p_szid = 8
  integer, parameter :: p_txid = 9, p_tyid = 10, p_tzid = 11
  integer, parameter :: p_cubnp = 4096

  integer :: argc, n, cubn, nelements, ntests, nq, cubnq, np, cubnp
  integer :: test, i, e, j, id, c, a, b, k, nidx, gid, element_id
  integer(int64) :: offset, clock_start, clock_end, clock_rate, rng_state, checksum_index
  real(real64) :: elapsed, gdof_per_second, checksum
  real(real64) :: s_cub_d(0:15, 0:15), s_cub_interp_t(0:7, 0:15)
  real(real64) :: s_u(0:7, 0:7), s_v(0:7, 0:7), s_w(0:7, 0:7)
  real(real64) :: s_u1(0:15, 0:15), s_v1(0:15, 0:15), s_w1(0:15, 0:15)
  real(real64) :: r_u(0:15), r_v(0:15), r_w(0:15)
  real(real64) :: r_ud(0:15), r_vd(0:15), r_wd(0:15)
  real(real64) :: u1_value, v1_value, w1_value, u2_value, v2_value, w2_value
  real(real64) :: udr, uds, udt, vdr, vds, vdt, wdr, wds, wdt
  real(real64) :: din, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz
  real(real64) :: jw, un, vn, wn, uhat, vhat, what, rhsv_u, rhsv_v, rhsv_w, ijw
  real(real64), allocatable :: vgeo(:), cubvgeo(:), cub_diff_interp_t(:)
  real(real64), allocatable :: cub_interp_t(:), u(:), adv_values(:)
  character(len=64) :: argument, dump_argument
  logical :: dump_values

  argc = command_argument_count()
  if (argc < 3) then
    print '(a)', 'Usage: ./adv N cubN numElements [nRepetitions]'
    stop 1
  end if

  call get_command_argument(1, argument)
  read(argument, *) n
  call get_command_argument(2, argument)
  read(argument, *) cubn
  call get_command_argument(3, argument)
  read(argument, *) nelements
  ntests = 1
  if (argc >= 4) then
    call get_command_argument(4, argument)
    read(argument, *) ntests
  end if
  dump_argument = ''
  call get_environment_variable('ADV_DUMP', dump_argument)
  dump_values = trim(dump_argument) == 'yes' .or. trim(dump_argument) == 'YES' .or. trim(dump_argument) == '1'

  nq = n + 1
  cubnq = cubn + 1
  np = nq * nq * nq
  cubnp = cubnq * cubnq * cubnq
  offset = int(nelements, int64) * int(np, int64)

  print '(a,i0)', 'Data type in bytes: ', storage_size(0.0_real64) / 8

  allocate(vgeo(0:int(np, int64) * nelements * p_nvgeo - 1))
  allocate(cubvgeo(0:int(cubnp, int64) * nelements * p_nvgeo - 1))
  allocate(cub_diff_interp_t(0:3_int64 * cubnp * nelements - 1))
  allocate(cub_interp_t(0:int(np, int64) * cubnp - 1))
  allocate(u(0:3_int64 * np * nelements - 1))
  allocate(adv_values(0:3_int64 * np * nelements - 1))

  call srand48(123_int64, rng_state)
  call drand_alloc(vgeo, rng_state)
  call drand_alloc(cubvgeo, rng_state)
  call drand_alloc(cub_diff_interp_t, rng_state)
  call drand_alloc(cub_interp_t, rng_state)
  call drand_alloc(u, rng_state)
  call drand_alloc(adv_values, rng_state)

!$omp target data map(to: vgeo, cubvgeo, cub_diff_interp_t, cub_interp_t, u) map(from: adv_values)
  call system_clock(clock_start, clock_rate)
  do test = 0, ntests - 1
!$omp target teams num_teams(nelements) thread_limit(256) private(e, i, j, id, c, a, b, k, nidx, gid, element_id, &
!$omp& s_cub_d, s_cub_interp_t, s_u, s_v, s_w, s_u1, s_v1, s_w1, r_u, r_v, r_w, r_ud, r_vd, r_wd, &
!$omp& u1_value, v1_value, w1_value, u2_value, v2_value, w2_value, udr, uds, udt, vdr, vds, vdt, &
!$omp& wdr, wds, wdt, din, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz, jw, un, vn, wn, &
!$omp& uhat, vhat, what, rhsv_u, rhsv_v, rhsv_w, ijw)

!$omp parallel private(i, j, id, c, a, b, k, nidx, gid, element_id, r_u, r_v, r_w, r_ud, r_vd, r_wd, &
!$omp& u1_value, v1_value, w1_value, u2_value, v2_value, w2_value, udr, uds, udt, vdr, vds, vdt, &
!$omp& wdr, wds, wdt, din, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz, jw, un, vn, wn, &
!$omp& uhat, vhat, what, rhsv_u, rhsv_v, rhsv_w, ijw) shared(s_cub_d, s_cub_interp_t, s_u, s_v, s_w, s_u1, s_v1, s_w1)
        element_id = omp_get_team_num()
        i = modulo(omp_get_thread_num(), 16)
        j = omp_get_thread_num() / 16
        id = j * 16 + i

        if (id < 8 * 16) s_cub_interp_t(j, i) = cub_interp_t(id)
        s_cub_d(j, i) = cub_diff_interp_t(id)

        r_u = 0.0_real64
        r_v = 0.0_real64
        r_w = 0.0_real64
        r_ud = 0.0_real64
        r_vd = 0.0_real64
        r_wd = 0.0_real64

        do c = 0, 7
          if (j < 8 .and. i < 8) then
            id = element_id * p_np + c * 8 * 8 + j * 8 + i
            s_u(j, i) = u(int(id, int64) + 0_int64 * offset)
            s_v(j, i) = u(int(id, int64) + 1_int64 * offset)
            s_w(j, i) = u(int(id, int64) + 2_int64 * offset)
          end if

!$omp barrier
          if (j < 8) then
            u1_value = 0.0_real64
            v1_value = 0.0_real64
            w1_value = 0.0_real64
            do a = 0, 7
              u1_value = u1_value + s_cub_interp_t(a, i) * s_u(j, a)
              v1_value = v1_value + s_cub_interp_t(a, i) * s_v(j, a)
              w1_value = w1_value + s_cub_interp_t(a, i) * s_w(j, a)
            end do
            s_u1(j, i) = u1_value
            s_v1(j, i) = v1_value
            s_w1(j, i) = w1_value
          else
            s_u1(j, i) = 0.0_real64
            s_v1(j, i) = 0.0_real64
            s_w1(j, i) = 0.0_real64
          end if

!$omp barrier
          u2_value = 0.0_real64
          v2_value = 0.0_real64
          w2_value = 0.0_real64
          do b = 0, 7
            u2_value = u2_value + s_cub_interp_t(b, j) * s_u1(b, i)
            v2_value = v2_value + s_cub_interp_t(b, j) * s_v1(b, i)
            w2_value = w2_value + s_cub_interp_t(b, j) * s_w1(b, i)
          end do
          do k = 0, 15
            r_u(k) = r_u(k) + s_cub_interp_t(c, k) * u2_value
            r_v(k) = r_v(k) + s_cub_interp_t(c, k) * v2_value
            r_w(k) = r_w(k) + s_cub_interp_t(c, k) * w2_value
          end do
          r_ud = r_u
          r_vd = r_v
          r_wd = r_w
        end do

        do k = 0, 15
          s_u1(j, i) = r_ud(k)
          s_v1(j, i) = r_vd(k)
          s_w1(j, i) = r_wd(k)

!$omp barrier
          udr = 0.0_real64
          uds = 0.0_real64
          udt = 0.0_real64
          vdr = 0.0_real64
          vds = 0.0_real64
          vdt = 0.0_real64
          wdr = 0.0_real64
          wds = 0.0_real64
          wdt = 0.0_real64
          do nidx = 0, 15
            din = s_cub_d(i, nidx)
            udr = udr + din * s_u1(j, nidx)
            vdr = vdr + din * s_v1(j, nidx)
            wdr = wdr + din * s_w1(j, nidx)
          end do
          do nidx = 0, 15
            din = s_cub_d(j, nidx)
            uds = uds + din * s_u1(nidx, i)
            vds = vds + din * s_v1(nidx, i)
            wds = wds + din * s_w1(nidx, i)
          end do
          do nidx = 0, 15
            din = s_cub_d(k, nidx)
            udt = udt + din * r_ud(nidx)
            vdt = vdt + din * r_vd(nidx)
            wdt = wdt + din * r_wd(nidx)
          end do

          gid = element_id * p_cubnp * p_nvgeo + k * 16 * 16 + j * 16 + i
          drdx = cubvgeo(gid + p_rxid * p_cubnp)
          drdy = cubvgeo(gid + p_ryid * p_cubnp)
          drdz = cubvgeo(gid + p_rzid * p_cubnp)
          dsdx = cubvgeo(gid + p_sxid * p_cubnp)
          dsdy = cubvgeo(gid + p_syid * p_cubnp)
          dsdz = cubvgeo(gid + p_szid * p_cubnp)
          dtdx = cubvgeo(gid + p_txid * p_cubnp)
          dtdy = cubvgeo(gid + p_tyid * p_cubnp)
          dtdz = cubvgeo(gid + p_tzid * p_cubnp)
          jw = cubvgeo(gid + p_jwid * p_cubnp)
          un = r_u(k)
          vn = r_v(k)
          wn = r_w(k)
          uhat = jw * (un * drdx + vn * drdy + wn * drdz)
          vhat = jw * (un * dsdx + vn * dsdy + wn * dsdz)
          what = jw * (un * dtdx + vn * dtdy + wn * dtdz)
          r_u(k) = uhat * udr + vhat * uds + what * udt
          r_v(k) = uhat * vdr + vhat * vds + what * vdt
          r_w(k) = uhat * wdr + vhat * wds + what * wdt
        end do

        do c = 0, 7
          rhsv_u = 0.0_real64
          rhsv_v = 0.0_real64
          rhsv_w = 0.0_real64
          do k = 0, 15
            rhsv_u = rhsv_u + s_cub_interp_t(c, k) * r_u(k)
            rhsv_v = rhsv_v + s_cub_interp_t(c, k) * r_v(k)
            rhsv_w = rhsv_w + s_cub_interp_t(c, k) * r_w(k)
          end do
          if (i < 8 .and. j < 8) then
            s_u(j, i) = rhsv_u
            s_v(j, i) = rhsv_v
            s_w(j, i) = rhsv_w
          end if

!$omp barrier
          if (j < 8) then
            rhsv_u = 0.0_real64
            rhsv_v = 0.0_real64
            rhsv_w = 0.0_real64
            do k = 0, 15
              if (k < 8 .and. i < 8) then
                rhsv_u = rhsv_u + s_cub_interp_t(j, k) * s_u(k, i)
                rhsv_v = rhsv_v + s_cub_interp_t(j, k) * s_v(k, i)
                rhsv_w = rhsv_w + s_cub_interp_t(j, k) * s_w(k, i)
              end if
            end do
            s_u1(j, i) = rhsv_u
            s_v1(j, i) = rhsv_v
            s_w1(j, i) = rhsv_w
          end if

!$omp barrier
          if (i < 8 .and. j < 8) then
            rhsv_u = 0.0_real64
            rhsv_v = 0.0_real64
            rhsv_w = 0.0_real64
            do k = 0, 15
              rhsv_u = rhsv_u + s_cub_interp_t(i, k) * s_u1(j, k)
              rhsv_v = rhsv_v + s_cub_interp_t(i, k) * s_v1(j, k)
              rhsv_w = rhsv_w + s_cub_interp_t(i, k) * s_w1(j, k)
            end do
            gid = element_id * p_np * p_nvgeo + c * 8 * 8 + j * 8 + i
            ijw = vgeo(gid + p_ijwid * p_np)
            id = element_id * p_np + c * 8 * 8 + j * 8 + i
            adv_values(int(id, int64) + 0_int64 * offset) = ijw * rhsv_u
            adv_values(int(id, int64) + 1_int64 * offset) = ijw * rhsv_v
            adv_values(int(id, int64) + 2_int64 * offset) = ijw * rhsv_w
          end if
        end do
!$omp end parallel
!$omp end target teams
  end do
  call system_clock(clock_end)
!$omp end target data

  elapsed = real(clock_end - clock_start, real64) * 1.0e9_real64 / real(clock_rate, real64) / real(ntests, real64)
  gdof_per_second = real(n * n * n * nelements, real64) / elapsed
  write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') ' NRepetitions=', ntests, ' N=', n, ' cubN=', cubn, &
       ' Nelements=', nelements, ' elapsed time=', elapsed, ' GDOF/s=', gdof_per_second

  checksum = 0.0_real64
  do checksum_index = 0, ubound(adv_values, 1)
    checksum = checksum + adv_values(checksum_index)
    if (dump_values) write (*, '(es24.16)') adv_values(checksum_index)
  end do
  write (*, '(a,es24.16)') 'Checksum=', checksum

  deallocate(vgeo, cubvgeo, cub_diff_interp_t, cub_interp_t, u, adv_values)

contains

  subroutine srand48(seed, state)
    integer(int64), intent(in) :: seed
    integer(int64), intent(out) :: state
    state = ior(shiftl(iand(seed, int(z'FFFFFFFF', int64)), 16), int(z'330E', int64))
  end subroutine srand48

  subroutine drand_alloc(values, state)
    real(real64), intent(out) :: values(0:)
    integer(int64), intent(inout) :: state
    integer(int64), parameter :: mask24 = int(z'FFFFFF', int64)
    integer(int64), parameter :: multiplier_low = int(z'ECE66D', int64)
    integer(int64), parameter :: multiplier_high = int(z'5DE', int64)
    integer(int64), parameter :: increment = int(z'B', int64)
    integer(int64) :: old_low, low, high, product, carry, index

    do index = 0, ubound(values, 1)
      old_low = iand(state, mask24)
      high = shiftr(state, 24)
      product = multiplier_low * old_low + increment
      carry = shiftr(product, 24)
      low = iand(product, mask24)
      high = iand(multiplier_low * high + multiplier_high * old_low + carry, mask24)
      state = ior(shiftl(high, 24), low)
      values(index) = real(state, real64) / 281474976710656.0_real64
    end do
  end subroutine drand_alloc

end program adv
