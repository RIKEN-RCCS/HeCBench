module snap_mod
  use iso_fortran_env, only: real64
  implicit none
  real(real64), parameter :: my_pi = 3.14159265358979323846_real64
  type :: complex_t
    real(real64) :: re = 0.0_real64
    real(real64) :: im = 0.0_real64
  end type complex_t
  type :: bindices_t
    integer :: j1 = 0, j2 = 0, j = 0
  end type bindices_t
contains
  integer function compute_ncoeff(twojmax)
    integer, intent(in) :: twojmax
    integer :: j1, j2, j
    compute_ncoeff = 0
    do j1 = 0, twojmax
      do j2 = 0, j1
        do j = abs(j1-j2), min(twojmax, j1+j2), 2
          if (j >= j1) compute_ncoeff = compute_ncoeff + 1
        end do
      end do
    end do
  end function compute_ncoeff

  real(real64) function compute_sfac(r, rcut, switch_flag)
    real(real64), intent(in) :: r, rcut
    integer, intent(in) :: switch_flag
    if (switch_flag == 0) then
      compute_sfac = 1.0_real64
    else if (switch_flag == 1) then
      if (r <= 0.0_real64) then
        compute_sfac = 1.0_real64
      else if (r > rcut) then
        compute_sfac = 0.0_real64
      else
        compute_sfac = 0.5_real64 * (cos((r - 0.0_real64) * my_pi / rcut) + 1.0_real64)
      end if
    else
      compute_sfac = 0.0_real64
    end if
  end function compute_sfac

  real(real64) function factorial(n)
    integer, intent(in) :: n
    integer :: i
    factorial = 1.0_real64
    do i = 2, n
      factorial = factorial * real(i, real64)
    end do
  end function factorial

  real(real64) function deltacg(j1, j2, j)
    integer, intent(in) :: j1, j2, j
    deltacg = sqrt(factorial((j1+j2-j)/2) * factorial((j1-j2+j)/2) * &
                   factorial((-j1+j2+j)/2) / factorial((j1+j2+j)/2 + 1))
  end function deltacg

  subroutine build_indices(twojmax, idxb, idxb_count, idxu_block, idxu_max, idxdu_block, idxdu_max)
    integer, intent(in) :: twojmax
    type(bindices_t), allocatable, intent(out) :: idxb(:)
    integer, allocatable, intent(out) :: idxu_block(:), idxdu_block(:)
    integer, intent(out) :: idxb_count, idxu_max, idxdu_max
    integer :: jdim, j1, j2, j, ma, mb, idxu_count, idxdu_count
    jdim = twojmax + 1
    idxb_count = compute_ncoeff(twojmax)
    allocate(idxb(0:idxb_count-1), idxu_block(0:jdim-1), idxdu_block(0:jdim-1))
    idxb_count = 0
    do j1 = 0, twojmax
      do j2 = 0, j1
        do j = abs(j1-j2), min(twojmax, j1+j2), 2
          if (j >= j1) then
            idxb(idxb_count)%j1 = j1
            idxb(idxb_count)%j2 = j2
            idxb(idxb_count)%j = j
            idxb_count = idxb_count + 1
          end if
        end do
      end do
    end do
    idxu_count = 0
    do j = 0, twojmax
      idxu_block(j) = idxu_count
      do mb = 0, j
        do ma = 0, j
          idxu_count = idxu_count + 1
        end do
      end do
    end do
    idxu_max = idxu_count
    idxdu_count = 0
    do j = 0, twojmax
      idxdu_block(j) = idxdu_count
      do mb = 0, j / 2
        do ma = 0, j
          idxdu_count = idxdu_count + 1
        end do
      end do
    end do
    idxdu_max = idxdu_count
  end subroutine build_indices

  subroutine compute_snap_phases(nsteps, nlocal, ninside, twojmax, idxu_max, idxdu_max, idxb_count, switch_flag, elapsed_ui, elapsed_yi, elapsed_duidrj, elapsed_deidrj, checksum)
    integer, intent(in) :: nsteps, nlocal, ninside, twojmax, idxu_max, idxdu_max, idxb_count, switch_flag
    real(real64), intent(out) :: elapsed_ui, elapsed_yi, elapsed_duidrj, elapsed_deidrj, checksum
    complex_t, allocatable :: ulist(:), ylist(:), dulist(:)
    real(real64), allocatable :: deidrj(:)
    integer :: natom, nbor, j, idx, step
    real(real64) :: r, sfac, t0, t1
    allocate(ulist(0:nlocal*ninside*idxu_max-1), ylist(0:nlocal*idxu_max-1), dulist(0:nlocal*ninside*idxdu_max*3-1), deidrj(0:nlocal*ninside*3-1))
    elapsed_ui = 0.0_real64; elapsed_yi = 0.0_real64; elapsed_duidrj = 0.0_real64; elapsed_deidrj = 0.0_real64
    do step = 1, nsteps
      t0 = wall_seconds()
!$omp target teams distribute parallel do collapse(2) thread_limit(256) map(from:ulist) private(natom,nbor,j,idx,r,sfac)
      do natom = 0, nlocal - 1
        do nbor = 0, ninside - 1
          r = sqrt(real((natom+1)*(nbor+1), real64)) * 0.001_real64 + 0.5_real64
          sfac = compute_sfac(r, 4.67637_real64, switch_flag)
          do j = 0, idxu_max - 1
            idx = natom + nlocal * nbor + nlocal * ninside * j
            ulist(idx)%re = sfac * cos(real(j + twojmax, real64) * r)
            ulist(idx)%im = sfac * sin(real(j + twojmax, real64) * r)
          end do
        end do
      end do
!$omp end target teams distribute parallel do
      t1 = wall_seconds(); elapsed_ui = elapsed_ui + t1 - t0
      t0 = wall_seconds()
!$omp target teams distribute parallel do collapse(2) thread_limit(256) map(to:ulist) map(from:ylist) private(natom,j,idx,nbor)
      do natom = 0, nlocal - 1
        do j = 0, idxu_max - 1
          idx = natom + nlocal * j
          ylist(idx)%re = 0.0_real64; ylist(idx)%im = 0.0_real64
          do nbor = 0, ninside - 1
            ylist(idx)%re = ylist(idx)%re + ulist(natom+nlocal*nbor+nlocal*ninside*j)%re
            ylist(idx)%im = ylist(idx)%im + ulist(natom+nlocal*nbor+nlocal*ninside*j)%im
          end do
        end do
      end do
!$omp end target teams distribute parallel do
      t1 = wall_seconds(); elapsed_yi = elapsed_yi + t1 - t0
      t0 = wall_seconds()
!$omp target teams distribute parallel do collapse(2) thread_limit(256) map(from:dulist) private(natom,nbor,j,idx)
      do natom = 0, nlocal - 1
        do nbor = 0, ninside - 1
          do j = 0, idxdu_max * 3 - 1
            idx = natom + nlocal * nbor + nlocal * ninside * j
            dulist(idx)%re = real(j + 1, real64) * 1.0e-6_real64
            dulist(idx)%im = -dulist(idx)%re
          end do
        end do
      end do
!$omp end target teams distribute parallel do
      t1 = wall_seconds(); elapsed_duidrj = elapsed_duidrj + t1 - t0
      t0 = wall_seconds()
!$omp target teams distribute parallel do collapse(2) thread_limit(256) map(to:ylist,dulist) map(from:deidrj) private(natom,nbor,j,idx)
      do natom = 0, nlocal - 1
        do nbor = 0, ninside - 1
          do j = 0, 2
            idx = natom + nlocal * nbor + nlocal * ninside * j
            deidrj(idx) = ylist(natom)%re * dulist(natom+nlocal*nbor+nlocal*ninside*j)%re - &
                          ylist(natom)%im * dulist(natom+nlocal*nbor+nlocal*ninside*j)%im
          end do
        end do
      end do
!$omp end target teams distribute parallel do
      t1 = wall_seconds(); elapsed_deidrj = elapsed_deidrj + t1 - t0
    end do
    checksum = sum(deidrj) + real(idxb_count, real64)
  end subroutine compute_snap_phases

  function wall_seconds() result(t)
    use iso_fortran_env, only: int64
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds
end module snap_mod
