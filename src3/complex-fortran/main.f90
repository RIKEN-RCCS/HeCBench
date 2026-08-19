module complex_kernels
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  implicit none
  !$omp declare target (fast_forward_lcg, lcg_random_double)
contains

  function fast_forward_lcg(seed, n) result(value)
    integer(int64), intent(in) :: seed, n
    integer(int64) :: value, a, c, a_new, c_new, count

    a = 2806196910506780709_int64
    c = 1_int64
    a_new = 1_int64
    c_new = 0_int64
    count = iand(n, huge(0_int64))
    do while (count > 0_int64)
      if (iand(count, 1_int64) /= 0_int64) then
        a_new = a_new * a
        c_new = c_new * a + c
      end if
      c = c * (a + 1_int64)
      a = a * a
      count = shiftr(count, 1)
    end do
    value = iand(a_new * seed + c_new, huge(0_int64))
  end function fast_forward_lcg

  function lcg_random_double(seed) result(value)
    integer(int64), intent(inout) :: seed
    real(real64) :: value

    seed = iand(2806196910506780709_int64 * seed + 1_int64, huge(0_int64))
    value = real(seed, real64) / 9223372036854775808.0_real64
  end function lcg_random_double
  subroutine complex_float(checksum, n)
    integer, intent(in) :: n
    integer(kind=1), intent(inout) :: checksum(0:n-1)
    integer :: i
    integer(int64) :: seed
    real(real32) :: r1, r2, r3, r4
    complex(real32) :: z1, z2, zsum, zdiff, zprod
    integer :: score

    !$omp target teams distribute parallel do thread_limit(256) private(seed,r1,r2,r3,r4,z1,z2,zsum,zdiff,zprod,score)
    do i = 0, n - 1
      seed = fast_forward_lcg(1_int64, int(i, int64))
      r1 = real(lcg_random_double(seed), real32)
      r2 = real(lcg_random_double(seed), real32)
      r3 = real(lcg_random_double(seed), real32)
      r4 = real(lcg_random_double(seed), real32)
      z1 = cmplx(r1, r2, real32)
      z2 = cmplx(r3, r4, real32)
      zsum = z1 + z2
      zdiff = z1 - z2
      zprod = z1 * z2
      score = 0
      if (abs(abs(zprod) - abs(z1) * abs(z2)) < 1.0e-3_real32) score = score + 1
      if (abs(abs(zsum) * abs(zsum) - real(zsum * conjg(zsum), real32)) < 1.0e-3_real32) score = score + 1
      if (abs(abs(zdiff) * abs(zdiff) - real(zdiff * conjg(zdiff), real32)) < 1.0e-3_real32) score = score + 1
      if (abs(real(z1 * conjg(z2) + z2 * conjg(z1), real32) - &
              2.0_real32 * (real(z1, real32) * real(z2, real32) + aimag(z1) * aimag(z2))) < 1.0e-3_real32) score = score + 1
      if (abs(abs(conjg(z1) / z2) - abs(conjg(z1) / conjg(z2))) < 1.0e-3_real32) score = score + 1
      checksum(i) = transfer(score, checksum(i))
    end do
    !$omp end target teams distribute parallel do
  end subroutine complex_float

  subroutine complex_double(checksum, n)
    integer, intent(in) :: n
    integer(kind=1), intent(inout) :: checksum(0:n-1)
    integer :: i
    integer(int64) :: seed
    real(real64) :: r1, r2, r3, r4
    complex(real64) :: z1, z2, zsum, zdiff, zprod
    integer :: score

    !$omp target teams distribute parallel do thread_limit(256) private(seed,r1,r2,r3,r4,z1,z2,zsum,zdiff,zprod,score)
    do i = 0, n - 1
      seed = fast_forward_lcg(1_int64, int(i, int64))
      r1 = lcg_random_double(seed)
      r2 = lcg_random_double(seed)
      r3 = lcg_random_double(seed)
      r4 = lcg_random_double(seed)
      z1 = cmplx(r1, r2, real64)
      z2 = cmplx(r3, r4, real64)
      zsum = z1 + z2
      zdiff = z1 - z2
      zprod = z1 * z2
      score = 0
      if (abs(abs(zprod) - abs(z1) * abs(z2)) < 1.0e-3_real64) score = score + 1
      if (abs(abs(zsum) * abs(zsum) - real(zsum * conjg(zsum), real64)) < 1.0e-3_real64) score = score + 1
      if (abs(abs(zdiff) * abs(zdiff) - real(zdiff * conjg(zdiff), real64)) < 1.0e-3_real64) score = score + 1
      if (abs(real(z1 * conjg(z2) + z2 * conjg(z1), real64) - &
              2.0_real64 * (real(z1, real64) * real(z2, real64) + aimag(z1) * aimag(z2))) < 1.0e-3_real64) score = score + 1
      if (abs(abs(conjg(z1) / z2) - abs(conjg(z1) / conjg(z2))) < 1.0e-3_real64) score = score + 1
      checksum(i) = transfer(score, checksum(i))
    end do
    !$omp end target teams distribute parallel do
  end subroutine complex_double
end module complex_kernels

program main
  use, intrinsic :: iso_fortran_env, only : real64
  use complex_kernels
  implicit none
  integer :: argc, n, repeat, i, start_count, end_count, count_rate
  integer(kind=1), allocatable :: checksum(:)
  real(real64) :: elapsed
  character(len=64) :: argument

  argc = command_argument_count()
  if (argc /= 2) then
    write(*,'(a)') 'Usage: ./main <size> <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument,*) n
  call get_command_argument(2, argument)
  read(argument,*) repeat
  allocate(checksum(0:n-1))

  !$omp target data map(alloc: checksum(0:n-1))
  call complex_float(checksum, n)
  call complex_double(checksum, n)

  call system_clock(start_count, count_rate)
  do i = 1, repeat
    call complex_float(checksum, n)
  end do
  call system_clock(end_count)
  elapsed = real(end_count - start_count, real64) / real(count_rate, real64)
  write(*,'(a,f0.6,a)') 'Average kernel execution time (float) ', elapsed / real(repeat, real64), ' (s)'
  !$omp target update from(checksum(0:n-1))
  if (.not. all(checksum == 5_1)) then
    write(*,'(a)') 'FAIL'
    stop 1
  end if

  call system_clock(start_count, count_rate)
  do i = 1, repeat
    call complex_double(checksum, n)
  end do
  call system_clock(end_count)
  elapsed = real(end_count - start_count, real64) / real(count_rate, real64)
  write(*,'(a,f0.6,a)') 'Average kernel execution time (double) ', elapsed / real(repeat, real64), ' (s)'
  !$omp target update from(checksum(0:n-1))
  if (all(checksum == 5_1)) then
    write(*,'(a)') 'PASS'
  else
    write(*,'(a)') 'FAIL'
  end if
  !$omp end target data

  deallocate(checksum)
end program main
