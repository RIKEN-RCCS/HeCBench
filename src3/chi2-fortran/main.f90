module chi2_support
  use, intrinsic :: iso_fortran_env, only : int8, int64, real32, real64
  implicit none
  integer(int64), parameter :: mt_mask = int(z'00000000FFFFFFFF', int64)
  integer(int64), parameter :: mt_upper = int(z'0000000080000000', int64)
  integer(int64), parameter :: mt_lower = int(z'000000007FFFFFFF', int64)
  integer(int64), parameter :: mt_matrix = int(z'000000009908B0DF', int64)
contains

  subroutine mt_seed(state, index, seed)
    integer(int64), intent(out) :: state(0:623)
    integer, intent(out) :: index
    integer(int64), intent(in) :: seed
    integer :: i
    state(0) = iand(seed, mt_mask)
    do i = 1, 623
      state(i) = iand(1812433253_int64 * ieor(state(i-1), ishft(state(i-1), -30)) + int(i, int64), mt_mask)
    end do
    index = 624
  end subroutine mt_seed

  subroutine mt_twist(state)
    integer(int64), intent(inout) :: state(0:623)
    integer :: i, next
    integer(int64) :: y
    do i = 0, 623
      next = modulo(i + 1, 624)
      y = ior(iand(state(i), mt_upper), iand(state(next), mt_lower))
      state(i) = ieor(state(modulo(i + 397, 624)), ishft(y, -1))
      if (iand(y, 1_int64) /= 0_int64) state(i) = ieor(state(i), mt_matrix)
    end do
  end subroutine mt_twist

  function mt_next(state, index) result(value)
    integer(int64), intent(inout) :: state(0:623)
    integer, intent(inout) :: index
    integer(int64) :: value
    if (index >= 624) then
      call mt_twist(state)
      index = 0
    end if
    value = state(index)
    value = ieor(value, ishft(value, -11))
    value = ieor(value, iand(ishft(value, 7), int(z'000000009D2C5680', int64)))
    value = ieor(value, iand(ishft(value, 15), int(z'00000000EFC60000', int64)))
    value = ieor(value, ishft(value, -18))
    value = iand(value, mt_mask)
    index = index + 1
  end function mt_next

  function uniform_0_2(state, index) result(value)
    integer(int64), intent(inout) :: state(0:623)
    integer, intent(inout) :: index
    integer :: value
    integer(int64) :: random_value
    do
      random_value = mt_next(state, index) / 1431655765_int64
      if (random_value <= 2_int64) exit
    end do
    value = int(random_value)
  end function uniform_0_2

  subroutine cpu_kernel(rows, cols, case_rows, control_rows, snpdata, results)
    integer, intent(in) :: rows, cols, case_rows, control_rows
    integer(int8), intent(in) :: snpdata(0:)
    real(real32), intent(out) :: results(0:)
    integer :: tid, m, n, p, cases(0:2), controls(0:2), tot_cases, tot_controls, total
    real(real32) :: chisquare, expected(0:2), control_expected(0:2), case_expected(0:2)
    real(real32) :: numerator1, numerator2

    do tid = 0, cols - 1
      chisquare = 0.0_real32
      cases = 1
      controls = 1
      do m = 0, case_rows - 1
        select case (snpdata(m * cols + tid))
        case (48_int8); cases(0) = cases(0) + 1
        case (49_int8); cases(1) = cases(1) + 1
        case (50_int8); cases(2) = cases(2) + 1
        end select
      end do
      do n = case_rows, case_rows + control_rows - 1
        select case (snpdata(n * cols + tid))
        case (48_int8); controls(0) = controls(0) + 1
        case (49_int8); controls(1) = controls(1) + 1
        case (50_int8); controls(2) = controls(2) + 1
        end select
      end do
      tot_cases = sum(cases)
      tot_controls = sum(controls)
      total = tot_cases + tot_controls
      do p = 0, 2
        expected(p) = real(cases(p), real32) + real(controls(p), real32)
        case_expected(p) = real(tot_cases, real32) * expected(p) / real(total, real32)
        control_expected(p) = real(tot_controls, real32) * expected(p) / real(total, real32)
        numerator1 = real(cases(p), real32) - case_expected(p)
        numerator2 = real(controls(p), real32) - control_expected(p)
        chisquare = chisquare + numerator1 * numerator1 / case_expected(p) + numerator2 * numerator2 / control_expected(p)
      end do
      results(tid) = chisquare
    end do
  end subroutine cpu_kernel
end module chi2_support

program chi2
  use, intrinsic :: iso_fortran_env, only : int8, int64, real32, real64
  use omp_lib, only: omp_get_wtime
  use chi2_support
  implicit none
  integer :: rows, cols, ncases, ncontrols, nthreads, repeat, i, m, n, p, ios
  integer :: cases(0:2), controls(0:2), tot_cases, tot_controls, total
  integer(int64) :: size, mt_state(0:623)
  integer :: mt_index
  character(len=64) :: argument
  integer(int8), allocatable :: datat(:)
  real(real32), allocatable :: h_results(:), cpu_results(:)
  real(real32) :: chisquare, expected(0:2), control_expected(0:2), case_expected(0:2), numerator1, numerator2
  real(real64) :: elapsed, start_time, stop_time
  logical :: passed

  if (command_argument_count() /= 6) then
    print '(a)', 'Usage: ./main <rows> <cols> <cases> <controls> <threads> <repeat>'
    error stop 1
  end if
  call read_integer_argument(1, rows)
  call read_integer_argument(2, cols)
  call read_integer_argument(3, ncases)
  call read_integer_argument(4, ncontrols)
  call read_integer_argument(5, nthreads)
  call read_integer_argument(6, repeat)
  print '(a,i0,a,i0,a,i0,a,i0,a,i0)', 'Individuals=', rows, ' SNPs=', cols, ' cases=', ncases, &
    ' controls=', ncontrols, ' nthreads=', nthreads
  size = int(rows, int64) * int(cols, int64)
  print '(a,i0)', 'Size of the data = ', size

  allocate(datat(0:size-1), h_results(0:cols-1), cpu_results(0:cols-1), stat=ios)
  if (ios /= 0) then
    print '(a)', 'ERROR: Memory for data not allocated.'
    error stop 1
  end if
  call mt_seed(mt_state, mt_index, 19937_int64)
  do i = 0, int(size) - 1
    datat(i) = int(48 + uniform_0_2(mt_state, mt_index), int8)
  end do

!$omp target data map(to:datat) map(from:h_results)
  start_time = omp_get_wtime()
  do i = 1, repeat
!$omp target teams distribute parallel do simd thread_limit(nthreads) &
!$omp& private(cases, controls, tot_cases, tot_controls, total, chisquare, expected, control_expected, &
!$omp& case_expected, numerator1, numerator2, m, n, p)
    do p = 0, cols - 1
      cases = 1
      controls = 1
      tot_cases = 1
      tot_controls = 1
      total = 1
      chisquare = 0.0_real32
      do m = 0, ncases - 1
        select case (datat(m * cols + p))
        case (48_int8); cases(0) = cases(0) + 1
        case (49_int8); cases(1) = cases(1) + 1
        case (50_int8); cases(2) = cases(2) + 1
        end select
      end do
      do n = ncases, ncases + ncontrols - 1
        select case (datat(n * cols + p))
        case (48_int8); controls(0) = controls(0) + 1
        case (49_int8); controls(1) = controls(1) + 1
        case (50_int8); controls(2) = controls(2) + 1
        end select
      end do
      do m = 0, 2
        tot_cases = tot_cases + cases(m)
        tot_controls = tot_controls + controls(m)
      end do
      total = tot_cases + tot_controls
      do m = 0, 2
        expected(m) = real(cases(m), real32) + real(controls(m), real32)
        case_expected(m) = real(tot_cases, real32) * expected(m) / real(total, real32)
        control_expected(m) = real(tot_controls, real32) * expected(m) / real(total, real32)
        numerator1 = real(cases(m), real32) - case_expected(m)
        numerator2 = real(controls(m), real32) - control_expected(m)
        chisquare = chisquare + numerator1 * numerator1 / case_expected(m) + numerator2 * numerator2 / control_expected(m)
      end do
      h_results(p) = chisquare
    end do
!$omp end target teams distribute parallel do simd
  end do
  stop_time = omp_get_wtime()
!$omp end target data
  elapsed = stop_time - start_time
  print '(a,f0.6,a)', 'Average kernel execution time = ', elapsed / real(repeat, real64), ' (s)'

  start_time = omp_get_wtime()
  call cpu_kernel(rows, cols, ncases, ncontrols, datat, cpu_results)
  stop_time = omp_get_wtime()
  elapsed = stop_time - start_time
  print '(a,f0.6,a)', 'Host execution time = ', elapsed, ' (s)'

  passed = all(abs(cpu_results - h_results) <= 1.0e-4_real32)
  if (passed) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
  deallocate(datat, h_results, cpu_results)
contains
  subroutine read_integer_argument(position, value)
    integer, intent(in) :: position
    integer, intent(out) :: value
    character(len=64) :: text
    integer :: status
    call get_command_argument(position, text, status=status)
    if (status /= 0) error stop 1
    read(text, *, iostat=status) value
    if (status /= 0) error stop 1
  end subroutine read_integer_argument
end program chi2
