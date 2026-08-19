program rainflow
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int64, real64
  use omp_lib
  use rainflow_mod
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
  end interface

  integer :: argc, num_history, repeat, n, i, error_count
  integer(int64) :: total_length
  integer, allocatable :: history_lengths(:), result_lengths(:), ref_result_lengths(:)
  integer, allocatable :: points(:)
  real(real64), allocatable :: history(:), extrema_buf(:)
  type(double3), allocatable :: results(:)
  real(real64) :: start_time, end_time
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <number of histories> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *) num_history
  call get_command_argument(2, arg)
  read(arg, *) repeat

  allocate(history_lengths(0:num_history), result_lengths(0:num_history-1), ref_result_lengths(0:num_history-1))

  call c_srand(123_c_int)
  total_length = 0_int64
  do n = 0, num_history - 1
    history_lengths(n) = int(total_length)
    total_length = total_length + int(mod(c_rand(), 10_c_int) + 1, int64) * 100_int64
  end do
  history_lengths(num_history) = int(total_length)

  print '(a,i0)', 'Total history length = ', total_length

  allocate(history(0:total_length-1), extrema_buf(0:total_length-1), points(0:total_length-1), results(0:total_length-1))
  do i = 0, int(total_length) - 1
    history(i) = real(c_rand(), real64) / 2147483647.0_real64
  end do

  !$omp target data map(to: history_lengths(0:num_history), history(0:total_length-1)) &
  !$omp& map(alloc: extrema_buf(0:total_length-1), points(0:total_length-1), results(0:total_length-1)) &
  !$omp& map(from: result_lengths(0:num_history-1))
    start_time = omp_get_wtime()
    do n = 0, repeat - 1
      !$omp target teams distribute parallel do thread_limit(256)
      do i = 0, num_history - 1
        call execute(history(history_lengths(i):), history_lengths(i + 1) - history_lengths(i), &
                     extrema_buf(history_lengths(i):), points(history_lengths(i):), &
                     results(history_lengths(i):), result_lengths(i))
      end do
      !$omp end target teams distribute parallel do
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time: ', ((end_time - start_time) * 1.0e6_real64) / repeat, ' (us)'
  !$omp end target data

  call reference(history, history_lengths, extrema_buf, points, results, ref_result_lengths, num_history)
  error_count = 0
  do i = 0, num_history - 1
    if (ref_result_lengths(i) /= result_lengths(i)) error_count = error_count + 1
  end do
  if (error_count == 0) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
end program rainflow
