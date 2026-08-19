program rfs
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use rfs_mod
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

  integer :: argc, n_arrays, n_elems, n, i
  integer :: input_size
  real(real32), allocatable :: arrays(:), max_val(:), result(:), factor(:), result_ref(:)
  real(real32) :: max_seen
  real(real64) :: start_time, end_time
  logical :: ok
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <number of arrays> <length of each array>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *) n_arrays
  call get_command_argument(2, arg)
  read(arg, *) n_elems

  input_size = n_arrays * n_elems
  allocate(arrays(0:input_size-1), max_val(0:n_arrays-1), result(0:n_arrays-1), &
           factor(0:n_arrays-1), result_ref(0:n_arrays-1))

  call c_srand(123_c_int)
  do n = 0, n_arrays - 1
    max_seen = 0.0_real32
    do i = 0, n_elems - 1
      arrays(n * n_elems + i) = real(c_rand(), real32) / 2147483647.0_real32
      if (mod(c_rand(), 2_c_int) /= 0) arrays(n * n_elems + i) = -arrays(n * n_elems + i)
      max_seen = max(abs(arrays(n * n_elems + i)), max_seen)
    end do
    factor(n) = create_rounding_factor(max_seen, n_elems)
    max_val(n) = max_seen
  end do

  do n = 0, n_arrays - 1
    result_ref(n) = 0.0_real32
    do i = 0, n_elems - 1
      result_ref(n) = result_ref(n) + truncate_with_rounding_factor(factor(n), arrays(n * n_elems + i))
    end do
  end do

  !$omp target data map(to: arrays(0:input_size-1), max_val(0:n_arrays-1)) map(alloc: result(0:n_arrays-1))
    !$omp target teams distribute parallel do thread_limit(256)
    do i = 0, n_arrays - 1
      result(i) = 0.0_real32
    end do
    !$omp end target teams distribute parallel do

    start_time = omp_get_wtime()
    do n = 0, n_arrays - 1
      call sum_array(factor(n), n_elems, arrays, n * n_elems, result, n)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time (sumArray): ', (end_time - start_time) / n_arrays, ' (s)'

    !$omp target update from(result(0:n_arrays-1))
    ok = all(result == result_ref)
    if (ok) then
      print '(a)', 'PASS'
    else
      print '(a)', 'FAIL'
    end if

    start_time = omp_get_wtime()
    call sum_arrays(n_arrays, n_elems, arrays, result, max_val)
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Kernel execution time (sumArrays): ', end_time - start_time, ' (s)'

    !$omp target update from(result(0:n_arrays-1))
    ok = all(result == result_ref)
    if (ok) then
      print '(a)', 'PASS'
    else
      print '(a)', 'FAIL'
    end if
  !$omp end target data
end program rfs
