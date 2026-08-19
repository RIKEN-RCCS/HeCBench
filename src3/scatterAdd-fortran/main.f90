program scatterAdd
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use scatteradd_kernels
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

  integer :: argc, batch_size, output_size, vector_dim, repeat
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 4) then
    print '(a)', 'Usage: ./main <batch size> <output size> <vector dimension> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) batch_size
  call get_command_argument(2, arg); read(arg, *) output_size
  call get_command_argument(3, arg); read(arg, *) vector_dim
  call get_command_argument(4, arg); read(arg, *) repeat

  print '(a,i0)', 'batch_size: ', batch_size
  print '(a,i0)', 'output_size (range of index values): ', output_size
  print '(a,i0)', 'vector_dimension: ', vector_dim
  call index_accumulate(batch_size, output_size, vector_dim, repeat)

contains

  subroutine index_accumulate(batch_size, output_size, vector_dim, repeat)
    integer, intent(in) :: batch_size, output_size, vector_dim, repeat
    integer :: i
    integer, allocatable :: index(:)
    real(real32), allocatable :: source(:), output(:), output_ref(:)
    real(real64) :: start_time, end_time
    logical :: ok

    allocate(index(0:batch_size-1), source(0:batch_size*vector_dim-1), &
             output(0:output_size*vector_dim-1), output_ref(0:output_size*vector_dim-1))
    call c_srand(2_c_int)
    do i = 0, batch_size - 1
      index(i) = mod(c_rand(), output_size)
    end do
    source = -1.0_real32
    output_ref = 0.0_real32
    call scatter_add_reference(batch_size, vector_dim, output_ref, index, source)

    !$omp target data map(to: source(0:batch_size*vector_dim-1), index(0:batch_size-1)) &
    !$omp& map(alloc: output(0:output_size*vector_dim-1))
      do i = 0, 9
        output = 0.0_real32
        !$omp target update to(output(0:output_size*vector_dim-1))
        call scatterAdd2_kernel(index, source, output, batch_size, output_size, vector_dim)
      end do

      !$omp target update from(output(0:output_size*vector_dim-1))
      ok = all(abs(output - output_ref) <= 1.0e-3_real32)
      if (ok) then
        print '(a)', 'PASS'
      else
        print '(a)', 'FAIL'
      end if

      start_time = omp_get_wtime()
      do i = 0, repeat - 1
        call scatterAdd_kernel(index, source, output, batch_size, output_size, vector_dim)
      end do
      end_time = omp_get_wtime()
      print '(a,f12.6,a)', 'Average execution time of kernel1: ', ((end_time-start_time)*1.0e6_real64)/repeat, ' (us)'

      start_time = omp_get_wtime()
      do i = 0, repeat - 1
        call scatterAdd2_kernel(index, source, output, batch_size, output_size, vector_dim)
      end do
      end_time = omp_get_wtime()
      print '(a,f12.6,a)', 'Average execution time of kernel2: ', ((end_time-start_time)*1.0e6_real64)/repeat, ' (us)'
    !$omp end target data
  end subroutine index_accumulate

end program scatterAdd
