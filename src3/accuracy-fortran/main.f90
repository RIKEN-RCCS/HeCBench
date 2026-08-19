program accuracy
  use iso_fortran_env, only: int32, int64, real32, real64
  use iso_c_binding, only: c_int
  use omp_lib
  implicit none
  integer :: nrows, ndims, top_k, repeat, data_size, i, row, col, ngrid
  integer :: ngt, count(0:0), count_ref, label_data
  integer(int64) :: engine_state
  character(len=64) :: arg
  real(real32) :: label_pred, pred
  real(real32), allocatable :: data(:)
  integer(int32), allocatable :: label(:)
  real(real64) :: start_time, elapsed

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

  if (command_argument_count() /= 4) then
    print '(a)', 'Usage: ./main <number of rows> <number of columns> <top K> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) nrows
  call get_command_argument(2, arg); read(arg, *) ndims
  call get_command_argument(3, arg); read(arg, *) top_k
  call get_command_argument(4, arg); read(arg, *) repeat

  data_size = nrows * ndims
  allocate(data(0:data_size-1), label(0:nrows-1))

  ! The OMP reference seeds both generators with 123.  This is the GNU
  ! default_random_engine recurrence used by that reference toolchain.
  engine_state = 123_int64
  do i = 0, data_size - 1
    engine_state = modulo(16807_int64 * engine_state, 2147483647_int64)
    data(i) = real(engine_state - 1_int64, real32) / real(2147483646_int64, real32)
  end do
  call c_srand(123_c_int)
  do i = 0, nrows - 1
    label(i) = modulo(c_rand(), int(ndims, c_int))
  end do

  count_ref = 0
  do row = 0, nrows - 1
    label_data = label(row)
    label_pred = data(row * ndims + label_data)
    ngt = 0
    do col = 0, ndims - 1
      pred = data(row * ndims + col)
      if (pred > label_pred .or. (pred == label_pred .and. col <= label_data)) ngt = ngt + 1
    end do
    if (ngt <= top_k) count_ref = count_ref + 1
  end do

!$omp target data map(to: label(0:nrows-1), data(0:data_size-1)) map(alloc: count(0:0))
  do ngrid = nrows / 4, nrows, nrows / 4
    print '(a,i0)', 'Grid size is ', ngrid
    start_time = omp_get_wtime()
    do i = 1, repeat
      count(0) = 0
!$omp target update to(count(0:0))
!$omp target teams distribute num_teams(ngrid) private(label_data,label_pred,ngt,col,pred)
      do row = 0, nrows - 1
        label_data = label(row)
        label_pred = data(row * ndims + label_data)
        ngt = 0
!$omp parallel do reduction(+:ngt) num_threads(256) private(pred)
        do col = 0, ndims - 1
          pred = data(row * ndims + col)
          if (pred > label_pred .or. (pred == label_pred .and. col <= label_data)) ngt = ngt + 1
        end do
!$omp end parallel do
        if (ngt <= top_k) then
!$omp atomic update
          count(0) = count(0) + 1
        end if
      end do
!$omp end target teams distribute
    end do
    elapsed = omp_get_wtime() - start_time
    print '(a,f12.6,a)', 'Average execution time of accuracy kernel: ', &
      elapsed * 1.0e6_real64 / real(repeat, real64), ' (us)'
!$omp target update from(count(0:0))
    if (count(0) == count_ref) then
      print '(a)', 'PASS'
    else
      print '(a)', 'FAIL'
    end if
  end do
!$omp end target data
end program accuracy
