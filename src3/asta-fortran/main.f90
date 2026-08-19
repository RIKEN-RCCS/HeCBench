program asta
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  use omp_lib
  implicit none

  integer(int32) :: blocks, threads, warmup, reps, matrix_height, matrix_width
  integer(int32) :: super_element_size, tiled_n, in_size, finished_size
  integer(int32) :: rep, status, arg_index
  integer(int32) :: lmem(0:1)
  integer(int32) :: tid, worker_count, cycle_length, next_in_cycle, index
  integer(int32), allocatable :: finished(:), head(:)
  real(real32), allocatable :: in_out(:), in_backup(:)
  real(real64) :: elapsed_ns, start_time, end_time
  real(real32) :: data1, data2, data3, data4, backup1, backup2, backup3, backup4
  character(len=256) :: argument

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand

    function c_rand() bind(C, name="rand") result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

  blocks = 16_int32
  threads = 64_int32
  warmup = 10_int32
  reps = 100_int32
  matrix_height = 197_int32
  matrix_width = 35588_int32
  super_element_size = 32_int32

  arg_index = 1_int32
  do while (arg_index <= command_argument_count())
    call get_command_argument(arg_index, argument)
    select case (trim(argument))
    case ('-h')
      call usage()
      stop
    case ('-i')
      call option_value(arg_index, threads)
    case ('-g')
      call option_value(arg_index, blocks)
    case ('-w')
      call option_value(arg_index, warmup)
    case ('-r')
      call option_value(arg_index, reps)
    case ('-m')
      call option_value(arg_index, matrix_height)
    case ('-n')
      call option_value(arg_index, matrix_width)
    case ('-s')
      call option_value(arg_index, super_element_size)
    case default
      write(*, '(a)') ''
      write(*, '(a)') 'Unrecognized option!'
      call usage()
      stop
    end select
    arg_index = arg_index + 1_int32
  end do

  if (threads > 256_int32) then
    write(*, '(a)') 'The thread block size is greater than the maximum thread block size that can be used on this device'
    error stop
  end if

  tiled_n = (matrix_width - 1_int32) / super_element_size + 1_int32
  in_size = matrix_height * tiled_n * super_element_size
  finished_size = matrix_height * tiled_n

  allocate(in_out(0:in_size - 1_int32))
  allocate(in_backup(0:in_size - 1_int32))
  allocate(finished(0:finished_size - 1_int32))
  allocate(head(0:0))

  call read_input(in_out, in_size)
  in_backup = in_out

  elapsed_ns = 0.0_real64

  !$omp target data map(alloc: in_out(0:in_size - 1_int32), finished(0:finished_size - 1_int32), head(0:0))
  do rep = 0_int32, warmup + reps - 1_int32
    in_out = in_backup
    finished = 0_int32
    head(0) = 0_int32

    !$omp target update to(in_out(0:in_size - 1_int32), finished(0:finished_size - 1_int32), head(0:0))

    start_time = omp_get_wtime()

    !$omp target teams num_teams(blocks) thread_limit(threads) private(lmem)

      !$omp parallel private(tid, worker_count, next_in_cycle, index, data1, data2, data3, data4, &
      !$omp& backup1, backup2, backup3, backup4)
      tid = omp_get_thread_num()
      worker_count = omp_get_num_threads()
      cycle_length = matrix_height * tiled_n - 1_int32

      if (tid == 0_int32) then
        !$omp atomic capture
        lmem(1) = head(0)
        head(0) = head(0) + 1_int32
        !$omp end atomic
      end if
      !$omp barrier

      do while (lmem(1) < cycle_length)
        next_in_cycle = (lmem(1) * matrix_height) - cycle_length * (lmem(1) / tiled_n)
        if (next_in_cycle == lmem(1)) then
          if (tid == 0_int32) then
            !$omp atomic capture
            lmem(1) = head(0)
            head(0) = head(0) + 1_int32
            !$omp end atomic
          end if
          !$omp barrier
          cycle
        end if

        index = tid
        if (index < super_element_size) data1 = in_out(lmem(1) * super_element_size + index)
        index = index + worker_count
        if (index < super_element_size) data2 = in_out(lmem(1) * super_element_size + index)
        index = index + worker_count
        if (index < super_element_size) data3 = in_out(lmem(1) * super_element_size + index)
        index = index + worker_count
        if (index < super_element_size) data4 = real(worker_count, real32)

        if (tid == 0_int32) then
          !$omp atomic read
          lmem(0) = finished(lmem(1))
          !$omp end atomic
        end if
        !$omp barrier

        do while (lmem(0) == 0_int32)
          index = tid
          if (index < super_element_size) backup1 = in_out(next_in_cycle * super_element_size + index)
          index = index + worker_count
          if (index < super_element_size) backup2 = in_out(next_in_cycle * super_element_size + index)
          index = index + worker_count
          if (index < super_element_size) backup3 = in_out(next_in_cycle * super_element_size + index)
          index = index + worker_count
          if (index < super_element_size) backup4 = in_out(next_in_cycle * super_element_size + index)

          if (tid == 0_int32) then
            !$omp atomic capture
            lmem(0) = finished(next_in_cycle)
            finished(next_in_cycle) = 1_int32
            !$omp end atomic
          end if
          !$omp barrier

          if (lmem(0) == 0_int32) then
            index = tid
            if (index < super_element_size) in_out(next_in_cycle * super_element_size + index) = data1
            index = index + worker_count
            if (index < super_element_size) in_out(next_in_cycle * super_element_size + index) = data2
            index = index + worker_count
            if (index < super_element_size) in_out(next_in_cycle * super_element_size + index) = data3
            index = index + worker_count
            if (index < super_element_size) in_out(next_in_cycle * super_element_size + index) = data4
          end if

          index = tid
          if (index < super_element_size) data1 = backup1
          index = index + worker_count
          if (index < super_element_size) data2 = backup2
          index = index + worker_count
          if (index < super_element_size) data3 = backup3
          index = index + worker_count
          if (index < super_element_size) data4 = backup4

          next_in_cycle = (next_in_cycle * matrix_height) - cycle_length * (next_in_cycle / tiled_n)
        end do

        if (tid == 0_int32) then
          !$omp atomic capture
          lmem(1) = head(0)
          head(0) = head(0) + 1_int32
          !$omp end atomic
        end if
        !$omp barrier
      end do
        !$omp end parallel
    !$omp end target teams

    end_time = omp_get_wtime()
    if (rep >= warmup) elapsed_ns = elapsed_ns + (end_time - start_time) * 1.0e9_real64

    !$omp target update from(in_out(0:in_size - 1_int32))
  end do
  !$omp end target data

  write(*, '(a,es16.8,a)') 'Average kernel execution time ', (elapsed_ns * 1.0e-9_real64) / real(reps, real64), ' (s)'

  status = verify_output(in_out, in_backup, tiled_n * super_element_size, matrix_height, super_element_size)
  if (status == 0_int32) then
    write(*, '(a)') 'PASS'
  else
    write(*, '(a)') 'FAIL'
  end if

  deallocate(in_out, in_backup, finished, head)

contains

  subroutine option_value(index, value)
    integer(int32), intent(inout) :: index
    integer(int32), intent(out) :: value
    character(len=256) :: text
    integer :: io_status

    if (index >= command_argument_count()) then
      write(*, '(a)') 'Unrecognized option!'
      call usage()
      error stop
    end if
    index = index + 1_int32
    call get_command_argument(index, text)
    read(text, *, iostat=io_status) value
    if (io_status /= 0) then
      write(*, '(a)') 'Unrecognized option!'
      call usage()
      error stop
    end if
  end subroutine option_value

  subroutine usage()
    write(*, '(a)') ''
    write(*, '(a)') 'Usage:  ./main [options]'
    write(*, '(a)') ''
    write(*, '(a)') 'General options:'
    write(*, '(a)') '    -h        help'
    write(*, '(a)') '    -i <I>    # of device threads per block (default=64)'
    write(*, '(a)') '    -g <G>    # of device blocks (default=16)'
    write(*, '(a)') '    -w <W>    # of warmup iterations (default=10)'
    write(*, '(a)') '    -r <R>    # of repetition iterations (default=100)'
    write(*, '(a)') ''
    write(*, '(a)') 'Benchmark-specific options:'
    write(*, '(a)') '    -m <M>    matrix height (default=197)'
    write(*, '(a)') '    -n <N>    matrix width (default=35588)'
    write(*, '(a)') '    -s <M>    super-element size (default=32)'
  end subroutine usage

  subroutine read_input(vector, vector_size)
    real(real32), intent(out) :: vector(0:)
    integer(int32), intent(in) :: vector_size
    integer(int32) :: i

    call c_srand(5432_c_int)
    do i = 0_int32, vector_size - 1_int32
      vector(i) = real(mod(c_rand(), 100_c_int), real32) / 100.0_real32
    end do
  end subroutine read_input

  integer(int32) function verify_output(input2, input, height, width, tile_size) result(result_status)
    real(real32), intent(in) :: input2(0:), input(0:)
    integer(int32), intent(in) :: height, width, tile_size
    real(real32), allocatable :: output(:)
    integer(int32) :: i, j, k, line
    real(real32) :: difference

    allocate(output(0:width * height - 1_int32))
    do k = 0_int32, width - 1_int32
      do i = 0_int32, height / tile_size - 1_int32
        do j = 0_int32, tile_size - 1_int32
          output(i * width * tile_size + k * tile_size + j) = input(k * height + i * tile_size + j)
        end do
      end do
    end do

    result_status = 0_int32
    do line = 0_int32, height * width - 1_int32
      difference = abs(output(line) - input2(line))
      if (difference > 0.00001_real32 .and. difference > 0.01_real32 * abs(output(line))) then
        write(*, '(a,i0,a,f0.6,a,f0.6,a,f0.6)') 'Failed at line: ', line, ' ref: ', output(line), &
          ' actual: ', input2(line), ' diff: ', difference
        result_status = 1_int32
        exit
      end if
    end do
    deallocate(output)
  end function verify_output

end program asta
