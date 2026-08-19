program snake
  use iso_fortran_env, only: int32, real64
  use omp_lib
  use snake_mod
  implicit none
  integer :: argc, read_length, num_reads, repeat, size_uint_bits, num_warps, concurrent_threads, num_blocks
  integer :: read_unit, ios, this_read, j, loop_par, error_threshold, accepted, i
  integer(int32), allocatable :: read_seq(:), ref_seq(:)
  integer, allocatable :: device_results(:), host_results(:)
  character(len=512) :: arg, file_path, line, token1, token2
  real(real64) :: t1, t2, elapsed_us
  logical :: error

  argc = command_argument_count()
  if (argc /= 4) then
    print '(a)', 'Usage: ./main [ReadLength] [ReadandRefFile] [#reads] [repeat]'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) read_length
  call get_command_argument(2, file_path)
  call get_command_argument(3, arg); read(arg, *) num_reads
  call get_command_argument(4, arg); read(arg, *) repeat
  size_uint_bits = 32
  num_warps = 8
  concurrent_threads = warp_size * num_warps
  num_blocks = (num_reads + concurrent_threads - 1) / concurrent_threads

  allocate(read_seq(0:num_reads*8-1), ref_seq(0:num_reads*8-1), device_results(0:num_reads-1), host_results(0:num_reads-1))
  read_seq = 0_int32
  ref_seq = 0_int32
  open(newunit=read_unit, file=trim(file_path), status='old', action='read', iostat=ios)
  if (ios /= 0 .and. index(trim(file_path), '../') == 1) then
    open(newunit=read_unit, file='../../src/'//trim(file_path(4:)), status='old', action='read', iostat=ios)
  end if
  if (ios /= 0) then
    print '(a,a)', 'The file does not exist or you do not have access permission: ', trim(file_path)
    stop
  end if
  do this_read = 0, num_reads - 1
    read(read_unit, '(a)', iostat=ios) line
    if (ios /= 0) exit
    call split_tab(line, token1, token2)
    do j = 0, read_length - 1
      call encode_base(token1(j+1:j+1), read_seq((j*2/size_uint_bits) + this_read*nbytes), 31 - (mod(j, size_uint_bits/2) * 2))
      call encode_base(token2(j+1:j+1), ref_seq((j*2/size_uint_bits) + this_read*nbytes), 31 - (mod(j, size_uint_bits/2) * 2))
    end do
  end do
  close(read_unit)

  !$omp target data map(to: read_seq(0:num_reads*8-1), ref_seq(0:num_reads*8-1)) map(alloc: device_results(0:num_reads-1))
    error = .false.
    do loop_par = 0, 25
      error_threshold = (loop_par * read_length) / 100
      t1 = omp_get_wtime()
      do i = 0, repeat - 1
        call sneaky_snake(num_blocks, concurrent_threads, read_seq, ref_seq, device_results, num_reads, error_threshold)
      end do
      t2 = omp_get_wtime()
      elapsed_us = (t2 - t1) * 1.0e6_real64
      !$omp target update from(device_results(0:num_reads-1))
      call sneaky_snake_ref(read_seq, ref_seq, host_results, num_reads, error_threshold)
      if (any(device_results /= host_results)) then
        error = .true.
        exit
      end if
      accepted = count(device_results == 1)
      write(*,'(a,i2,a,f10.4,a,i10,a,i10)') 'Error threshold: ', error_threshold, ' | Average kernel time (us): ', elapsed_us / repeat, &
        ' | Accepted: ', accepted, ' | Rejected: ', num_reads - accepted
    end do
    if (error) then
      print '(a)', 'FAIL'
    else
      print '(a)', 'PASS'
    end if
  !$omp end target data

contains
  subroutine split_tab(line, a, b)
    character(len=*), intent(in) :: line
    character(len=*), intent(out) :: a, b
    integer :: p
    p = index(line, char(9))
    if (p == 0) p = index(line, ' ')
    a = adjustl(line(:p-1))
    b = adjustl(line(p+1:))
  end subroutine split_tab

  subroutine encode_base(ch, word, bit)
    character(len=1), intent(in) :: ch
    integer(int32), intent(inout) :: word
    integer, intent(in) :: bit
    select case (ch)
    case ('C')
      word = set_bit(word, bit - 1)
    case ('G')
      word = set_bit(word, bit)
    case ('T')
      word = set_bit(set_bit(word, bit), bit - 1)
    case default
    end select
  end subroutine encode_base
end program snake
