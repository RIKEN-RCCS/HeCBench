module crc64_module
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real64
  implicit none
  integer(int64), parameter :: crc_poly = int(z'C96C5795D7870F42', int64)
  integer, parameter :: min_thread_bytes = 1024
!$omp declare target (crc64_bytes, crc64_multiply, crc64_x_power, crc64_combine)
contains

  subroutine build_crc_table(table)
    integer(int64), intent(out) :: table(0:255)
    integer :: i, bit
    integer(int64) :: value
    do i = 0, 255
      value = int(i, int64)
      do bit = 1, 8
        if (btest(value, 0)) then
          value = ieor(shiftr(value, 1), crc_poly)
        else
          value = shiftr(value, 1)
        end if
      end do
      table(i) = value
    end do
  end subroutine build_crc_table

  function crc64_bytes(data, start, nbytes, table) result(checksum)
    integer(int8), intent(in) :: data(0:*)
    integer, intent(in) :: start, nbytes
    integer(int64), intent(in) :: table(0:255)
    integer(int64) :: checksum
    integer :: position, index
    checksum = not(0_int64)
    do position = start, start + nbytes - 1
      index = int(iand(ieor(checksum, int(data(position), int64)), int(z'FF', int64)))
      checksum = ieor(table(index), shiftr(checksum, 8))
    end do
    checksum = not(checksum)
  end function crc64_bytes

  function crc64_multiply(left, right) result(product)
    integer(int64), intent(in) :: left, right
    integer(int64) :: product, a, b, top
    a = left; b = right; product = 0_int64; top = shiftl(1_int64, 63)
    do while (a /= 0_int64)
      if (iand(a, top) /= 0_int64) then
        product = ieor(product, b)
        a = ieor(a, top)
      end if
      if (btest(b, 0)) then
        b = ieor(shiftr(b, 1), crc_poly)
      else
        b = shiftr(b, 1)
      end if
      a = shiftl(a, 1)
    end do
  end function crc64_multiply

  function crc64_x_power(n) result(power)
    integer(int64), intent(in) :: n
    integer(int64) :: power, base, exponent
    power = shiftl(1_int64, 63)
    base = shiftl(1_int64, 62)
    exponent = n
    do while (exponent /= 0_int64)
      if (btest(exponent, 0)) power = crc64_multiply(power, base)
      base = crc64_multiply(base, base)
      exponent = shiftr(exponent, 1)
    end do
  end function crc64_x_power

  function crc64_combine(first, second, second_bytes) result(checksum)
    integer(int64), intent(in) :: first, second, second_bytes
    integer(int64) :: checksum
    checksum = ieor(second, crc64_multiply(first, crc64_x_power(8_int64 * second_bytes)))
  end function crc64_combine
  function crc64_omp(nbytes, data, table) result(checksum)
    integer, intent(in) :: nbytes
    integer(int8), intent(in) :: data(0:nbytes-1)
    integer(int64), intent(in) :: table(0:255)
    integer(int64) :: checksum
    integer(int64), allocatable :: thread_checksum(:), thread_size(:)
    integer :: nthreads, tid, bytes_per_thread, first, last, size
    if (nbytes > 2 * min_thread_bytes) then
      nthreads = 96 * 8 * 32
      if (nbytes < nthreads * min_thread_bytes) nthreads = nbytes / min_thread_bytes
      allocate(thread_checksum(0:nthreads-1), thread_size(0:nthreads-1))
!$omp target data map(from:thread_checksum,thread_size) map(to:data,table)
!$omp target teams distribute parallel do num_teams(nthreads/64) thread_limit(64) private(bytes_per_thread,first,last,size)
      do tid = 0, nthreads - 1
        bytes_per_thread = nbytes / nthreads
        first = bytes_per_thread * tid
        if (tid /= nthreads - 1) then
          last = first + bytes_per_thread
        else
          last = nbytes
        end if
        size = last - first
        thread_size(tid) = int(size, int64)
        thread_checksum(tid) = crc64_bytes(data, first, size, table)
      end do
!$omp end target teams distribute parallel do
!$omp end target data
      checksum = thread_checksum(0)
      do tid = 1, nthreads - 1
        checksum = crc64_combine(checksum, thread_checksum(tid), thread_size(tid))
      end do
      deallocate(thread_checksum, thread_size)
    else
      checksum = crc64_bytes(data, 0, nbytes, table)
    end if
  end function crc64_omp

  subroutine crc64_invert(checksum, bytes)
    integer(int64), intent(in) :: checksum
    integer(int8), intent(out) :: bytes(0:7)
    integer(int64) :: value
    integer :: i
    value = not(checksum)
    do i = 0, 7
      bytes(i) = int(iand(shiftr(value, 8*i), int(z'FF', int64)), int8)
    end do
  end subroutine crc64_invert
end module crc64_module

program crc64_test
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real64
  use, intrinsic :: iso_c_binding, only : c_int, c_double
  use omp_lib
  use crc64_module
  implicit none
  interface
    subroutine c_srand48(seed) bind(C, name='srand48')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand48
    function c_drand48() bind(C, name='drand48') result(value)
      import :: c_double
      real(c_double) :: value
    end function c_drand48
  end interface
  integer :: ntests, seed, max_test_length, argc, ntest, test_length, div_point, i
  integer(int64) :: checksum, combined, first_checksum, second_checksum
  integer(int64) :: table(0:255)
  integer(int8), allocatable :: buffer(:)
  real(real64) :: total_time, total_bytes, begin_time, end_time, elapsed
  character(len=64) :: argument

  ntests = 10; seed = 5; max_test_length = 2097152
  argc = command_argument_count()
  if (argc > 0) then; call get_command_argument(1, argument); read(argument, *) ntests; end if
  if (argc > 1) then; call get_command_argument(2, argument); read(argument, *) seed; end if
  if (argc > 2) then; call get_command_argument(3, argument); read(argument, *) max_test_length; end if
  print '(a,i0,a,i0)', 'Running ', ntests, ' tests with seed ', seed
  call c_srand48(int(seed, c_int)); call build_crc_table(table)
  total_time = 0.0_real64; total_bytes = 0.0_real64
  do ntest = 1, ntests
    test_length = int(real(max_test_length, real64) * (c_drand48() + 1.0_real64))
    allocate(buffer(0:test_length+7)); buffer = 0_int8
    do i = 0, test_length - 1
      buffer(i) = int(255.0_real64 * c_drand48(), int8)
    end do
    begin_time = omp_get_wtime()
    checksum = crc64_omp(test_length, buffer, table)
    end_time = omp_get_wtime(); elapsed = end_time - begin_time
    if (ntest > 1) then
      total_time = total_time + elapsed
      total_bytes = total_bytes + real(test_length, real64)
    end if
    call crc64_invert(checksum, buffer(test_length:test_length+7))
    combined = crc64_bytes(buffer, 0, test_length+8, table)
    write(*,'(i0,1x,i0,1x,a)', advance='no') ntest, test_length, merge('pass ', 'fail ', combined == not(0_int64))
    div_point = int(real(test_length, real64) * c_drand48())
    first_checksum = crc64_bytes(buffer, 0, div_point, table)
    second_checksum = crc64_bytes(buffer, div_point, test_length-div_point, table)
    combined = crc64_combine(first_checksum, second_checksum, int(test_length-div_point, int64))
    write(*,'(a)') merge('pass', 'fail', combined == checksum)
    deallocate(buffer)
  end do
  print '(f0.6,a)', (total_bytes/(1024.0_real64*1024.0_real64))/total_time, ' MB/s'
end program crc64_test
