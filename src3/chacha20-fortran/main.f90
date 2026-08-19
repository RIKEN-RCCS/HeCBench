module chacha20_device
  use iso_fortran_env, only: int8, int32, int64
  use omp_lib
  implicit none
  integer(int64), parameter :: mask32 = int(z'00000000FFFFFFFF', int64)
  !$omp declare target (add32, rotl32, pack4, unpack4, quarter, chacha_block, chacha_crypt, hex_to_raw)
contains
  integer(int64) function add32(a, b)
    integer(int64), intent(in) :: a, b
    add32 = iand(a + b, mask32)
  end function add32

  integer(int64) function rotl32(x, n)
    integer(int64), intent(in) :: x
    integer, intent(in) :: n
    rotl32 = iand(ior(shiftl(x, n), shiftr(x, 32 - n)), mask32)
  end function rotl32

  integer(int64) function pack4(a, first)
    integer(int8), intent(in) :: a(*)
    integer, intent(in) :: first
    pack4 = ior(ior(shiftl(iand(int(a(first), int64), 255_int64), 0), &
                    shiftl(iand(int(a(first + 1), int64), 255_int64), 8)), &
                ior(shiftl(iand(int(a(first + 2), int64), 255_int64), 16), &
                    shiftl(iand(int(a(first + 3), int64), 255_int64), 24)))
  end function pack4

  subroutine unpack4(src, dst, first)
    integer(int64), intent(in) :: src
    integer(int8), intent(inout) :: dst(*)
    integer, intent(in) :: first
    integer(int64) :: byte_value
    integer :: i
    do i = 0, 3
      byte_value = iand(shiftr(src, 8*i), 255_int64)
      if (byte_value >= 128_int64) byte_value = byte_value - 256_int64
      dst(first + i) = int(byte_value, int8)
    end do
  end subroutine unpack4

  subroutine quarter(x, a, b, c, d)
    integer(int64), intent(inout) :: x(16)
    integer, intent(in) :: a, b, c, d
    x(a) = add32(x(a), x(b)); x(d) = rotl32(ieor(x(d), x(a)), 16)
    x(c) = add32(x(c), x(d)); x(b) = rotl32(ieor(x(b), x(c)), 12)
    x(a) = add32(x(a), x(b)); x(d) = rotl32(ieor(x(d), x(a)), 8)
    x(c) = add32(x(c), x(d)); x(b) = rotl32(ieor(x(b), x(c)), 7)
  end subroutine quarter

  subroutine chacha_block(key, nonce, counter, stream)
    integer(int8), intent(in) :: key(32), nonce(8)
    integer(int64), intent(in) :: counter
    integer(int8), intent(out) :: stream(64)
    integer(int64) :: state(16), work(16)
    integer(int8) :: magic(16)
    integer :: i, round
    magic = [int(iachar('e'),int8), int(iachar('x'),int8), int(iachar('p'),int8), int(iachar('a'),int8), &
             int(iachar('n'),int8), int(iachar('d'),int8), int(iachar(' '),int8), int(iachar('3'),int8), &
             int(iachar('2'),int8), int(iachar('-'),int8), int(iachar('b'),int8), int(iachar('y'),int8), &
             int(iachar('t'),int8), int(iachar('e'),int8), int(iachar(' '),int8), int(iachar('k'),int8)]
    do i = 1, 4
      state(i) = pack4(magic, 4 * (i - 1) + 1)
    end do
    do i = 1, 8
      state(i + 4) = pack4(key, 4 * (i - 1) + 1)
    end do
    state(13) = iand(counter, mask32)
    state(14) = iand(shiftr(counter, 32), mask32)
    state(15) = pack4(nonce, 1); state(16) = pack4(nonce, 5)
    work = state
    do round = 1, 10
      call quarter(work, 1, 5, 9, 13);  call quarter(work, 2, 6, 10, 14)
      call quarter(work, 3, 7, 11, 15); call quarter(work, 4, 8, 12, 16)
      call quarter(work, 1, 6, 11, 16); call quarter(work, 2, 7, 12, 13)
      call quarter(work, 3, 8, 9, 14);  call quarter(work, 4, 5, 10, 15)
    end do
    do i = 1, 16
      call unpack4(add32(work(i), state(i)), stream, 4 * (i - 1) + 1)
    end do
  end subroutine chacha_block

  subroutine chacha_crypt(bytes, n_bytes, key, nonce)
    integer(int8), intent(inout) :: bytes(*)
    integer, intent(in) :: n_bytes
    integer(int8), intent(in) :: key(32), nonce(8)
    integer(int8) :: stream(64)
    integer(int64) :: counter
    integer :: i, position
    counter = 0_int64; position = 65
    do i = 1, n_bytes
      if (position > 64) then
        call chacha_block(key, nonce, counter, stream)
        counter = counter + 1_int64
        position = 1
      end if
      bytes(i) = ieor(bytes(i), stream(position))
      position = position + 1
    end do
  end subroutine chacha_crypt
  subroutine hex_to_raw(src, n, dst, table)
    integer(int8), intent(in) :: src(*)
    integer, intent(in) :: n
    integer(int8), intent(inout) :: dst(*)
    integer(int8), intent(in) :: table(0:255)
    integer :: i, tid, nthreads
    tid = omp_get_thread_num(); nthreads = omp_get_num_threads()
    do i = tid + 1, n / 2, nthreads
      dst(i) = transfer(ior(shiftl(int(table(int(src(2*i-1),int32)), int32), 4), &
                            int(table(int(src(2*i),int32)), int32)), dst(i))
    end do
  end subroutine hex_to_raw

  subroutine test_keystreams(text_key, key_length, text_nonce, nonce_length, text_keystream, stream_length, table, raw_key, raw_nonce, raw_keystream, result)
    integer(int8), intent(in) :: text_key(*), text_nonce(*), text_keystream(*)
    integer, intent(in) :: key_length, nonce_length, stream_length
    integer(int8), intent(in) :: table(0:255)
    integer(int8), intent(inout) :: raw_key(*), raw_nonce(*), raw_keystream(*), result(*)
    integer :: tid
    !$omp target teams num_teams(1) thread_limit(256) &
    !$omp& map(to:text_key(1:key_length),text_nonce(1:nonce_length),text_keystream(1:stream_length),table(0:255)) &
    !$omp& map(tofrom:raw_key(1:key_length/2),raw_nonce(1:nonce_length/2), &
    !$omp& raw_keystream(1:stream_length/2),result(1:stream_length/2))
    !$omp parallel num_threads(256) private(tid)
    call hex_to_raw(text_key, key_length, raw_key, table)
    call hex_to_raw(text_nonce, nonce_length, raw_nonce, table)
    call hex_to_raw(text_keystream, stream_length, raw_keystream, table)
    !$omp barrier
    tid = omp_get_thread_num()
    if (tid == 0) then
      call chacha_crypt(result, stream_length / 2, raw_key(1:32), raw_nonce(1:8))
    end if
    !$omp end parallel
    !$omp end target teams
  end subroutine test_keystreams
end module chacha20_device

program main
  use iso_fortran_env, only: int8, int32, int64, real32, real64
  use omp_lib
  use chacha20_device
  implicit none
  character(len=*), parameter :: key_string = '000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
  character(len=*), parameter :: nonce_string = '0001020304050607'
  character(len=*), parameter :: stream_string = &
    'f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c94400491769' // &
    '05d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd' // &
    '9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc3' // &
    '85245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d7' // &
    '0eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b5409' // &
    '8746d4524d38407a6deb3ab78fab78c9'
  integer :: repeat, i, j
  integer(int8) :: table(0:255)
  integer(int8) :: key_text(len(key_string)), nonce_text(len(nonce_string)), stream_text(len(stream_string))
  integer(int8), allocatable :: raw_key(:), raw_nonce(:), raw_stream(:), result(:)
  real(real32) :: elapsed_us
  real(real64) :: start_time, stop_time
  character(len=32) :: arg

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    error stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) repeat
  table = 0_int8
  do i = 0, 9
    table(iachar('0') + i) = int(i, int8)
  end do
  do i = 0, 25
    table(iachar('a') + i) = int(i + 10, int8)
    table(iachar('A') + i) = int(i + 10, int8)
  end do
  do i = 1, len(key_string); key_text(i) = int(iachar(key_string(i:i)), int8); end do
  do i = 1, len(nonce_string); nonce_text(i) = int(iachar(nonce_string(i:i)), int8); end do
  do i = 1, len(stream_string); stream_text(i) = int(iachar(stream_string(i:i)), int8); end do
  allocate(raw_key(len(key_string)/2), raw_nonce(len(nonce_string)/2), &
           raw_stream(len(stream_string)/2), result(len(stream_string)/2))
!$omp target data map(to:table(0:255), key_text(1:len(key_string)), nonce_text(1:len(nonce_string)), &
!$omp& stream_text(1:len(stream_string))) &
!$omp& map(alloc:raw_key(1:size(raw_key)),raw_nonce(1:size(raw_nonce))) &
!$omp& map(from:raw_stream(1:size(raw_stream)),result(1:size(result)))
  start_time = omp_get_wtime()
  do i = 1, repeat
    !$omp target teams distribute parallel do
    do j = 1, size(result)
      result(j) = 0_int8
    end do
    !$omp end target teams distribute parallel do
    call test_keystreams(key_text, size(key_text), nonce_text, size(nonce_text), &
      stream_text, size(stream_text), table, raw_key, raw_nonce, raw_stream, result)
  end do
  stop_time = omp_get_wtime()
!$omp end target data
  elapsed_us = real((stop_time - start_time) * 1.0e6_real64 / real(repeat, real64), real32)
  print '(a,f0.6,a)', 'Average execution time of kernels: ', elapsed_us, ' (us)'
  if (all(result == raw_stream)) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
end program main
