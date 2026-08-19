module c_random_mod
  use iso_c_binding, only: c_int, c_long
  implicit none
  interface
    subroutine srandom(seed) bind(C, name="srandom")
      import c_int
      integer(c_int), value :: seed
    end subroutine srandom
    function random() result(r) bind(C, name="random")
      import c_long
      integer(c_long) :: r
    end function random
  end interface
end module c_random_mod

module md5_mod
  use iso_fortran_env, only: int8, int32
  implicit none
  integer(int32), parameter :: s_tab(0:63) = [ &
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22, &
    5,9,14,20, 5,9,14,20, 5,9,14,20, 5,9,14,20, &
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23, &
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21]
  integer(int32), parameter :: k_tab(0:63) = [ &
    int(z'd76aa478',int32), int(z'e8c7b756',int32), int(z'242070db',int32), int(z'c1bdceee',int32), &
    int(z'f57c0faf',int32), int(z'4787c62a',int32), int(z'a8304613',int32), int(z'fd469501',int32), &
    int(z'698098d8',int32), int(z'8b44f7af',int32), int(z'ffff5bb1',int32), int(z'895cd7be',int32), &
    int(z'6b901122',int32), int(z'fd987193',int32), int(z'a679438e',int32), int(z'49b40821',int32), &
    int(z'f61e2562',int32), int(z'c040b340',int32), int(z'265e5a51',int32), int(z'e9b6c7aa',int32), &
    int(z'd62f105d',int32), int(z'02441453',int32), int(z'd8a1e681',int32), int(z'e7d3fbc8',int32), &
    int(z'21e1cde6',int32), int(z'c33707d6',int32), int(z'f4d50d87',int32), int(z'455a14ed',int32), &
    int(z'a9e3e905',int32), int(z'fcefa3f8',int32), int(z'676f02d9',int32), int(z'8d2a4c8a',int32), &
    int(z'fffa3942',int32), int(z'8771f681',int32), int(z'6d9d6122',int32), int(z'fde5380c',int32), &
    int(z'a4beea44',int32), int(z'4bdecfa9',int32), int(z'f6bb4b60',int32), int(z'bebfbc70',int32), &
    int(z'289b7ec6',int32), int(z'eaa127fa',int32), int(z'd4ef3085',int32), int(z'04881d05',int32), &
    int(z'd9d4d039',int32), int(z'e6db99e5',int32), int(z'1fa27cf8',int32), int(z'c4ac5665',int32), &
    int(z'f4292244',int32), int(z'432aff97',int32), int(z'ab9423a7',int32), int(z'fc93a039',int32), &
    int(z'655b59c3',int32), int(z'8f0ccc92',int32), int(z'ffeff47d',int32), int(z'85845dd1',int32), &
    int(z'6fa87e4f',int32), int(z'fe2ce6e0',int32), int(z'a3014314',int32), int(z'4e0811a1',int32), &
    int(z'f7537e82',int32), int(z'bd3af235',int32), int(z'2ad7d2bb',int32), int(z'eb86d391',int32)]
!$omp declare target (s_tab, k_tab, f_fun, g_fun, h_fun, i_fun, leftrotate, md5_2words, index_to_key)
contains
  pure integer(int32) function f_fun(x,y,z)
    integer(int32), intent(in) :: x,y,z
    f_fun = ior(iand(x,y), iand(not(x),z))
  end function f_fun
  pure integer(int32) function g_fun(x,y,z)
    integer(int32), intent(in) :: x,y,z
    g_fun = ior(iand(x,z), iand(not(z),y))
  end function g_fun
  pure integer(int32) function h_fun(x,y,z)
    integer(int32), intent(in) :: x,y,z
    h_fun = ieor(ieor(x,y),z)
  end function h_fun
  pure integer(int32) function i_fun(x,y,z)
    integer(int32), intent(in) :: x,y,z
    i_fun = ieor(y, ior(x, not(z)))
  end function i_fun
  pure integer(int32) function leftrotate(x, c)
    integer(int32), intent(in) :: x
    integer, intent(in) :: c
    leftrotate = ior(shiftl(x,c), shiftr(x,32-c))
  end function leftrotate

  subroutine md5_2words(key, len, digest)
    integer(int32), intent(in) :: key(0:7)
    integer, intent(in) :: len
    integer(int32), intent(out) :: digest(0:3)
    integer(int32) :: a,b,c,d,h0,h1,h2,h3,fval,temp,w(0:15)
    integer :: r, g
    w = 0_int32
    w(0) = ior(ior(ior(iand(key(0),255_int32), shiftl(iand(key(1),255_int32),8)), shiftl(iand(key(2),255_int32),16)), shiftl(iand(key(3),255_int32),24))
    w(1) = ior(ior(ior(iand(key(4),255_int32), shiftl(iand(key(5),255_int32),8)), shiftl(iand(key(6),255_int32),16)), shiftl(iand(key(7),255_int32),24))
    select case (len)
    case (0); w(0) = ior(w(0), int(z'00000080',int32))
    case (1); w(0) = ior(w(0), int(z'00008000',int32))
    case (2); w(0) = ior(w(0), int(z'00800000',int32))
    case (3); w(0) = ior(w(0), int(z'80000000',int32))
    case (4); w(1) = ior(w(1), int(z'00000080',int32))
    case (5); w(1) = ior(w(1), int(z'00008000',int32))
    case (6); w(1) = ior(w(1), int(z'00800000',int32))
    case (7); w(1) = ior(w(1), int(z'80000000',int32))
    end select
    w(14) = int(len*8, int32)
    h0 = int(z'67452301',int32); h1 = int(z'efcdab89',int32)
    h2 = int(z'98badcfe',int32); h3 = int(z'10325476',int32)
    a = h0; b = h1; c = h2; d = h3
    do r = 0, 63
      if (r < 16) then
        fval = f_fun(b,c,d); g = r
      else if (r < 32) then
        fval = g_fun(b,c,d); g = mod(5*r + 1, 16)
      else if (r < 48) then
        fval = h_fun(b,c,d); g = mod(3*r + 5, 16)
      else
        fval = i_fun(b,c,d); g = mod(7*r, 16)
      end if
      temp = d
      d = c
      c = b
      b = b + leftrotate(a + fval + k_tab(r) + w(g), int(s_tab(r)))
      a = temp
    end do
    digest(0) = h0 + a
    digest(1) = h1 + b
    digest(2) = h2 + c
    digest(3) = h3 + d
  end subroutine md5_2words

  integer function find_keyspace_size(byte_length, vals_per_byte)
    integer, intent(in) :: byte_length, vals_per_byte
    integer :: i
    find_keyspace_size = 1
    do i = 1, byte_length
      if (find_keyspace_size >= int(z'7fffffff') / vals_per_byte) then
        find_keyspace_size = -1
        return
      end if
      find_keyspace_size = find_keyspace_size * vals_per_byte
    end do
  end function find_keyspace_size

  subroutine index_to_key(index_in, byte_length, vals_per_byte, vals)
    integer, intent(in) :: index_in, byte_length, vals_per_byte
    integer(int32), intent(out) :: vals(0:7)
    integer :: idx, i
    idx = index_in
    vals = 0_int32
    do i = 0, 7
      vals(i) = int(mod(idx, vals_per_byte), int32)
      idx = idx / vals_per_byte
    end do
  end subroutine index_to_key

  subroutine find_key_cpu(search_digest, byte_length, vals_per_byte, found_index, found_key, found_digest)
    integer(int32), intent(in) :: search_digest(0:3)
    integer, intent(in) :: byte_length, vals_per_byte
    integer, intent(out) :: found_index
    integer(int32), intent(out) :: found_key(0:7), found_digest(0:3)
    integer :: keyspace, idx, j
    integer(int32) :: key(0:7), digest(0:3)
    keyspace = find_keyspace_size(byte_length, vals_per_byte)
    do idx = 0, keyspace-1, vals_per_byte
      call index_to_key(idx, byte_length, vals_per_byte, key)
      do j = 0, vals_per_byte-1
        call md5_2words(key, byte_length, digest)
        if (all(digest == search_digest)) then
          found_index = idx + j
          found_key = key
          found_digest = digest
        end if
        key(0) = key(0) + 1_int32
      end do
    end do
  end subroutine find_key_cpu

  subroutine find_key_gpu(search_digest, byte_length, vals_per_byte, found_index, found_key, found_digest)
    integer(int32), intent(in) :: search_digest(0:3)
    integer, intent(in) :: byte_length, vals_per_byte
    integer, intent(inout) :: found_index
    integer(int32), intent(inout) :: found_key(0:7), found_digest(0:3)
    integer :: keyspace, nthreads, nblocks, threadid, startindex, j
    integer(int32) :: key(0:7), digest(0:3), sd0, sd1, sd2, sd3
    keyspace = find_keyspace_size(byte_length, vals_per_byte)
    nthreads = 256
    nblocks = ceiling(real(keyspace) / real(vals_per_byte))
    sd0 = search_digest(0); sd1 = search_digest(1); sd2 = search_digest(2); sd3 = search_digest(3)
!$omp target map(from:found_index,found_key(0:7),found_digest(0:3))
!$omp teams distribute parallel do simd thread_limit(256) private(startindex,key,digest,j)
    do threadid = 0, nblocks-1
      startindex = threadid * vals_per_byte
      call index_to_key(startindex, byte_length, vals_per_byte, key)
      do j = 0, vals_per_byte-1
        if (startindex+j < keyspace) then
          call md5_2words(key, byte_length, digest)
          if (digest(0) == sd0 .and. digest(1) == sd1 .and. digest(2) == sd2 .and. digest(3) == sd3) then
            found_index = startindex + j
            found_key = key
            found_digest = digest
          end if
          key(0) = key(0) + 1_int32
        end if
      end do
    end do
!$omp end teams distribute parallel do simd
!$omp end target
  end subroutine find_key_gpu

  subroutine print_hex_bytes(label, vals, nbytes)
    character(len=*), intent(in) :: label
    integer(int32), intent(in) :: vals(0:)
    integer, intent(in) :: nbytes
    integer :: i
    write(*,'(a)',advance='no') trim(label)//'0x'
    do i = 0, nbytes-1
      write(*,'(z2.2)',advance='no') iand(vals(i),255_int32)
    end do
    print *
  end subroutine print_hex_bytes

  subroutine digest_to_bytes(digest, bytes)
    integer(int32), intent(in) :: digest(0:3)
    integer(int32), intent(out) :: bytes(0:15)
    integer :: i
    do i = 0, 3
      bytes(4*i+0) = iand(digest(i),255_int32)
      bytes(4*i+1) = iand(shiftr(digest(i),8),255_int32)
      bytes(4*i+2) = iand(shiftr(digest(i),16),255_int32)
      bytes(4*i+3) = iand(shiftr(digest(i),24),255_int32)
    end do
  end subroutine digest_to_bytes
end module md5_mod

program MD5Hash
  use iso_fortran_env, only: int32
  use omp_lib
  use c_random_mod
  use md5_mod
  implicit none
  integer, parameter :: sizes_byte_length(0:3) = [7,5,6,5]
  integer, parameter :: sizes_vals_per_byte(0:3) = [10,35,25,70]
  integer :: offload, passes, size, byte_length, vals_per_byte, keyspace, pass, random_index
  integer :: found_index
  integer(int32) :: random_key(0:7), found_key(0:7), random_digest(0:3), found_digest(0:3), digest_bytes(0:15)
  character(len=64) :: arg
  real(8) :: start_time, elapsed_ms, rate
  logical :: ok

  call get_command_argument(1,arg); read(arg,*) offload
  call get_command_argument(2,arg); read(arg,*) passes
  do size = 1, 4
    byte_length = sizes_byte_length(size-1)
    vals_per_byte = sizes_vals_per_byte(size-1)
    print '(a,i0,a,i0,a)', 'Searching keys of length ', byte_length, ' bytes and ', vals_per_byte, ' values per byte'
    keyspace = find_keyspace_size(byte_length, vals_per_byte)
    if (keyspace < 0 .or. byte_length > 7) stop 1
    print '(a,i0,a,i0,a)', '|keyspace| = ', keyspace, ' (', int(keyspace/1.0e6), 'M)'
    call srandom(12345)
    do pass = 0, passes-1
      random_index = mod(int(random()), keyspace)
      call index_to_key(random_index, byte_length, vals_per_byte, random_key)
      call md5_2words(random_key, byte_length, random_digest)
      print *
      print '(a,i0,a)', '--- iteration ', pass, ' ---'
      print '(a)', 'Looking for random key:'
      print '(a,i0)', ' randomIndex = ', random_index
      call print_hex_bytes(' randomKey   = ', random_key, 8)
      call digest_to_bytes(random_digest, digest_bytes)
      call print_hex_bytes(' randomDigest= ', digest_bytes, 16)
      found_digest = 0_int32; found_key = 0_int32; found_index = -1
      start_time = omp_get_wtime()
      if (offload == 0) then
        call find_key_cpu(random_digest, byte_length, vals_per_byte, found_index, found_key, found_digest)
      else
        call find_key_gpu(random_digest, byte_length, vals_per_byte, found_index, found_key, found_digest)
      end if
      elapsed_ms = (omp_get_wtime() - start_time) * 1000.0d0
      rate = real(keyspace,8) / (elapsed_ms/1000.0d0) / 1.0d9
      print '(a,f0.3,a,es12.5,a)', 'time = ', elapsed_ms, ' ms, rate = ', rate, ' GHash/sec'
      ok = found_index == random_index .and. all(found_key == random_key) .and. all(found_digest == random_digest)
      if (.not. ok) rate = huge(rate)
      if (ok) print '(a)', 'Successfully found match (index, key, hash):'
      print '(a,i0)', ' foundIndex  = ', found_index
      call print_hex_bytes(' foundKey    = ', found_key, 8)
      call digest_to_bytes(found_digest, digest_bytes)
      call print_hex_bytes(' foundDigest = ', digest_bytes, 16)
      print *
      print '(a)', merge('PASS', 'FAIL', ok)
    end do
  end do
end program MD5Hash
