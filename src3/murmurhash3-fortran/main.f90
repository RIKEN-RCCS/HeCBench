module murmurhash3_mod
  use iso_fortran_env, only: int32, int64, real64
  implicit none
  integer, parameter :: block_size = 256
  integer(int64), parameter :: c1 = int(z'87c37b91114253d5', int64)
  integer(int64), parameter :: c2 = int(z'4cf5ad432745937f', int64)
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  pure integer(int64) function rotl64(x, r) result(v)
    integer(int64), intent(in) :: x
    integer, intent(in) :: r
    v = ior(ishft(x, r), shiftr(x, 64-r))
  end function

  pure integer(int64) function getblock64(data, base, iblock) result(s)
    integer(int32), intent(in) :: data(0:)
    integer, intent(in) :: base, iblock
    integer :: n
    s = 0_int64
    do n = 0, 7
      s = ior(s, ishft(int(iand(data(base + 8*iblock + n),255), int64), n*8))
    end do
  end function

  pure integer(int64) function fmix64(k0) result(k)
    integer(int64), intent(in) :: k0
    k = k0
    k = ieor(k, shiftr(k, 33))
    k = k * int(z'ff51afd7ed558ccd', int64)
    k = ieor(k, shiftr(k, 33))
    k = k * int(z'c4ceb9fe1a85ec53', int64)
    k = ieor(k, shiftr(k, 33))
  end function

  subroutine murmurhash3_x64_128(data, base, len, seed, out0, out1)
    integer(int32), intent(in) :: data(0:)
    integer, intent(in) :: base
    integer(int32), intent(in) :: len, seed
    integer(int64), intent(out) :: out0, out1
    integer(int32) :: nblocks
    integer :: i, tail
    integer(int64) :: h1, h2, k1, k2
    nblocks = len / 16
    h1 = int(seed, int64); h2 = int(seed, int64)
    do i = 0, nblocks-1
      k1 = getblock64(data, base, i*2)
      k2 = getblock64(data, base, i*2+1)
      k1 = k1 * c1; k1 = rotl64(k1, 31); k1 = k1 * c2; h1 = ieor(h1, k1)
      h1 = rotl64(h1, 27); h1 = h1 + h2; h1 = h1*5_int64 + int(z'52dce729', int64)
      k2 = k2 * c2; k2 = rotl64(k2, 33); k2 = k2 * c1; h2 = ieor(h2, k2)
      h2 = rotl64(h2, 31); h2 = h2 + h1; h2 = h2*5_int64 + int(z'38495ab5', int64)
    end do
    tail = base + nblocks*16
    k1 = 0_int64; k2 = 0_int64
    select case (iand(len, 15_int32))
    case (15); k2 = ieor(k2, ishft(int(data(tail+14),int64),48)); k2 = ieor(k2, ishft(int(data(tail+13),int64),40)); k2 = ieor(k2, ishft(int(data(tail+12),int64),32)); k2 = ieor(k2, ishft(int(data(tail+11),int64),24)); k2 = ieor(k2, ishft(int(data(tail+10),int64),16)); k2 = ieor(k2, ishft(int(data(tail+9),int64),8)); k2 = ieor(k2, int(data(tail+8),int64))
    case (14); k2 = ieor(k2, ishft(int(data(tail+13),int64),40)); k2 = ieor(k2, ishft(int(data(tail+12),int64),32)); k2 = ieor(k2, ishft(int(data(tail+11),int64),24)); k2 = ieor(k2, ishft(int(data(tail+10),int64),16)); k2 = ieor(k2, ishft(int(data(tail+9),int64),8)); k2 = ieor(k2, int(data(tail+8),int64))
    case (13); k2 = ieor(k2, ishft(int(data(tail+12),int64),32)); k2 = ieor(k2, ishft(int(data(tail+11),int64),24)); k2 = ieor(k2, ishft(int(data(tail+10),int64),16)); k2 = ieor(k2, ishft(int(data(tail+9),int64),8)); k2 = ieor(k2, int(data(tail+8),int64))
    case (12); k2 = ieor(k2, ishft(int(data(tail+11),int64),24)); k2 = ieor(k2, ishft(int(data(tail+10),int64),16)); k2 = ieor(k2, ishft(int(data(tail+9),int64),8)); k2 = ieor(k2, int(data(tail+8),int64))
    case (11); k2 = ieor(k2, ishft(int(data(tail+10),int64),16)); k2 = ieor(k2, ishft(int(data(tail+9),int64),8)); k2 = ieor(k2, int(data(tail+8),int64))
    case (10); k2 = ieor(k2, ishft(int(data(tail+9),int64),8)); k2 = ieor(k2, int(data(tail+8),int64))
    case (9);  k2 = ieor(k2, int(data(tail+8),int64))
    end select
    if (iand(len,15_int32) >= 9) then
      k2 = k2*c2; k2 = rotl64(k2,33); k2 = k2*c1; h2 = ieor(h2,k2)
    end if
    select case (min(iand(len,15_int32), 8_int32))
    case (8); k1 = ieor(k1, ishft(int(data(tail+7),int64),56)); k1 = ieor(k1, ishft(int(data(tail+6),int64),48)); k1 = ieor(k1, ishft(int(data(tail+5),int64),40)); k1 = ieor(k1, ishft(int(data(tail+4),int64),32)); k1 = ieor(k1, ishft(int(data(tail+3),int64),24)); k1 = ieor(k1, ishft(int(data(tail+2),int64),16)); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (7); k1 = ieor(k1, ishft(int(data(tail+6),int64),48)); k1 = ieor(k1, ishft(int(data(tail+5),int64),40)); k1 = ieor(k1, ishft(int(data(tail+4),int64),32)); k1 = ieor(k1, ishft(int(data(tail+3),int64),24)); k1 = ieor(k1, ishft(int(data(tail+2),int64),16)); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (6); k1 = ieor(k1, ishft(int(data(tail+5),int64),40)); k1 = ieor(k1, ishft(int(data(tail+4),int64),32)); k1 = ieor(k1, ishft(int(data(tail+3),int64),24)); k1 = ieor(k1, ishft(int(data(tail+2),int64),16)); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (5); k1 = ieor(k1, ishft(int(data(tail+4),int64),32)); k1 = ieor(k1, ishft(int(data(tail+3),int64),24)); k1 = ieor(k1, ishft(int(data(tail+2),int64),16)); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (4); k1 = ieor(k1, ishft(int(data(tail+3),int64),24)); k1 = ieor(k1, ishft(int(data(tail+2),int64),16)); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (3); k1 = ieor(k1, ishft(int(data(tail+2),int64),16)); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (2); k1 = ieor(k1, ishft(int(data(tail+1),int64),8)); k1 = ieor(k1, int(data(tail),int64))
    case (1); k1 = ieor(k1, int(data(tail),int64))
    end select
    if (iand(len,15_int32) >= 1) then
      k1 = k1*c1; k1 = rotl64(k1,31); k1 = k1*c2; h1 = ieor(h1,k1)
    end if
    h1 = ieor(h1, int(len,int64)); h2 = ieor(h2, int(len,int64))
    h1 = h1 + h2; h2 = h2 + h1
    h1 = fmix64(h1); h2 = fmix64(h2)
    h1 = h1 + h2; h2 = h2 + h1
    out0 = h1; out1 = h2
  end subroutine
end module

program main
  use murmurhash3_mod
  implicit none
  integer :: argc, ios, repeat, r
  integer(int32) :: num_keys, i, c, total_length
  character(len=64) :: arg
  integer(int32), allocatable :: length(:), d_length(:), d_keys(:)
  integer(int64), allocatable :: out(:), ref(:)
  real(real64) :: t0, t1
  logical :: error
  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: main <number of keys> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) num_keys
  call get_command_argument(2,arg); read(arg,*,iostat=ios) repeat
  allocate(length(0:num_keys-1), d_length(0:num_keys), ref(0:2*num_keys-1), out(0:2*num_keys-1))
  call c_srand(3_int32)
  total_length = 0
  do i = 0, num_keys-1
    length(i) = mod(c_rand(), 10000)
    total_length = total_length + length(i)
  end do
  allocate(d_keys(0:total_length-1))
  d_length(0) = 0
  do i = 0, num_keys-1
    d_length(i+1) = d_length(i) + length(i)
    do c = 0, length(i)-1
      d_keys(d_length(i)+c) = mod(c, 256)
    end do
    call murmurhash3_x64_128(d_keys, d_length(i), length(i), i, ref(2*i), ref(2*i+1))
  end do
  !$omp target data map(to:d_keys(0:total_length-1),d_length(0:num_keys),length(0:num_keys-1)) map(from:out(0:2*num_keys-1))
  t0 = seconds()
  do r = 1, repeat
    !$omp target teams distribute parallel do thread_limit(block_size)
    do i = 0, num_keys-1
      call murmurhash3_x64_128(d_keys, d_length(i), length(i), i, out(2*i), out(2*i+1))
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time ', (t1-t0)/real(repeat,real64), ' (s)'
  !$omp end target data
  error = .false.
  do i = 0, num_keys-1
    if (out(2*i) /= ref(2*i) .or. out(2*i+1) /= ref(2*i+1)) then
      error = .true.; exit
    end if
  end do
  print '(a)', merge('FAIL   ', 'SUCCESS', error)
end program
