module quantbnb_mod
  use iso_fortran_env, only: int8, int32, int64, real32, real64
  implicit none
  integer, parameter :: mt_n = 624, mt_m = 397
  integer(int64), parameter :: mask32 = int(z'00000000FFFFFFFF', int64)
  integer(int64), parameter :: upper_mask = int(z'0000000080000000', int64)
  integer(int64), parameter :: lower_mask = int(z'000000007FFFFFFF', int64)
  integer(int64), parameter :: matrix_a = int(z'000000009908B0DF', int64)
  integer(int8), parameter :: byte_value(0:255) = [ &
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, &
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, &
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47, &
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63, &
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79, &
    80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95, &
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111, &
    112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127, &
    -128,-127,-126,-125,-124,-123,-122,-121,-120,-119,-118,-117,-116,-115,-114,-113, &
    -112,-111,-110,-109,-108,-107,-106,-105,-104,-103,-102,-101,-100,-99,-98,-97, &
    -96,-95,-94,-93,-92,-91,-90,-89,-88,-87,-86,-85,-84,-83,-82,-81, &
    -80,-79,-78,-77,-76,-75,-74,-73,-72,-71,-70,-69,-68,-67,-66,-65, &
    -64,-63,-62,-61,-60,-59,-58,-57,-56,-55,-54,-53,-52,-51,-50,-49, &
    -48,-47,-46,-45,-44,-43,-42,-41,-40,-39,-38,-37,-36,-35,-34,-33, &
    -32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17, &
    -16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ]
!$omp declare target (byte_value, dquantize)
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine load_code(code)
    real(real32), intent(out) :: code(0:255)
    character(len=:), allocatable :: text
    character(len=1) :: ch
    character(len=64) :: token
    character(len=32768) :: line
    integer :: u, ios, pos, n, first, last
    logical :: in_number
    code = 0.0_real32
    text = ''
    open(newunit=u, file='../quantBnB-cuda/code.h', status='old', action='read', iostat=ios)
    if (ios /= 0) stop 'Cannot open ../quantBnB-cuda/code.h'
    do
      read(u, '(a)', iostat=ios) line
      if (ios /= 0) exit
      text = text // trim(line) // ' '
    end do
    close(u)
    first = index(text, '{') + 1
    last = index(text, '}')
    if (first <= 1 .or. last < first) stop 'Malformed ../quantBnB-cuda/code.h'
    token = ''; pos = 0; n = 0; in_number = .false.
    do u = first, last
      ch = text(u:u)
      if ((ch >= '0' .and. ch <= '9') .or. ch == '-' .or. ch == '+' .or. ch == '.' .or. &
          ((ch == 'e' .or. ch == 'E') .and. in_number)) then
        pos = pos + 1; token(pos:pos) = ch; in_number = .true.
      else if (in_number) then
        read(token(1:pos), *) code(n)
        n = n + 1; pos = 0; token = ''; in_number = .false.
        if (n > 255) exit
      end if
    end do
    if (n /= 256) stop 'Expected 256 values in ../quantBnB-cuda/code.h'
  end subroutine

  integer(int8) function dquantize(code, rnd, x) result(out)
    real(real32), intent(in) :: code(0:255), rnd, x
    integer :: pivot, upper_pivot, lower_pivot, step
    real(real32) :: lower, upper, val, midpoint, dist_to_upper, dist_to_lower, dist_full
    pivot = 127; upper_pivot = 255; lower_pivot = 0
    lower = -1.0_real32; upper = 1.0_real32; val = code(pivot)
    step = 64
    do while (step > 0)
      if (x > val) then
        lower_pivot = pivot; lower = val; pivot = pivot + step
      else
        upper_pivot = pivot; upper = val; pivot = pivot - step
      end if
      val = code(pivot); step = step / 2
    end do
    if (upper_pivot == 255) upper = code(upper_pivot)
    if (lower_pivot == 0) lower = code(lower_pivot)
    if (x > val) then
      midpoint = (upper + val) * 0.5_real32
      if (x > midpoint) then; out = byte_value(upper_pivot); else; out = byte_value(pivot); end if
    else
      midpoint = (lower + val) * 0.5_real32
      if (x < midpoint) then; out = byte_value(lower_pivot); else; out = byte_value(pivot); end if
    end if
    dist_to_upper = rnd; dist_to_lower = rnd; dist_full = rnd
  end function

  subroutine mt_seed(state, index, seed)
    integer(int64), intent(out) :: state(0:mt_n-1)
    integer, intent(out) :: index
    integer(int64), intent(in) :: seed
    integer :: i
    state(0) = iand(seed, mask32)
    do i = 1, mt_n-1
      state(i) = iand(1812433253_int64 * ieor(state(i-1), ishft(state(i-1), -30)) + int(i, int64), mask32)
    end do
    index = mt_n
  end subroutine

  subroutine mt_twist(state)
    integer(int64), intent(inout) :: state(0:mt_n-1)
    integer :: i
    integer(int64) :: y
    do i = 0, mt_n-1
      y = ior(iand(state(i), upper_mask), iand(state(mod(i+1,mt_n)), lower_mask))
      state(i) = ieor(state(mod(i+mt_m,mt_n)), ishft(y,-1))
      if (iand(y,1_int64) /= 0_int64) state(i) = ieor(state(i), matrix_a)
      state(i) = iand(state(i), mask32)
    end do
  end subroutine

  integer(int64) function mt_next(state, index) result(value)
    integer(int64), intent(inout) :: state(0:mt_n-1)
    integer, intent(inout) :: index
    if (index >= mt_n) then
      call mt_twist(state)
      index = 0
    end if
    value = state(index)
    index = index + 1
    value = ieor(value, ishft(value,-11))
    value = ieor(value, iand(ishft(value,7), int(z'000000009D2C5680',int64)))
    value = ieor(value, iand(ishft(value,15), int(z'00000000EFC60000',int64)))
    value = ieor(value, ishft(value,-18))
    value = iand(value, mask32)
  end function

  real(real32) function uniform01(state, index) result(value)
    integer(int64), intent(inout) :: state(0:mt_n-1)
    integer, intent(inout) :: index
    value = real(real(mt_next(state,index),real64) * 2.3283064365386962890625e-10_real64, real32)
  end function

  subroutine fill_normal(a)
    real(real32), intent(out) :: a(0:)
    integer :: i, mt_index
    integer(int64) :: mt_state(0:mt_n-1)
    real(real32) :: x, y, radius, multiplier, saved
    logical :: saved_available
    call mt_seed(mt_state, mt_index, 19937_int64)
    saved_available = .false.
    i = 0
    do while (i <= ubound(a,1))
      if (saved_available) then
        a(i) = saved
        saved_available = .false.
      else
        do
          x = 2.0_real32*uniform01(mt_state,mt_index)-1.0_real32
          y = 2.0_real32*uniform01(mt_state,mt_index)-1.0_real32
          radius = x*x + y*y
          if (radius > 0.0_real32 .and. radius <= 1.0_real32) exit
        end do
        multiplier = sqrt(-2.0_real32*log(radius)/radius)
        saved = x*multiplier
        saved_available = .true.
        a(i) = y*multiplier
      end if
      i = i + 1
    end do
  end subroutine
end module

program main
  use quantbnb_mod
  implicit none
  integer(int64) :: n, i
  integer :: repeat, r, ios, block_size, grid
  character(len=64) :: arg
  real(real32), allocatable :: a(:), code(:)
  integer(int8), allocatable :: out(:), ref(:)
  real(real64) :: t0, t1
  logical :: ok

  if (command_argument_count() /= 2) then
    print '(a)', 'Usage: main <number of elements> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) n
  call get_command_argument(2,arg); read(arg,*,iostat=ios) repeat
  allocate(a(0:n-1), out(0:n-1), ref(0:n-1), code(0:255))
  call load_code(code)
  call fill_normal(a)
  do i = 0, n-1
    ref(i) = dquantize(code, 0.0_real32, a(i))
  end do
  block_size = 256
  grid = int((n + block_size - 1) / block_size)

  !$omp target data map(to:a(0:n-1),code(0:255)) map(from:out(0:n-1))
  t0 = seconds()
  do r = 1, repeat
    !$omp target teams distribute parallel do num_teams(grid) num_threads(block_size)
    do i = 0, n-1
      out(i) = dquantize(code, 0.0_real32, a(i))
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,i0,a,f10.6,a)', 'Average execution time of kQuantize kernel with block size ', &
    block_size, ': ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  !$omp end target data

  ok = .true.
  do i = 0, n-1
    if (out(i) /= ref(i)) then
      ok = .false.; exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
end program
