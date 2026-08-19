program main
  use iso_fortran_env, only: real64, int64
  implicit none
  integer :: argc, m, nsys, block_size, repeat, i, j, rep, stat
  integer(int64) :: matrix_size, pos
  character(len=128) :: arg
  real(real64), allocatable :: u_seq(:), d_seq(:), l_seq(:), rhs_seq(:)
  real(real64), allocatable :: u_input(:), d_input(:), l_input(:), rhs_input(:)
  real(real64), allocatable :: u_host(:), d_host(:), l_host(:), rhs_host(:)
  real(real64), allocatable :: rhs_seq_output(:), rhs_seq_interleave(:)
  real(real64), allocatable :: a(:), b(:), d(:), rhs(:)
  real(real64) :: t0, t1

  argc = command_argument_count()
  if (argc /= 4) then
    print '(a)', 'Usage: ./main [system size] [#systems] [thread block size] [repeat]'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) m
  call get_command_argument(2, arg); read(arg, *, iostat=stat) nsys
  call get_command_argument(3, arg); read(arg, *, iostat=stat) block_size
  call get_command_argument(4, arg); read(arg, *, iostat=stat) repeat
  matrix_size = int(m, int64) * int(nsys, int64)
  allocate(a(0:m-1), b(0:m-1), d(0:m-1), rhs(0:m-1))
  allocate(u_seq(0:matrix_size-1), d_seq(0:matrix_size-1), l_seq(0:matrix_size-1), rhs_seq(0:matrix_size-1))
  allocate(u_input(0:matrix_size-1), d_input(0:matrix_size-1), l_input(0:matrix_size-1), rhs_input(0:matrix_size-1))
  allocate(u_host(0:matrix_size-1), d_host(0:matrix_size-1), l_host(0:matrix_size-1), rhs_host(0:matrix_size-1))
  allocate(rhs_seq_output(0:matrix_size-1), rhs_seq_interleave(0:matrix_size-1))
  call load_thomas_matrix_syn(m, a, b, d, rhs)
  call initialize_rows(m, nsys, a, b, d, rhs, u_seq, l_seq, d_seq, rhs_seq, u_input, l_input, d_input, rhs_input)
  t0 = wall_seconds()
  do rep = 1, repeat
    call solve_seq(l_seq, d_seq, u_seq, rhs_seq, m, nsys)
  end do
  t1 = wall_seconds()
  print '(a,f12.6,a)', 'Average serial execution time: ', (t1 - t0) * 1000.0_real64 / real(repeat, real64), ' (ms)'
  rhs_seq_output = rhs_seq
  call initialize_rows(m, nsys, a, b, d, rhs, u_seq, l_seq, d_seq, rhs_seq, u_input, l_input, d_input, rhs_input)
  do i = 0, m - 1
    do j = 0, nsys - 1
      u_host(i * nsys + j) = u_input(j * m + i)
      l_host(i * nsys + j) = l_input(j * m + i)
      d_host(i * nsys + j) = d_input(j * m + i)
      rhs_host(i * nsys + j) = rhs_input(j * m + i)
      rhs_seq_interleave(i * nsys + j) = rhs_seq_output(j * m + i)
    end do
  end do
!$omp target data map(to:l_host,d_host,u_host) map(tofrom:rhs_host)
  t0 = wall_seconds()
  do rep = 1, repeat
!$omp target teams distribute parallel do thread_limit(block_size)
    do j = 0, nsys - 1
      call solve_interleaved_one(l_host, d_host, u_host, rhs_host, m, nsys, j)
    end do
!$omp end target teams distribute parallel do
  end do
  t1 = wall_seconds()
  print '(a,f12.6,a)', 'Average kernel execution time: ', (t1 - t0) * 1000.0_real64 / real(repeat, real64), ' (ms)'
!$omp end target data
  call calc_error(rhs_seq_interleave, rhs_host, int(matrix_size))
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine load_thomas_matrix_syn(size, a, b, d, rhs)
    integer, intent(in) :: size
    real(real64), intent(out) :: a(0:), b(0:), d(0:), rhs(0:)
    integer :: i
    do i = 0, size - 1
      a(i) = frand(-2.0_real64, 2.0_real64, i * 4 + 1)
      b(i) = frand(-2.0_real64, 2.0_real64, i * 4 + 2)
      d(i) = frand(5.0_real64, 10.0_real64, i * 4 + 3)
      rhs(i) = frand(-2.0_real64, 2.0_real64, i * 4 + 4)
    end do
  end subroutine load_thomas_matrix_syn

  pure real(real64) function frand(lo, hi, seed)
    real(real64), intent(in) :: lo, hi
    integer, intent(in) :: seed
    frand = lo + real(mod(seed * 1103515245 + 12345, 2147483647), real64) / 2147483647.0_real64 * (hi - lo)
  end function frand

  subroutine initialize_rows(m, nsys, a, b, d, rhs, u_seq, l_seq, d_seq, rhs_seq, u_input, l_input, d_input, rhs_input)
    integer, intent(in) :: m, nsys
    real(real64), intent(in) :: a(0:), b(0:), d(0:), rhs(0:)
    real(real64), intent(out) :: u_seq(0:), l_seq(0:), d_seq(0:), rhs_seq(0:), u_input(0:), l_input(0:), d_input(0:), rhs_input(0:)
    integer :: i, j, idx
    do i = 0, nsys - 1
      do j = 0, m - 1
        idx = i * m + j
        u_seq(idx) = a(j); u_input(idx) = a(j)
        d_seq(idx) = d(j); d_input(idx) = d(j)
        l_seq(idx) = b(j); l_input(idx) = b(j)
        rhs_seq(idx) = rhs(j); rhs_input(idx) = rhs(j)
      end do
    end do
  end subroutine initialize_rows

  subroutine solve_seq(l, d, u, rhs, m, nsys)
    integer, intent(in) :: m, nsys
    real(real64), intent(in) :: l(0:), d(0:)
    real(real64), intent(inout) :: u(0:), rhs(0:)
    integer :: j, i, first, last
    do j = 0, nsys - 1
      first = j * m
      last = first + m - 1
      u(first) = u(first) / d(first)
      rhs(first) = rhs(first) / d(first)
      do i = first + 1, last - 1
        u(i) = u(i) / (d(i) - l(i) * u(i - 1))
        rhs(i) = (rhs(i) - l(i) * rhs(i - 1)) / (d(i) - l(i) * u(i - 1))
      end do
      rhs(last) = (rhs(last) - l(last) * rhs(last - 1)) / (d(last) - l(last) * u(last - 1))
      do i = last - 1, first, -1
        rhs(i) = rhs(i) - u(i) * rhs(i + 1)
      end do
    end do
  end subroutine solve_seq

  subroutine solve_interleaved_one(l, d, u, rhs, m, nsys, tid)
    integer, intent(in) :: m, nsys, tid
    real(real64), intent(in) :: l(0:), d(0:)
    real(real64), intent(inout) :: u(0:), rhs(0:)
    integer :: first, last, i
    first = tid
    last = nsys * (m - 1) + tid
    u(first) = u(first) / d(first)
    rhs(first) = rhs(first) / d(first)
    do i = first + nsys, last - nsys, nsys
      u(i) = u(i) / (d(i) - l(i) * u(i - nsys))
      rhs(i) = (rhs(i) - l(i) * rhs(i - nsys)) / (d(i) - l(i) * u(i - nsys))
    end do
    rhs(last) = (rhs(last) - l(last) * rhs(last - nsys)) / (d(last) - l(last) * u(last - nsys))
    do i = last - nsys, first, -nsys
      rhs(i) = rhs(i) - u(i) * rhs(i + nsys)
    end do
  end subroutine solve_interleaved_one

  subroutine calc_error(src, dst, n)
    integer, intent(in) :: n
    real(real64), intent(in) :: src(0:), dst(0:)
    integer :: i
    real(real64) :: err
    err = 0.0_real64
    do i = 0, n - 1
      err = max(err, abs(abs(src(i)) - abs(dst(i))))
    end do
    print '(a,es12.5)', 'Maximum error: ', err
  end subroutine calc_error
end program main
