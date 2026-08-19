module attention_rng
  use, intrinsic :: iso_fortran_env, only : int64, real32
  implicit none
  integer, parameter :: mt_n = 624, mt_m = 397
  integer(int64), parameter :: mask32 = int(z'00000000FFFFFFFF', int64)
  integer(int64), parameter :: upper_mask = int(z'0000000080000000', int64)
  integer(int64), parameter :: lower_mask = int(z'000000007FFFFFFF', int64)
  integer(int64), parameter :: matrix_a = int(z'000000009908B0DF', int64)
  type :: mt19937_state
    integer(int64) :: state(0:mt_n-1)
    integer :: index = mt_n
  end type mt19937_state
contains
  subroutine seed_mt(generator, seed)
    type(mt19937_state), intent(out) :: generator
    integer(int64), intent(in) :: seed
    integer :: i
    generator%state(0) = iand(seed, mask32)
    do i = 1, mt_n - 1
      generator%state(i) = iand(1812433253_int64 * ieor(generator%state(i-1), ishft(generator%state(i-1), -30)) + int(i, int64), mask32)
    end do
    generator%index = mt_n
  end subroutine seed_mt

  subroutine twist_mt(generator)
    type(mt19937_state), intent(inout) :: generator
    integer :: i, next
    integer(int64) :: y
    do i = 0, mt_n - 1
      next = mod(i + 1, mt_n)
      y = ior(iand(generator%state(i), upper_mask), iand(generator%state(next), lower_mask))
      generator%state(i) = ieor(generator%state(mod(i + mt_m, mt_n)), ishft(y, -1))
      if (iand(y, 1_int64) /= 0_int64) generator%state(i) = ieor(generator%state(i), matrix_a)
      generator%state(i) = iand(generator%state(i), mask32)
    end do
    generator%index = 0
  end subroutine twist_mt

  function next_u32(generator) result(value)
    type(mt19937_state), intent(inout) :: generator
    integer(int64) :: value
    if (generator%index >= mt_n) call twist_mt(generator)
    value = generator%state(generator%index)
    generator%index = generator%index + 1
    value = ieor(value, ishft(value, -11))
    value = ieor(value, iand(ishft(value, 7), int(z'000000009D2C5680', int64)))
    value = ieor(value, iand(ishft(value, 15), int(z'00000000EFC60000', int64)))
    value = ieor(value, ishft(value, -18))
    value = iand(value, mask32)
  end function next_u32

  function uniform_minus_point_zero_one_to_point_zero_one(generator) result(value)
    type(mt19937_state), intent(inout) :: generator
    real(real32) :: value
    ! libstdc++ generate_canonical<float,24> uses one mt19937 draw and divides by 2^32.
    value = -0.01_real32 + 0.02_real32 * real(next_u32(generator), real32) / 4294967296.0_real32
  end function uniform_minus_point_zero_one_to_point_zero_one
end module attention_rng

module attention_kernels
  use, intrinsic :: iso_fortran_env, only : real32
  implicit none
contains
  subroutine attention_host(key, value, query, n, d, output)
    integer, intent(in) :: n, d
    real(real32), intent(in) :: key(0:), value(0:), query(0:d-1)
    real(real32), intent(out) :: output(0:d-1)
    real(real32), allocatable :: dot_product(:), score(:)
    real(real32) :: sum
    integer :: i, j
    allocate(dot_product(0:n-1), score(0:n-1))
    do i = 0, n - 1
      sum = 0.0_real32
      do j = 0, d - 1
        sum = sum + key(i * d + j) * query(j)
      end do
      dot_product(i) = sum
    end do
    sum = 0.0_real32
    do i = 0, n - 1
      sum = sum + exp(dot_product(i))
    end do
    do i = 0, n - 1
      score(i) = exp(dot_product(i)) / sum
    end do
    do j = 0, d - 1
      sum = 0.0_real32
      do i = 0, n - 1
        sum = sum + score(i) * value(i * d + j)
      end do
      output(j) = sum
    end do
    deallocate(dot_product, score)
  end subroutine attention_host

  subroutine attention_device(key, value, query, n, d, repeat, output, average_ms)
    integer, intent(in) :: n, d, repeat
    real(real32), intent(in) :: key(0:), value(0:), query(0:d-1)
    real(real32), intent(out) :: output(0:d-1), average_ms
    real(real32), allocatable :: dot_product(:), score(:), exp_sum(:)
    real(real32) :: sum
    integer :: i, j, k
    integer(kind=8) :: start_count, end_count, clock_rate
    allocate(dot_product(0:n-1), score(0:n-1), exp_sum(0:0))
!$omp target data map(to: key(0:n*d-1), value(0:n*d-1), query(0:d-1)) map(alloc: dot_product(0:n-1), score(0:n-1), exp_sum(0:0)) map(from: output(0:d-1))
    call system_clock(start_count, clock_rate)
    do k = 0, repeat - 1
      exp_sum(0) = 0.0_real32
!$omp target update to(exp_sum(0:0))
!$omp target teams distribute parallel do thread_limit(256) private(sum)
      do i = 0, n - 1
        sum = 0.0_real32
        do j = 0, d - 1
          sum = sum + key(i * d + j) * query(j)
        end do
        dot_product(i) = sum
!$omp atomic update
        exp_sum(0) = exp_sum(0) + exp(sum)
      end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256)
      do i = 0, n - 1
        score(i) = exp(dot_product(i)) / exp_sum(0)
      end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do thread_limit(256) private(sum)
      do j = 0, d - 1
        sum = 0.0_real32
        do i = 0, n - 1
          sum = sum + score(i) * value(i * d + j)
        end do
        output(j) = sum
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(end_count)
!$omp end target data
    average_ms = real(end_count - start_count, real32) * 1000.0_real32 / real(clock_rate, real32) / real(repeat, real32)
    deallocate(dot_product, score, exp_sum)
  end subroutine attention_device
end module attention_kernels

program attention
  use, intrinsic :: iso_fortran_env, only : real32, int64
  use attention_rng
  use attention_kernels
  implicit none
  integer :: n, d, repeat, ios, i
  integer(int64) :: kv_size
  character(len=64) :: argument
  logical :: ok
  real(real32) :: average_ms
  real(real32), allocatable :: key(:), value(:), query(:), host_output(:), device_output(:)
  type(mt19937_state) :: generator

  if (command_argument_count() /= 3) then
    write(*, '(A)') 'Usage: ./main <rows> <columns> <repeat>'
    error stop 1
  end if
  call get_command_argument(1, argument); read(argument, *, iostat=ios) n; if (ios /= 0) error stop 1
  call get_command_argument(2, argument); read(argument, *, iostat=ios) d; if (ios /= 0) error stop 1
  call get_command_argument(3, argument); read(argument, *, iostat=ios) repeat; if (ios /= 0 .or. repeat <= 0) error stop 1
  kv_size = int(n, int64) * int(d, int64)
  allocate(key(0:kv_size-1), value(0:kv_size-1), query(0:d-1), host_output(0:d-1), device_output(0:d-1))
  call seed_mt(generator, 19937_int64)
  do i = 0, int(kv_size) - 1
    key(i) = uniform_minus_point_zero_one_to_point_zero_one(generator)
    value(i) = uniform_minus_point_zero_one_to_point_zero_one(generator)
    query(mod(i, d)) = uniform_minus_point_zero_one_to_point_zero_one(generator)
  end do
  call attention_host(key, value, query, n, d, host_output)
  call attention_device(key, value, query, n, d, repeat, device_output, average_ms)
  write(*, '(A,F0.6,A)') 'Average execution time of kernels ', average_ms, ' (ms)'
  ok = .true.
  do i = 0, d - 1
    if (abs(host_output(i) - device_output(i)) > 1.0e-3_real32) then
      ok = .false.
      exit
    end if
  end do
  if (ok) then; write(*, '(A)') 'PASS'; else; write(*, '(A)') 'FAIL'; end if
  deallocate(key, value, query, host_output, device_output)
end program attention
