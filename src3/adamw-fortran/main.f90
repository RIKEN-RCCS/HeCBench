module adamw_rng
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  implicit none
  integer(int64), parameter :: u32_mask = int(z'FFFFFFFF', int64)
  integer(int64), parameter :: mt_matrix_a = int(z'9908B0DF', int64)
  integer(int64), parameter :: mt_upper_mask = int(z'80000000', int64)
  integer(int64), parameter :: mt_lower_mask = int(z'7FFFFFFF', int64)
contains
  subroutine mt_seed(state, index, seed)
    integer(int64), intent(out) :: state(0:623)
    integer, intent(out) :: index
    integer(int64), intent(in) :: seed
    integer :: i
    state(0) = iand(seed, u32_mask)
    do i = 1, 623
      state(i) = iand(1812433253_int64 * ieor(state(i - 1), ishft(state(i - 1), -30)) + int(i, int64), u32_mask)
    end do
    index = 624
  end subroutine mt_seed

  subroutine mt_twist(state)
    integer(int64), intent(inout) :: state(0:623)
    integer :: i, j
    integer(int64) :: y
    do i = 0, 623
      j = modulo(i + 1, 624)
      y = ior(iand(state(i), mt_upper_mask), iand(state(j), mt_lower_mask))
      j = modulo(i + 397, 624)
      state(i) = ieor(state(j), ishft(y, -1))
      if (iand(y, 1_int64) /= 0_int64) state(i) = ieor(state(i), mt_matrix_a)
      state(i) = iand(state(i), u32_mask)
    end do
  end subroutine mt_twist

  function mt_next(state, index) result(value)
    integer(int64), intent(inout) :: state(0:623)
    integer, intent(inout) :: index
    integer(int64) :: value
    if (index >= 624) then
      call mt_twist(state)
      index = 0
    end if
    value = state(index)
    index = index + 1
    value = ieor(value, ishft(value, -11))
    value = ieor(value, iand(ishft(value, 7), int(z'9D2C5680', int64)))
    value = ieor(value, iand(ishft(value, 15), int(z'EFC60000', int64)))
    value = ieor(value, ishft(value, -18))
    value = iand(value, u32_mask)
  end function mt_next

  function uniform01(state, index) result(value)
    integer(int64), intent(inout) :: state(0:623)
    integer, intent(inout) :: index
    real(real32) :: value
    ! libstdc++'s uniform_real_distribution<float>(0,1) consumes one mt19937
    ! word and scales it by 2^-32; retain the same seed and draw order.
    value = real(real(mt_next(state, index), real64) * 2.3283064365386962890625e-10_real64, real32)
  end function uniform01
end module adamw_rng

program adamw
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real32
  use adamw_rng, only : mt_seed, uniform01
  use adamw_kernels, only : fused_4bit_kernel
  use adamw_reference, only : reference_kernel
  implicit none
  character(len=64) :: argument
  integer :: argc, time_step, step, blocks_per_grid, num_teams, num_threads
  integer(int64) :: vector_size, size, i, mt_state(0:623), clock_start, clock_end, clock_rate
  integer :: mt_index
  real(real32), allocatable :: g(:), p(:), p_ref(:), m_qscale(:), v_qscale(:), r(:)
  integer(int8), allocatable :: m(:), v(:)
  real(real32) :: lr, weight_decay, beta1, beta2, eps, resid_beta1, resid_beta2, weight_decay_update
  real(real32) :: correction1, correction2_sqrt, step_size, absmax_error, elapsed_ms, random_value

  argc = command_argument_count()
  if (argc /= 2) then
    write(*,'(A)') 'Usage: ./main <vector size> <number of time steps>'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument, *) vector_size
  call get_command_argument(2, argument)
  read(argument, *) time_step

  size = vector_size * 2_int64
  allocate(g(0:size - 1), p(0:size - 1), p_ref(0:size - 1), m_qscale(0:size - 1), v_qscale(0:size - 1), r(0:size - 1))
  allocate(m(0:vector_size - 1), v(0:vector_size - 1))
  call mt_seed(mt_state, mt_index, 19937_int64)
  do i = 0, size - 1
    m_qscale(i) = uniform01(mt_state, mt_index)
    v_qscale(i) = uniform01(mt_state, mt_index)
    g(i) = uniform01(mt_state, mt_index)
    random_value = uniform01(mt_state, mt_index)
    r(i) = random_value
    p(i) = random_value
    p_ref(i) = random_value
  end do
  do i = 0, vector_size - 1
    m(i) = transfer(int(256.0_real32 * uniform01(mt_state, mt_index), int32), m(i))
    v(i) = transfer(int(256.0_real32 * uniform01(mt_state, mt_index), int32), v(i))
  end do

  num_threads = 64
  blocks_per_grid = int((vector_size + 63_int64) / 64_int64)
  num_teams = blocks_per_grid
  lr = 1.0e-3_real32
  weight_decay = 1.0e-2_real32
  beta1 = 0.9_real32
  beta2 = 0.999_real32
  eps = 1.0e-8_real32
  resid_beta1 = 1.0_real32 - beta1
  resid_beta2 = 1.0_real32 - beta2
  weight_decay_update = 1.0_real32 - lr * weight_decay

!$omp target data map(to: m_qscale(0:size - 1), v_qscale(0:size - 1), g(0:size - 1), p(0:size - 1), &
!$omp& m(0:vector_size - 1), v(0:vector_size - 1))
  do step = 1, time_step
    correction1 = 1.0_real32 - beta1**step
    correction2_sqrt = sqrt(1.0_real32 - beta2**step)
    step_size = lr / correction1
    call fused_4bit_kernel(num_teams, num_threads, p, g, m_qscale, v_qscale, m, v, beta1, beta2, lr, &
        weight_decay, eps, step, vector_size, correction1, correction2_sqrt, step_size, weight_decay_update, resid_beta1, resid_beta2)
    call reference_kernel(blocks_per_grid, p_ref, g, m_qscale, v_qscale, m, v, beta1, beta2, lr, weight_decay, eps, &
        step, vector_size, correction1, correction2_sqrt, step_size, weight_decay_update, resid_beta1, resid_beta2)
  end do

!$omp target update from(p(0:size - 1))
  absmax_error = 0.0_real32
  do i = 0, size - 1
    absmax_error = max(absmax_error, abs(p(i) - p_ref(i)))
  end do
  write(*,'(A,F0.6)') 'Absolute maximum error: ', absmax_error
  if (absmax_error > 1.0e-3_real32) then
    write(*,'(A)') 'FAIL'
  else
    write(*,'(A)') 'PASS'
  end if

  call system_clock(count=clock_start, count_rate=clock_rate)
  do step = 1, time_step
    correction1 = 1.0_real32 - beta1**step
    correction2_sqrt = sqrt(1.0_real32 - beta2**step)
    step_size = lr / correction1
    call fused_4bit_kernel(num_teams, num_threads, p, g, m_qscale, v_qscale, m, v, beta1, beta2, lr, &
        weight_decay, eps, step, vector_size, correction1, correction2_sqrt, step_size, weight_decay_update, resid_beta1, resid_beta2)
  end do
  call system_clock(count=clock_end)
  elapsed_ms = real(clock_end - clock_start, real32) * 1000.0_real32 / real(clock_rate, real32) / real(time_step, real32)
  write(*,'(A,F0.6,A)') 'Average kernel execution time ', elapsed_ms, ' (ms)'
!$omp end target data

  deallocate(g, p, p_ref, m_qscale, v_qscale, m, v, r)
end program adamw
