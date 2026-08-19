module adamw_kernels
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real32
  implicit none

  integer(int32), parameter :: bitmask = 15_int32
  integer(int32), parameter :: right_pack_bitmask = 240_int32
  real(real32), parameter :: exp_qmap(0:15) = [ &
    -0.8875_real32, -0.6625_real32, -0.4375_real32, -0.2125_real32, &
    -0.0775_real32, -0.0325_real32, -0.0055_real32,  0.0000_real32, &
     0.0055_real32,  0.0325_real32,  0.0775_real32,  0.2125_real32, &
     0.4375_real32,  0.6625_real32,  0.8875_real32,  1.0000_real32 ]
  real(real32), parameter :: exp_qmidpt(0:14) = [ &
    -0.775_real32, -0.55_real32, -0.325_real32, -0.145_real32, &
    -0.055_real32, -0.019_real32, -0.00275_real32, 0.00275_real32, &
     0.019_real32, 0.055_real32, 0.145_real32, 0.325_real32, &
     0.55_real32, 0.775_real32, 0.94375_real32 ]
  real(real32), parameter :: sq_qmap(0:15) = [ &
    0.0625_real32, 0.1250_real32, 0.1875_real32, 0.2500_real32, &
    0.3125_real32, 0.3750_real32, 0.4375_real32, 0.5000_real32, &
    0.5625_real32, 0.6250_real32, 0.6875_real32, 0.7500_real32, &
    0.8125_real32, 0.8750_real32, 0.9375_real32, 1.0000_real32 ]
  real(real32), parameter :: sq_qmidpt(0:14) = [ &
    0.09375_real32, 0.15625_real32, 0.21875_real32, 0.28125_real32, &
    0.34375_real32, 0.40625_real32, 0.46875_real32, 0.53125_real32, &
    0.59375_real32, 0.65625_real32, 0.71875_real32, 0.78125_real32, &
    0.84375_real32, 0.90625_real32, 0.96875_real32 ]

!$omp declare target (q_mapping)
contains

  integer(int32) function q_mapping(qmap, qmidpt, x) result(index)
    real(real32), intent(in) :: qmap(0:15), qmidpt(0:14), x
    integer(int32) :: low, high, middle

    low = 0_int32
    high = 15_int32
    if (x <= qmap(low)) then
      index = low
      return
    end if
    if (qmap(high) <= x) then
      index = high
      return
    end if
    do while (low < high)
      middle = ishft(low + high, -1)
      if (qmap(middle) <= x) then
        low = middle + 1_int32
      else
        high = middle
      end if
    end do
    if (qmidpt(low - 1_int32) < x) then
      index = low
    else
      index = low - 1_int32
    end if
  end function q_mapping

  subroutine fused_4bit_kernel(num_teams, num_threads, p, g, exp_qscale, sq_qscale, exp, sq, &
      beta1, beta2, lr, weight_decay, eps, step, total_size, correction1, correction2_sqrt, &
      step_size, weight_decay_update, resid_beta1, resid_beta2)
    integer, intent(in) :: num_teams, num_threads, step
    integer(int64), intent(in) :: total_size
    real(real32), intent(inout) :: p(0:), exp_qscale(0:), sq_qscale(0:)
    real(real32), intent(in) :: g(0:)
    integer(int8), intent(inout) :: exp(0:), sq(0:)
    real(real32), intent(in) :: beta1, beta2, lr, weight_decay, eps
    real(real32), intent(in) :: correction1, correction2_sqrt, step_size
    real(real32), intent(in) :: weight_decay_update, resid_beta1, resid_beta2
    integer :: block_id, thread_id
    integer(int64) :: global_id
    integer(int32) :: exp_full, sq_full, exp_left_index, sq_left_index
    integer(int32) :: exp_right_index, sq_right_index, packed_exp, packed_sq
    integer(int32) :: q_exp_left, q_sq_left, q_exp_right, q_sq_right
    real(real32) :: absmax_exp, absmax_sq, exp_average_scale
    real(real32) :: exp_left, sq_left, exp_right, sq_right
    real(real32) :: local_absmax_exp, local_absmax_sq
    real(real32) :: p_left, p_right, g_left, g_right
    real(real32) :: local_exp_left(0:63), local_sq_left(0:63)
    real(real32) :: local_exp_right(0:63), local_sq_right(0:63)

    ! This preserves the C++ target-teams outer loop followed by two separate
    ! parallel-for phases.  The four local arrays are shared within one team.
!$omp target teams distribute num_teams(num_teams) thread_limit(num_threads) &
!$omp& private(absmax_exp, absmax_sq, exp_average_scale, exp_left, sq_left, exp_right, sq_right, &
!$omp& local_absmax_exp, local_absmax_sq, p_left, p_right, g_left, g_right, exp_full, sq_full, &
!$omp& exp_left_index, sq_left_index, exp_right_index, sq_right_index, global_id, local_exp_left, &
!$omp& local_sq_left, local_exp_right, local_sq_right, packed_exp, packed_sq, q_exp_left, q_sq_left, &
!$omp& q_exp_right, q_sq_right)
    do block_id = 0, num_teams - 1
      absmax_exp = 0.0_real32
      absmax_sq = 0.0_real32
!$omp parallel do reduction(max:absmax_exp,absmax_sq) num_threads(num_threads) &
!$omp& private(global_id, exp_full, sq_full, exp_left_index, sq_left_index, exp_right_index, sq_right_index, &
!$omp& p_left, p_right, g_left, g_right, exp_average_scale, exp_left, sq_left, exp_right, sq_right, &
!$omp& local_absmax_exp, local_absmax_sq)
      do thread_id = 0, 63
        global_id = int(block_id, int64) * 64_int64 + int(thread_id, int64)
        if (global_id < total_size) then
          exp_full = int(exp(global_id), int32)
          sq_full = int(sq(global_id), int32)
          p_left = p(2_int64 * global_id)
          p_right = p(2_int64 * global_id + 1_int64)
          g_left = g(2_int64 * global_id)
          g_right = g(2_int64 * global_id + 1_int64)

          exp_left_index = iand(exp_full, bitmask)
          sq_left_index = iand(sq_full, bitmask)
          p_left = p_left * weight_decay_update
          exp_average_scale = exp_qscale(block_id)
          exp_left = beta1 * (exp_qmap(exp_left_index) * exp_average_scale) + resid_beta1 * g_left
          sq_left = beta2 * (sq_qmap(sq_left_index) * sq_qscale(block_id)) + resid_beta2 * (g_left * g_left)
          p(2_int64 * global_id) = p_left - step_size * (exp_left / (sqrt(sq_left) / correction2_sqrt + eps))

          exp_right_index = iand(ishft(exp_full, -4), bitmask)
          sq_right_index = iand(ishft(sq_full, -4), bitmask)
          p_right = p_right * weight_decay_update
          exp_right = beta1 * (exp_qmap(exp_right_index) * exp_average_scale) + resid_beta1 * g_right
          sq_right = beta2 * (sq_qmap(sq_right_index) * sq_qscale(block_id)) + resid_beta2 * (g_right * g_right)
          p(2_int64 * global_id + 1_int64) = p_right - step_size * (exp_right / (sqrt(sq_right) / correction2_sqrt + eps))

          local_exp_left(thread_id) = exp_left
          local_sq_left(thread_id) = sq_left
          local_exp_right(thread_id) = exp_right
          local_sq_right(thread_id) = sq_right
          local_absmax_exp = max(exp_left, exp_right)
          local_absmax_sq = max(sq_left, sq_right)
          absmax_exp = max(absmax_exp, local_absmax_exp)
          absmax_sq = max(absmax_sq, local_absmax_sq)
        end if
      end do
!$omp end parallel do

      exp_qscale(block_id) = absmax_exp
      sq_qscale(block_id) = absmax_sq
!$omp parallel do num_threads(num_threads) &
!$omp& private(global_id, packed_exp, packed_sq, q_exp_left, q_sq_left, q_exp_right, q_sq_right)
      do thread_id = 0, 63
        global_id = int(block_id, int64) * 64_int64 + int(thread_id, int64)
        if (global_id < total_size) then
          packed_exp = 0_int32
          packed_sq = 0_int32
          q_exp_left = q_mapping(exp_qmap, exp_qmidpt, local_exp_left(thread_id) / absmax_exp)
          q_sq_left = q_mapping(sq_qmap, sq_qmidpt, local_sq_left(thread_id) / absmax_sq)
          packed_exp = ior(packed_exp, iand(q_exp_left, bitmask))
          packed_sq = ior(packed_sq, iand(q_sq_left, bitmask))
          q_exp_right = q_mapping(exp_qmap, exp_qmidpt, local_exp_right(thread_id) / absmax_exp)
          q_sq_right = q_mapping(sq_qmap, sq_qmidpt, local_sq_right(thread_id) / absmax_sq)
          packed_exp = ior(packed_exp, iand(ishft(q_exp_right, 4), right_pack_bitmask))
          packed_sq = ior(packed_sq, iand(ishft(q_sq_right, 4), right_pack_bitmask))
          exp(global_id) = transfer(packed_exp, exp(global_id))
          sq(global_id) = transfer(packed_sq, sq(global_id))
        end if
      end do
!$omp end parallel do
    end do
!$omp end target teams distribute
  end subroutine fused_4bit_kernel
end module adamw_kernels
