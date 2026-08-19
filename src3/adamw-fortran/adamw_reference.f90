module adamw_reference
  use, intrinsic :: iso_fortran_env, only : int8, int32, int64, real32
  use adamw_kernels, only : bitmask, right_pack_bitmask, exp_qmap, exp_qmidpt, sq_qmap, sq_qmidpt, q_mapping
  implicit none
contains
  subroutine reference_kernel(grid_size, p, g, exp_qscale, sq_qscale, exp, sq, beta1, beta2, lr, &
      weight_decay, eps, step, total_size, correction1, correction2_sqrt, step_size, weight_decay_update, resid_beta1, resid_beta2)
    integer, intent(in) :: grid_size, step
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
    real(real32) :: absmax_exp, absmax_sq, exp_average_scale
    real(real32) :: exp_left, sq_left, exp_right, sq_right
    real(real32) :: p_left, p_right, g_left, g_right
    real(real32) :: local_exp_left(0:63), local_sq_left(0:63)
    real(real32) :: local_exp_right(0:63), local_sq_right(0:63)

    do block_id = 0, grid_size - 1
      absmax_exp = 0.0_real32
      absmax_sq = 0.0_real32
      do thread_id = 0, 63
        global_id = int(block_id, int64) * 64_int64 + int(thread_id, int64)
        if (global_id >= total_size) exit
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
        absmax_exp = max(absmax_exp, max(exp_left, exp_right))
        absmax_sq = max(absmax_sq, max(sq_left, sq_right))
      end do
      exp_qscale(block_id) = absmax_exp
      sq_qscale(block_id) = absmax_sq
      do thread_id = 0, 63
        global_id = int(block_id, int64) * 64_int64 + int(thread_id, int64)
        if (global_id >= total_size) exit
        packed_exp = 0_int32
        packed_sq = 0_int32
        packed_exp = ior(packed_exp, iand(q_mapping(exp_qmap, exp_qmidpt, local_exp_left(thread_id) / absmax_exp), bitmask))
        packed_sq = ior(packed_sq, iand(q_mapping(sq_qmap, sq_qmidpt, local_sq_left(thread_id) / absmax_sq), bitmask))
        packed_exp = ior(packed_exp, iand(ishft(q_mapping(exp_qmap, exp_qmidpt, local_exp_right(thread_id) / absmax_exp), 4), right_pack_bitmask))
        packed_sq = ior(packed_sq, iand(ishft(q_mapping(sq_qmap, sq_qmidpt, local_sq_right(thread_id) / absmax_sq), 4), right_pack_bitmask))
        exp(global_id) = transfer(packed_exp, exp(global_id))
        sq(global_id) = transfer(packed_sq, sq(global_id))
      end do
    end do
  end subroutine reference_kernel
end module adamw_reference
