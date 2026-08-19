module scel_mod
  use iso_fortran_env, only: real32
  implicit none
  integer, parameter :: gpu_num_threads = 256

contains

  real(real32) function bool01(cond) result(v)
    logical, intent(in) :: cond
    v = merge(1.0_real32, 0.0_real32, cond)
  end function bool01

  real(real32) function sigmoid_xent_forward(lgt, tgt) result(v)
    real(real32), intent(in) :: lgt, tgt
    v = lgt * (tgt - bool01(lgt >= 0.0_real32)) - log(1.0_real32 + exp(lgt - 2.0_real32 * lgt * bool01(lgt >= 0.0_real32)))
  end function sigmoid_xent_forward

  real(real32) function sigmoid_partition(lgt) result(v)
    real(real32), intent(in) :: lgt
    v = lgt * bool01(lgt >= 0.0_real32) + log(1.0_real32 + exp(lgt - 2.0_real32 * lgt * bool01(lgt >= 0.0_real32)))
  end function sigmoid_partition

  real(real32) function sigmoid_xent_forward_with_log_d_trick(lgt, tgt) result(v)
    real(real32), intent(in) :: lgt, tgt
    v = (2.0_real32 * tgt - 1.0_real32) * (lgt - sigmoid_partition(lgt))
  end function sigmoid_xent_forward_with_log_d_trick

  real(real32) function unjoined_sigmoid_xent_forward(lgt, tgt) result(v)
    real(real32), intent(in) :: lgt, tgt
    v = lgt * tgt + (tgt - 1.0_real32) * lgt * bool01(lgt >= 0.0_real32) - &
        (1.0_real32 - tgt) * log(1.0_real32 + exp(lgt - 2.0_real32 * lgt * bool01(lgt >= 0.0_real32)))
  end function unjoined_sigmoid_xent_forward

  subroutine sigmoid_cross_entropy_with_logits_kernel(outer_size, inner_size, log_d_trick, unjoined_lr_loss, logits, targets, out)
    integer, intent(in) :: outer_size, inner_size
    logical, intent(in) :: log_d_trick, unjoined_lr_loss
    real(real32), intent(in) :: logits(0:), targets(0:)
    real(real32), intent(out) :: out(0:)
    integer :: i, in_idx
    real(real32) :: value, lgt, tgt

    !$omp target teams distribute num_teams(outer_size)
    do i = 0, outer_size - 1
      value = 0.0_real32
      !$omp parallel do reduction(+:value) num_threads(gpu_num_threads) private(lgt,tgt)
      do in_idx = i * inner_size, (i + 1) * inner_size - 1
        lgt = logits(in_idx)
        tgt = targets(in_idx)
        if (unjoined_lr_loss) then
          value = value + unjoined_sigmoid_xent_forward(lgt, tgt)
        else if (log_d_trick) then
          value = value + sigmoid_xent_forward_with_log_d_trick(lgt, tgt)
        else
          value = value + sigmoid_xent_forward(lgt, tgt)
        end if
      end do
      !$omp end parallel do
      out(i) = -value / real(inner_size, real32)
    end do
    !$omp end target teams distribute
  end subroutine sigmoid_cross_entropy_with_logits_kernel

  subroutine reference(outer_size, inner_size, log_d_trick, unjoined_lr_loss, logits, targets, out)
    integer, intent(in) :: outer_size, inner_size
    logical, intent(in) :: log_d_trick, unjoined_lr_loss
    real(real32), intent(in) :: logits(0:), targets(0:)
    real(real32), intent(out) :: out(0:)
    integer :: i, in_idx
    real(real32) :: value, lgt, tgt

    do i = 0, outer_size - 1
      value = 0.0_real32
      do in_idx = i * inner_size, (i + 1) * inner_size - 1
        lgt = logits(in_idx)
        tgt = targets(in_idx)
        if (unjoined_lr_loss) then
          value = value + unjoined_sigmoid_xent_forward(lgt, tgt)
        else if (log_d_trick) then
          value = value + sigmoid_xent_forward_with_log_d_trick(lgt, tgt)
        else
          value = value + sigmoid_xent_forward(lgt, tgt)
        end if
      end do
      out(i) = -value / real(inner_size, real32)
    end do
  end subroutine reference

end module scel_mod
