module rainflow_mod
  use iso_fortran_env, only: real64
  implicit none

  type :: double3
    real(real64) :: x
    real(real64) :: y
    real(real64) :: z
  end type double3

!$omp declare target (extrema, execute)
contains

  subroutine extrema(history, history_length, result, result_length)
    real(real64), intent(in) :: history(0:)
    integer, intent(in) :: history_length
    real(real64), intent(out) :: result(0:)
    integer, intent(out) :: result_length
    integer :: i, eidx

    result(0) = history(0)
    eidx = 0
    do i = 1, history_length - 2
      if ((history(i) > result(eidx) .and. history(i) > history(i + 1)) .or. &
          (history(i) < result(eidx) .and. history(i) < history(i + 1))) then
        eidx = eidx + 1
        result(eidx) = history(i)
      end if
    end do
    eidx = eidx + 1
    result(eidx) = history(history_length - 1)
    result_length = eidx + 1
  end subroutine extrema

  subroutine execute(history, history_length, extrema_buf, points, results, results_length)
    real(real64), intent(in) :: history(0:)
    integer, intent(in) :: history_length
    real(real64), intent(inout) :: extrema_buf(0:)
    integer, intent(inout) :: points(0:)
    type(double3), intent(inout) :: results(0:)
    integer, intent(out) :: results_length
    integer :: extrema_length
    integer :: pidx, eidx, ridx, i
    real(real64) :: x_range, y_range, y_mean, range, mean

    call extrema(history, history_length, extrema_buf, extrema_length)

    pidx = -1
    eidx = -1
    ridx = -1
    do i = 0, extrema_length - 1
      pidx = pidx + 1
      eidx = eidx + 1
      points(pidx) = eidx
      do while (pidx >= 2)
        x_range = abs(extrema_buf(points(pidx - 1)) - extrema_buf(points(pidx)))
        y_range = abs(extrema_buf(points(pidx - 2)) - extrema_buf(points(pidx - 1)))
        if (x_range < y_range) exit

        y_mean = 0.5_real64 * (extrema_buf(points(pidx - 2)) + extrema_buf(points(pidx - 1)))
        ridx = ridx + 1
        if (pidx == 2) then
          results(ridx)%x = 0.5_real64
          results(ridx)%y = y_range
          results(ridx)%z = y_mean
          points(0) = points(1)
          points(1) = points(2)
          pidx = 1
        else
          results(ridx)%x = 1.0_real64
          results(ridx)%y = y_range
          results(ridx)%z = y_mean
          points(pidx - 2) = points(pidx)
          pidx = pidx - 2
        end if
      end do
    end do

    do i = 0, pidx - 1
      range = abs(extrema_buf(points(i)) - extrema_buf(points(i + 1)))
      mean = 0.5_real64 * (extrema_buf(points(i)) + extrema_buf(points(i + 1)))
      ridx = ridx + 1
      results(ridx)%x = 0.5_real64
      results(ridx)%y = range
      results(ridx)%z = mean
    end do

    results_length = ridx + 1
  end subroutine execute

  subroutine reference(history, history_lengths, extrema_buf, points, results, ref_result_lengths, num_history)
    real(real64), intent(in) :: history(0:)
    integer, intent(in) :: history_lengths(0:)
    real(real64), intent(inout) :: extrema_buf(0:)
    integer, intent(inout) :: points(0:)
    type(double3), intent(inout) :: results(0:)
    integer, intent(out) :: ref_result_lengths(0:)
    integer, intent(in) :: num_history
    integer :: i, offset, history_length

    do i = 0, num_history - 1
      offset = history_lengths(i)
      history_length = history_lengths(i + 1) - offset
      call execute(history(offset:), history_length, extrema_buf(offset:), points(offset:), &
                   results(offset:), ref_result_lengths(i))
    end do
  end subroutine reference

end module rainflow_mod
