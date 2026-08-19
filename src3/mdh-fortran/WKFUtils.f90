module wkfutils_mod
  use omp_lib
  implicit none
  type :: wkf_timerhandle
    real(8) :: start_time = 0.0d0
    real(8) :: stop_time = 0.0d0
  end type wkf_timerhandle
contains
  subroutine wkf_timer_start(timer)
    type(wkf_timerhandle), intent(inout) :: timer
    timer%start_time = omp_get_wtime()
  end subroutine wkf_timer_start

  subroutine wkf_timer_stop(timer)
    type(wkf_timerhandle), intent(inout) :: timer
    timer%stop_time = omp_get_wtime()
  end subroutine wkf_timer_stop

  real(8) function wkf_timer_time(timer)
    type(wkf_timerhandle), intent(in) :: timer
    wkf_timer_time = timer%stop_time - timer%start_time
  end function wkf_timer_time
end module wkfutils_mod
