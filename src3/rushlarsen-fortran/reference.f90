module rushlarsen_reference
  use iso_fortran_env, only: real64
  use rushlarsen_kernels, only: rush_step
  implicit none
contains
  subroutine forward_rush_larsen(states, t, dt, parameters, n)
    real(real64), intent(inout) :: states(0:)
    real(real64), intent(in) :: t, dt, parameters(0:)
    integer, intent(in) :: n
    call rush_step(states, t, dt, parameters, n)
  end subroutine forward_rush_larsen
end module rushlarsen_reference
