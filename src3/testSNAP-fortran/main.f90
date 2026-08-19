program main
  use iso_fortran_env, only: real64
  use snap_mod
  implicit none
  integer :: nsteps, i, nlocal, nghost, ninside, ncoeff, twojmax, idxu_max, idxdu_max, idxb_count, switch_flag
  real(real64) :: elapsed_ui, elapsed_yi, elapsed_duidrj, elapsed_deidrj, checksum
  type(bindices_t), allocatable :: idxb(:)
  integer, allocatable :: idxu_block(:), idxdu_block(:)
  nsteps = option_int('--nsteps', 100)
  switch_flag = 1
  twojmax = 14
  ninside = 64
  nlocal = 512
  nghost = 128
  ncoeff = compute_ncoeff(twojmax)
  call build_indices(twojmax, idxb, idxb_count, idxu_block, idxu_max, idxdu_block, idxdu_max)
  print '(a,i0)', 'ninside = ', ninside
  print '(a,i0)', 'ncoeff = ', ncoeff
  print '(a,i0)', 'nlocal = ', nlocal
  print '(a,i0)', 'nghost = ', nghost
  print '(a,i0)', 'twojmax = ', twojmax
  call compute_snap_phases(nsteps, nlocal, ninside, twojmax, idxu_max, idxdu_max, idxb_count, switch_flag, &
    elapsed_ui, elapsed_yi, elapsed_duidrj, elapsed_deidrj, checksum)
  print '(a,f12.6)', 'compute_ui: ', elapsed_ui
  print '(a,f12.6)', 'compute_yi: ', elapsed_yi
  print '(a,f12.6)', 'compute_duidrj: ', elapsed_duidrj
  print '(a,f12.6)', 'compute_deidrj: ', elapsed_deidrj
  print '(a,es24.16)', 'SNAP checksum: ', checksum
contains
  integer function option_int(name, default)
    character(len=*), intent(in) :: name
    integer, intent(in) :: default
    integer :: n, i, stat
    character(len=128) :: arg, nxt
    option_int = default
    n = command_argument_count()
    do i = 1, n - 1
      call get_command_argument(i, arg)
      if (trim(arg) == trim(name)) then
        call get_command_argument(i + 1, nxt)
        read(nxt, *, iostat=stat) option_int
        if (stat /= 0) option_int = default
      end if
    end do
  end function option_int
end program main
