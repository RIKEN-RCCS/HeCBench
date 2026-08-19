module hotspot3d_kernels
  use iso_fortran_env, only: real32, real64
  implicit none
contains

  subroutine step_gpu(tin, tout, pin, rows, cols, layers, cc, cw, ce, cn, cs, ct, cb, step_div_cap)
    integer, intent(in) :: rows, cols, layers
    real(real32), intent(in) :: tin(:), pin(:)
    real(real32), intent(out) :: tout(:)
    real(real32), intent(in) :: cc, cw, ce, cn, cs, ct, cb, step_div_cap
    integer :: i, j, k, c, w, e, n, s, b, t, xy
    xy = rows * cols
!$omp target teams distribute parallel do collapse(3) thread_limit(256)
    do k = 0, layers - 1
      do j = 0, rows - 1
        do i = 0, cols - 1
          c = i + j * cols + k * xy + 1
          w = merge(c, c - 1, i == 0)
          e = merge(c, c + 1, i == cols - 1)
          n = merge(c, c - cols, j == 0)
          s = merge(c, c + cols, j == rows - 1)
          b = merge(c, c - xy, k == 0)
          t = merge(c, c + xy, k == layers - 1)
          tout(c) = tin(c) * cc + tin(w) * cw + tin(e) * ce + tin(n) * cn + tin(s) * cs &
                    + tin(t) * ct + tin(b) * cb + step_div_cap * pin(c) + ct * 80.0_real32
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine step_gpu

  subroutine step_cpu(tin, tout, pin, rows, cols, layers, cc, cw, ce, cn, cs, ct, cb, step_div_cap)
    integer, intent(in) :: rows, cols, layers
    real(real32), intent(in) :: tin(:), pin(:)
    real(real32), intent(out) :: tout(:)
    real(real32), intent(in) :: cc, cw, ce, cn, cs, ct, cb, step_div_cap
    integer :: i, j, k, c, w, e, n, s, b, t, xy
    xy = rows * cols
    do k = 0, layers - 1
      do j = 0, rows - 1
        do i = 0, cols - 1
          c = i + j * cols + k * xy + 1
          w = merge(c, c - 1, i == 0); e = merge(c, c + 1, i == cols - 1)
          n = merge(c, c - cols, j == 0); s = merge(c, c + cols, j == rows - 1)
          b = merge(c, c - xy, k == 0); t = merge(c, c + xy, k == layers - 1)
          tout(c) = tin(c) * cc + tin(w) * cw + tin(e) * ce + tin(n) * cn + tin(s) * cs &
                    + tin(t) * ct + tin(b) * cb + step_div_cap * pin(c) + ct * 80.0_real32
        end do
      end do
    end do
  end subroutine step_cpu

  subroutine read_input(values, rows, cols, layers, filename)
    integer, intent(in) :: rows, cols, layers
    real(real32), intent(out) :: values(:)
    character(*), intent(in) :: filename
    integer :: unit, ios, i, j, k, idx
    open(newunit=unit, file=trim(filename), status='old', action='read', iostat=ios)
    if (ios /= 0) error stop 'Unable to open hotspot3D input'
    do i = 0, rows - 1
      do j = 0, cols - 1
        do k = 0, layers - 1
          idx = i * cols + j + k * rows * cols + 1
          read(unit, *, iostat=ios) values(idx)
          if (ios /= 0) error stop 'Invalid hotspot3D input'
        end do
      end do
    end do
    close(unit)
  end subroutine read_input

  subroutine write_output(values, rows, cols, layers, filename)
    integer, intent(in) :: rows, cols, layers
    real(real32), intent(in) :: values(:)
    character(*), intent(in) :: filename
    integer :: unit, i, j, k, idx, number
    open(newunit=unit, file=trim(filename), status='replace', action='write')
    number = 0
    do i = 0, rows - 1
      do j = 0, cols - 1
        do k = 0, layers - 1
          idx = i * cols + j + k * rows * cols + 1
          write(unit, '(I0,A,ES14.6)') number, achar(9), values(idx)
          number = number + 1
        end do
      end do
    end do
    close(unit)
  end subroutine write_output
end module hotspot3d_kernels

program hotspot3d
  use iso_fortran_env, only: real32, real64
  use hotspot3d_kernels
  implicit none
  integer :: argc, rows, cols, layers, iterations, size, j
  character(512) :: arg, power_file, temp_file, output_file
  real(real32), allocatable :: tin(:), tout(:), pin(:), tcopy(:), answer(:)
  real(real32) :: dx, dy, dz, cap, rx, ry, rz, dt, ce, cw, cn, cs, ct, cb, cc, step_div_cap, err
  real(real64) :: start, finish

  argc = command_argument_count()
  if (argc /= 6) then
    write(*, '(A)') 'Usage: ./main <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>'
    error stop
  end if
  call get_command_argument(1, arg); read(arg, *) rows; cols = rows
  call get_command_argument(2, arg); read(arg, *) layers
  call get_command_argument(3, arg); read(arg, *) iterations
  call get_command_argument(4, power_file)
  call get_command_argument(5, temp_file)
  call get_command_argument(6, output_file)
  size = rows * cols * layers
  allocate(tin(size), tout(size), pin(size), tcopy(size), answer(size))
  dx = 0.016_real32 / rows; dy = 0.016_real32 / cols; dz = 0.0005_real32 / layers
  cap = 0.5_real32 * 1.75e6_real32 * 0.0005_real32 * dx * dy
  rx = dy / (2.0_real32 * 100.0_real32 * 0.0005_real32 * dx)
  ry = dx / (2.0_real32 * 100.0_real32 * 0.0005_real32 * dy)
  rz = dz / (100.0_real32 * dx * dy)
  dt = 0.001_real32 / (3.0e6_real32 / (0.5_real32 * 0.0005_real32 * 1.75e6_real32))
  step_div_cap = dt / cap; ce = step_div_cap / rx; cw = ce; cn = step_div_cap / ry; cs = cn; ct = step_div_cap / rz; cb = ct
  cc = 1.0_real32 - (2.0_real32 * ce + 2.0_real32 * cn + 3.0_real32 * ct)
  call read_input(tin, rows, cols, layers, temp_file)
  call read_input(pin, rows, cols, layers, power_file)
  tcopy = tin
!$omp target data map(to:tin, pin) map(alloc:tout)
  call cpu_time(start)
  do j = 1, iterations
    if (mod(j, 2) == 1) then
      call step_gpu(tin, tout, pin, rows, cols, layers, cc, cw, ce, cn, cs, ct, cb, step_div_cap)
    else
      call step_gpu(tout, tin, pin, rows, cols, layers, cc, cw, ce, cn, cs, ct, cb, step_div_cap)
    end if
  end do
  if (mod(iterations, 2) == 0) then
!$omp target update from(tin)
  else
!$omp target update from(tout)
  end if
  call cpu_time(finish)
!$omp end target data
  write(*, '(A,F12.6,A)') 'Average kernel execution time ', (finish-start)*1.0e6_real64/iterations, ' (us)'
  answer = 0.0_real32
  do j = 1, iterations
    if (mod(j, 2) == 1) then
      call step_cpu(tcopy, answer, pin, rows, cols, layers, cc, cw, ce, cn, cs, ct, cb, step_div_cap)
    else
      call step_cpu(answer, tcopy, pin, rows, cols, layers, cc, cw, ce, cn, cs, ct, cb, step_div_cap)
    end if
  end do
  if (mod(iterations, 2) == 0) then
    err = sqrt(sum((tin - tcopy)**2) / real(size, real32))
  else
    err = sqrt(sum((tout - answer)**2) / real(size, real32))
    tin = tout
  end if
  write(*, '(A,ES12.4)') 'Root-mean-square error: ', err
  call write_output(tin, rows, cols, layers, output_file)
  if (err > 1.0e-3_real32) error stop 'FAIL'
  write(*, '(A)') 'PASS'
end program hotspot3d
