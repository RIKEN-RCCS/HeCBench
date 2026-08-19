program main
  use iso_fortran_env, only: real64, int64
  use su3_types
  use lattice_mod
  use mat_nn_openmp_mod
  implicit none
  integer(int64) :: iterations, warmups, ldim, total_sites
  integer :: threads_per_group, verbose, device
  type(site), allocatable :: a(:), c(:)
  type(su3_matrix), allocatable :: b(:)
  type(complex_t) :: one, third, cc
  real(real64) :: ttotal, tflop, memory_usage
  integer(int64) :: i
  integer :: j, k, l, m
  logical :: ok

  iterations = int(flag_int('-i', 1000), int64)
  ldim = int(flag_int('-l', 32), int64)
  threads_per_group = flag_int('-t', 128)
  verbose = flag_int('-v', 1)
  device = flag_int('-d', -1)
  warmups = int(flag_int('-w', 1), int64)
  if (device /= -999999 .and. verbose >= 3) print '(a,i0)', 'Requested device = ', device
  total_sites = ldim * ldim * ldim * ldim
  allocate(a(0:total_sites-1), c(0:total_sites-1), b(0:3))
  one%real = 1.0_real64; one%imag = 0.0_real64
  third%real = 1.0_real64 / 3.0_real64; third%imag = 0.0_real64
  call make_lattice(a, int(ldim), one)
  call init_link(b, third)
  if (verbose >= 1) then
    print '(a,i0,a)', 'Number of sites = ', ldim, '^4'
    print '(a,i0,a,i0,a)', 'Executing ', iterations, ' iterations with ', warmups, ' warmups'
    if (threads_per_group /= 0) print '(a,i0)', 'Threads per group = ', threads_per_group
  end if
  ttotal = su3_mat_nn(a, b, c, total_sites, iterations, threads_per_group, warmups, verbose)
  if (verbose >= 1) print '(a,f12.6,a)', 'Total kernel execution time = ', ttotal, ' (s)'
  tflop = real(iterations, real64) * real(total_sites, real64) * 864.0_real64
  print '(a,f8.3)', 'Total GFLOP/s = ', tflop / ttotal / 1.0e9_real64
  memory_usage = real(storage_size(a(0)) / 8, real64) * real(size(a) + size(c), real64) + &
                 real(storage_size(b(0)) / 8, real64) * real(size(b), real64)
  print '(a,f8.3)', 'Total GByte/s (GPU memory)  = ', real(iterations, real64) * memory_usage / ttotal / 1.0e9_real64
  ok = .true.
  do i = 0, total_sites - 1
    do j = 0, 3
      do k = 0, 2
        do l = 0, 2
          cc%real = 0.0_real64; cc%imag = 0.0_real64
          do m = 0, 2
            call cmulsum(a(i)%link(j)%e(k,m), b(j)%e(m,l), cc)
          end do
          if (abs(c(i)%link(j)%e(k,l)%real - cc%real) >= 1.0e-6_real64 .or. &
              abs(c(i)%link(j)%e(k,l)%imag - cc%imag) >= 1.0e-6_real64) ok = .false.
        end do
      end do
    end do
  end do
  if (.not. ok) stop 1
contains
  integer function flag_int(name, default)
    character(len=*), intent(in) :: name
    integer, intent(in) :: default
    integer :: n, p, stat
    character(len=128) :: arg, nxt
    flag_int = default
    n = command_argument_count()
    do p = 1, n
      call get_command_argument(p, arg)
      if (trim(arg) == trim(name) .and. p < n) then
        call get_command_argument(p + 1, nxt)
        read(nxt, *, iostat=stat) flag_int
        if (stat /= 0) flag_int = default
      end if
      if (trim(arg) == '-h') then
        print '(a)', 'Usage: ./main [-i iterations] [-l lattice dimension] [-t threads per workgroup] [-d device] [-v verbosity level [0,1,2,3]] [-w warmups]'
        stop 1
      end if
    end do
  end function flag_int
end program main
