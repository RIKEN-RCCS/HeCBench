module c_rng
  use iso_c_binding, only: c_int
  implicit none
  interface
    subroutine srand(seed) bind(C, name="srand")
      import c_int
      integer(c_int), value :: seed
    end subroutine srand
    function rand() result(r) bind(C, name="rand")
      import c_int
      integer(c_int) :: r
    end function rand
  end interface
end module c_rng

module mask_kernels
  use iso_fortran_env, only: int32
  implicit none
  integer, parameter :: GPU_THREADS = 256
contains
  subroutine sequence_mask(n, m, b, input, seq_lengths, fill_val, output)
    integer, intent(in) :: n, m, b, fill_val, seq_lengths(0:), input(0:)
    integer, intent(out) :: output(0:)
    integer :: index, i, j, k, ind
!$omp target teams distribute parallel do num_teams(m*n) num_threads(GPU_THREADS) private(i,j,k,ind)
    do index = 0, b*n*m-1
      k = mod(index, m)
      j = mod((index-k)/m, n)
      i = (index - m*j - k) / (n*m)
      ind = n*m*i + m*j + k
      output(ind) = merge(fill_val, input(ind), k >= seq_lengths(j))
    end do
!$omp end target teams distribute parallel do
  end subroutine sequence_mask

  subroutine window_mask(n, m, b, input, window_centers, radius, fill_val, output)
    integer, intent(in) :: n, m, b, radius, fill_val, window_centers(0:), input(0:)
    integer, intent(out) :: output(0:)
    integer :: index, i, j, k, ind
!$omp target teams distribute parallel do num_teams(m*n) num_threads(GPU_THREADS) private(i,j,k,ind)
    do index = 0, b*n*m-1
      k = mod(index, m)
      j = mod((index-k)/m, n)
      i = (index - m*j - k) / (n*m)
      ind = n*m*i + m*j + k
      output(ind) = merge(fill_val, input(ind), k < window_centers(j)-radius .or. k > window_centers(j)+radius)
    end do
!$omp end target teams distribute parallel do
  end subroutine window_mask

  subroutine triangular_mask(kind, n, m, b, input, fill_val, output)
    integer, intent(in) :: kind, n, m, b, fill_val, input(0:)
    integer, intent(out) :: output(0:)
    integer :: index, i, j, k, ind
    logical :: masked
!$omp target teams distribute parallel do num_teams(m*n) num_threads(GPU_THREADS) private(i,j,k,ind,masked)
    do index = 0, b*n*m-1
      k = mod(index, m)
      j = mod((index-k)/m, n)
      i = (index - m*j - k) / (n*m)
      ind = n*m*i + m*j + k
      select case (kind)
      case (1); masked = k > j
      case (2); masked = k < j
      case (3); masked = k >= j
      case default; masked = k <= j
      end select
      output(ind) = merge(fill_val, input(ind), masked)
    end do
!$omp end target teams distribute parallel do
  end subroutine triangular_mask
end module mask_kernels

program mask
  use omp_lib
  use c_rng
  use mask_kernels
  implicit none
  integer :: m, n, b, batch_dim, repeat, radius, fill_val, data_size, seq_len, window_size
  integer :: i, iter, cnt_fill
  character(len=64) :: arg
  integer, allocatable :: h_in(:), h_out(:), out_ref(:), h_seq_len(:), h_window(:)
  real(8) :: start_time, elapsed

  if (command_argument_count() /= 4) then
    print '(a)', 'Usage: ./main <sequence length> <sequence length> <batch size> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) m
  call get_command_argument(2,arg); read(arg,*) n
  call get_command_argument(3,arg); read(arg,*) b
  call get_command_argument(4,arg); read(arg,*) repeat

  fill_val = -1
  radius = m / 4
  batch_dim = merge(1, b, b <= 0)
  print *
  print '(a,i0,a,i0,a,i0)', 'M = ', m, ', N = ', n, ', B = ', batch_dim
  data_size = n * m * batch_dim
  seq_len = n
  window_size = n
  allocate(h_in(0:data_size-1), h_out(0:data_size-1), out_ref(0:data_size-1), h_seq_len(0:seq_len-1), h_window(0:window_size-1))
  call srand(123)
  do i = 0, seq_len-1
    h_seq_len(i) = mod(rand(), m/2)
  end do
  do i = 0, window_size-1
    h_window(i) = mod(rand(), m)
  end do
  do i = 0, data_size-1
    h_in(i) = mod(rand(), m*n)
  end do

!$omp target data map(to:h_in(0:data_size-1),h_seq_len(0:seq_len-1),h_window(0:window_size-1)) map(alloc:h_out(0:data_size-1))
  call sequence_ref(n, m, batch_dim, h_in, h_seq_len, fill_val, out_ref)
  start_time = omp_get_wtime()
  do iter = 1, repeat
    call sequence_mask(n, m, batch_dim, h_in, h_seq_len, fill_val, h_out)
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average execution time of sequenceMask kernel: ', elapsed*1.0e6/repeat, ' (us)'
  call print_mask_ratio(h_out, out_ref, fill_val, data_size)

  call window_ref(n, m, batch_dim, h_in, h_window, radius, fill_val, out_ref)
  start_time = omp_get_wtime()
  do iter = 1, repeat
    call window_mask(n, m, batch_dim, h_in, h_window, radius, fill_val, h_out)
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average execution time of windowMask kernel: ', elapsed*1.0e6/repeat, ' (us)'
  call print_mask_ratio(h_out, out_ref, fill_val, data_size)

  do i = 1, 4
    call triangular_ref(i, n, m, batch_dim, h_in, fill_val, out_ref)
    start_time = omp_get_wtime()
    do iter = 1, repeat
      call triangular_mask(i, n, m, batch_dim, h_in, fill_val, h_out)
    end do
    elapsed = omp_get_wtime() - start_time
    select case (i)
    case (1); print '(a,f0.6,a)', 'Average execution time of upperMask kernel: ', elapsed*1.0e6/repeat, ' (us)'
    case (2); print '(a,f0.6,a)', 'Average execution time of lowerMask kernel: ', elapsed*1.0e6/repeat, ' (us)'
    case (3); print '(a,f0.6,a)', 'Average execution time of upperDiagMask kernel: ', elapsed*1.0e6/repeat, ' (us)'
    case (4); print '(a,f0.6,a)', 'Average execution time of lowerDiagMask kernel: ', elapsed*1.0e6/repeat, ' (us)'
    end select
    call print_mask_ratio(h_out, out_ref, fill_val, data_size)
  end do
!$omp end target data

contains
  subroutine print_mask_ratio(out, ref, fill, ndata)
    integer, intent(inout) :: out(0:)
    integer, intent(in) :: ref(0:), fill, ndata
    integer :: ii, cnt
!$omp target update from(out(0:ndata-1))
    cnt = 0
    do ii = 0, ndata-1
      if (out(ii) == fill) cnt = cnt + 1
    end do
    print '(a,a,f0.6)', merge('PASS', 'FAIL', all(out(0:ndata-1) == ref(0:ndata-1))), ', Mask ratio: ', real(cnt)/real(ndata)
  end subroutine print_mask_ratio

  subroutine sequence_ref(n, m, b, input, seq_lengths, fill_val, output)
    integer, intent(in) :: n, m, b, input(0:), seq_lengths(0:), fill_val
    integer, intent(out) :: output(0:)
    integer :: index, i, j, k, ind
!$omp parallel do private(i,j,k,ind)
    do index = 0, b*n*m-1
      k = mod(index, m); j = mod((index-k)/m, n); i = (index-m*j-k)/(n*m); ind = n*m*i + m*j + k
      output(ind) = merge(fill_val, input(ind), k >= seq_lengths(j))
    end do
  end subroutine sequence_ref

  subroutine window_ref(n, m, b, input, window_centers, radius, fill_val, output)
    integer, intent(in) :: n, m, b, input(0:), window_centers(0:), radius, fill_val
    integer, intent(out) :: output(0:)
    integer :: index, i, j, k, ind
!$omp parallel do private(i,j,k,ind)
    do index = 0, b*n*m-1
      k = mod(index, m); j = mod((index-k)/m, n); i = (index-m*j-k)/(n*m); ind = n*m*i + m*j + k
      output(ind) = merge(fill_val, input(ind), k < window_centers(j)-radius .or. k > window_centers(j)+radius)
    end do
  end subroutine window_ref

  subroutine triangular_ref(kind, n, m, b, input, fill_val, output)
    integer, intent(in) :: kind, n, m, b, input(0:), fill_val
    integer, intent(out) :: output(0:)
    integer :: index, i, j, k, ind
    logical :: masked
!$omp parallel do private(i,j,k,ind,masked)
    do index = 0, b*n*m-1
      k = mod(index, m); j = mod((index-k)/m, n); i = (index-m*j-k)/(n*m); ind = n*m*i + m*j + k
      select case (kind)
      case (1); masked = k > j
      case (2); masked = k < j
      case (3); masked = k >= j
      case default; masked = k <= j
      end select
      output(ind) = merge(fill_val, input(ind), masked)
    end do
  end subroutine triangular_ref
end program mask
