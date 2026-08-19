module pagerank_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  real(real32), parameter :: d_factor = 0.85_real32
  integer, parameter :: block_size_const = 256
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: int32
      integer(int32), value :: seed
    end subroutine
    function c_rand() bind(C, name="rand") result(v)
      import :: int32
      integer(int32) :: v
    end function
  end interface
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  subroutine usage()
    print '(a)', 'Usage: main [-n number of pages] [-i max iterations] [-t threshold] [-q divisor for zero density]'
  end subroutine

  subroutine parse_args(n, iter, thresh, divisor)
    integer, intent(inout) :: n, iter, divisor
    real(real32), intent(inout) :: thresh
    integer :: i, argc, ios
    character(len=64) :: arg, val
    argc = command_argument_count(); i = 1
    do while (i <= argc)
      call get_command_argument(i, arg)
      if (trim(arg) == '-n' .or. trim(arg) == '--number of pages') then
        i = i + 1; call get_command_argument(i, val); read(val,*,iostat=ios) n
      else if (trim(arg) == '-i' .or. trim(arg) == '--max number of iterations') then
        i = i + 1; call get_command_argument(i, val); read(val,*,iostat=ios) iter
      else if (trim(arg) == '-t' .or. trim(arg) == '--minimum threshold') then
        i = i + 1; call get_command_argument(i, val); read(val,*,iostat=ios) thresh
      else if (trim(arg) == '-q' .or. trim(arg) == '--divisor for zero density') then
        i = i + 1; call get_command_argument(i, val); read(val,*,iostat=ios) divisor
      else
        call usage(); stop 1
      end if
      i = i + 1
    end do
  end subroutine

  subroutine random_pages(n, pages, noutlinks, divisor)
    integer, intent(in) :: n, divisor
    integer(int32), intent(out) :: pages(0:n*n-1)
    integer(int32), intent(out) :: noutlinks(0:n-1)
    integer :: i, j, k
    if (divisor <= 0) stop "ERROR: Invalid divisor"
    pages = 0
    do i = 0, n-1
      noutlinks(i) = 0
      do j = 0, n-1
        if (i /= j .and. mod(abs(c_rand()), divisor) == 0) then
          pages(i*n+j) = 1
          noutlinks(i) = noutlinks(i) + 1
        end if
      end do
      if (noutlinks(i) == 0) then
        do
          k = mod(abs(c_rand()), n)
          if (k /= i) exit
        end do
        pages(i*n+k) = 1
        noutlinks(i) = 1
      end if
    end do
  end subroutine

  real(real32) function maximum_dif(difs, n) result(mx)
    integer, intent(in) :: n
    real(real32), intent(in) :: difs(0:n-1)
    integer :: i
    mx = 0.0_real32
    do i = 0, n-1
      mx = max(mx, difs(i))
    end do
  end function

  subroutine map_ref(pages, page_ranks, maps, noutlinks, n)
    integer, intent(in) :: n
    integer(int32), intent(in) :: pages(0:n*n-1), noutlinks(0:n-1)
    real(real32), intent(in) :: page_ranks(0:n-1)
    real(real32), intent(out) :: maps(0:n*n-1)
    integer :: i, j
    real(real32) :: outbound_rank
    do i = 0, n-1
      outbound_rank = page_ranks(i) / real(noutlinks(i), real32)
      do j = 0, n-1
        maps(i*n+j) = real(pages(i*n+j), real32) * outbound_rank
      end do
    end do
  end subroutine

  subroutine reduce_ref(page_ranks, maps, n, diffs)
    integer, intent(in) :: n
    real(real32), intent(inout) :: page_ranks(0:n-1), diffs(0:n-1)
    real(real32), intent(in) :: maps(0:n*n-1)
    integer :: i, j
    real(real32) :: old_rank, new_rank
    do j = 0, n-1
      old_rank = page_ranks(j); new_rank = 0.0_real32
      do i = 0, n-1
        new_rank = new_rank + maps(i*n+j)
      end do
      new_rank = ((1.0_real32 - d_factor) / real(n,real32)) + d_factor * new_rank
      diffs(j) = max(abs(new_rank - old_rank), diffs(j))
      page_ranks(j) = new_rank
    end do
  end subroutine
end module

program main
  use pagerank_mod
  implicit none
  integer :: n, iter, divisor, t, i, j, block_size
  integer(int32), allocatable :: pages(:), noutlinks(:)
  real(real32), allocatable :: maps(:), page_ranks(:), diffs(:)
  real(real32) :: thresh, max_diff, max_diff_ref, outbound_rank, old_rank, new_rank
  real(real64) :: ktime, t0, t1
  logical :: ok

  n = 1000; iter = 1000; thresh = 1.0e-16_real32; divisor = 2
  call parse_args(n, iter, thresh, divisor)
  allocate(page_ranks(0:n-1), maps(0:n*n-1), noutlinks(0:n-1), pages(0:n*n-1), diffs(0:n-1))
  call c_srand(1_int32)
  call random_pages(n, pages, noutlinks, divisor)
  page_ranks = 1.0_real32 / real(n, real32)
  diffs = 0.0_real32
  max_diff = 99.0_real32; max_diff_ref = 99.0_real32
  block_size = min(n, block_size_const)
  ktime = 0.0_real64

  !$omp target data map(to:pages(0:n*n-1),page_ranks(0:n-1),noutlinks(0:n-1),diffs(0:n-1)) map(alloc:maps(0:n*n-1))
  t = 1
  do while (t <= iter .and. max_diff >= thresh)
    t0 = seconds()
    !$omp target teams distribute parallel do private(j,outbound_rank) thread_limit(block_size)
    do i = 0, n-1
      outbound_rank = page_ranks(i) / real(noutlinks(i), real32)
      do j = 0, n-1
        maps(i*n+j) = real(pages(i*n+j), real32) * outbound_rank
      end do
    end do
    !$omp end target teams distribute parallel do

    !$omp target teams distribute parallel do private(i,old_rank,new_rank) thread_limit(block_size)
    do j = 0, n-1
      old_rank = page_ranks(j); new_rank = 0.0_real32
      do i = 0, n-1
        new_rank = new_rank + maps(i*n+j)
      end do
      new_rank = ((1.0_real32 - d_factor) / real(n,real32)) + d_factor * new_rank
      diffs(j) = max(abs(new_rank - old_rank), diffs(j))
      page_ranks(j) = new_rank
    end do
    !$omp end target teams distribute parallel do
    t1 = seconds(); ktime = ktime + (t1 - t0)
    !$omp target update from(diffs(0:n-1))
    max_diff = maximum_dif(diffs, n)
    t = t + 1
  end do
  write(0,'(a,f0.6,a,i0)') 'Max difference ', max_diff, ' is reached at iteration ', t
  print '(a,i0,a,i0,a,f0.6,a,f0.6,a)', '"Options": "-n ', n, ' -i ', iter, ' -t ', thresh, &
    '". Total kernel execution time: ', ktime, ' (s)'
  !$omp end target data

  diffs = 0.0_real32
  t = 1
  do while (t <= iter .and. max_diff_ref >= thresh)
    call map_ref(pages, page_ranks, maps, noutlinks, n)
    call reduce_ref(page_ranks, maps, n, diffs)
    max_diff_ref = maximum_dif(diffs, n)
    t = t + 1
  end do
  ok = abs(max_diff - max_diff_ref) < 1.0e-3_real32
  print '(a)', merge('PASS', 'FAIL', ok)
end program
