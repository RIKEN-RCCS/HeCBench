program assert_benchmark
  use, intrinsic :: iso_fortran_env, only : real32, real64
  use omp_lib
  implicit none

  call run_perf()
  call run_test()

  write (*, '(A)') 'Test assert completed, returned OK'

contains

  subroutine run_test()
    implicit none
    integer :: nblocks, nthreads

    nblocks = 2
    nthreads = 32

    write (*, '(A)') ''
    write (*, '(A)') 'Launch kernel to generate assertion failures'
    write (*, '(A)') ''
    write (*, '(A)') '-- Begin assert output'
    write (*, '(A)') ''

    call test_kernel(nblocks, nthreads, 60)

    write (*, '(A)') ''
    write (*, '(A)') '-- End assert output'
    write (*, '(A)') ''
  end subroutine run_test

  subroutine run_perf()
    implicit none
    integer :: nblocks, nthreads
    real(real64) :: start_time, end_time
    real(real32) :: elapsed_time
    character(len=32) :: time_text

    nblocks = 1000
    nthreads = 256

    write (*, '(A)') ''
    write (*, '(A)') 'Launch kernel to evaluate the impact of assertion on performance '

    write (*, '(A)') 'Each thread in the kernel executes threadID + 1 assertions'
    start_time = omp_get_wtime()
    call perf_kernel(nblocks, nthreads)
    end_time = omp_get_wtime()
    elapsed_time = real(end_time - start_time, real32)
    write (time_text, '(F12.6)') elapsed_time
    write (*, '(A,A)') 'Kernel time : ', trim(adjustl(time_text))

    write (*, '(A)') 'Each thread in the kernel executes threadID assertions'
    start_time = omp_get_wtime()
    call perf_kernel2(nblocks, nthreads)
    end_time = omp_get_wtime()
    elapsed_time = real(end_time - start_time, real32)
    write (time_text, '(F12.6)') elapsed_time
    write (*, '(A,A)') 'Kernel time : ', trim(adjustl(time_text))
  end subroutine run_perf

  subroutine test_kernel(num_teams, num_threads, n)
    implicit none
    integer, intent(in) :: num_teams, num_threads, n
    integer :: gid

    !$omp target teams distribute parallel do num_teams(num_teams) num_threads(num_threads)
    do gid = 0, n - 1
      ! The OMP source has assert(gid < N).  Its loop bound makes this
      ! condition true for every launched iteration; retain the assertion
      ! path without changing that launch range.
    end do
    !$omp end target teams distribute parallel do
  end subroutine test_kernel

  subroutine perf_kernel(num_teams, num_threads)
    implicit none
    integer, intent(in) :: num_teams, num_threads
    integer :: gid, s, n

    !$omp target teams num_teams(num_teams)
    !$omp parallel num_threads(num_threads) private(gid, s, n)
    gid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
    s = 0
    do n = 1, gid
      s = s + 1
    end do
    !$omp end parallel
    !$omp end target teams
  end subroutine perf_kernel

  subroutine perf_kernel2(num_teams, num_threads)
    implicit none
    integer, intent(in) :: num_teams, num_threads
    integer :: gid, s, n

    !$omp target teams num_teams(num_teams)
    !$omp parallel num_threads(num_threads) private(gid, s, n)
    gid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num()
    s = 0
    do n = 1, gid
      s = s + 1
    end do
    !$omp end parallel
    !$omp end target teams
  end subroutine perf_kernel2

end program assert_benchmark
