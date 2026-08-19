module mat_nn_openmp_mod
  use iso_fortran_env, only: real64, int64
  use su3_types
  implicit none
contains
  function su3_mat_nn(a, b, c, total_sites, iterations, threads_per_team, warmups, verbose) result(ttotal)
    type(site), intent(in) :: a(0:)
    type(su3_matrix), intent(in) :: b(0:3)
    type(site), intent(inout) :: c(0:)
    integer(int64), intent(in) :: total_sites, iterations, warmups
    integer, intent(inout) :: threads_per_team
    integer, intent(in) :: verbose
    real(real64) :: ttotal, t0, t1, checksum
    integer(int64) :: num_work_items, id, site_id
    integer :: iters, j, k, l, m
    type(complex_t) :: cc
    if (threads_per_team == 0) threads_per_team = 36
    num_work_items = total_sites * int(threads_per_team, int64)
    if (verbose >= 1) then
      print '(a,i0)', 'Number of teams = ', total_sites
      print '(a,i0)', 'Threads per team = ', threads_per_team
      print '(a,i0)', 'Number of work items = ', num_work_items
    end if
!$omp target data map(to:a,b) map(from:c)
    t0 = wall_seconds()
    do iters = 0, int(iterations + warmups) - 1
      if (iters == warmups) t0 = wall_seconds()
!$omp target teams distribute parallel do num_teams(total_sites) thread_limit(threads_per_team) private(id,site_id,j,k,l,m,cc)
      do id = 0_int64, num_work_items - 1_int64
        site_id = id / 36_int64
        if (site_id < total_sites) then
          j = int(mod(id, 36_int64) / 9_int64)
          k = int(mod(id, 9_int64) / 3_int64)
          l = int(mod(id, 3_int64))
          cc%real = 0.0_real64
          cc%imag = 0.0_real64
          do m = 0, 2
            call cmulsum(a(site_id)%link(j)%e(k,m), b(j)%e(m,l), cc)
          end do
          c(site_id)%link(j)%e(k,l) = cc
        end if
      end do
!$omp end target teams distribute parallel do
    end do
    t1 = wall_seconds()
!$omp end target data
    ttotal = t1 - t0
    checksum = 0.0_real64
    do site_id = 0, total_sites - 1
      do j = 0, 3
        do k = 0, 2
          do l = 0, 2
            checksum = checksum + c(site_id)%link(j)%e(k,l)%real
          end do
        end do
      end do
    end do
    checksum = checksum / real(total_sites, real64)
    if (abs(checksum - 36.0_real64) < 1.0e-6_real64) then
      print '(a,f5.0)', 'Checksum SUCCESS... though please be diligent and check the following value is not NaN: checksum=', checksum
    else
      print '(a)', 'Checksum FAILURE'
    end if
  end function su3_mat_nn

  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds
end module mat_nn_openmp_mod
