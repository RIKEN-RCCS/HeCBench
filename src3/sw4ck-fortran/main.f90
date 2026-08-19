program main
  use iso_fortran_env, only: real64
  use sw4ck_utils
  use curvilinear4sg_mod
  implicit none
  integer :: argc, repeat, unit, stat, ds, i, p, n, cof_size, sg_str_size, nkg
  character(len=256) :: path
  character(len=128) :: arg
  integer :: onesided(0:13,0:1)
  type(sarray_t) :: alpha(0:1), mu(0:1), lambda(0:1), met(0:1), jac(0:1), uacc(0:1)
  real(real64), allocatable :: cof(:), sg_str(:)
  real(real64), parameter :: exact_norm(0:1) = [2.2502232733796421194_real64, 202.0512747393526638_real64]
  real(real64) :: t0, t1, time_sum, norm, err
  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <path to file> <repeat>'
    stop 1
  end if
  call get_command_argument(1, path)
  call get_command_argument(2, arg); read(arg, *, iostat=stat) repeat
  print '(a,a)', 'Reading from file ', trim(path)
  open(newunit=unit, file=trim(path), status='old', action='read', iostat=stat)
  if (stat /= 0) stop 1
  do ds = 0, 1
    read(unit, *) onesided(:,ds)
    call read_named(unit, alpha(ds), 'a_AlphaVE_0')
    call read_named(unit, mu(ds), 'mMuVE_0')
    call read_named(unit, lambda(ds), 'mLambdaVE_0')
    call read_named(unit, met(ds), 'mMetric')
    call read_named(unit, jac(ds), 'mJ')
    call read_named(unit, uacc(ds), 'a_Uacc')
    do i = 1, 9
      read(unit, *, iostat=stat)
      if (stat /= 0) exit
    end do
    call init_sarray(alpha(ds)); call init_sarray(mu(ds)); call init_sarray(lambda(ds))
    call init_sarray(met(ds)); call init_sarray(jac(ds)); call init_sarray(uacc(ds))
  end do
  close(unit)
  cof_size = 6 + 384 + 24 + 48 + 6 + 384 + 6 + 6
  allocate(cof(0:cof_size-1))
  do i = 0, cof_size - 1
    cof(i) = real(i, real64) / 1000.0_real64
  end do
!$omp target enter data map(to:cof)
  do ds = 0, 1
    sg_str_size = onesided(7,ds) - onesided(6,ds) + onesided(9,ds) - onesided(8,ds) + 2
    allocate(sg_str(0:sg_str_size-1))
    do n = 0, sg_str_size - 1
      sg_str(n) = real(n, real64) / 1000.0_real64
    end do
    nkg = onesided(12,ds)
!$omp target data map(to:cof,alpha(ds)%data,mu(ds)%data,lambda(ds)%data,met(ds)%data,jac(ds)%data,sg_str) map(alloc:uacc(ds)%data)
    time_sum = 0.0_real64
    do p = 1, repeat
!$omp target update to(uacc(ds)%data)
      t0 = wall_seconds()
      call curvilinear4sg_ci(onesided(6,ds), onesided(7,ds), onesided(8,ds), onesided(9,ds), onesided(10,ds), onesided(11,ds), &
        alpha(ds)%data, mu(ds)%data, lambda(ds)%data, met(ds)%data, jac(ds)%data, uacc(ds)%data, onesided(:,ds), cof, sg_str, nkg, '-', &
        alpha(ds)%nc, alpha(ds)%ni, alpha(ds)%nj, alpha(ds)%nk)
      t1 = wall_seconds()
      time_sum = time_sum + (t1 - t0)
    end do
    print '(/a,f12.6,a/)', 'Average execution time of sw4ck kernels: ', time_sum * 1000.0_real64 / real(repeat, real64), ' milliseconds'
!$omp target update from(uacc(ds)%data)
!$omp end target data
    norm = norm_sarray(uacc(ds))
    err = (norm - exact_norm(ds)) / exact_norm(ds) * 100.0_real64
    print '(a,f12.6,a)', 'Error = ', err, ' %'
    deallocate(sg_str)
  end do
!$omp target exit data map(delete:cof)
contains
  subroutine read_named(unit, s, wanted)
    integer, intent(in) :: unit
    type(sarray_t), intent(out) :: s
    character(len=*), intent(in) :: wanted
    character(len=32) :: name
    read(unit, *) name, s%g, s%nc, s%ni, s%nj, s%nk, s%ib, s%ie, s%jb, s%je, s%kb, s%ke, s%base, s%offi, s%offj, s%offk, s%offc, s%npts
    s%name = name
  end subroutine read_named
end program main
