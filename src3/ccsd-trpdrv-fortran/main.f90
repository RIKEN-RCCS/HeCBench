module ccsd_precision
  use iso_fortran_env, only : real64, int64
  use iso_c_binding, only : c_double, c_long
  implicit none
  integer, parameter :: dp = real64

  interface
    subroutine c_srand48(seed) bind(C, name='srand48')
      import :: c_long
      integer(c_long), value :: seed
    end subroutine c_srand48

    function c_drand48() bind(C, name='drand48') result(value)
      import :: c_double
      real(c_double) :: value
    end function c_drand48
  end interface
contains

  subroutine make_array(a)
    real(dp), intent(out) :: a(0:)
    integer :: p
    do p = 0, ubound(a, 1)
      a(p) = c_drand48()
    end do
  end subroutine make_array

  subroutine ccsd_tengy_reference(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, &
       dintc1, dintx1, t1v1, dintc2, dintx2, t1v2, eorb, eaijk, &
       emp4i, emp5i, emp4k, emp5k, ncor, nocc, nvir)
    integer, intent(in) :: ncor, nocc, nvir
    real(dp), intent(in) :: f1n(0:), f1t(0:), f2n(0:), f2t(0:), f3n(0:), f3t(0:), f4n(0:), f4t(0:)
    real(dp), intent(in) :: dintc1(0:), dintx1(0:), t1v1(0:), dintc2(0:), dintx2(0:), t1v2(0:), eorb(0:)
    real(dp), intent(in) :: eaijk
    real(dp), intent(out) :: emp4i, emp5i, emp4k, emp5k
    integer :: b, c, bc, cb
    real(dp) :: denom, f1nbc, f1tbc, f1ncb, f1tcb, f2nbc, f2tbc, f2ncb, f2tcb
    real(dp) :: f3nbc, f3tbc, f3ncb, f3tcb, f4nbc, f4tbc, f4ncb, f4tcb
    real(dp) :: t1v1b, t1v2b, dintx1c, dintx2c, dintc1c, dintc2c

    emp5i = 0.0_dp; emp4i = 0.0_dp; emp5k = 0.0_dp; emp4k = 0.0_dp
    do c = 0, nvir - 1
      do b = 0, nvir - 1
        denom = -1.0_dp / (eorb(ncor+nocc+b) + eorb(ncor+nocc+c) + eaijk)
        bc = b + c*nvir; cb = c + b*nvir
        f1nbc=f1n(bc); f1tbc=f1t(bc); f1ncb=f1n(cb); f1tcb=f1t(cb)
        f2nbc=f2n(bc); f2tbc=f2t(bc); f2ncb=f2n(cb); f2tcb=f2t(cb)
        f3nbc=f3n(bc); f3tbc=f3t(bc); f3ncb=f3n(cb); f3tcb=f3t(cb)
        f4nbc=f4n(bc); f4tbc=f4t(bc); f4ncb=f4n(cb); f4tcb=f4t(cb)
        emp4i = emp4i + denom*(f1tbc+f1ncb+f2tcb+f3nbc+f4ncb)*(f1tbc-2.0_dp*f2tbc-2.0_dp*f3tbc+f4tbc) &
          - denom*(f1nbc+f1tcb+f2ncb+f3ncb)*(2.0_dp*f1tbc-f2tbc-f3tbc+2.0_dp*f4tbc) &
          + denom*3.0_dp*(f1nbc*(f1nbc+f3ncb+2.0_dp*f4tcb)+f2nbc*f2tcb+f3nbc*f4tbc)
        emp4k = emp4k + denom*(f1nbc+f1tcb+f2ncb+f3tbc+f4tcb)*(f1nbc-2.0_dp*f2nbc-2.0_dp*f3nbc+f4nbc) &
          - denom*(f1tbc+f1ncb+f2tcb+f3tcb)*(2.0_dp*f1nbc-f2nbc-f3nbc+2.0_dp*f4nbc) &
          + denom*3.0_dp*(f1tbc*(f1tbc+f3tcb+2.0_dp*f4ncb)+f2tbc*f2ncb+f3tbc*f4nbc)
        t1v1b=t1v1(b); t1v2b=t1v2(b); dintx1c=dintx1(c); dintx2c=dintx2(c); dintc1c=dintc1(c); dintc2c=dintc2(c)
        emp5i = emp5i + denom*t1v1b*dintx1c*(f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2.0_dp &
          +(f3nbc+f4tbc+f1ncb)*4.0_dp) + denom*t1v1b*dintc1c*(f1nbc+f4nbc+f1tcb-(f2nbc+f3nbc+f2tcb)*2.0_dp)
        emp5k = emp5k + denom*t1v2b*dintx2c*(f1nbc+f2tbc+f4tcb-(f3nbc+f4tbc+f2tcb+f1tbc+f2nbc+f3tcb)*2.0_dp &
          +(f3tbc+f4nbc+f1tcb)*4.0_dp) + denom*t1v2b*dintc2c*(f1tbc+f4tbc+f1ncb-(f2tbc+f3tbc+f2ncb)*2.0_dp)
      end do
    end do
  end subroutine ccsd_tengy_reference

  subroutine ccsd_tengy_gpu(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, &
       dintc1, dintx1, t1v1, dintc2, dintx2, t1v2, eorb, eaijk, &
       emp4i, emp5i, emp4k, emp5k, ncor, nocc, nvir, time_ns)
    use omp_lib, only : omp_get_wtime
    integer, intent(in) :: ncor, nocc, nvir
    real(dp), intent(in) :: f1n(0:), f1t(0:), f2n(0:), f2t(0:), f3n(0:), f3t(0:), f4n(0:), f4t(0:)
    real(dp), intent(in) :: dintc1(0:), dintx1(0:), t1v1(0:), dintc2(0:), dintx2(0:), t1v2(0:), eorb(0:)
    real(dp), intent(in) :: eaijk
    real(dp), intent(out) :: emp4i, emp5i, emp4k, emp5k
    integer(int64), intent(out) :: time_ns
    integer :: b, c, bc, cb, lnvv
    real(dp) :: denom, f1nbc, f1tbc, f1ncb, f1tcb, f2nbc, f2tbc, f2ncb, f2tcb
    real(dp) :: f3nbc, f3tbc, f3ncb, f3tcb, f4nbc, f4tbc, f4ncb, f4tcb
    real(dp) :: t1v1b, t1v2b, dintx1c, dintx2c, dintc1c, dintc2c, t0, t1

    lnvv = nvir*nvir
    emp5i = 0.0_dp; emp4i = 0.0_dp; emp5k = 0.0_dp; emp4k = 0.0_dp
    !$omp target data map(to: f1n(0:lnvv-1), f2n(0:lnvv-1), f3n(0:lnvv-1), f4n(0:lnvv-1), &
    !$omp& f1t(0:lnvv-1), f2t(0:lnvv-1), f3t(0:lnvv-1), f4t(0:lnvv-1), &
    !$omp& dintc1(0:nvir-1), dintc2(0:nvir-1), dintx1(0:nvir-1), dintx2(0:nvir-1), &
    !$omp& t1v1(0:nvir-1), t1v2(0:nvir-1), eorb(0:ncor+nocc+nvir-1)) &
    !$omp& map(tofrom: emp5i, emp4i, emp5k, emp4k)
    t0 = omp_get_wtime()
    !$omp target teams distribute parallel do collapse(2) thread_limit(256) reduction(+:emp5i,emp4i,emp5k,emp4k)
    do b = 0, nvir - 1
      do c = 0, nvir - 1
        denom = -1.0_dp / (eorb(ncor+nocc+b) + eorb(ncor+nocc+c) + eaijk)
        bc = b+c*nvir; cb = c+b*nvir
        f1nbc=f1n(bc); f1tbc=f1t(bc); f1ncb=f1n(cb); f1tcb=f1t(cb)
        f2nbc=f2n(bc); f2tbc=f2t(bc); f2ncb=f2n(cb); f2tcb=f2t(cb)
        f3nbc=f3n(bc); f3tbc=f3t(bc); f3ncb=f3n(cb); f3tcb=f3t(cb)
        f4nbc=f4n(bc); f4tbc=f4t(bc); f4ncb=f4n(cb); f4tcb=f4t(cb)
        emp4i = emp4i + denom*(f1tbc+f1ncb+f2tcb+f3nbc+f4ncb)*(f1tbc-2.0_dp*f2tbc-2.0_dp*f3tbc+f4tbc) &
          - denom*(f1nbc+f1tcb+f2ncb+f3ncb)*(2.0_dp*f1tbc-f2tbc-f3tbc+2.0_dp*f4tbc) &
          + denom*3.0_dp*(f1nbc*(f1nbc+f3ncb+2.0_dp*f4tcb)+f2nbc*f2tcb+f3nbc*f4tbc)
        emp4k = emp4k + denom*(f1nbc+f1tcb+f2ncb+f3tbc+f4tcb)*(f1nbc-2.0_dp*f2nbc-2.0_dp*f3nbc+f4nbc) &
          - denom*(f1tbc+f1ncb+f2tcb+f3tcb)*(2.0_dp*f1nbc-f2nbc-f3nbc+2.0_dp*f4nbc) &
          + denom*3.0_dp*(f1tbc*(f1tbc+f3tcb+2.0_dp*f4ncb)+f2tbc*f2ncb+f3tbc*f4nbc)
        t1v1b=t1v1(b); t1v2b=t1v2(b); dintx1c=dintx1(c); dintx2c=dintx2(c); dintc1c=dintc1(c); dintc2c=dintc2(c)
        emp5i = emp5i + denom*t1v1b*dintx1c*(f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2.0_dp &
          +(f3nbc+f4tbc+f1ncb)*4.0_dp) + denom*t1v1b*dintc1c*(f1nbc+f4nbc+f1tcb-(f2nbc+f3nbc+f2tcb)*2.0_dp)
        emp5k = emp5k + denom*t1v2b*dintx2c*(f1nbc+f2tbc+f4tcb-(f3nbc+f4tbc+f2tcb+f1tbc+f2nbc+f3tcb)*2.0_dp &
          +(f3tbc+f4nbc+f1tcb)*4.0_dp) + denom*t1v2b*dintc2c*(f1tbc+f4tbc+f1ncb-(f2tbc+f3tbc+f2ncb)*2.0_dp)
      end do
    end do
    !$omp end target teams distribute parallel do
    t1 = omp_get_wtime()
    time_ns = int((t1-t0)*1.0e9_dp, int64)
    !$omp end target data
  end subroutine ccsd_tengy_gpu

  subroutine ccsd_trpdrv_gpu(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, eorb, ncor, nocc, nvir, &
       emp4, emp5, a, i, j, k, dintc1, dintx1, t1v1, dintc2, dintx2, t1v2, time_ns)
    integer, intent(in) :: ncor, nocc, nvir, a, i, j, k
    real(dp), intent(in) :: f1n(0:), f1t(0:), f2n(0:), f2t(0:), f3n(0:), f3t(0:), f4n(0:), f4t(0:), eorb(0:)
    real(dp), intent(in) :: dintc1(0:), dintx1(0:), t1v1(0:), dintc2(0:), dintx2(0:), t1v2(0:)
    real(dp), intent(inout) :: emp4, emp5
    integer(int64), intent(out) :: time_ns
    real(dp) :: eaijk, emp4i, emp5i, emp4k, emp5k
    eaijk = eorb(a-1) - (eorb(ncor+i-1) + eorb(ncor+j-1) + eorb(ncor+k-1))
    call ccsd_tengy_gpu(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,eaijk, &
      emp4i,emp5i,emp4k,emp5k,ncor,nocc,nvir,time_ns)
    emp4 = emp4 + emp4i; emp5 = emp5 + emp5i
    if (i /= k) then
      emp4 = emp4 + emp4k; emp5 = emp5 + emp5k
    end if
  end subroutine ccsd_trpdrv_gpu

  subroutine ccsd_trpdrv_ref(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, eorb, ncor, nocc, nvir, &
       emp4, emp5, a, i, j, k, dintc1, dintx1, t1v1, dintc2, dintx2, t1v2)
    integer, intent(in) :: ncor, nocc, nvir, a, i, j, k
    real(dp), intent(in) :: f1n(0:), f1t(0:), f2n(0:), f2t(0:), f3n(0:), f3t(0:), f4n(0:), f4t(0:), eorb(0:)
    real(dp), intent(in) :: dintc1(0:), dintx1(0:), t1v1(0:), dintc2(0:), dintx2(0:), t1v2(0:)
    real(dp), intent(inout) :: emp4, emp5
    real(dp) :: eaijk, emp4i, emp5i, emp4k, emp5k
    eaijk = eorb(a-1) - (eorb(ncor+i-1) + eorb(ncor+j-1) + eorb(ncor+k-1))
    call ccsd_tengy_reference(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,eorb,eaijk, &
      emp4i,emp5i,emp4k,emp5k,ncor,nocc,nvir)
    emp4 = emp4 + emp4i; emp5 = emp5 + emp5i
    if (i /= k) then
      emp4 = emp4 + emp4k; emp5 = emp5 + emp5k
    end if
  end subroutine ccsd_trpdrv_ref
end module ccsd_precision

program main
  use iso_fortran_env, only : real64, int64
  use iso_c_binding, only : c_long
  use ccsd_precision
  implicit none
  integer :: argc, ncor, nocc, nvir, maxiter, nkpass, nbf, lnvv, lnov, kchunk, ntimers
  integer :: klo, khi, a, i, j, k, iter, ios
  integer(int64) :: elapsed_ns
  real(real64) :: memory, emp4, emp5, emp4_r, emp5_r, tsum, tmax, tmin, tavg, dgemm_flops, dgemm_mops, tengy_ops
  real(real64), allocatable :: eorb(:), f1n(:), f1t(:), f2n(:), f2t(:), f3n(:), f3t(:), f4n(:), f4t(:)
  real(real64), allocatable :: tij(:), tkj(:), tia(:), tka(:), xia(:), xka(:), jia(:), jka(:), kia(:), kka(:), jij(:), jkj(:), kij(:), kkj(:)
  real(real64), allocatable :: dja(:), djka(:), djia(:), dintc1(:), dintx1(:), t1v1(:), dintc2(:), dintx2(:), t1v2(:), timers(:)
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc < 2) then
    print '(a)', 'Usage: ./test_cbody nocc nvir [maxiter] [nkpass]'
    error stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=ios) nocc
  if (ios /= 0) error stop 1
  call get_command_argument(2, arg); read(arg, *, iostat=ios) nvir
  if (ios /= 0) error stop 1
  ncor = 0; maxiter = 100; nkpass = 1
  if (argc > 2) then
    call get_command_argument(3, arg); read(arg, *, iostat=ios) maxiter
    if (ios /= 0) error stop 1
    if (maxiter < 0) maxiter = shiftl(1, 30)
  end if
  if (argc > 3) then
    call get_command_argument(4, arg); read(arg, *, iostat=ios) nkpass
    if (ios /= 0) error stop 1
  end if
  if (nocc < 1 .or. nvir < 1) then
    print '(a)', 'Arguments must be non-negative!'
    error stop 1
  end if
  print '(a,i0,a,i0,a,i0,a,i0)', 'Test driver for cbody with nocc=',nocc,', nvir=',nvir,', maxiter=',maxiter,', nkpass=',nkpass
  nbf=ncor+nocc+nvir; lnvv=nvir*nvir; lnov=nocc*nvir; kchunk=(nocc-1)/nkpass+1
  memory = real(nbf,real64)+8.0_real64*lnvv+lnvv+kchunk*lnvv+lnov*nocc+kchunk*lnov+lnov*nocc+kchunk*lnov+lnvv+ &
    kchunk*lnvv+lnvv+kchunk*lnvv+lnov*nocc+kchunk*lnov+lnov*nocc+kchunk*lnov+lnov+nvir*kchunk+nvir*nocc+6.0_real64*lnvv
  memory = memory * 8.0_real64
  print '(a,f0.6,a)', 'This test requires ', 1.0e-9_real64*memory, ' GB of memory.'
  if (1.0e-9_real64*memory > 4.0_real64) then
    print '(a)', 'You need to increase MAX_MEM (4)'; print '(a,i0,a)', 'or set nkpass (',nkpass,') to a larger number.'
    error stop 4
  end if
  allocate(eorb(0:nbf-1),f1n(0:lnvv-1),f2n(0:lnvv-1),f3n(0:lnvv-1),f4n(0:lnvv-1),f1t(0:lnvv-1),f2t(0:lnvv-1),f3t(0:lnvv-1),f4t(0:lnvv-1))
  allocate(tij(0:lnvv-1),tkj(0:kchunk*lnvv-1),tia(0:lnov*nocc-1),tka(0:kchunk*lnov-1),xia(0:lnov*nocc-1),xka(0:kchunk*lnov-1))
  allocate(jia(0:lnvv-1),jka(0:kchunk*lnvv-1),kia(0:lnvv-1),kka(0:kchunk*lnvv-1),jij(0:lnov*nocc-1),jkj(0:kchunk*lnov-1),kij(0:lnov*nocc-1),kkj(0:kchunk*lnov-1))
  allocate(dja(0:lnov-1),djka(0:nvir*kchunk-1),djia(0:nvir*nocc-1),dintc1(0:lnvv-1),dintc2(0:lnvv-1),dintx1(0:lnvv-1),dintx2(0:lnvv-1),t1v1(0:lnvv-1),t1v2(0:lnvv-1))
  call c_srand48(2_c_long)
  call make_array(eorb); call make_array(f1n); call make_array(f2n); call make_array(f3n); call make_array(f4n); call make_array(f1t); call make_array(f2t); call make_array(f3t); call make_array(f4t)
  call make_array(tij); call make_array(tkj); call make_array(tia); call make_array(tka); call make_array(xia); call make_array(xka); call make_array(jia); call make_array(jka); call make_array(kia); call make_array(kka); call make_array(jij); call make_array(jkj); call make_array(kij); call make_array(kkj); call make_array(dja); call make_array(djka); call make_array(djia); call make_array(dintc1); call make_array(dintc2); call make_array(dintx1); call make_array(dintx2); call make_array(t1v1); call make_array(t1v2)
  ntimers=min(maxiter,nocc*nocc*nocc*nocc); allocate(timers(0:ntimers-1)); timers=0.0_real64
  emp4=0.0_real64; emp5=0.0_real64; emp4_r=0.0_real64; emp5_r=0.0_real64; iter=0; a=1
  all_loops: do klo=1,nocc,kchunk
    khi=min(nocc,klo+kchunk-1)
    do j=1,nocc
      do i=1,nocc
        do k=klo,min(khi,i)
          call ccsd_trpdrv_gpu(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,eorb,ncor,nocc,nvir,emp4,emp5,a,i,j,k,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2,elapsed_ns)
          timers(iter)=real(elapsed_ns,real64)*1.0e-9_real64
          call ccsd_trpdrv_ref(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,eorb,ncor,nocc,nvir,emp4_r,emp5_r,a,i,j,k,dintc1,dintx1,t1v1,dintc2,dintx2,t1v2)
          iter=iter+1
          if (iter == maxiter) then
            print '(a,i0,a)', 'Stopping after ',iter,' iterations...'; exit all_loops
          end if
          if (emp4 > 1000.0_real64) emp4=emp4-1000.0_real64
          if (emp4 < -1000.0_real64) emp4=emp4+1000.0_real64
          if (emp5 > 1000.0_real64) emp5=emp5-1000.0_real64
          if (emp5 < -1000.0_real64) emp5=emp5+1000.0_real64
          if (emp4_r > 1000.0_real64) emp4_r=emp4_r-1000.0_real64
          if (emp4_r < -1000.0_real64) emp4_r=emp4_r+1000.0_real64
          if (emp5_r > 1000.0_real64) emp5_r=emp5_r-1000.0_real64
          if (emp5_r < -1000.0_real64) emp5_r=emp5_r+1000.0_real64
        end do
      end do
    end do
  end do all_loops
  tsum=sum(timers(0:iter-1)); tmax=maxval(timers(0:iter-1)); tmin=minval(timers(0:iter-1)); tavg=tsum/real(iter,real64)
  print '(a,f0.6,a,f0.6,a,f0.6)', 'Kernel timing: min=',tmin,', max=',tmax,', avg=',tavg
  dgemm_flops=(8.0_real64*nvir)*nvir*(nvir+nocc); dgemm_mops=8.0_real64*(4.0_real64*nvir*nvir+2.0_real64*nvir*nocc); tengy_ops=(1.0_real64*nvir)*nvir*(86.0_real64+8.0_real64)
  print '(a,es10.3,a,es10.3,a,es10.3)', 'OPS: dgemm_flops=',dgemm_flops,' dgemm_mops=',dgemm_mops,' tengy_ops=',tengy_ops
  print '(a,es10.3,a,es10.3)', 'PERF: GF/s=',1.0e-9_real64*(dgemm_flops+tengy_ops)/tavg,' GB/s=',8.0e-9_real64*(dgemm_mops+tengy_ops)/tavg
  print '(a) ', 'These are meaningless but should not vary for a particular input:'
  print '(a,f0.6,a,f0.6)', 'emp4=',emp4,' emp5=',emp5
  if (abs(emp4_r-emp4) < 1.0e-6_real64 .and. abs(emp5_r-emp5) < 1.0e-6_real64) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
end program main
