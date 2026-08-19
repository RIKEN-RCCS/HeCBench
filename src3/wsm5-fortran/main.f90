program main
  use iso_fortran_env, only: real32, real64, int64
  use wsm5_kernel_mod
  implicit none
  integer :: argc, repeat, stat, i, rep
  character(len=128) :: arg
  real(real32), allocatable :: th(:), pii(:), q(:), qc(:), qi(:), qr(:), qs(:), den(:), p(:), delz(:)
  real(real32), allocatable :: rain(:), rainncv(:), sr(:), snow(:), snowncv(:)
  real(real32) :: delt, rain_sum, snow_sum
  integer :: ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe, d3, d2, dips, dipe, djps, djpe, dkps, dkpe
  integer :: remx, remy, teamX, teamY
  real(real64) :: t0, t1, time_sum
  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *, iostat=stat) repeat
  delt = 10.0_real32
  ims = 0; ime = 59; jms = 0; jme = 45; kms = 0; kme = 2
  ips = 0; ipe = 59; jps = 0; jpe = 45; kps = 0; kpe = 2
  d3 = (ime-ims+1) * (jme-jms+1) * (kme-kms+1)
  d2 = (ime-ims+1) * (jme-jms+1)
  dips = 0; dipe = ipe-ips+1; djps = 0; djpe = jpe-jps+1; dkps = 0; dkpe = kpe-kps+1
  remx = merge(1, 0, mod(ipe-ips+1, xxx) /= 0)
  remy = merge(1, 0, mod(jpe-jps+1, yyy) /= 0)
  teamX = (ipe-ips+1) / xxx + remx
  teamY = (jpe-jps+1) / yyy + remy
  time_sum = 0.0_real64
  do rep = 1, repeat
    call alloc_all()
!$omp target data map(to:th,pii,q,qc,qi,qr,qs,den,p,delz,rainncv,snowncv,sr) map(tofrom:rain,snow)
    t0 = wall_seconds()
    call wsm(th, pii, q, qc, qi, qr, qs, den, p, delz, rain, rainncv, sr, snow, snowncv, delt, &
      dips+1, ipe-ips+1, djps+1, jpe-jps+1, dkps+1, kpe-kps+1, &
      dips+1, dipe, djps+1, djpe, dkps+1, dkpe, dips+1, dipe, djps+1, djpe, dkps+1, dkpe, teamX, teamY)
    t1 = wall_seconds()
    time_sum = time_sum + (t1 - t0)
!$omp end target data
    rain_sum = sum(rain)
    snow_sum = sum(snow)
    deallocate(th,pii,q,qc,qi,qr,qs,den,p,delz,rain,rainncv,sr,snow,snowncv)
  end do
  print '(a,f12.6,a)', 'Average kernel execution time: ', time_sum * 1000.0_real64 / real(repeat, real64), ' (ms)'
  print '(a,f12.6,a,f12.6)', 'Checksum: rain = ', rain_sum, ' snow = ', snow_sum
contains
  function wall_seconds() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function wall_seconds

  subroutine alloc_all()
    allocate(th(0:d3-1), pii(0:d3-1), q(0:d3-1), qc(0:d3-1), qi(0:d3-1), qr(0:d3-1), qs(0:d3-1))
    allocate(den(0:d3-1), p(0:d3-1), delz(0:d3-1), rain(0:d2-1), rainncv(0:d2-1), sr(0:d2-1), snow(0:d2-1), snowncv(0:d2-1))
    th=0.001_real32; pii=0.001_real32; q=0.001_real32; qc=0.001_real32; qi=0.001_real32
    qr=0.001_real32; qs=0.001_real32; den=0.001_real32; p=0.001_real32; delz=0.001_real32
    rain=0.001_real32; rainncv=0.001_real32; sr=0.001_real32; snow=0.001_real32; snowncv=0.001_real32
  end subroutine alloc_all
end program main
