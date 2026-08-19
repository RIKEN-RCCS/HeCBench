module c_rng
  use iso_c_binding, only: c_int
  implicit none
  integer, parameter :: RAND_MAX_F = 2147483647
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
contains
  real function rand_unit()
    rand_unit = real(rand()) / real(RAND_MAX_F)
  end function rand_unit
end module c_rng

module mdh_kernels
  use iso_fortran_env, only: real32
  use omp_lib, only: omp_get_wtime
  implicit none
contains
  subroutine run_gpu_kernel(wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val, avg)
    integer, intent(in) :: wgsize, itmax, ngrid, natom, ngadj
    real(real32), intent(in) :: ax(0:), ay(0:), az(0:), gx(0:), gy(0:), gz(0:), charge(0:), size(0:), xkappa, pre1
    real(real32), intent(inout) :: val(0:)
    real(8), intent(out) :: avg
    integer :: n, igrid, iatom
    real(real32) :: sumv, lgx, lgy, lgz, dist
    real(8) :: t0, t1
!$omp target data map(to:ax(0:natom-1),ay(0:natom-1),az(0:natom-1),charge(0:natom-1),size(0:natom-1),gx(0:ngadj-1),gy(0:ngadj-1),gz(0:ngadj-1)) map(alloc:val(0:ngadj-1))
    t0 = omp_get_wtime()
    do n = 1, itmax
!$omp target teams distribute thread_limit(wgsize) private(sumv,lgx,lgy,lgz,iatom,dist)
      do igrid = 0, ngrid-1
        sumv = 0.0_real32
        lgx = gx(igrid); lgy = gy(igrid); lgz = gz(igrid)
!$omp parallel do reduction(+:sumv)
        do iatom = 0, natom-1
          dist = sqrt((lgx-ax(iatom))*(lgx-ax(iatom)) + (lgy-ay(iatom))*(lgy-ay(iatom)) + (lgz-az(iatom))*(lgz-az(iatom)))
          sumv = sumv + pre1 * (charge(iatom)/dist) * exp(-xkappa*(dist-size(iatom))) / (1.0_real32 + xkappa*size(iatom))
        end do
!$omp end parallel do
        val(igrid) = sumv
      end do
!$omp end target teams distribute
    end do
    t1 = omp_get_wtime()
    avg = (t1 - t0) / real(itmax, 8)
!$omp target update from(val(0:ngrid-1))
!$omp end target data
  end subroutine run_gpu_kernel

  subroutine run_cpu_kernel(itmax, ngrid, natom, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val, avg)
    integer, intent(in) :: itmax, ngrid, natom
    real(real32), intent(in) :: ax(0:), ay(0:), az(0:), gx(0:), gy(0:), gz(0:), charge(0:), size(0:), xkappa, pre1
    real(real32), intent(out) :: val(0:)
    real(8), intent(out) :: avg
    integer :: n, igrid, iatom
    real(real32) :: sumv, lgx, lgy, lgz, dist
    real(8) :: t0, t1
    t0 = omp_get_wtime()
    do n = 1, itmax
!$omp parallel do private(sumv,lgx,lgy,lgz,iatom,dist)
      do igrid = 0, ngrid-1
        sumv = 0.0_real32
        lgx = gx(igrid); lgy = gy(igrid); lgz = gz(igrid)
!$omp simd reduction(+:sumv)
        do iatom = 0, natom-1
          dist = sqrt((lgx-ax(iatom))*(lgx-ax(iatom)) + (lgy-ay(iatom))*(lgy-ay(iatom)) + (lgz-az(iatom))*(lgz-az(iatom)))
          sumv = sumv + pre1 * (charge(iatom)/dist) * exp(-xkappa*(dist-size(iatom))) / (1.0_real32 + xkappa*size(iatom))
        end do
        val(igrid) = sumv
      end do
!$omp end parallel do
    end do
    t1 = omp_get_wtime()
    avg = (t1 - t0) / real(itmax, 8)
  end subroutine run_cpu_kernel
end module mdh_kernels

program mdh
  use iso_fortran_env, only: real32
  use omp_lib
  use c_rng
  use wkfutils_mod
  use mdh_kernels
  implicit none
  integer :: itmax, wgsize, argc, i
  integer, parameter :: natom = 5877, ngrid = 134918
  integer :: ngadj
  character(len=128) :: arg
  real(real32), allocatable :: ax(:), ay(:), az(:), gx(:), gy(:), gz(:), charge(:), size(:), val_cpu(:), val_gpu(:)
  real(real32) :: pre1, xkappa
  real(8) :: cpu_avg, gpu_avg, t0, t1
  logical :: ok

  itmax = 100
  wgsize = 256
  argc = command_argument_count()
  i = 1
  do while (i <= argc)
    call get_command_argument(i, arg)
    select case (trim(arg))
    case ('-itmax')
      i = i + 1; call get_command_argument(i, arg); read(arg, *) itmax
    case ('-wgsize')
      i = i + 1; call get_command_argument(i, arg); read(arg, *) wgsize
    end select
    i = i + 1
  end do
  print '(a)', 'Run parameters:'
  print '(a,i0)', '  kernel loop count: ', itmax
  print '(a,i0)', '     workgroup size: ', wgsize

  ngadj = ngrid + (512 - iand(ngrid, 511))
  pre1 = 4.46184985145e19_real32
  xkappa = 0.0735516324639_real32
  allocate(ax(0:natom-1), ay(0:natom-1), az(0:natom-1), charge(0:natom-1), size(0:natom-1))
  allocate(gx(0:ngadj-1), gy(0:ngadj-1), gz(0:ngadj-1), val_cpu(0:ngadj-1), val_gpu(0:ngadj-1))
  ax = 0.0; ay = 0.0; az = 0.0; charge = 0.0; size = 0.0; gx = 0.0; gy = 0.0; gz = 0.0

  print '(a)', 'Generating Data.. '
  do i = 0, natom-1
    ax(i) = rand_unit(); ay(i) = rand_unit(); az(i) = rand_unit()
    charge(i) = rand_unit(); size(i) = real(natom, real32)
  end do
  do i = 0, ngrid-1
    gx(i) = rand_unit(); gy(i) = rand_unit(); gz(i) = rand_unit()
  end do
  print '(a)', 'Done generating inputs.'
  print *

  t0 = omp_get_wtime()
  call run_cpu_kernel(itmax, ngadj, natom, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val_cpu, cpu_avg)
  t1 = omp_get_wtime()
  print '(a,es16.8,a,i0,a)', 'Average kernel execution time: ', cpu_avg, ''
  print '(a,es16.8,a,i0,a)', 'CPU Time: ', t1 - t0, ' (Number of tests = ', itmax, ')'
  print *

  t0 = omp_get_wtime()
  call run_gpu_kernel(wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val_gpu, gpu_avg)
  t1 = omp_get_wtime()
  print '(a,es16.8)', 'Average kernel time on the device: ', gpu_avg
  print '(a,es16.8,a,i0,a)', 'GPU Time: ', t1 - t0, ' (Number of tests = ', itmax, ')'
  print *

  ok = all(abs(val_cpu(0:ngrid-1) - val_gpu(0:ngrid-1)) <= 1.0e-3_real32)
  print '(a)', merge('PASS', 'FAIL', ok)
end program mdh
