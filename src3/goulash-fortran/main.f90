module goulash_module
  use, intrinsic :: iso_fortran_env, only : int64, real64
  implicit none
  integer, parameter :: mhu_l = 10, mhu_m = 5, tau_m = 18
  real(real64), parameter :: mhu_a(0:mhu_l+mhu_m-1) = [ &
    9.9632117206253790e-01_real64, 4.0825738726469545e-02_real64, 6.3401613233199589e-04_real64, &
    4.4158436861700431e-06_real64, 1.1622058324043520e-08_real64, 1.0000000000000000e+00_real64, &
    4.0568375699663400e-02_real64, 6.4216825832642788e-04_real64, 4.2661664422410096e-06_real64, &
    1.3559930396321903e-08_real64, -1.3573468728873069e-11_real64, -4.2594802366702580e-13_real64, &
    7.6779952208246166e-15_real64, 1.4260675804433780e-16_real64, -2.6656212072499249e-18_real64 ]
  real(real64), parameter :: tau_a(0:tau_m-1) = [ &
    1.7765862602413648e+01_real64*0.02_real64, 5.0010202770602419e-02_real64*0.02_real64, &
   -7.8002064070783474e-04_real64*0.02_real64,-6.9399661775931530e-05_real64*0.02_real64, &
    1.6936588308244311e-06_real64*0.02_real64, 5.4629017090963798e-07_real64*0.02_real64, &
   -1.3805420990037933e-08_real64*0.02_real64,-8.0678945216155694e-10_real64*0.02_real64, &
    1.6209833004622630e-11_real64*0.02_real64, 6.5130101230170358e-13_real64*0.02_real64, &
   -6.9931705949674988e-15_real64*0.02_real64,-3.1161210504114690e-16_real64*0.02_real64, &
    5.0166191902609083e-19_real64*0.02_real64, 7.8608831661430381e-20_real64*0.02_real64, &
    4.3936315597226053e-22_real64*0.02_real64,-7.0535966258003289e-24_real64*0.02_real64, &
   -9.0473475495087118e-26_real64*0.02_real64,-2.9878427692323621e-28_real64*0.02_real64 ]
contains
  subroutine gate(m_gate, n_cells, vm)
    integer(int64), intent(in) :: n_cells
    real(real64), intent(inout) :: m_gate(0:)
    real(real64), intent(in) :: vm(0:)
    integer(int64) :: i
    integer :: j, k
    real(real64) :: sum1, sum2, x, mhu, tau_r
!$omp target teams distribute parallel do thread_limit(256) private(sum1,sum2,x,mhu,tau_r,j,k)
    do i = 0_int64, n_cells - 1_int64
      x = vm(i)
      sum1 = 0.0_real64
      do j = mhu_m - 1, 0, -1
        sum1 = mhu_a(j) + x * sum1
      end do
      sum2 = 0.0_real64
      k = mhu_m + mhu_l - 1
      do j = k, mhu_m, -1
        sum2 = mhu_a(j) + x * sum2
      end do
      mhu = sum1 / sum2
      sum1 = 0.0_real64
      do j = tau_m - 1, 0, -1
        sum1 = tau_a(j) + x * sum1
      end do
      tau_r = sum1
      m_gate(i) = m_gate(i) + (mhu - m_gate(i)) * (1.0_real64 - exp(-tau_r))
    end do
!$omp end target teams distribute parallel do
  end subroutine gate

  subroutine reference(m_gate, n_cells, vm)
    integer(int64), intent(in) :: n_cells
    real(real64), intent(inout) :: m_gate(0:)
    real(real64), intent(in) :: vm(0:)
    integer(int64) :: i
    integer :: j, k
    real(real64) :: sum1, sum2, x, mhu, tau_r
    do i = 0_int64, n_cells - 1_int64
      x = vm(i)
      sum1 = 0.0_real64
      do j = mhu_m - 1, 0, -1
        sum1 = mhu_a(j) + x * sum1
      end do
      sum2 = 0.0_real64
      k = mhu_m + mhu_l - 1
      do j = k, mhu_m, -1
        sum2 = mhu_a(j) + x * sum2
      end do
      mhu = sum1 / sum2
      sum1 = 0.0_real64
      do j = tau_m - 1, 0, -1
        sum1 = tau_a(j) + x * sum1
      end do
      tau_r = sum1
      m_gate(i) = m_gate(i) + (mhu - m_gate(i)) * (1.0_real64 - exp(-tau_r))
    end do
  end subroutine reference
end module goulash_module

program goulash
  use, intrinsic :: iso_fortran_env, only : int64, real64
  use omp_lib
  use goulash_module
  implicit none
  integer :: argc
  integer(int64) :: iterations, itime, n_cells, i
  real(real64) :: kernel_mem_used, kernel_starttime, kernel_endtime, kernel_runtime
  real(real64), allocatable :: m_gate(:), m_gate_h(:), vm(:)
  character(len=128) :: argument

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <Iterations> <Kernel_GBs_used>'
    stop 1
  end if
  call get_command_argument(1, argument); read(argument, *) iterations
  call get_command_argument(2, argument); read(argument, *) kernel_mem_used
  n_cells = int((kernel_mem_used * 1024.0_real64 * 1024.0_real64 * 1024.0_real64) / (8.0_real64 * 2.0_real64), int64)
  print '(a,i0)', 'Number of cells: ', n_cells

  allocate(m_gate(0:n_cells-1), m_gate_h(0:n_cells-1), vm(0:n_cells-1))
  m_gate = 0.0_real64
  m_gate_h = 0.0_real64
  vm = 0.0_real64

!$omp target data map(to:m_gate(0:n_cells),vm(0:n_cells))
  do itime = 0_int64, iterations
    if (itime == 1_int64) then
!$omp target update from(m_gate(0:n_cells))
      kernel_starttime = omp_get_wtime()
    end if
    call gate(m_gate, n_cells, vm)
  end do
  kernel_endtime = omp_get_wtime()
  kernel_runtime = kernel_endtime - kernel_starttime
  print '(a,f0.6,a,i0,a)', 'total kernel time ', kernel_runtime, '(s) for ', iterations - 1_int64, ' iterations'
!$omp end target data

  call reference(m_gate_h, n_cells, vm)
  do i = 0_int64, n_cells - 1_int64
    if (abs(m_gate(i) - m_gate_h(i)) > 1.0e-6_real64) then
      print '(a)', 'FAIL'
      stop 0
    end if
  end do
  print '(a)', 'PASS'
end program goulash
