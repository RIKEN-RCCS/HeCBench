module divergence_module
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  integer, parameter :: np = 4, dims = 2, block_size = 16
  real(real64), parameter :: rrearth = 1.5683814303638645e-7_real64
  type :: element_type
    real(real64) :: metdet(0:np*np-1), dinv(0:np*np*dims*dims-1), rmetdet(0:np*np-1)
  end type element_type
  type :: derivative_type
    real(real64) :: dvv(0:np*np-1)
  end type derivative_type
contains
  subroutine read_velocity(v, unit)
    real(real64), intent(out) :: v(0:np*np*dims-1)
    integer, intent(in) :: unit
    integer :: i, j, k
    do i = 0, dims-1
      do j = 0, np-1
        do k = 0, np-1
          read(unit, *) v(k*np*dims + dims*j + i)
        end do
      end do
    end do
  end subroutine read_velocity

  subroutine read_element(elem, unit)
    type(element_type), intent(out) :: elem
    integer, intent(in) :: unit
    integer :: i, j, k, l
    do i = 0, np-1
      do j = 0, np-1
        read(unit, *) elem%metdet(i*np+j)
        elem%rmetdet(i*np+j) = 1.0_real64 / elem%metdet(i*np+j)
      end do
    end do
    do i = 0, dims-1
      do j = 0, dims-1
        do k = 0, np-1
          do l = 0, np-1
            read(unit, *) elem%dinv(4*np*l + 4*k + dims*i + j)
          end do
        end do
      end do
    end do
  end subroutine read_element

  subroutine read_derivative(deriv, unit)
    type(derivative_type), intent(out) :: deriv
    integer, intent(in) :: unit
    integer :: i, j
    do i = 0, np-1
      do j = 0, np-1
        read(unit, *) deriv%dvv(j*np+i)
      end do
    end do
  end subroutine read_derivative

  subroutine read_divergence(divergence, unit)
    real(real64), intent(out) :: divergence(0:np*np-1)
    integer, intent(in) :: unit
    integer :: i, j
    do i = 0, np-1
      do j = 0, np-1
        read(unit, *) divergence(i*np+j)
      end do
    end do
  end subroutine read_divergence

  subroutine divergence_sphere_cpu(v, deriv, elem, divergence)
    real(real64), intent(in) :: v(0:np*np*dims-1)
    type(derivative_type), intent(in) :: deriv
    type(element_type), intent(in) :: elem
    real(real64), intent(out) :: divergence(0:np*np-1)
    real(real64) :: gv(0:np*np*dims-1), vvtemp(0:np*np-1), dudx00, dvdy00
    integer :: i, j, k, l
    do j = 0, np-1
      do i = 0, np-1
        do k = 0, dims-1
          gv(j*np*dims+i*dims+k) = elem%metdet(j*np+i) * &
            (elem%dinv(j*np*dims*dims+i*dims*dims+k*dims)*v(j*dims*np+dims*i) + &
             elem%dinv(j*np*dims*dims+i*dims*dims+k*dims+1)*v(j*dims*np+dims*i+1))
        end do
      end do
    end do
    do l = 0, np-1
      do j = 0, np-1
        dudx00 = 0.0_real64; dvdy00 = 0.0_real64
        do i = 0, np-1
          dudx00 = dudx00 + deriv%dvv(l*np+i)*gv(j*np*dims+i*dims)
          dvdy00 = dvdy00 + deriv%dvv(l*np+i)*gv(i*np*dims+j*dims+1)
        end do
        divergence(j*np+l) = dudx00
        vvtemp(l*np+j) = dvdy00
      end do
    end do
    do i = 0, np-1
      do j = 0, np-1
        divergence(i*np+j) = (divergence(i*np+j)+vvtemp(i*np+j))*elem%rmetdet(i*np+j)*rrearth
      end do
    end do
  end subroutine divergence_sphere_cpu

  subroutine divergence_sphere_gpu(v, deriv, elem, divergence)
    real(real64), intent(in) :: v(0:np*np*dims-1)
    type(derivative_type), intent(in) :: deriv
    type(element_type), intent(in) :: elem
    real(real64), intent(out) :: divergence(0:np*np-1)
    real(real64) :: gv(0:np*np*dims-1), vvtemp(0:np*np-1), dudx00, dvdy00
    integer :: i, j, k, l
    do j = 0, np-1
      do i = 0, np-1
        do k = 0, dims-1
          gv(j*np*dims+i*dims+k) = elem%metdet(j*np+i) * &
            (elem%dinv(j*np*dims*dims+i*dims*dims+k*dims)*v(j*dims*np+dims*i) + &
             elem%dinv(j*np*dims*dims+i*dims*dims+k*dims+1)*v(j*dims*np+dims*i+1))
        end do
      end do
    end do
!$omp target data map(to:gv,deriv%dvv,elem%rmetdet) map(from:divergence) map(alloc:vvtemp)
!$omp target teams distribute parallel do collapse(2) thread_limit(block_size*block_size) private(dudx00,dvdy00,i)
    do j = 0, np-1
      do l = 0, np-1
        dudx00 = 0.0_real64; dvdy00 = 0.0_real64
        do i = 0, np-1
          dudx00 = dudx00 + deriv%dvv(l*np+i)*gv(j*np*dims+i*dims)
          dvdy00 = dvdy00 + deriv%dvv(l*np+i)*gv(i*np*dims+j*dims+1)
        end do
        divergence(j*np+l) = dudx00
        vvtemp(l*np+j) = dvdy00
      end do
    end do
!$omp end target teams distribute parallel do
!$omp target teams distribute parallel do collapse(2) thread_limit(block_size*block_size)
    do l = 0, np-1
      do j = 0, np-1
        divergence(l*np+j) = (divergence(l*np+j)+vvtemp(l*np+j))*elem%rmetdet(l*np+j)*rrearth
      end do
    end do
!$omp end target teams distribute parallel do
!$omp end target data
  end subroutine divergence_sphere_gpu
end module divergence_module

program divergence
  use, intrinsic :: iso_fortran_env, only : real64
  use omp_lib
  use divergence_module
  implicit none
  real(real64) :: v(0:np*np*dims-1), divergence_expected(0:np*np-1)
  real(real64) :: divergence_cpu(0:np*np-1), divergence_gpu(0:np*np-1)
  type(element_type) :: elem
  type(derivative_type) :: deriv
  integer :: argc, unit, numtests, iteration, i, j
  real(real64) :: start_time, stop_time, cpu_time_ms, gpu_time_ms
  character(len=512) :: filename, argument
  argc = command_argument_count()
  if (argc > 0) then
    call get_command_argument(1, filename)
    open(newunit=unit, file=trim(filename), status='old', action='read')
  else
    unit = 5
  end if
  call read_velocity(v, unit); call read_element(elem, unit); call read_derivative(deriv, unit)
  call read_divergence(divergence_expected, unit)
  if (argc > 0) close(unit)
  numtests = 100000
  if (argc > 1) then; call get_command_argument(2, argument); read(argument, *) numtests; end if
  print '(a)', 'Divergence on the CPU'
  do iteration = 1, numtests; call divergence_sphere_cpu(v, deriv, elem, divergence_cpu); end do
  start_time = omp_get_wtime()
  do iteration = 1, numtests; call divergence_sphere_cpu(v, deriv, elem, divergence_cpu); end do
  stop_time = omp_get_wtime(); cpu_time_ms = (stop_time-start_time)*1000.0_real64
  print '(a)', 'Divergence on the GPU'
  do iteration = 1, numtests; call divergence_sphere_gpu(v, deriv, elem, divergence_gpu); end do
  start_time = omp_get_wtime()
  do iteration = 1, numtests; call divergence_sphere_gpu(v, deriv, elem, divergence_gpu); end do
  stop_time = omp_get_wtime(); gpu_time_ms = (stop_time-start_time)*1000.0_real64
  print '(a)', 'Divergence Errors'
  print '(a)', 'CPU             GPU'
  do i = 0, np-1
    do j = 0, np-1
      print '(es16.8,a,es16.8)', divergence_cpu(i*np+j)-divergence_expected(i*np+j), '    ', &
        divergence_gpu(i*np+j)-divergence_expected(i*np+j)
    end do
    print '(a)', ''
  end do
  print '(a,f0.6,a)', 'Total CPU Time: ', cpu_time_ms, ' (ms)'
  print '(a,f0.6,a)', 'Total GPU Time: ', gpu_time_ms, ' (ms)'
end program divergence
