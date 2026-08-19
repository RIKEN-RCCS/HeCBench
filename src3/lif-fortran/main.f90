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

module lif_kernels
  use iso_fortran_env, only: real32
  implicit none
contains
  subroutine reference(num_neurons, neurons_per_item, dt, encode_result, voltage_array, reftime_array, tau_rc, tau_ref, bias, gain, spikes)
    integer, intent(in) :: num_neurons, neurons_per_item
    real(real32), intent(in) :: dt, tau_rc, tau_ref
    real(real32), intent(in) :: encode_result(0:), bias(0:), gain(0:)
    real(real32), intent(inout) :: voltage_array(0:), reftime_array(0:)
    real(real32), intent(out) :: spikes(0:)
    integer :: i
!$omp parallel do
    do i = 0, num_neurons-1
      call lif_cell(i, neurons_per_item, dt, encode_result, voltage_array, reftime_array, tau_rc, tau_ref, bias, gain, spikes)
    end do
!$omp end parallel do
  end subroutine reference

  subroutine test(num_neurons, neurons_per_item, dt, encode_result, voltage_array, reftime_array, tau_rc, tau_ref, bias, gain, spikes)
    integer, intent(in) :: num_neurons, neurons_per_item
    real(real32), intent(in) :: dt, tau_rc, tau_ref
    real(real32), intent(in) :: encode_result(0:), bias(0:), gain(0:)
    real(real32), intent(inout) :: voltage_array(0:), reftime_array(0:)
    real(real32), intent(out) :: spikes(0:)
    integer :: i
!$omp target teams distribute parallel do thread_limit(256)
    do i = 0, num_neurons-1
      call lif_cell(i, neurons_per_item, dt, encode_result, voltage_array, reftime_array, tau_rc, tau_ref, bias, gain, spikes)
    end do
!$omp end target teams distribute parallel do
  end subroutine test

  subroutine lif_cell(i, neurons_per_item, dt, encode_result, voltage_array, reftime_array, tau_rc, tau_ref, bias, gain, spikes)
    integer, intent(in) :: i, neurons_per_item
    real(real32), intent(in) :: dt, tau_rc, tau_ref
    real(real32), intent(in) :: encode_result(0:), bias(0:), gain(0:)
    real(real32), intent(inout) :: voltage_array(0:), reftime_array(0:)
    real(real32), intent(out) :: spikes(0:)
    integer :: neuron_index, item_index
    real(real32) :: voltage, ref_time, current, dv, spike, mult
    neuron_index = mod(i, neurons_per_item)
    item_index = i / neurons_per_item
    voltage = voltage_array(i)
    ref_time = reftime_array(i)
    current = bias(neuron_index) + gain(neuron_index) * encode_result(item_index)
    dv = -expm1(-dt / tau_rc) * (current - voltage)
    voltage = max(voltage + dv, 0.0_real32)
    ref_time = ref_time - dt
    mult = ref_time
    mult = mult * (-1.0_real32 / dt)
    mult = mult + 1.0_real32
    mult = min(mult, 1.0_real32)
    mult = max(mult, 0.0_real32)
    voltage = voltage * mult
    if (voltage > 1.0_real32) then
      spike = 1.0_real32 / dt
      ref_time = tau_ref + dt * (1.0_real32 - (voltage - 1.0_real32) / dv)
      voltage = 0.0_real32
    else
      spike = 0.0_real32
    end if
    reftime_array(i) = ref_time
    voltage_array(i) = voltage
    spikes(i) = spike
  end subroutine lif_cell
end module lif_kernels

program main
  use iso_fortran_env, only: real32
  use omp_lib
  use c_rng
  use lif_kernels
  implicit none
  integer :: neurons_per_item, num_items, num_steps, num_neurons, step, i
  character(len=64) :: arg
  real(real32), allocatable :: encode_result(:), bias(:), gain(:), voltage(:), reftime(:), spikes(:)
  real(real32), allocatable :: voltage_gold(:), reftime_gold(:), spikes_gold(:)
  real(real32) :: dt, tau_rc, tau_ref
  real(8) :: start_time, elapsed
  logical :: ok

  if (command_argument_count() /= 3) then
    print '(a)', 'Usage: ./main <neurons per item> <num_items> <num_steps>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) neurons_per_item
  call get_command_argument(2,arg); read(arg,*) num_items
  call get_command_argument(3,arg); read(arg,*) num_steps
  num_neurons = neurons_per_item * num_items
  dt = 0.1_real32
  tau_rc = 10.0_real32
  tau_ref = 2.0_real32
  allocate(encode_result(0:num_items-1), bias(0:neurons_per_item-1), gain(0:neurons_per_item-1))
  allocate(voltage(0:num_neurons-1), reftime(0:num_neurons-1), spikes(0:num_neurons-1))
  allocate(voltage_gold(0:num_neurons-1), reftime_gold(0:num_neurons-1), spikes_gold(0:num_neurons-1))

  call srand(123)
  do i = 0, num_items-1
    encode_result(i) = rand_unit()
  end do
  do i = 0, num_neurons-1
    voltage(i) = 1.0_real32 + rand_unit()
    voltage_gold(i) = voltage(i)
    reftime(i) = real(mod(rand(), 5), real32) / 10.0_real32
    reftime_gold(i) = reftime(i)
  end do
  do i = 0, neurons_per_item-1
    bias(i) = rand_unit()
    gain(i) = rand_unit() + 0.5_real32
  end do

!$omp target data map(to:encode_result(0:num_items-1),bias(0:neurons_per_item-1),gain(0:neurons_per_item-1)) map(from:spikes(0:num_neurons-1)) map(tofrom:voltage(0:num_neurons-1),reftime(0:num_neurons-1))
  start_time = omp_get_wtime()
  do step = 1, num_steps
    call test(num_neurons, neurons_per_item, dt, encode_result, voltage, reftime, tau_rc, tau_ref, bias, gain, spikes)
  end do
  elapsed = omp_get_wtime() - start_time
  print '(a,f0.6,a)', 'Average kernel execution time: ', (elapsed * 1.0e6) / num_steps, ' (us)'
!$omp end target data

  do step = 1, num_steps
    call reference(num_neurons, neurons_per_item, dt, encode_result, voltage_gold, reftime_gold, tau_rc, tau_ref, bias, gain, spikes_gold)
  end do
  ok = .true.
  do i = 0, num_neurons-1
    if (abs(spikes(i) - spikes_gold(i)) > 1.0e-3_real32) then
      print '(a,i0,a,f0.6,1x,f0.6)', '@', i, ': ', spikes(i), spikes_gold(i)
      ok = .false.
      exit
    end if
  end do
  print '(a)', merge('PASS', 'FAIL', ok)
end program main
