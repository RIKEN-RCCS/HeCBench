module pointwise_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  type :: checksum
    real(real64) :: i, c, h
  end type
contains
  real(real64) function seconds() result(t)
    integer(int64) :: c, r
    call system_clock(c, r)
    t = real(c, real64) / real(r, real64)
  end function

  real(real32) function lcg_random(seed) result(v)
    integer(int32), intent(inout) :: seed
    integer(int64), parameter :: m = 2147483648_int64, a = 26757677_int64, c = 1_int64
    seed = int(mod(a * int(seed,int64) + c, m), int32)
    v = real(seed, real32) / real(m, real32)
  end function

  subroutine init(data, size)
    integer, intent(in) :: size
    real(real32), intent(out) :: data(0:size-1)
    integer :: index
    integer(int32) :: seed
    !$omp target teams distribute parallel do private(seed) thread_limit(256)
    do index = 0, size-1
      seed = ieor(int(index,int32), int(size,int32))
      data(index) = lcg_random(seed)
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine elementwise(hidden_size, mini_batch, tmp_h, tmp_i, bias, linear_gates, h_out, i_out, c_in, c_out)
    integer, intent(in) :: hidden_size, mini_batch
    real(real32), intent(in) :: tmp_h(0:), tmp_i(0:), bias(0:), c_in(0:)
    real(real32), intent(inout) :: linear_gates(0:), h_out(0:), i_out(0:), c_out(0:)
    integer :: num_elements, index, batch, gate_index, gidx
    real(real32) :: g0, g1, g2, g3, in_gate, forget_gate, in_gate2, out_gate, val
    num_elements = mini_batch * hidden_size
    !$omp target teams distribute parallel do private(batch,gate_index,gidx,g0,g1,g2,g3,in_gate,forget_gate,in_gate2,out_gate,val) thread_limit(256)
    do index = 0, num_elements-1
      batch = index / hidden_size
      gate_index = mod(index, hidden_size) + 4 * batch * hidden_size
      gidx = mod(index, hidden_size)
      g0 = tmp_i(gate_index) + tmp_h(gate_index) + bias(gidx) + bias(4*hidden_size + gidx)
      g1 = tmp_i(hidden_size + gate_index) + tmp_h(hidden_size + gate_index) + bias(hidden_size + gidx) + bias(5*hidden_size + gidx)
      g2 = tmp_i(2*hidden_size + gate_index) + tmp_h(2*hidden_size + gate_index) + bias(2*hidden_size + gidx) + bias(6*hidden_size + gidx)
      g3 = tmp_i(3*hidden_size + gate_index) + tmp_h(3*hidden_size + gate_index) + bias(3*hidden_size + gidx) + bias(7*hidden_size + gidx)
      linear_gates(gate_index) = g0
      linear_gates(gate_index + hidden_size) = g1
      linear_gates(gate_index + 2*hidden_size) = g2
      linear_gates(gate_index + 3*hidden_size) = g3
      in_gate = 1.0_real32 / (1.0_real32 + exp(-g0))
      forget_gate = 1.0_real32 / (1.0_real32 + exp(-g1))
      in_gate2 = tanh(g2)
      out_gate = 1.0_real32 / (1.0_real32 + exp(-g3))
      val = forget_gate * c_in(index) + in_gate * in_gate2
      c_out(index) = val
      val = out_gate * tanh(val)
      h_out(index) = val
      i_out(index) = val
    end do
    !$omp end target teams distribute parallel do
  end subroutine

  subroutine test(hidden_size, mini_batch, seq_length, num_layers, cs, time)
    integer, intent(in) :: hidden_size, mini_batch, seq_length, num_layers
    type(checksum), intent(out) :: cs
    real(real64), intent(inout) :: time
    integer :: num_elements, hc_size, i_size, bias_size, tmp_h_size, tmp_i_size, act_size
    integer :: lstart, lend, rstart, rend, recur_batch_size, layer, step, m, j, k
    real(real64) :: ktime, t0, t1
    real(real32), allocatable :: h_data(:), i_data(:), c_data(:), bias(:), tmp_h(:), tmp_i(:), linear_gates(:)
    real(real32), allocatable :: test_output_i(:), test_output_h(:), test_output_c(:)
    num_elements = hidden_size * mini_batch
    hc_size = (seq_length + 1) * num_layers * num_elements
    i_size = seq_length * (num_layers + 1) * num_elements
    bias_size = num_layers * hidden_size * 8
    tmp_h_size = 4 * num_layers * num_elements
    tmp_i_size = 4 * seq_length * num_elements
    act_size = 4 * seq_length * num_layers * num_elements
    allocate(h_data(0:hc_size-1), i_data(0:i_size-1), c_data(0:hc_size-1), bias(0:bias_size-1), &
      tmp_h(0:tmp_h_size-1), tmp_i(0:tmp_i_size-1), linear_gates(0:act_size-1))

    !$omp target data map(alloc:h_data(0:hc_size-1),i_data(0:i_size-1),c_data(0:hc_size-1),bias(0:bias_size-1),tmp_h(0:tmp_h_size-1),tmp_i(0:tmp_i_size-1),linear_gates(0:act_size-1))
    call init(tmp_h, tmp_h_size); call init(tmp_i, tmp_i_size); call init(c_data, hc_size); call init(bias, bias_size)
    lstart = 0; lend = 0; rstart = 0; rend = 0; recur_batch_size = 2; ktime = 0.0_real64
    do
      if (lend == 0) then
        lstart = 0; lend = 1; rstart = 0
      else
        lstart = lstart + 1; lend = lend + 1; rstart = rstart - recur_batch_size
        if (lend > num_layers .or. rstart < 0) then
          rstart = rstart + (lstart + 1) * recur_batch_size
          lstart = 0; lend = 1
        end if
        do while (rstart >= seq_length .and. lend <= num_layers)
          lstart = lstart + 1; lend = lend + 1; rstart = rstart - recur_batch_size
        end do
        if (lend > num_layers .or. rstart < 0) exit
      end if
      rend = min(rstart + recur_batch_size, seq_length)
      t0 = seconds()
      do layer = lstart, lend-1
        do step = rstart, rend-1
          call elementwise(hidden_size, mini_batch, &
            tmp_h(4*layer*num_elements:), tmp_i(4*step*num_elements:), bias(8*layer*hidden_size:), &
            linear_gates(4*(step*num_elements + layer*seq_length*num_elements):), &
            h_data((step+1)*num_elements + layer*(seq_length+1)*num_elements:), &
            i_data(step*num_elements + (layer+1)*seq_length*num_elements:), &
            c_data(step*num_elements + layer*(seq_length+1)*num_elements:), &
            c_data((step+1)*num_elements + layer*(seq_length+1)*num_elements:))
        end do
      end do
      t1 = seconds(); ktime = ktime + (t1-t0)*1.0e9_real64
    end do
    time = time + ktime
    !$omp target update from(i_data(0:i_size-1))
    !$omp target update from(h_data(0:hc_size-1))
    !$omp target update from(c_data(0:hc_size-1))
    !$omp end target data

    allocate(test_output_i(0:num_elements*seq_length-1), test_output_h(0:num_elements*num_layers-1), test_output_c(0:num_elements*num_layers-1))
    test_output_i = i_data(num_layers*seq_length*num_elements:num_layers*seq_length*num_elements+seq_length*num_elements-1)
    do layer = 0, num_layers-1
      test_output_h(layer*num_elements:(layer+1)*num_elements-1) = h_data(seq_length*num_elements + layer*(seq_length+1)*num_elements:seq_length*num_elements + layer*(seq_length+1)*num_elements + num_elements-1)
      test_output_c(layer*num_elements:(layer+1)*num_elements-1) = c_data(seq_length*num_elements + layer*(seq_length+1)*num_elements:seq_length*num_elements + layer*(seq_length+1)*num_elements + num_elements-1)
    end do
    cs%i = 0.0_real64; cs%h = 0.0_real64; cs%c = 0.0_real64
    do m = 0, mini_batch-1
      do j = 0, seq_length-1
        do k = 0, hidden_size-1
          cs%i = cs%i + test_output_i(j*num_elements + m*hidden_size + k)
        end do
      end do
      do j = 0, num_layers-1
        do k = 0, hidden_size-1
          cs%h = cs%h + test_output_h(j*num_elements + m*hidden_size + k)
          cs%c = cs%c + test_output_c(j*num_elements + m*hidden_size + k)
        end do
      end do
    end do
  end subroutine
end module

program main
  use pointwise_mod
  implicit none
  integer :: seq_length, num_layers, hidden_size, mini_batch, num_runs, run, ios
  character(len=64) :: arg
  type(checksum) :: cs
  real(real64) :: time
  if (command_argument_count() == 5) then
    call get_command_argument(1,arg); read(arg,*,iostat=ios) seq_length
    call get_command_argument(2,arg); read(arg,*,iostat=ios) num_layers
    call get_command_argument(3,arg); read(arg,*,iostat=ios) hidden_size
    call get_command_argument(4,arg); read(arg,*,iostat=ios) mini_batch
    call get_command_argument(5,arg); read(arg,*,iostat=ios) num_runs
  else if (command_argument_count() == 0) then
    print '(a)', 'Running with default settings'
    seq_length = 100; num_layers = 4; hidden_size = 512; mini_batch = 64; num_runs = 1
  else
    print '(a)', 'Usage: main <seqLength> <numLayers> <hiddenSize> <miniBatch> <repeat>'
    stop 1
  end if
  print '(a,i0,a,i0,a,i0,a,i0)', 'seqLength ', seq_length, ', numLayers ', num_layers, &
    ', hiddenSize ', hidden_size, ', miniBatch ', mini_batch
  time = 0.0_real64
  do run = 1, num_runs
    call test(hidden_size, mini_batch, seq_length, num_layers, cs, time)
  end do
  print '(a,f10.6,a)', 'Average kernel execution time: ', (time*1.0e-9_real64)/real(num_runs,real64), ' (s)'
  print '(a,es14.6,a,es14.6,a,es14.6)', 'i checksum ', cs%i, '     c checksum ', cs%c, '     h checksum ', cs%h
end program
