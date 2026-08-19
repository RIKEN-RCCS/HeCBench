module backprop_fortran
  use, intrinsic :: iso_c_binding, only : c_int
  use, intrinsic :: iso_fortran_env, only : real32, real64
  implicit none

  integer, parameter :: sp = real32
  integer, parameter :: width = 16, height = 16, block_size = 16
  real(sp), parameter :: eta = 0.3_sp, momentum = 0.3_sp

  interface
    subroutine c_srand(seed) bind(C, name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface

contains

  pure integer function widx(i, j, hid) result(index)
    integer, intent(in) :: i, j, hid
    index = i * (hid + 1) + j
  end function widx

  real(sp) function random_weight() result(value)
    value = real(c_rand(), sp) / real(2147483647, sp)
  end function random_weight

  pure real(sp) function squash(value) result(result_value)
    real(sp), intent(in) :: value
    result_value = 1.0_sp / (1.0_sp + exp(-value))
  end function squash

  subroutine layerforward(l1, l2, conn, n1, n2)
    integer, intent(in) :: n1, n2
    real(sp), intent(inout) :: l1(0:), l2(0:)
    real(sp), intent(in) :: conn(0:)
    integer :: j, k
    real(sp) :: sum

    l1(0) = 1.0_sp
    !$omp parallel do private(k, sum) schedule(static)
    do j = 1, n2
      sum = 0.0_sp
      do k = 0, n1
        sum = sum + conn(widx(k, j, n2)) * l1(k)
      end do
      l2(j) = squash(sum)
    end do
    !$omp end parallel do
  end subroutine layerforward

  subroutine output_error(delta, target, output, nj, err)
    integer, intent(in) :: nj
    real(sp), intent(out) :: delta(0:), err
    real(sp), intent(in) :: target(0:), output(0:)
    integer :: j
    real(sp) :: o, t, errsum

    errsum = 0.0_sp
    do j = 1, nj
      o = output(j)
      t = target(j)
      delta(j) = o * (1.0_sp - o) * (t - o)
      errsum = errsum + abs(delta(j))
    end do
    err = errsum
  end subroutine output_error

  subroutine hidden_error(delta_h, nh, delta_o, no, who, hidden, err)
    integer, intent(in) :: nh, no
    real(sp), intent(out) :: delta_h(0:), err
    real(sp), intent(in) :: delta_o(0:), who(0:), hidden(0:)
    integer :: j, k
    real(sp) :: h, sum, errsum

    errsum = 0.0_sp
    do j = 1, nh
      h = hidden(j)
      sum = 0.0_sp
      do k = 1, no
        sum = sum + delta_o(k) * who(widx(j, k, no))
      end do
      delta_h(j) = h * (1.0_sp - h) * sum
      errsum = errsum + abs(delta_h(j))
    end do
    err = errsum
  end subroutine hidden_error

  subroutine adjust_weights(delta, ndelta, ly, nly, weights, oldweights)
    integer, intent(in) :: ndelta, nly
    real(sp), intent(in) :: delta(0:)
    real(sp), intent(inout) :: ly(0:), weights(0:), oldweights(0:)
    integer :: j, k, index
    real(sp) :: new_dw

    ly(0) = 1.0_sp
    !$omp parallel do private(k, index, new_dw)
    do j = 1, ndelta
      do k = 0, nly
        index = widx(k, j, ndelta)
        new_dw = eta * delta(j) * ly(k) + momentum * oldweights(index)
        weights(index) = weights(index) + new_dw
        oldweights(index) = new_dw
      end do
    end do
    !$omp end parallel do
  end subroutine adjust_weights

  subroutine layerforward_reference(input, input_weights, partial_sum, num_blocks, hid)
    integer, intent(in) :: num_blocks, hid
    real(sp), intent(in) :: input(0:)
    real(sp), intent(inout) :: input_weights(0:), partial_sum(0:)
    real(sp) :: input_node(0:height-1), matrix(0:height-1, 0:width-1)
    integer :: by, ty, tx, stride, index, index_in

    do by = 0, num_blocks - 1
      do ty = 0, block_size - 1
        do tx = 0, block_size - 1
          index = (hid + 1) * height * by + (hid + 1) * ty + tx + 1 + (hid + 1)
          index_in = height * by + ty + 1
          if (tx == 0) input_node(ty) = input(index_in)
          matrix(ty, tx) = input_weights(index)
        end do
      end do
      matrix = matrix * spread(input_node, dim=2, ncopies=width)
      stride = 1
      do while (stride <= height)
        do ty = 0, block_size - 1
          do tx = 0, block_size - 1
            if (mod(ty, stride) == 0) matrix(ty, tx) = matrix(ty, tx) + matrix(ty + stride / 2, tx)
          end do
        end do
        stride = stride * 2
      end do
      do ty = 0, block_size - 1
        do tx = 0, block_size - 1
          index = (hid + 1) * height * by + (hid + 1) * ty + tx + 1 + (hid + 1)
          input_weights(index) = matrix(ty, tx)
        end do
      end do
      do ty = 0, block_size - 1
        partial_sum(by * hid + ty) = matrix(0, ty)
      end do
    end do
  end subroutine layerforward_reference

  subroutine adjust_weights_reference(ly, weights, delta, oldweights, num_blocks, hid)
    integer, intent(in) :: num_blocks, hid
    real(sp), intent(in) :: ly(0:), delta(0:)
    real(sp), intent(inout) :: weights(0:), oldweights(0:)
    integer :: by, ty, tx, index, index_y, index_x
    real(sp) :: new_dw

    do by = 0, num_blocks - 1
      do ty = 0, block_size - 1
        do tx = 0, block_size - 1
          index = (hid + 1) * height * by + (hid + 1) * ty + tx + 1 + (hid + 1)
          index_y = height * by + ty + 1
          index_x = tx + 1
          new_dw = eta * delta(index_x) * ly(index_y) + momentum * oldweights(index)
          weights(index) = weights(index) + new_dw
          oldweights(index) = new_dw
        end do
      end do
      do tx = 0, block_size - 1
        index_x = tx + 1
        new_dw = eta * delta(index_x) + momentum * oldweights(index_x)
        weights(index_x) = weights(index_x) + new_dw
        oldweights(index_x) = new_dw
      end do
    end do
  end subroutine adjust_weights_reference

  subroutine reference_execution(in, hid, out, input, input_weights, input_prev_weights, &
       partial_sum, hidden_units, output_units, hidden_delta, output_delta, target, hidden_weights, hidden_prev_weights)
    integer, intent(in) :: in, hid, out
    real(sp), intent(in) :: input(0:), input_prev_weights(0:), target(0:)
    real(sp), intent(inout) :: input_weights(0:), partial_sum(0:), hidden_units(0:), output_units(0:), &
      hidden_delta(0:), output_delta(0:), hidden_weights(0:), hidden_prev_weights(0:)
    integer :: num_blocks, j, k
    real(sp) :: sum, out_err, hid_err
    real(sp), allocatable :: host_weights(:), host_partial(:), host_prev(:)

    write(*, '(A)') 'Performing host execution '
    num_blocks = in / block_size
    allocate(host_weights(0:(in + 1) * (hid + 1) - 1), host_partial(0:num_blocks * width - 1), &
      host_prev(0:(in + 1) * (hid + 1) - 1))
    host_weights = input_weights
    call layerforward_reference(input, host_weights, host_partial, num_blocks, hid)
    partial_sum = host_partial
    do j = 1, hid
      sum = 0.0_sp
      do k = 0, num_blocks - 1
        sum = sum + partial_sum(k * hid + j - 1)
      end do
      sum = sum + input_weights(widx(0, j, hid))
      hidden_units(j) = squash(sum)
    end do
    call layerforward(hidden_units, output_units, hidden_weights, hid, out)
    call output_error(output_delta, target, output_units, out, out_err)
    call hidden_error(hidden_delta, hid, output_delta, out, hidden_weights, hidden_units, hid_err)
    call adjust_weights(output_delta, out, hidden_units, hid, hidden_weights, hidden_prev_weights)
    host_weights = input_weights
    host_prev = input_prev_weights
    call adjust_weights_reference(input, host_weights, hidden_delta, host_prev, num_blocks, hid)
    input_weights = host_weights
    deallocate(host_weights, host_partial, host_prev)
  end subroutine reference_execution

  subroutine train_kernel(in, hid, out, input, input_weights, input_prev_weights, hidden_units, output_units, &
       hidden_delta, output_delta, target, hidden_weights, hidden_prev_weights)
    integer, intent(in) :: in, hid, out
    real(sp), intent(in) :: input(0:), target(0:)
    real(sp), intent(inout) :: input_weights(0:), input_prev_weights(0:), hidden_units(0:), output_units(0:), &
      hidden_delta(0:), output_delta(0:), hidden_weights(0:), hidden_prev_weights(0:)
    integer :: num_blocks, by, ty, tx, stride, index, index_in, index_y, index_x, j, k
    integer :: weight_count, differences
    real(sp) :: sum, out_err, hid_err, new_dw
    real(real64) :: start_time, end_time
    real(sp), allocatable :: input_weights_reference(:), input_prev_weights_reference(:), partial_sum(:), scratch(:,:,:)

    num_blocks = in / block_size
    weight_count = (in + 1) * (hid + 1)
    allocate(input_weights_reference(0:weight_count-1), input_prev_weights_reference(0:weight_count-1), &
       partial_sum(0:num_blocks * width - 1), scratch(0:num_blocks-1, 0:height-1, 0:width-1))
    input_weights_reference = input_weights
    input_prev_weights_reference = input_prev_weights
    write(*, '(A)') 'Performing device offload'
    call cpu_time(start_time)

    !$omp target data map(to: input(0:in), hidden_delta(0:hid), input_prev_weights(0:weight_count-1)) &
    !$omp& map(tofrom: input_weights(0:weight_count-1)) map(alloc: partial_sum(0:num_blocks*width-1), scratch)
    !$omp target teams num_teams(num_blocks) thread_limit(block_size*block_size)
    !$omp distribute
    do by = 0, num_blocks - 1
      !$omp parallel do collapse(2) private(index, index_in)
      do ty = 0, block_size - 1
        do tx = 0, block_size - 1
          index = (hid + 1) * height * by + (hid + 1) * ty + tx + 1 + (hid + 1)
          index_in = height * by + ty + 1
          scratch(by, ty, tx) = input_weights(index) * input(index_in)
        end do
      end do
      !$omp end parallel do
      stride = 1
      do while (stride <= height)
        !$omp parallel do collapse(2)
        do ty = 0, block_size - 1
          do tx = 0, block_size - 1
            if (mod(ty, stride) == 0) scratch(by, ty, tx) = scratch(by, ty, tx) + scratch(by, ty + stride / 2, tx)
          end do
        end do
        !$omp end parallel do
        stride = stride * 2
      end do
      !$omp parallel do collapse(2) private(index)
      do ty = 0, block_size - 1
        do tx = 0, block_size - 1
          index = (hid + 1) * height * by + (hid + 1) * ty + tx + 1 + (hid + 1)
          input_weights(index) = scratch(by, ty, tx)
        end do
      end do
      !$omp end parallel do
      !$omp parallel do
      do ty = 0, block_size - 1
        partial_sum(by * hid + ty) = scratch(by, 0, ty)
      end do
      !$omp end parallel do
    end do
    !$omp end distribute
    !$omp end target teams
    !$omp target update from(partial_sum(0:num_blocks*width-1))

    do j = 1, hid
      sum = 0.0_sp
      do k = 0, num_blocks - 1
        sum = sum + partial_sum(k * hid + j - 1)
      end do
      sum = sum + input_weights(widx(0, j, hid))
      hidden_units(j) = squash(sum)
    end do
    call layerforward(hidden_units, output_units, hidden_weights, hid, out)
    call output_error(output_delta, target, output_units, out, out_err)
    call hidden_error(hidden_delta, hid, output_delta, out, hidden_weights, hidden_units, hid_err)
    call adjust_weights(output_delta, out, hidden_units, hid, hidden_weights, hidden_prev_weights)

    !$omp target update to(input_weights(0:weight_count-1))
    !$omp target teams num_teams(num_blocks) thread_limit(block_size*block_size)
    !$omp distribute
    do by = 0, num_blocks - 1
      !$omp parallel do collapse(2) private(index, index_y, index_x, new_dw)
      do ty = 0, block_size - 1
        do tx = 0, block_size - 1
          index = (hid + 1) * height * by + (hid + 1) * ty + tx + 1 + (hid + 1)
          index_y = height * by + ty + 1
          index_x = tx + 1
          new_dw = eta * hidden_delta(index_x) * input(index_y) + momentum * input_prev_weights(index)
          input_weights(index) = input_weights(index) + new_dw
          input_prev_weights(index) = new_dw
        end do
      end do
      !$omp end parallel do
      !$omp parallel do private(index_x, new_dw)
      do tx = 0, block_size - 1
        if (by == 0) then
          index_x = tx + 1
          new_dw = eta * hidden_delta(index_x) + momentum * input_prev_weights(index_x)
          input_weights(index_x) = input_weights(index_x) + new_dw
          input_prev_weights(index_x) = new_dw
        end if
      end do
      !$omp end parallel do
    end do
    !$omp end distribute
    !$omp end target teams
    !$omp end target data
    call cpu_time(end_time)
    write(*, '(A,F0.6,A)') 'Device offloading time = ', end_time - start_time, '(s)'

    call reference_execution(in, hid, out, input, input_weights_reference, input_prev_weights_reference, partial_sum, &
      hidden_units, output_units, hidden_delta, output_delta, target, hidden_weights, hidden_prev_weights)
    differences = count(abs(input_weights - input_weights_reference) >= 1.0e-3_sp)
    if (differences == 0) then
      write(*, '(A)') 'PASS'
    else
      write(*, '(A)') 'FAIL'
    end if
    deallocate(input_weights_reference, input_prev_weights_reference, partial_sum, scratch)
  end subroutine train_kernel

end module backprop_fortran

program main
  use, intrinsic :: iso_c_binding, only : c_int
  use backprop_fortran
  implicit none
  integer :: argc, in, hid, out, i, j, ios
  character(len=64) :: argument
  real(sp), allocatable :: input(:), hidden_units(:), output_units(:), hidden_delta(:), output_delta(:), target(:)
  real(sp), allocatable :: input_weights(:), hidden_weights(:), input_prev_weights(:), hidden_prev_weights(:)

  argc = command_argument_count()
  if (argc /= 1) then
    write(*, '(A)') 'Usage: ./main <number of input nodes>'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument, *, iostat=ios) in
  if (ios /= 0 .or. mod(in, 16) /= 0) then
    write(*, '(A)') 'The number of input nodes must be divided by 16'
    stop 1
  end if
  hid = 16
  out = 1
  write(*, '(A,I0)') 'Random number generator seed: ', 7
  call c_srand(7_c_int)
  allocate(input(0:in), hidden_units(0:hid), output_units(0:out), hidden_delta(0:hid), output_delta(0:out), target(0:out))
  allocate(input_weights(0:(in+1)*(hid+1)-1), input_prev_weights(0:(in+1)*(hid+1)-1))
  allocate(hidden_weights(0:(hid+1)*(out+1)-1), hidden_prev_weights(0:(hid+1)*(out+1)-1))
  do i = 0, in
    do j = 0, hid
      input_weights(widx(i, j, hid)) = random_weight()
      input_prev_weights(widx(i, j, hid)) = 0.0_sp
    end do
  end do
  do i = 0, hid
    do j = 0, out
      hidden_weights(widx(i, j, out)) = random_weight()
      hidden_prev_weights(widx(i, j, out)) = 0.0_sp
    end do
  end do
  target = 0.1_sp
  do i = 1, in
    input(i) = random_weight()
  end do
  write(*, '(A,I0)') 'Input layer size : ', in
  write(*, '(A)') 'Starting training kernel'
  call train_kernel(in, hid, out, input, input_weights, input_prev_weights, hidden_units, output_units, hidden_delta, &
    output_delta, target, hidden_weights, hidden_prev_weights)
  write(*, '(A)') ''
  write(*, '(A)') 'Finish the training for one iteration'
  deallocate(input, hidden_units, output_units, hidden_delta, output_delta, target, input_weights, input_prev_weights, hidden_weights, hidden_prev_weights)
end program main
