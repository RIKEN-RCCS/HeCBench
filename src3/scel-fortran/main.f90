program scel
  use iso_fortran_env, only: real32, real64
  use omp_lib
  use scel_mod
  implicit none

  integer :: argc, outer_size, inner_size, repeat, input_size, output_size
  integer :: i, j, unjoined_lr_loss, log_d, log_d_trick
  real(real32), allocatable :: h_logits(:), h_targets(:), h_out(:), r_out(:)
  real(real64) :: start_time, end_time
  logical :: ok, mode_ok
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage: ./main <outer size> <inner_size> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) outer_size
  call get_command_argument(2, arg); read(arg, *) inner_size
  call get_command_argument(3, arg); read(arg, *) repeat

  input_size = (outer_size + 1) * inner_size
  output_size = outer_size
  allocate(h_logits(0:input_size-1), h_targets(0:input_size-1), h_out(0:output_size-1), r_out(0:output_size-1))

  call random_seed()
  do i = 0, input_size - 1
    call normal_pair(h_logits(i), h_targets(i))
    h_targets(i) = h_targets(i) + 1.0_real32
  end do

  ok = .true.
  !$omp target data map(to: h_logits(0:input_size-1), h_targets(0:input_size-1)) map(alloc: h_out(0:output_size-1))
    do unjoined_lr_loss = 0, 1
      log_d = merge(1, 0, unjoined_lr_loss == 0)
      do log_d_trick = 0, log_d
        start_time = omp_get_wtime()
        do i = 0, repeat - 1
          call sigmoid_cross_entropy_with_logits_kernel(outer_size, inner_size, log_d_trick /= 0, unjoined_lr_loss /= 0, h_logits, h_targets, h_out)
        end do
        end_time = omp_get_wtime()
        print '(a,f12.6,a)', 'Average execution time of SigmoidCrossEntropyWithLogits kernel: ', ((end_time-start_time)*1.0e6_real64)/repeat, ' (us)'

        !$omp target update from(h_out(0:output_size-1))
        call reference(outer_size, inner_size, log_d_trick /= 0, unjoined_lr_loss /= 0, h_logits, h_targets, r_out)
        mode_ok = .true.
        do j = 0, output_size - 1
          if (abs(r_out(j) - h_out(j)) > 1.0e-3_real32) then
            mode_ok = .false.
            exit
          end if
        end do
        if (.not. mode_ok) ok = .false.
      end do
    end do
  !$omp end target data

  if (ok) then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if

contains

  subroutine normal_pair(a, b)
    real(real32), intent(out) :: a, b
    real(real32) :: u1, u2, r
    call random_number(u1)
    call random_number(u2)
    u1 = max(u1, tiny(1.0_real32))
    r = sqrt(-2.0_real32 * log(u1))
    a = r * cos(6.283185307179586_real32 * u2)
    b = r * sin(6.283185307179586_real32 * u2)
  end subroutine normal_pair

end program scel
