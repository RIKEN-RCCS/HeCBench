program secp256k1
  use iso_fortran_env, only: real64
  use omp_lib
  use secp256k1_mod
  use secp256k1_prec_mod, only: init_prec
  implicit none
  integer :: argc, repeat, n, i
  type(secp256k1_ge_storage) :: prec(0:511)
  integer :: output(0:31)
  character(len=64) :: arg
  character(len=64) :: result
  character(len=*), parameter :: hex = '0123456789abcdef'
  real(real64) :: start_time, end_time

  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *) repeat
  call init_prec(prec)
  !$omp target data map(to: prec(0:511)) map(from: output(0:31))
    start_time = omp_get_wtime()
    do n = 0, repeat - 1
      call secp256k1_kernel(prec, output)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time: ', (end_time-start_time)/repeat, ' (s)'
  !$omp end target data
  result = ''
  do i = 0, 31
    result(2*i+1:2*i+1) = hex(output(i)/16 + 1:output(i)/16 + 1)
    result(2*i+2:2*i+2) = hex(mod(output(i), 16) + 1:mod(output(i), 16) + 1)
  end do
  print '(a,a)', 'result = ', trim(result)
  if (trim(result) == 'bbde464b6355ee6de6deba5ae860f8a66524937eee81dde224a0214efd795d09') then
    print '(a)', 'PASS'
  else
    print '(a)', 'FAIL'
  end if
end program secp256k1
