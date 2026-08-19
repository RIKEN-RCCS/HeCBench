program main
  use qrg_reference
  implicit none
  integer :: repeat, ios, i, dim, bit, pos
  character(len=64) :: arg
  integer(int32), allocatable :: table(:)
  real(real32), allocatable :: output(:)
  integer :: sz_workgroup
  integer(int32) :: seed, result, data, distance, d
  real(real64) :: t0, t1, sum_delta, sum_ref, delta, ref, l1norm
  logical :: pass_flag
  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: main <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) repeat
  allocate(table(0:qrng_dimensions*qrng_resolution-1), output(0:qrng_dimensions*n-1))
  print '(a)', 'Initializing QRNG tables...'
  call init_quasirandom_generator(table)
  print '(a)', '>>>Launch QuasirandomGenerator kernel...'
  print *
  sz_workgroup = 64 * (256 / qrng_dimensions) / 64
  !$omp target data map(alloc:output(0:qrng_dimensions*n-1)) map(to:table(0:qrng_dimensions*qrng_resolution-1))
  seed = 0
  t0 = seconds()
  do i = 1, repeat
    !$omp target teams distribute parallel do collapse(2) private(result,data,bit) thread_limit(sz_workgroup)
    do pos = 0, n-1
      do dim = 0, qrng_dimensions-1
        result = 0; data = seed + pos
        do bit = 0, qrng_resolution-1
          if (iand(data,1_int32) /= 0) result = ieor(result, table(bit + dim*qrng_resolution))
          data = shiftr(data,1)
        end do
        output(dim*n+pos) = real(result + 1_int32, real32) * int_scale
      end do
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (qrng): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  print *
  print '(a)', 'Read back results...'
  !$omp target update from(output(0:qrng_dimensions*n-1))
  print '(a)', 'Comparing to the CPU results...'
  sum_delta = 0.0_real64; sum_ref = 0.0_real64
  do dim = 0, qrng_dimensions-1
    do pos = 0, n-1
      ref = get_quasirandom_value63(int(pos,int64), dim)
      delta = real(output(dim*n+pos),real64) - ref
      sum_delta = sum_delta + abs(delta); sum_ref = sum_ref + abs(ref)
    end do
  end do
  l1norm = sum_delta / sum_ref
  print '(a,es14.6)', '  L1 norm: ', l1norm
  print '(a,a,a)', '  ckQuasirandomGenerator deviations ', merge('WITHIN', 'ABOVE ', l1norm < 1.0e-6_real64), ' Allowable Tolerance'
  print *
  pass_flag = l1norm < 1.0e-6_real64
  print '(a)', '>>>Launch InverseCND kernel...'
  print *
  sz_workgroup = 128
  distance = int(int(z'FFFFFFFF',int64) / int(qrng_dimensions*n + 1,int64), int32)
  t0 = seconds()
  do i = 1, repeat
    !$omp target teams distribute parallel do private(d) thread_limit(sz_workgroup)
    do pos = 0, qrng_dimensions*n-1
      d = int(int(pos + 1,int64) * int(distance,int64), int32)
      output(pos) = moro_inv_cnd_gpu(d)
    end do
    !$omp end target teams distribute parallel do
  end do
  t1 = seconds()
  print '(a,f10.6,a)', 'Average kernel execution time (icnd): ', ((t1-t0)*1.0e6_real64)/real(repeat,real64), ' (us)'
  print '(a)', 'Read back results...'
  !$omp target update from(output(0:qrng_dimensions*n-1))
  print '(a)', 'Comparing to the CPU results...'
  sum_delta = 0.0_real64; sum_ref = 0.0_real64
  do pos = 0, qrng_dimensions*n-1
    d = int(int(pos + 1,int64) * int(distance,int64), int32)
    ref = moro_inv_cnd_cpu(d)
    delta = real(output(pos),real64) - ref
    sum_delta = sum_delta + abs(delta); sum_ref = sum_ref + abs(ref)
  end do
  l1norm = sum_delta / sum_ref
  print '(a,es14.6)', '  L1 norm: ', l1norm
  print '(a,a,a)', '  ckInverseCNDGPU deviations ', merge('WITHIN', 'ABOVE ', l1norm < 1.0e-6_real64), ' Allowable Tolerance'
  pass_flag = pass_flag .and. (l1norm < 1.0e-6_real64)
  print '(a)', merge('PASS', 'FAIL', pass_flag)
  !$omp end target data
end program
