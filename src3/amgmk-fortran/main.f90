program amgmk
  use omp_lib
  use amgmk_types
  use amgmk_vector
  use amgmk_csr_matrix
  use amgmk_laplace
  use amgmk_csr_matvec
  use amgmk_relax
  implicit none
  type(csr_matrix) :: a
  type(seq_vector) :: rhs, x, sol, y
  integer :: iter, max_threads
  real(real64) :: start_time, elapsed, error

  write(*,'(a)') ''
  write(*,'(a)') '//------------ '
  write(*,'(a)') '//  CORAL  AMGmk Benchmark Version 1.0 '
  write(*,'(a)') '//------------ '
  write(*,'(a,i0)') ' testIter   = ', test_iter
  !$omp parallel
  !$omp master
  max_threads = omp_get_num_threads()
  !$omp end master
  !$omp end parallel
  write(*,'(a,i0)') 'max_num_threads = ', max_threads

  call generate_seq_laplacian(a, rhs, x, sol)
  call vector_create(y, nrows)
  call vector_set_constant(x, 1.0_real64)
  call vector_set_constant(y, 0.0_real64)
  start_time = omp_get_wtime()
  do iter = 1, test_iter
    call csr_matvec(1.0_real64, a, x, 0.0_real64, y)
  end do
  elapsed = omp_get_wtime() - start_time
  write(*,'(a)') '//------------ '
  write(*,'(a)') '//   MATVEC'
  write(*,'(a,f0.6,a)') 'Wall time = ', elapsed, ' seconds.'

  call vector_set_constant(x, 1.0_real64)
  start_time = omp_get_wtime()
  do iter = 1, test_iter
    call relax_gpu(a%data, a%row_ptr, a%col_ind, sol%data, x%data, a%num_nonzeros)
  end do
  elapsed = omp_get_wtime() - start_time
  error = maxval(abs(x%data - 1.0_real64))
  if (error > 1.0e-10_real64) write(*,'(a,es14.6)') ' Relax: error: ', error
  write(*,'(a)') '//------------ '
  write(*,'(a)') '//   Relax'
  write(*,'(a,f0.6,a)') 'Wall time = ', elapsed, ' seconds.'

  call vector_set_constant(x, 1.0_real64)
  call vector_set_constant(y, 1.0_real64)
  start_time = omp_get_wtime()
  do iter = 1, test_iter
    y%data = y%data + 0.5_real64 * x%data
  end do
  elapsed = omp_get_wtime() - start_time
  error = maxval(abs(y%data - (1.0_real64 + 0.5_real64 * real(test_iter,real64))))
  if (error > 1.0e-10_real64) write(*,'(a,es14.6)') ' Axpy: error: ', error
  write(*,'(a)') '//------------ '
  write(*,'(a)') '//   Axpy'
  write(*,'(a,f0.6,a)') 'Wall time = ', elapsed, ' seconds.'

  call vector_destroy(rhs); call vector_destroy(x); call vector_destroy(sol); call vector_destroy(y)
  call csr_destroy(a)
end program amgmk
