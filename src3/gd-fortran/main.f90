program gd
  use iso_fortran_env, only: real32, int32, int64
  use omp_lib
  use gd_data
  implicit none
  type(classification_data_crs) :: a
  character(len=1024) :: filename, arg
  integer :: argc, iterations, k, i, j, t
  integer :: correct(1)
  integer(int64) :: start_count, end_count, count_rate
  real(real32) :: lambda, alpha, xp, v, accum, temp, total_obj(1), l2_norm, obj_val, train_error
  real(real32) :: ref_l2_norm, ref_obj_val, ref_train_error
  real(real32), allocatable :: x(:), grad(:), ref_x(:), ref_grad(:)
  integer(int32), allocatable :: row_ptr(:), col_index(:), y_label(:)
  real(real32), allocatable :: values(:)
  integer(int32) :: m, n, nnz

  argc = command_argument_count()
  if (argc /= 4) then
    write(*,'(a)') 'Usage: ./main <path to file> <lambda> <alpha> <repeat>'
    error stop 1
  end if
  call get_command_argument(1, filename)
  call get_command_argument(2, arg); read(arg, *) lambda
  call get_command_argument(3, arg); read(arg, *) alpha
  call get_command_argument(4, arg); read(arg, *) iterations
  call load_libsvm(trim(filename), a)
  m = a%m; n = a%n; nnz = a%nnz
  allocate(x(n), grad(n), ref_x(n), ref_grad(n), row_ptr(0:m), col_index(nnz), y_label(m), values(nnz))
  x = 0.0_real32; grad = 0.0_real32; ref_x = 0.0_real32; ref_grad = 0.0_real32
  row_ptr = a%row_ptr; col_index = a%col_index; y_label = a%y_label; values = a%values

  call system_clock(start_count, count_rate)
  !$omp target data map(to: x(1:n), row_ptr(0:m), col_index(1:nnz), values(1:nnz), y_label(1:m)) &
  !$omp& map(alloc: grad(1:n), total_obj(1:1), correct(1:1))
  do k = 1, iterations
    total_obj(1) = 0.0_real32; correct(1) = 0; l2_norm = 0.0_real32
    grad = 0.0_real32
    !$omp target update to(total_obj(1:1), correct(1:1), grad(1:n))
    !$omp target teams distribute parallel do thread_limit(256) &
    !$omp& map(to:row_ptr(0:m),col_index(1:nnz),values(1:nnz),y_label(1:m),x(1:n)) &
    !$omp& map(tofrom:grad(1:n),total_obj(1:1),correct(1:1))
    do i = 1, m
      xp = 0.0_real32
      do j = row_ptr(i-1)+1, row_ptr(i)
        xp = xp + values(j) * x(col_index(j))
      end do
      v = log(1.0_real32 + exp(-xp * real(y_label(i), real32)))
      !$omp atomic update
      total_obj(1) = total_obj(1) + v
      if (1.0_real32/(1.0_real32 + exp(-xp)) >= 0.5_real32) then
        if (y_label(i) == 1) then
          !$omp atomic update
          correct(1) = correct(1) + 1
        end if
      else
        if (y_label(i) == -1) then
          !$omp atomic update
          correct(1) = correct(1) + 1
        end if
      end if
      accum = exp(-real(y_label(i), real32)*xp)
      accum = accum / (1.0_real32 + accum)
      do j = row_ptr(i-1)+1, row_ptr(i)
        temp = -accum*values(j)*real(y_label(i), real32)
        !$omp atomic update
        grad(col_index(j)) = grad(col_index(j)) + temp
      end do
    end do
    !$omp target teams distribute parallel do reduction(+:l2_norm) thread_limit(256) &
    !$omp& map(tofrom:x(1:n),grad(1:n),l2_norm)
    do i = 1, n
      l2_norm = l2_norm + x(i)*x(i)
      x(i) = x(i) - alpha*(grad(i)/real(m,real32) + lambda*x(i))
    end do
  end do
  !$omp target update from(total_obj(1:1), correct(1:1))
  !$omp end target data
  call system_clock(end_count)
  write(*,'(a,f12.6,a,i0,a)') 'Training time takes ', &
    real(end_count-start_count,real32)/real(count_rate,real32), ' (s) for ', iterations, ' iterations'
  obj_val = total_obj(1)/real(m,real32) + 0.5_real32*lambda*l2_norm
  train_error = 1.0_real32 - real(correct(1),real32)/real(m,real32)
  write(*,'(a,f12.6,a,f12.6)') 'object value = ', obj_val, ' train_error = ', train_error

  ! Independent host reference, from the same zero vector, mirrors reference.h.
  do k = 1, iterations
    total_obj(1) = 0.0_real32; correct(1) = 0; ref_grad = 0.0_real32
    do i = 1, m
      xp = 0.0_real32
      do j = row_ptr(i-1)+1, row_ptr(i)
        xp = xp + values(j)*ref_x(col_index(j))
      end do
      total_obj(1) = total_obj(1) + log(1.0_real32 + exp(-xp*real(y_label(i),real32)))
      t = merge(1, -1, 1.0_real32/(1.0_real32+exp(-xp)) >= 0.5_real32)
      if (y_label(i) == t) correct(1) = correct(1) + 1
      accum = exp(-real(y_label(i),real32)*xp); accum = accum/(1.0_real32+accum)
      do j = row_ptr(i-1)+1, row_ptr(i)
        ref_grad(col_index(j)) = ref_grad(col_index(j)) - accum*values(j)*real(y_label(i),real32)
      end do
    end do
    ref_l2_norm = sum(ref_x*ref_x)
    ref_obj_val = total_obj(1)/real(m,real32) + 0.5_real32*lambda*ref_l2_norm
    ref_train_error = 1.0_real32 - real(correct(1),real32)/real(m,real32)
    do i = 1, n
      ref_x(i) = ref_x(i) - alpha*(ref_grad(i)/real(m,real32)+lambda*ref_x(i))
    end do
  end do
  if (abs(obj_val-ref_obj_val) < 1.0e-3_real32 .and. &
      abs(train_error-ref_train_error) < 1.0e-3_real32) then
    write(*,'(a)') 'PASS'
  else
    write(*,'(a,2es12.4)') 'FAIL: objective/error differences = ', &
      abs(obj_val-ref_obj_val), abs(train_error-ref_train_error)
    error stop 2
  end if
end program gd
