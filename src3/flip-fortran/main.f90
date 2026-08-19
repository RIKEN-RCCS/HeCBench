module flip_kernels
  use iso_fortran_env, only: int64, real32, real64
  implicit none

  interface flip_device
    module procedure flip_device_sp, flip_device_dp
  end interface
  interface flip_cpu
    module procedure flip_cpu_sp, flip_cpu_dp
  end interface

contains

  subroutine flip_device_sp(input, output, n, flip_dims, flip_dims_size, strides, strides_contiguous, shape, total_dims)
    integer(int64), intent(in) :: n, flip_dims_size, total_dims
    real(real32), intent(in) :: input(0:n-1)
    real(real32), intent(out) :: output(0:n-1)
    integer(int64), intent(in) :: flip_dims(0:total_dims-1), strides(0:total_dims-1)
    integer(int64), intent(in) :: strides_contiguous(0:total_dims-1), shape(0:total_dims-1)
    integer(int64) :: linear_index, cur_indices, rem, dst_offset, temp, i, j
!$omp target teams distribute parallel do thread_limit(256) private(cur_indices,rem,dst_offset,temp,i,j)
    do linear_index = 0, n - 1
      cur_indices = linear_index
      dst_offset = 0_int64
      do i = 0, total_dims - 1
        temp = cur_indices
        cur_indices = cur_indices / strides_contiguous(i)
        rem = temp - cur_indices * strides_contiguous(i)
        do j = 0, flip_dims_size - 1
          if (i == flip_dims(j)) cur_indices = shape(i) - 1_int64 - cur_indices
        end do
        dst_offset = dst_offset + cur_indices * strides(i)
        cur_indices = rem
      end do
      output(linear_index) = input(dst_offset)
    end do
!$omp end target teams distribute parallel do
  end subroutine

  subroutine flip_device_dp(input, output, n, flip_dims, flip_dims_size, strides, strides_contiguous, shape, total_dims)
    integer(int64), intent(in) :: n, flip_dims_size, total_dims
    real(real64), intent(in) :: input(0:n-1)
    real(real64), intent(out) :: output(0:n-1)
    integer(int64), intent(in) :: flip_dims(0:total_dims-1), strides(0:total_dims-1)
    integer(int64), intent(in) :: strides_contiguous(0:total_dims-1), shape(0:total_dims-1)
    integer(int64) :: linear_index, cur_indices, rem, dst_offset, temp, i, j
!$omp target teams distribute parallel do thread_limit(256) private(cur_indices,rem,dst_offset,temp,i,j)
    do linear_index = 0, n - 1
      cur_indices = linear_index
      dst_offset = 0_int64
      do i = 0, total_dims - 1
        temp = cur_indices
        cur_indices = cur_indices / strides_contiguous(i)
        rem = temp - cur_indices * strides_contiguous(i)
        do j = 0, flip_dims_size - 1
          if (i == flip_dims(j)) cur_indices = shape(i) - 1_int64 - cur_indices
        end do
        dst_offset = dst_offset + cur_indices * strides(i)
        cur_indices = rem
      end do
      output(linear_index) = input(dst_offset)
    end do
!$omp end target teams distribute parallel do
  end subroutine

  subroutine flip_cpu_sp(input, output, n, flip_dims, flip_dims_size, strides, strides_contiguous, shape, total_dims)
    integer(int64), intent(in) :: n, flip_dims_size, total_dims
    real(real32), intent(in) :: input(0:n-1)
    real(real32), intent(out) :: output(0:n-1)
    integer(int64), intent(in) :: flip_dims(0:total_dims-1), strides(0:total_dims-1)
    integer(int64), intent(in) :: strides_contiguous(0:total_dims-1), shape(0:total_dims-1)
    integer(int64) :: linear_index, cur_indices, rem, dst_offset, temp, i, j
!$omp parallel do private(cur_indices,rem,dst_offset,temp,i,j)
    do linear_index = 0, n - 1
      cur_indices = linear_index; dst_offset = 0_int64
      do i = 0, total_dims - 1
        temp = cur_indices; cur_indices = cur_indices / strides_contiguous(i)
        rem = temp - cur_indices * strides_contiguous(i)
        do j = 0, flip_dims_size - 1
          if (i == flip_dims(j)) cur_indices = shape(i) - 1_int64 - cur_indices
        end do
        dst_offset = dst_offset + cur_indices * strides(i); cur_indices = rem
      end do
      output(linear_index) = input(dst_offset)
    end do
!$omp end parallel do
  end subroutine

  subroutine flip_cpu_dp(input, output, n, flip_dims, flip_dims_size, strides, strides_contiguous, shape, total_dims)
    integer(int64), intent(in) :: n, flip_dims_size, total_dims
    real(real64), intent(in) :: input(0:n-1)
    real(real64), intent(out) :: output(0:n-1)
    integer(int64), intent(in) :: flip_dims(0:total_dims-1), strides(0:total_dims-1)
    integer(int64), intent(in) :: strides_contiguous(0:total_dims-1), shape(0:total_dims-1)
    integer(int64) :: linear_index, cur_indices, rem, dst_offset, temp, i, j
!$omp parallel do private(cur_indices,rem,dst_offset,temp,i,j)
    do linear_index = 0, n - 1
      cur_indices = linear_index; dst_offset = 0_int64
      do i = 0, total_dims - 1
        temp = cur_indices; cur_indices = cur_indices / strides_contiguous(i)
        rem = temp - cur_indices * strides_contiguous(i)
        do j = 0, flip_dims_size - 1
          if (i == flip_dims(j)) cur_indices = shape(i) - 1_int64 - cur_indices
        end do
        dst_offset = dst_offset + cur_indices * strides(i); cur_indices = rem
      end do
      output(linear_index) = input(dst_offset)
    end do
!$omp end parallel do
  end subroutine
end module

program flip
  use iso_fortran_env, only: int32, int64, real32, real64
  use omp_lib
  use flip_kernels
  implicit none
  integer :: argc, repeat, i
  integer(int64) :: num_dims, dim_size, num_flip_dims, n, t0, t1, count_rate
  integer(int64), allocatable :: shape(:), flip_dims(:), stride(:)
  real(real32), allocatable :: input_sp(:), output_sp(:), reference_sp(:)
  real(real64), allocatable :: input_dp(:), output_dp(:), reference_dp(:)
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 3) then
    print '(a)', 'Usage: ./main <number of dimensions> <size of each dimension> <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) num_dims
  call get_command_argument(2,arg); read(arg,*) dim_size
  call get_command_argument(3,arg); read(arg,*) repeat
  if (num_dims /= 3_int64) then
    print '(a)', 'This benchmark supports the original three-dimensional layout only.'
    stop 1
  end if
  num_flip_dims = num_dims
  n = dim_size ** num_dims
  allocate(shape(0:num_dims-1), flip_dims(0:num_dims-1), stride(0:num_dims-1))
  shape = dim_size
  do i = 0, int(num_dims)-1
    flip_dims(i) = i
  end do
  stride(0) = shape(1) * shape(2); stride(1) = shape(2); stride(2) = 1_int64
  print '(a,3(i0,1x),a)', 'shape: ( ', shape, ')'
  print '(a,3(i0,1x),a)', 'flip_dims: ( ', flip_dims, ')'
  print '(a,3(i0,1x),a)', 'stride: ( ', stride, ')'

  print '(a)', '=========== Data type is FP32 =========='
  allocate(input_sp(0:n-1), output_sp(0:n-1), reference_sp(0:n-1))
  do i = 0, int(n)-1; input_sp(i) = real(i,real32); end do
!$omp target data map(to:input_sp(0:n-1),shape(0:num_dims-1),flip_dims(0:num_dims-1),stride(0:num_dims-1)) map(alloc:output_sp(0:n-1))
  call flip_device(input_sp,output_sp,n,flip_dims,num_flip_dims,stride,stride,shape,num_dims)
  call flip_cpu(input_sp,reference_sp,n,flip_dims,num_flip_dims,stride,stride,shape,num_dims)
!$omp target update from(output_sp(0:n-1))
  if (all(output_sp == reference_sp)) then; print '(a)', 'PASS'; else; print '(a)', 'FAIL'; end if
  call system_clock(t0,count_rate); do i=1,repeat; call flip_device(input_sp,output_sp,n,flip_dims,num_flip_dims,stride,stride,shape,num_dims); end do; call system_clock(t1)
  print '(a,f0.6,a)', 'Average execution time of the flip kernel: ', real(t1-t0,real64)*1000.0_real64/real(count_rate,real64)/repeat, ' (ms)'
!$omp end target data
  deallocate(input_sp,output_sp,reference_sp)

  print '(a)', '=========== Data type is FP64 =========='
  allocate(input_dp(0:n-1), output_dp(0:n-1), reference_dp(0:n-1))
  do i = 0, int(n)-1; input_dp(i) = real(i,real64); end do
!$omp target data map(to:input_dp(0:n-1),shape(0:num_dims-1),flip_dims(0:num_dims-1),stride(0:num_dims-1)) map(alloc:output_dp(0:n-1))
  call flip_device(input_dp,output_dp,n,flip_dims,num_flip_dims,stride,stride,shape,num_dims)
  call flip_cpu(input_dp,reference_dp,n,flip_dims,num_flip_dims,stride,stride,shape,num_dims)
!$omp target update from(output_dp(0:n-1))
  if (all(output_dp == reference_dp)) then; print '(a)', 'PASS'; else; print '(a)', 'FAIL'; end if
  call system_clock(t0,count_rate); do i=1,repeat; call flip_device(input_dp,output_dp,n,flip_dims,num_flip_dims,stride,stride,shape,num_dims); end do; call system_clock(t1)
  print '(a,f0.6,a)', 'Average execution time of the flip kernel: ', real(t1-t0,real64)*1000.0_real64/real(count_rate,real64)/repeat, ' (ms)'
!$omp end target data
end program
