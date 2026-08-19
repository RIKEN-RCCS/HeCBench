program mallocfree
  use iso_c_binding, only: c_ptr, c_size_t, c_int, c_null_ptr, c_sizeof
  use omp_lib
  implicit none

  integer, parameter :: NUM_SIZE = 19
  integer, parameter :: NUM_ITER = 500
  integer(c_size_t) :: total_global_mem
  integer(c_size_t) :: sizes(0:NUM_SIZE-1)
  type(c_ptr) :: ad(0:NUM_ITER-1)
  integer(c_int), allocatable :: a(:)
  integer :: argc, i, j, num, device_num, words
  character(len=128) :: arg
  real(8) :: start_time, end_time

  argc = command_argument_count()
  if (argc /= 1) then
    print '(a)', 'Usage: ./main <total global memory size in bytes>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *) total_global_mem

  num = NUM_SIZE
  do i = 0, NUM_SIZE-1
    sizes(i) = int(2_c_size_t, c_size_t) ** int(i + 6, c_size_t)
    if (int(NUM_ITER + 1, c_size_t) * sizes(i) > total_global_mem) then
      num = i
      exit
    end if
  end do

  words = max(1, int(sizes(num-1) / c_sizeof(0_c_int)))
  allocate(a(0:words-1))
  a = 1_c_int
  device_num = 0

  call test_init(sizes(0), device_num)

  do i = 0, num-1
    start_time = omp_get_wtime()
    do j = 0, NUM_ITER-1
      ad(j) = omp_target_alloc(sizes(i), device_num)
    end do
    end_time = omp_get_wtime()
    print '(a,i0,a,f0.6,a)', 'omp_target_alloc(', sizes(i), ') takes ', &
      (end_time - start_time) * 1.0e6 / NUM_ITER, ' us'

    start_time = omp_get_wtime()
    do j = 0, NUM_ITER-1
      call omp_target_free(ad(j), device_num)
      ad(j) = c_null_ptr
    end do
    end_time = omp_get_wtime()
    print '(a,i0,a,f0.6,a)', 'omp_target_free(', sizes(i), ') takes ', &
      (end_time - start_time) * 1.0e6 / NUM_ITER, ' us'
  end do

contains
  subroutine test_init(nbytes, dev)
    integer(c_size_t), intent(in) :: nbytes
    integer, intent(in) :: dev
    type(c_ptr) :: ptr
    real(8) :: t0, t1

    print '(a)', 'Initial allocation and deallocation'
    t0 = omp_get_wtime()
    ptr = omp_target_alloc(nbytes, dev)
    t1 = omp_get_wtime()
    print '(a,i0,a,f0.6,a)', 'omp_target_alloc(', nbytes, ') takes ', (t1-t0)*1.0e6, ' us'
    t0 = omp_get_wtime()
    call omp_target_free(ptr, dev)
    t1 = omp_get_wtime()
    print '(a,i0,a,f0.6,a)', 'omp_target_free(', nbytes, ') takes ', (t1-t0)*1.0e6, ' us'
    print *
  end subroutine test_init
end program mallocfree
