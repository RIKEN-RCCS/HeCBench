program aligned_types
  use, intrinsic :: iso_c_binding, only : c_int8_t
  use, intrinsic :: iso_fortran_env, only : real64
  use omp_lib, only : omp_get_wtime
  implicit none

  integer, parameter :: mem_size = 50000000
  integer, parameter :: num_iterations = 1000
  integer :: i, memory_size, total_failures
  integer :: status
  character(len=256) :: program_name
  integer(c_int8_t), allocatable :: h_idata_cpu(:), d_odata(:)

  call get_command_argument(0, program_name)
  write (*, '(A)') '[' // trim(program_name) // '] - Starting...'
  write (*, '(A)') 'Allocating memory...'

  ! The C++ source masks MEM_SIZE with 0xffffff00, forcing a 256-byte
  ! multiple.  All element sizes below divide this value exactly.
  memory_size = iand(mem_size, int(z'FFFFFF00'))
  allocate(h_idata_cpu(0:memory_size - 1), d_odata(0:memory_size - 1), stat=status)
  if (status /= 0) error stop 'Allocation failed'

  write (*, '(A)') 'Generating host input data array...'
  do i = 0, memory_size - 1
    h_idata_cpu(i) = unsigned_byte_from_int(iand(i, 255) + 1)
  end do

  write (*, '(A)') 'Uploading input data to GPU memory...'
  total_failures = 0

  ! This is intentionally one target-data region for all thirteen tests, as
  ! in aligned-types-omp: input is copied once and output is device-allocated
  ! once.  run_test only performs an update from after its timed launches.
  !$omp target data map(to: h_idata_cpu(0:memory_size - 1)) &
  !$omp& map(alloc: d_odata(0:memory_size - 1))
    write (*, '(A)') 'Testing misaligned types...'
    write (*, '(A)') 'uchar_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 1, 1, memory_size)

    write (*, '(A)') 'uchar4_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 4, 4, memory_size)

    write (*, '(A)') 'uchar4_aligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 4, 4, memory_size)

    write (*, '(A)') 'ushort_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 2, 2, memory_size)

    write (*, '(A)') 'uint_aligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 4, 4, memory_size)

    write (*, '(A)') 'uint2_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 8, 8, memory_size)

    write (*, '(A)') 'uint2_aligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 8, 8, memory_size)

    ! uint3_aligned has a 16-byte stride but only twelve payload bytes; the
    ! fourth component models the C++ structure's alignment padding.
    write (*, '(A)') 'uint3_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 12, 12, memory_size)

    write (*, '(A)') 'uint3_aligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 16, 12, memory_size)

    write (*, '(A)') 'uint4_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 16, 16, memory_size)

    write (*, '(A)') 'uint4_aligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 16, 16, memory_size)

    write (*, '(A)') 'uint8_misaligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 32, 32, memory_size)

    write (*, '(A)') 'uint8_aligned...'
    total_failures = total_failures + run_test(h_idata_cpu, d_odata, 32, 32, memory_size)

    write (*, '(/, A, I0, A)') '[alignedTypes] -> Test Results: ', total_failures, ' Failures'
    write (*, '(A)') 'Shutting down...'
  !$omp end target data

  deallocate(d_odata, h_idata_cpu)
  if (total_failures /= 0) then
    write (*, '(A)') 'Test failed!'
    error stop 1
  end if
  write (*, '(A)') 'Test passed'

contains

  pure function unsigned_byte_from_int(value) result(byte)
    integer, intent(in) :: value
    integer(c_int8_t) :: byte
    integer :: signed_value

    signed_value = value
    if (signed_value > 127) signed_value = signed_value - 256
    byte = int(signed_value, c_int8_t)
  end function unsigned_byte_from_int

  integer function run_test(d_idata, d_odata, element_size, packed_element_size, memory_size) result(failure)
    integer(c_int8_t), intent(in) :: d_idata(0:)
    integer(c_int8_t), intent(inout) :: d_odata(0:)
    integer, intent(in) :: element_size, packed_element_size, memory_size
    integer :: i, iteration, pos, base, num_elements, total_mem_size_aligned
    integer :: flag
    real(real64) :: start_time, end_time, gpu_time, throughput

    total_mem_size_aligned = memory_size - mod(memory_size, element_size)
    num_elements = memory_size / element_size

    !$omp target teams distribute parallel do thread_limit(256)
    do i = 0, memory_size - 1
      d_odata(i) = 0_c_int8_t
    end do
    !$omp end target teams distribute parallel do

    start_time = omp_get_wtime()
    do iteration = 1, num_iterations
      !$omp target teams distribute parallel do thread_limit(256) private(base)
      do pos = 0, num_elements - 1
        base = pos * element_size
        ! Array-section assignment is the direct Fortran equivalent of the
        ! C++ TData assignment: each work item copies one full object stride,
        ! including the four padding bytes of uint3_aligned.
        d_odata(base:base + element_size - 1) = d_idata(base:base + element_size - 1)
      end do
      !$omp end target teams distribute parallel do
    end do
    end_time = omp_get_wtime()
    gpu_time = (end_time - start_time) / real(num_iterations, real64)
    throughput = real(total_mem_size_aligned, real64) / (gpu_time * 1073741824.0_real64)

    write (*, '(A, F0.6, A, F0.6, A)') 'Avg. time: ', gpu_time * 1000.0_real64, &
      ' ms / Copy throughput: ', throughput, ' GB/s.'

    !$omp target update from(d_odata(0:memory_size - 1))
    flag = test_cpu(d_odata, d_idata, num_elements, element_size, packed_element_size)
    if (flag == 1) then
      write (*, '(A)') achar(9) // 'TEST PASS'
    else
      write (*, '(A)') achar(9) // 'TEST FAIL'
    end if
    failure = 1 - flag
  end function run_test

  integer function test_cpu(h_odata, h_idata, num_elements, element_size, packed_element_size) result(flag)
    integer(c_int8_t), intent(in) :: h_odata(0:), h_idata(0:)
    integer, intent(in) :: num_elements, element_size, packed_element_size
    integer :: pos, byte, base

    flag = 1
    do pos = 0, num_elements - 1
      base = pos * element_size
      do byte = 0, packed_element_size - 1
        if (h_odata(base + byte) /= h_idata(base + byte)) then
          flag = 0
          return
        end if
      end do
    end do
  end function test_cpu

end program aligned_types
