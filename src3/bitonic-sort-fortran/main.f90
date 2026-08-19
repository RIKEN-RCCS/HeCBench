program bitonic_sort
  use iso_c_binding, only: c_int
  use omp_lib
  implicit none
  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(value)
      import c_int
      integer(c_int) :: value
    end function c_rand
  end interface
  integer :: n, seed, array_size, i, ios, unequal
  character(len=64) :: arg
  integer, allocatable :: data_cpu(:), data_gpu(:)

  if (command_argument_count() /= 2) then
    print *, 'Usage: ./main n k'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) n
  if (ios /= 0 .or. n < 0 .or. n >= bit_size(0)-1) then
    print *, 'Incorrect parameters'; stop 1
  end if
  call get_command_argument(2,arg); read(arg,*,iostat=ios) seed
  if (ios /= 0) then; print *, 'Incorrect parameters'; stop 1; end if
  array_size = shiftl(1,n)
  allocate(data_cpu(0:array_size-1), data_gpu(0:array_size-1))
  call c_srand(int(seed,c_int))
  do i=0,array_size-1
    data_gpu(i) = modulo(c_rand(),1000_c_int)
    data_cpu(i) = data_gpu(i)
  end do
  print *, 'Array size:', array_size, ', seed:', seed
  print *, 'Bitonic sort (parallel)..'
  call parallel_bitonic_sort(data_gpu,n)
  print *, 'Bitonic sort (serial)..'
  call bitonic_sort_cpu(data_cpu,n)
  unequal = 0
  do i=0,array_size-1
    if (data_gpu(i) /= data_cpu(i)) unequal = 1
  end do
  if (unequal == 0) then; print *, 'PASS'; else; print *, 'FAIL'; end if
contains
  subroutine parallel_bitonic_sort(input,n)
    integer, intent(inout) :: input(0:)
    integer, intent(in) :: n
    integer :: step, stage, seq_len, two_power, i, nsize, seq_num, swapped_ele, h_len, odd, temp
    logical :: increasing
    real(8) :: start, finish
    nsize=shiftl(1,n)
    !$omp target data map(alloc:input(0:nsize-1))
    !$omp target update to(input(0:nsize-1))
    start=omp_get_wtime()
    do step=0,n-1
      do stage=step,0,-1
        seq_len=shiftl(1,stage+1); two_power=shiftl(1,step-stage)
        !$omp target teams distribute parallel do thread_limit(256) private(seq_num,swapped_ele,h_len,odd,increasing,temp)
        do i=0,nsize-1
          seq_num=i/seq_len; swapped_ele=-1; h_len=seq_len/2
          if (i < seq_len*seq_num+h_len) swapped_ele=i+h_len
          odd=seq_num/two_power; increasing=(modulo(odd,2)==0)
          if (swapped_ele /= -1) then
            if ((input(i)>input(swapped_ele) .and. increasing) .or. (input(i)<input(swapped_ele) .and. .not.increasing)) then
              temp=input(i); input(i)=input(swapped_ele); input(swapped_ele)=temp
            end if
          end if
        end do
        !$omp end target teams distribute parallel do
      end do
    end do
    finish=omp_get_wtime()
    print '(a,f12.6,a)', 'Total kernel execution time: ',(finish-start)*1.0d3,' (ms)'
    !$omp target update from(input(0:nsize-1))
    !$omp end target data
  end subroutine
  subroutine bitonic_sort_cpu(a,n)
    integer, intent(inout) :: a(0:)
    integer, intent(in) :: n
    integer :: step,stage,num_sequence,seq_len,seq_num,odd,h_len,i,swapped,temp
    logical :: increasing
    do step=0,n-1
      do stage=step,0,-1
        num_sequence=shiftl(1,n-stage-1); seq_len=shiftl(1,stage+1); h_len=seq_len/2
        do seq_num=0,num_sequence-1
          odd=seq_num/shiftl(1,step-stage); increasing=(modulo(odd,2)==0)
          do i=seq_num*seq_len,seq_num*seq_len+h_len-1
            swapped=i+h_len
            if ((a(i)>a(swapped) .and. increasing) .or. (a(i)<a(swapped) .and. .not.increasing)) then
              temp=a(i); a(i)=a(swapped); a(swapped)=temp
            end if
          end do
        end do
      end do
    end do
  end subroutine
end program
