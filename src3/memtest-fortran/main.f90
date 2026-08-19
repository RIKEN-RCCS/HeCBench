module c_rng
  use iso_c_binding, only: c_int
  implicit none
  interface
    subroutine srand(seed) bind(C, name="srand")
      import c_int
      integer(c_int), value :: seed
    end subroutine srand
    function rand() result(r) bind(C, name="rand")
      import c_int
      integer(c_int) :: r
    end function rand
  end interface
end module c_rng

module memtest_kernels
  use iso_fortran_env, only: int32, int64
  implicit none
  integer, parameter :: MAX_ERR_RECORD_COUNT = 10
  integer(int64), parameter :: BLOCKSIZE = 1024_int64*1024_int64
contains
  subroutine record_err(err_count, err_addr, err_expect, err_current, err_second_read, addr, expect, current)
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:)
    integer(int64), intent(in) :: addr, expect, current
    integer(int32) :: idx
!$omp atomic capture
    idx = err_count(0); err_count(0) = err_count(0) + 1
!$omp end atomic
    idx = mod(idx, MAX_ERR_RECORD_COUNT)
    err_addr(idx) = addr
    err_expect(idx) = expect
    err_current(idx) = current
    err_second_read(idx) = current
  end subroutine record_err

  subroutine kernel0_write(buf, mem_size)
    integer(int64), intent(inout) :: buf(0:)
    integer(int64), intent(in) :: mem_size
    integer(int64) :: nblocks, base_word, end_word, pword, mask
    integer(int32) :: i, pattern
    nblocks = mem_size / BLOCKSIZE
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64) private(base_word,end_word,pword,pattern,mask)
    do i = 0, int(nblocks)-1
      base_word = (int(i,int64)*BLOCKSIZE) / 8_int64
      end_word = (int(i+1,int64)*BLOCKSIZE) / 8_int64
      pword = base_word
      pattern = 1
      mask = 8
      buf(pword) = int(pattern, int64)
      pattern = shiftl(pattern, 1)
      do
        pword = base_word + mask/8_int64
        if (pword == base_word) then
          mask = shiftl(mask, 1)
          if (mask == 0) exit
          cycle
        end if
        if (pword >= end_word) exit
        buf(pword) = int(pattern, int64)
        pattern = shiftl(pattern, 1)
        mask = shiftl(mask, 1)
        if (mask == 0) exit
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel0_write

  subroutine kernel0_read(buf, mem_size, err_count, err_addr, err_expect, err_current, err_second_read)
    integer(int64), intent(in) :: buf(0:)
    integer(int64), intent(in) :: mem_size
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:)
    integer(int64) :: nblocks, base_word, end_word, pword, mask
    integer(int32) :: i, pattern
    nblocks = mem_size / BLOCKSIZE
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64) private(base_word,end_word,pword,pattern,mask)
    do i = 0, int(nblocks)-1
      base_word = (int(i,int64)*BLOCKSIZE) / 8_int64
      end_word = (int(i+1,int64)*BLOCKSIZE) / 8_int64
      pword = base_word; pattern = 1; mask = 8
      if (buf(pword) /= int(pattern,int64)) call record_err(err_count,err_addr,err_expect,err_current,err_second_read,pword,int(pattern,int64),buf(pword))
      pattern = shiftl(pattern, 1)
      do
        pword = base_word + mask/8_int64
        if (pword == base_word) then
          mask = shiftl(mask, 1); if (mask == 0) exit; cycle
        end if
        if (pword >= end_word) exit
        if (buf(pword) /= int(pattern,int64)) call record_err(err_count,err_addr,err_expect,err_current,err_second_read,pword,int(pattern,int64),buf(pword))
        pattern = shiftl(pattern, 1); mask = shiftl(mask, 1); if (mask == 0) exit
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel0_read

  subroutine kernel1_write(buf, mem_size)
    integer(int64), intent(inout) :: buf(0:)
    integer(int64), intent(in) :: mem_size
    integer(int64) :: n
    integer :: i
    n = mem_size / 8_int64
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64)
    do i = 0, int(n)-1
      buf(i) = int(i, int64)
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel1_write

  subroutine kernel1_read(buf, mem_size, err_count, err_addr, err_expect, err_current, err_second_read)
    integer(int64), intent(in) :: buf(0:)
    integer(int64), intent(in) :: mem_size
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:)
    integer(int64) :: n
    integer :: i
    n = mem_size / 8_int64
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64)
    do i = 0, int(n)-1
      if (buf(i) /= int(i,int64)) call record_err(err_count,err_addr,err_expect,err_current,err_second_read,int(i,int64),int(i,int64),buf(i))
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel1_read

  subroutine kernel_write(buf, mem_size, p1)
    integer(int64), intent(inout) :: buf(0:)
    integer(int64), intent(in) :: mem_size, p1
    integer(int64) :: n
    integer :: i
    n = mem_size / 8_int64
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64)
    do i = 0, int(n)-1
      buf(i) = p1
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel_write

  subroutine kernel_read_write(buf, mem_size, p1, p2, err_count, err_addr, err_expect, err_current, err_second_read)
    integer(int64), intent(inout) :: buf(0:)
    integer(int64), intent(in) :: mem_size, p1, p2
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:)
    integer(int64) :: n, localp
    integer :: i
    n = mem_size / 8_int64
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64) private(localp)
    do i = 0, int(n)-1
      localp = buf(i)
      if (localp /= p1) call record_err(err_count,err_addr,err_expect,err_current,err_second_read,int(i,int64),p1,localp)
      buf(i) = p2
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel_read_write

  subroutine kernel_read(buf, mem_size, p1, err_count, err_addr, err_expect, err_current, err_second_read)
    integer(int64), intent(in) :: buf(0:)
    integer(int64), intent(in) :: mem_size, p1
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:)
    integer(int64) :: n, localp
    integer :: i
    n = mem_size / 8_int64
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64) private(localp)
    do i = 0, int(n)-1
      localp = buf(i)
      if (localp /= p1) call record_err(err_count,err_addr,err_expect,err_current,err_second_read,int(i,int64),p1,localp)
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel_read

  subroutine kernel5_init(buf32, mem_size)
    integer(int32), intent(inout) :: buf32(0:)
    integer(int64), intent(in) :: mem_size
    integer(int64) :: n
    integer(int32) :: p1, p2
    integer :: i
    n = mem_size / 64_int64
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64) private(p1,p2)
    do i = 0, int(n)-1
      p1 = shiftl(1_int32, mod(i,32))
      p2 = not(p1)
      buf32(i*16+0)=p1; buf32(i*16+1)=p1; buf32(i*16+2)=p2; buf32(i*16+3)=p2
      buf32(i*16+4)=p1; buf32(i*16+5)=p1; buf32(i*16+6)=p2; buf32(i*16+7)=p2
      buf32(i*16+8)=p1; buf32(i*16+9)=p1; buf32(i*16+10)=p2; buf32(i*16+11)=p2
      buf32(i*16+12)=p1; buf32(i*16+13)=p1; buf32(i*16+14)=p2; buf32(i*16+15)=p2
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel5_init

  subroutine kernel5_move(buf32, mem_size)
    integer(int32), intent(inout) :: buf32(0:)
    integer(int64), intent(in) :: mem_size
    integer, parameter :: half_count = int(BLOCKSIZE/4_int64/2_int64)
    integer(int64) :: n, base, mid
    integer :: i, j
    n = mem_size / BLOCKSIZE
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64) private(base,mid,j)
    do i = 0, int(n)-1
      base = int(i,int64) * (BLOCKSIZE/4_int64)
      mid = base + BLOCKSIZE/4_int64/2_int64
      do j = 0, half_count-1
        buf32(mid+j) = buf32(base+j)
      end do
      do j = 0, half_count-9
        buf32(base+j+8) = buf32(mid+j)
      end do
      do j = 0, 7
        buf32(base+j) = buf32(mid+half_count-8+j)
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel5_move

  subroutine kernel5_check(buf32, mem_size, err_count, err_addr, err_expect, err_current, err_second_read)
    integer(int32), intent(in) :: buf32(0:)
    integer(int64), intent(in) :: mem_size
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:)
    integer(int64) :: n
    integer :: i
    n = mem_size / (2_int64*4_int64)
!$omp target teams distribute parallel do num_teams(1024) thread_limit(64)
    do i = 0, int(n)-1
      if (buf32(2*i) /= buf32(2*i+1)) call record_err(err_count,err_addr,err_expect,err_current,err_second_read,int(2*i,int64),int(buf32(2*i+1),int64),int(buf32(2*i),int64))
    end do
!$omp end target teams distribute parallel do
  end subroutine kernel5_check
end module memtest_kernels

program main
  use iso_fortran_env, only: int32, int64
  use omp_lib
  use c_rng
  use memtest_kernels
  implicit none
  integer :: repeat, i
  integer(int64), parameter :: mem_size = 2_int64*1024_int64*1024_int64*1024_int64
  integer(int64), allocatable :: dev_mem(:), err_addr(:), err_expect(:), err_current(:), err_second_read(:)
  integer(int32), allocatable :: dev_mem32(:)
  integer(int32) :: err_cnt(0:0)
  integer(int64) :: p1, p2
  character(len=64) :: arg
  real(8) :: start_time, elapsed

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) repeat
  allocate(err_addr(0:MAX_ERR_RECORD_COUNT-1), err_expect(0:MAX_ERR_RECORD_COUNT-1), err_current(0:MAX_ERR_RECORD_COUNT-1), err_second_read(0:MAX_ERR_RECORD_COUNT-1))
  allocate(dev_mem(0:mem_size/8_int64-1))
  allocate(dev_mem32(0:mem_size/4_int64-1))
  err_cnt = 0

!$omp target data map(to:err_cnt(0:0)) map(alloc:err_addr(0:MAX_ERR_RECORD_COUNT-1),err_expect(0:MAX_ERR_RECORD_COUNT-1),err_current(0:MAX_ERR_RECORD_COUNT-1),err_second_read(0:MAX_ERR_RECORD_COUNT-1),dev_mem(0:mem_size/8_int64-1),dev_mem32(0:mem_size/4_int64-1))
  write(*,'(a)',advance='no') new_line('a')//'test0: '
  do i = 1, repeat
    call kernel0_write(dev_mem, mem_size)
    call kernel0_read(dev_mem, mem_size, err_cnt, err_addr, err_expect, err_current, err_second_read)
  end do
  call check(err_cnt)

  write(*,'(a)',advance='no') new_line('a')//'test1: '
  do i = 1, repeat
    call kernel1_write(dev_mem, mem_size)
    call kernel1_read(dev_mem, mem_size, err_cnt, err_addr, err_expect, err_current, err_second_read)
  end do
  call check(err_cnt)

  write(*,'(a)',advance='no') new_line('a')//'test2: '
  do i = 1, repeat
    p1 = 0_int64; p2 = not(p1)
    call moving_inversion(err_cnt, err_addr, err_expect, err_current, err_second_read, dev_mem, mem_size, p1)
    call moving_inversion(err_cnt, err_addr, err_expect, err_current, err_second_read, dev_mem, mem_size, p2)
  end do

  write(*,'(a)',advance='no') new_line('a')//'test3: '
  do i = 1, repeat
    p1 = int(z'8080808080808080', int64); p2 = not(p1)
    call moving_inversion(err_cnt, err_addr, err_expect, err_current, err_second_read, dev_mem, mem_size, p1)
    call moving_inversion(err_cnt, err_addr, err_expect, err_current, err_second_read, dev_mem, mem_size, p2)
  end do

  write(*,'(a)',advance='no') new_line('a')//'test4: '
  call srand(123)
  do i = 1, repeat
    p1 = ior(shiftl(int(rand(),int64), 32), int(rand(),int64))
    call moving_inversion(err_cnt, err_addr, err_expect, err_current, err_second_read, dev_mem, mem_size, p1)
  end do

  write(*,'(a)',advance='no') new_line('a')//'test5: '
  start_time = omp_get_wtime()
  do i = 1, repeat
    call kernel5_init(dev_mem32, mem_size)
    call kernel5_move(dev_mem32, mem_size)
    call kernel5_check(dev_mem32, mem_size, err_cnt, err_addr, err_expect, err_current, err_second_read)
  end do
  elapsed = omp_get_wtime() - start_time
  call check(err_cnt)
  print '(a,f0.6,a)', new_line('a')//'Average kernel execution time (test5): ', elapsed / repeat, ' (s)'
!$omp end target data

contains
  subroutine check(err_count)
    integer(int32), intent(inout) :: err_count(0:)
!$omp target update from(err_count(0:0))
    write(*,'(a)',advance='no') merge('x', '.', err_count(0) /= 0)
!$omp target
    err_count(0) = 0
!$omp end target
  end subroutine check

  subroutine moving_inversion(err_count, err_addr, err_expect, err_current, err_second_read, mem, nbytes, pat1)
    integer(int32), intent(inout) :: err_count(0:)
    integer(int64), intent(inout) :: err_addr(0:), err_expect(0:), err_current(0:), err_second_read(0:), mem(0:)
    integer(int64), intent(in) :: nbytes
    integer(int64), intent(inout) :: pat1
    integer(int64) :: pat2
    integer :: k
    pat2 = not(pat1)
    call kernel_write(mem, nbytes, pat1)
    do k = 1, 10
      call kernel_read_write(mem, nbytes, pat1, pat2, err_count, err_addr, err_expect, err_current, err_second_read)
      pat1 = pat2
      pat2 = not(pat1)
    end do
    call kernel_read(mem, nbytes, pat1, err_count, err_addr, err_expect, err_current, err_second_read)
    call check(err_count)
  end subroutine moving_inversion
end program main
