program conversion
  use, intrinsic :: iso_fortran_env, only : int8, int32, real32, real64
  use omp_lib, only : omp_get_wtime
  implicit none

  integer(int32) :: nelems, niters, ios
  character(len=64) :: argument

  if (command_argument_count() /= 2) then
    write(*,'(a)') 'Usage: ./main <number of elements> <repeat>'
    stop 1
  end if
  call get_command_argument(1, argument)
  read(argument,*,iostat=ios) nelems
  if (ios /= 0 .or. nelems <= 0) error stop 'number of elements must be positive'
  call get_command_argument(2, argument)
  read(argument,*,iostat=ios) niters
  if (ios /= 0 .or. niters <= 0) error stop 'repeat must be positive'

  write(*,'(a)') 'float -> float'
  call convert_rr(nelems, niters)
  write(*,'(a)') 'float -> int'
  call convert_ir(nelems, niters)
  write(*,'(a)') 'float -> char'
  call convert_br(nelems, niters)
  write(*,'(a)') 'float -> uchar'
  call convert_ur(nelems, niters)

  write(*,'(a)') 'int -> int'
  call convert_ii(nelems, niters)
  write(*,'(a)') 'int -> float'
  call convert_ri(nelems, niters)
  write(*,'(a)') 'int -> char'
  call convert_bi(nelems, niters)
  write(*,'(a)') 'int -> uchar'
  call convert_ui(nelems, niters)

  write(*,'(a)') 'char -> int'
  call convert_ib(nelems, niters)
  write(*,'(a)') 'char -> float'
  call convert_rb(nelems, niters)
  write(*,'(a)') 'char -> char'
  call convert_bb(nelems, niters)
  write(*,'(a)') 'char -> uchar'
  call convert_ub(nelems, niters)

  write(*,'(a)') 'uchar -> int'
  call convert_iu(nelems, niters)
  write(*,'(a)') 'uchar -> float'
  call convert_ru(nelems, niters)
  write(*,'(a)') 'uchar -> char'
  call convert_bu(nelems, niters)
  write(*,'(a)') 'uchar -> uchar'
  call convert_uu(nelems, niters)

contains

  subroutine print_result(nelems, source_bytes, destination_bytes, elapsed)
    integer(int32), intent(in) :: nelems, source_bytes, destination_bytes
    real(real64), intent(in) :: elapsed
    real(real64) :: size_gb
    size_gb = real(source_bytes + destination_bytes, real64) * real(nelems, real64) / 1.0e9_real64
    write(*,'(a,f0.2,a,f0.6,a,f0.6)') 'size(GB):', size_gb, ', average time(sec):', elapsed, ', BW:', size_gb / elapsed
  end subroutine print_result

  subroutine convert_rr(n, repeats)
    integer(int32), intent(in) :: n, repeats
    real(real32), allocatable :: src(:), dst(:)
    integer(int32) :: i, k, ls, gs
    real(real64) :: begin_time, elapsed
    allocate(src(0:n-1), dst(0:n-1)); ls=min(n,256_int32); gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1; dst(i)=src(i); end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime()
    do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1; dst(i)=src(i); end do
!$omp end target teams distribute parallel do
    end do
    elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,4_int32,elapsed); deallocate(src,dst)
  end subroutine convert_rr

  subroutine convert_ir(n, repeats)
    integer(int32), intent(in) :: n, repeats
    real(real32), allocatable :: src(:); integer(int32), allocatable :: dst(:)
    integer(int32) :: i,k,ls,gs; real(real64) :: begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1)); ls=min(n,256_int32); gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1; dst(i)=int(src(i),int32); end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime(); do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1; dst(i)=int(src(i),int32); end do
!$omp end target teams distribute parallel do
    end do; elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,4_int32,elapsed); deallocate(src,dst)
  end subroutine convert_ir

  subroutine convert_br(n, repeats)
    integer(int32), intent(in) :: n,repeats
    real(real32), allocatable :: src(:); integer(int8), allocatable :: dst(:)
    integer(int32)::i,k,ls,gs; real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=transfer(int(src(i),int32),dst(i));end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=transfer(int(src(i),int32),dst(i));end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_br

  subroutine convert_ur(n,repeats)
    integer(int32),intent(in)::n,repeats
    real(real32),allocatable::src(:);integer(int8),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=transfer(int(src(i),int32),dst(i));end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=transfer(int(src(i),int32),dst(i));end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ur

  subroutine convert_ii(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int32),allocatable::src(:),dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,4_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ii

  subroutine convert_ri(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int32),allocatable::src(:);real(real32),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=real(src(i),real32);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=real(src(i),real32);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,4_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ri

  subroutine convert_bi(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int32),allocatable::src(:);integer(int8),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=transfer(src(i),dst(i));end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=transfer(src(i),dst(i));end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_bi

  subroutine convert_ui(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int32),allocatable::src(:);integer(int8),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=transfer(src(i),dst(i));end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=transfer(src(i),dst(i));end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,4_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ui

  subroutine convert_ib(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:);integer(int32),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=int(src(i),int32);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=int(src(i),int32);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,4_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ib

  subroutine convert_rb(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:);real(real32),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=real(src(i),real32);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=real(src(i),real32);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,4_int32,elapsed);deallocate(src,dst)
  end subroutine convert_rb

  subroutine convert_bb(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:),dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_bb

  subroutine convert_ub(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:),dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ub

  subroutine convert_iu(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:);integer(int32),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=int(src(i),int32);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=int(src(i),int32);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,4_int32,elapsed);deallocate(src,dst)
  end subroutine convert_iu

  subroutine convert_ru(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:);real(real32),allocatable::dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=real(src(i),real32);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=real(src(i),real32);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,4_int32,elapsed);deallocate(src,dst)
  end subroutine convert_ru

  subroutine convert_bu(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:),dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_bu

  subroutine convert_uu(n,repeats)
    integer(int32),intent(in)::n,repeats
    integer(int8),allocatable::src(:),dst(:)
    integer(int32)::i,k,ls,gs;real(real64)::begin_time,elapsed
    allocate(src(0:n-1),dst(0:n-1));ls=min(n,256_int32);gs=(n+ls-1)/ls
!$omp target data map(alloc:src(0:n-1),dst(0:n-1))
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
    do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    begin_time=omp_get_wtime();do k=1,repeats
!$omp target teams distribute parallel do num_teams(gs) thread_limit(ls)
      do i=0,n-1;dst(i)=src(i);end do
!$omp end target teams distribute parallel do
    end do;elapsed=(omp_get_wtime()-begin_time)/real(repeats,real64)
!$omp end target data
    call print_result(n,1_int32,1_int32,elapsed);deallocate(src,dst)
  end subroutine convert_uu
end program conversion
