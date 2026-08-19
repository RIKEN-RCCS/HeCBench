module keccak32
  use iso_fortran_env, only: int32
  implicit none
  integer(int32), parameter :: rc(0:21)=[ &
    int(z'00000001',int32),int(z'00008082',int32),int(z'0000808A',int32),int(z'80008000',int32), &
    int(z'0000808B',int32),int(z'80000001',int32),int(z'80008081',int32),int(z'00008009',int32), &
    int(z'0000008A',int32),int(z'00000088',int32),int(z'80008009',int32),int(z'8000000A',int32), &
    int(z'8000808B',int32),int(z'0000008B',int32),int(z'00008089',int32),int(z'00008003',int32), &
    int(z'00008002',int32),int(z'00000080',int32),int(z'0000800A',int32),int(z'8000000A',int32), &
    int(z'80008081',int32),int(z'00008080',int32)]
  integer, parameter :: rho(0:4,0:4)=reshape([ &
    0,1,30,28,27, 4,12,6,23,20, 3,10,11,25,7, &
    9,13,15,21,8, 18,2,29,24,14],[5,5])
  !$omp declare target(rc,rho)
  !$omp declare target(rol32,keccakf)
contains
  pure integer(int32) function rol32(x,n)
    integer(int32),intent(in)::x
    integer,intent(in)::n
    if(n==0) then
      rol32=x
    else
      rol32=ior(shiftl(x,n),shiftr(x,32-n))
    end if
  end function rol32

  subroutine keccakf(a)
    integer(int32),intent(inout)::a(0:24)
    integer(int32)::b(0:24),c(0:4),d(0:4)
    integer::r,x,y,nx,ny
    do r=0,21
      do x=0,4
        c(x)=ieor(ieor(a(x),a(x+5)),ieor(a(x+10),ieor(a(x+15),a(x+20))))
      end do
      do x=0,4
        d(x)=ieor(c(modulo(x+4,5)),rol32(c(modulo(x+1,5)),1))
      end do
      do y=0,4
        do x=0,4
          a(x+5*y)=ieor(a(x+5*y),d(x))
        end do
      end do
      do y=0,4
        do x=0,4
          nx=y; ny=modulo(2*x+3*y,5)
          b(nx+5*ny)=rol32(a(x+5*y),rho(x,y))
        end do
      end do
      do y=0,4
        do x=0,4
          a(x+5*y)=ieor(b(x+5*y),iand(not(b(modulo(x+1,5)+5*y)),b(modulo(x+2,5)+5*y)))
        end do
      end do
      a(0)=ieor(a(0),rc(r))
    end do
  end subroutine keccakf

  subroutine tree_kernel(inp,out,nthreads,nblocks,ninput)
    integer(int32),intent(in)::inp(0:)
    integer(int32),intent(inout)::out(0:)
    integer,intent(in)::nthreads,nblocks,ninput
    integer::blk,thr,k,w,input_index,output_index,input_count,output_count
    integer(int32)::state(0:24)
    input_count=8*nthreads*nblocks*ninput
    output_count=8*nthreads*nblocks
    !$omp target update to(inp(0:input_count-1))
    !$omp target teams distribute parallel do collapse(2) num_teams(nblocks) thread_limit(nthreads) &
    !$omp& private(state,input_index,output_index,k,w)
    do blk=0,nblocks-1
      do thr=0,nthreads-1
        state=0_int32
        do k=0,ninput-1
          do w=0,7
            input_index=thr+w*nthreads+k*nthreads*8+blk*nthreads*8*ninput
            state(w)=ieor(state(w),inp(input_index))
          end do
          call keccakf(state)
        end do
        do w=0,7
          output_index=thr+w*nthreads+blk*nthreads*8
          out(output_index)=state(w)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
    !$omp target update from(out(0:output_count-1))
  end subroutine tree_kernel
end module keccak32

program main
  use iso_fortran_env, only:int32,int64,real64
  use keccak32
  implicit none
  integer,parameter::nthreads=64,nblocks=64
  integer::ninput,imax,argc,i,rep,clk0,clk1,rate
  integer(int32),allocatable::inp(:),gpu(:),cpu(:)
  integer(int32)::kscpu(0:24),ksgpu(0:24)
  integer(int64)::input_bytes
  real(real64)::cpu_seconds,gpu_seconds,cpu_speed,gpu_speed
  character(len=32)::arg

  ninput=1024; imax=8
  argc=command_argument_count()
  if(argc>=1) then; call get_command_argument(1,arg);read(arg,*) imax;end if
  if(argc>=2) then; call get_command_argument(2,arg);read(arg,*) ninput;end if
  print '(a,i0)', 'Number of threads per block             NB_THREADS           ',nthreads
  print '(a,i0)', 'Number of thread blocks                 NB_THREADS_BLOCKS    ',nblocks
  print '(a,i0)', 'Input block size of Keccak (in Byte)    INPUT_BLOCK_SIZE_B   ',32
  print '(a,i0)', 'Output block size of Keccak (in Byte)   OUTPUT_BLOCK_SIZE_B  ',32
  print '(a,i0)', 'Number of input blocks                  NB_INPUT_BLOCK       ',ninput

  allocate(inp(0:8*nthreads*nblocks*ninput-1),gpu(0:8*nthreads*nblocks-1),cpu(0:8*nthreads*nblocks-1))
  do i=0,size(inp)-1; inp(i)=int(i,int32); end do
  kscpu=0_int32; ksgpu=0_int32

  print '(a)','CPU speed test started'
  call system_clock(clk0,rate)
  do rep=1,imax
    call cpu_tree(inp,cpu,ninput)
    call top_hash(kscpu,cpu,nthreads*nblocks)
  end do
  call system_clock(clk1)
  cpu_seconds=real(clk1-clk0,real64)/real(rate,real64)
  input_bytes=int(32,int64)*nthreads*nblocks*ninput*imax
  cpu_speed=real(input_bytes,real64)/(1000.0_real64*cpu_seconds)
  write(*,'(a,f0.2,a)') 'CPU speed : ',cpu_speed,' kB/s'
  write(*,'(a,f0.5,a)') 'CPU time : ',cpu_seconds,' s'

  print '(a)','GPU speed test started'
  !$omp target data map(alloc:inp(0:size(inp)-1),gpu(0:size(gpu)-1))
  call system_clock(clk0,rate)
  do rep=1,imax
    call tree_kernel(inp,gpu,nthreads,nblocks,ninput)
    call top_hash(ksgpu,gpu,nthreads*nblocks)
  end do
  call system_clock(clk1)
  !$omp end target data
  gpu_seconds=real(clk1-clk0,real64)/real(rate,real64)
  gpu_speed=real(input_bytes,real64)/(1000.0_real64*gpu_seconds)
  write(*,'(a,f0.2,a)') 'GPU speed : ',gpu_speed,' kB/s'
  write(*,'(a,f0.5,a)') 'GPU time : ',gpu_seconds,' s'
  if(all(kscpu==ksgpu)) then
    print '(a)','PASS'
  else
    print '(a)','FAIL'; error stop 2
  end if
contains
  subroutine cpu_tree(x,y,ni)
    integer(int32),intent(in)::x(0:)
    integer(int32),intent(out)::y(0:)
    integer,intent(in)::ni
    integer::blk,thr,k,w,input_index,output_index
    integer(int32)::s(0:24)
    do blk=0,nblocks-1
      do thr=0,nthreads-1
        s=0_int32
        do k=0,ni-1
          do w=0,7
            input_index=thr+w*nthreads+k*nthreads*8+blk*nthreads*8*ni
            s(w)=ieor(s(w),x(input_index))
          end do
          call keccakf(s)
        end do
        do w=0,7
          output_index=thr+w*nthreads+blk*nthreads*8
          y(output_index)=s(w)
        end do
      end do
    end do
  end subroutine cpu_tree

  subroutine top_hash(s,x,count)
    integer(int32),intent(inout)::s(0:24)
    integer(int32),intent(in)::x(0:)
    integer,intent(in)::count
    integer::k,w
    do k=0,count-1
      do w=0,7
        s(w)=ieor(s(w),x(w+8*k))
      end do
      call keccakf(s)
    end do
  end subroutine top_hash
end program main
