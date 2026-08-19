module concat_kernel
  use iso_fortran_env, only: real32, real64
  implicit none
contains
  subroutine concat(inp1, inp2, output, sz0, sz2, sz11, sz12)
    integer, intent(in) :: sz0, sz2, sz11, sz12
    real(real32), intent(in) :: inp1(0:), inp2(0:)
    real(real32), intent(inout) :: output(0:)
    integer :: idx, idx_x, idx_y, idx1, idx0, nele
    nele = sz0 * sz2 * (sz11 + sz12)
!$omp target teams distribute parallel do thread_limit(256) private(idx_x,idx_y,idx1,idx0)
    do idx = 0, nele - 1
      idx_x = modulo(idx, sz2); idx_y = idx / sz2
      idx1 = modulo(idx_y, sz11 + sz12); idx0 = idx_y / (sz11 + sz12)
      if (idx1 < sz11) then
        output(idx) = inp1((idx0 * sz11 + idx1) * sz2 + idx_x)
      else
        idx1 = idx1 - sz11
        output(idx) = inp2((idx0 * sz12 + idx1) * sz2 + idx_x)
      end if
    end do
!$omp end target teams distribute parallel do
  end subroutine
  subroutine concat_cpu(inp1, inp2, output, sz0, sz2, sz11, sz12)
    integer, intent(in) :: sz0,sz2,sz11,sz12
    real(real32), intent(in) :: inp1(0:),inp2(0:)
    real(real32), intent(out) :: output(0:)
    integer :: idx,idx_x,idx_y,idx1,idx0,nele
    nele=sz0*sz2*(sz11+sz12)
    do idx=0,nele-1
      idx_x=modulo(idx,sz2); idx_y=idx/sz2; idx1=modulo(idx_y,sz11+sz12); idx0=idx_y/(sz11+sz12)
      if(idx1<sz11) then; output(idx)=inp1((idx0*sz11+idx1)*sz2+idx_x)
      else; output(idx)=inp2((idx0*sz12+idx1-sz11)*sz2+idx_x); end if
    end do
  end subroutine
end module
program main
  use iso_fortran_env, only: real32,real64,int64
  use iso_c_binding, only:c_int
  use omp_lib, only:omp_get_wtime
  use concat_kernel
  implicit none
  integer::repeat,nhead,seq_len,batch_size,hidden_dim,head_dim,sl1,sl2,beam_size,sz0,i
  integer::inp1_size,inp2_size,outp_size
  real(real32),allocatable::inp1(:),inp2(:),outp(:),outref(:)
  real(real64)::start,stop,avg,size_bytes
  character(32)::arg
  interface
    subroutine c_srand(seed) bind(C,name='srand'); import c_int; integer(c_int),value::seed; end subroutine
    function c_rand() bind(C,name='rand') result(v); import c_int; integer(c_int)::v; end function
  end interface
  if(command_argument_count()/=1) error stop 'Usage: ./main <repeat>'
  call get_command_argument(1,arg); read(arg,*)repeat
  do nhead=6,48,6
    if(nhead/=6 .and. nhead/=12 .and. nhead/=24 .and. nhead/=48) cycle
    call c_srand(int(nhead,c_int)); seq_len=1024; batch_size=8; hidden_dim=nhead*128; head_dim=hidden_dim/nhead; beam_size=8
    sl1=mod(c_rand(),seq_len-1)+1; sl2=seq_len-sl1; sz0=batch_size*beam_size*nhead
    inp1_size=sz0*head_dim*sl1; inp2_size=sz0*head_dim*sl2; outp_size=sz0*head_dim*seq_len
    size_bytes=real(2_int64*outp_size*4,real32)*1.0e-9_real32
    write(*,'(/,A,I0,A,I0,A,I0,A,I0,A,I0)') 'num_head = ',nhead,achar(9)//'seq_len = ',seq_len, &
      achar(9)//'batch_size = ',batch_size,achar(9)//'hidden_dimension = ',hidden_dim,achar(9)//'beam_size = ',beam_size
    write(*,'(A,F0.2)') 'Total device memory usage (GB) = ',size_bytes
    allocate(inp1(0:inp1_size-1),inp2(0:inp2_size-1),outp(0:outp_size-1),outref(0:outp_size-1))
    do i=0,inp1_size-1; inp1(i)=real(mod(c_rand(),inp1_size),real32); end do
    do i=0,inp2_size-1; inp2(i)=real(mod(c_rand(),inp2_size),real32); end do
!$omp target data map(to:inp1,inp2) map(alloc:outp)
    call concat(inp1,inp2,outp,sz0,head_dim,sl1,sl2)
!$omp target update from(outp)
    call concat_cpu(inp1,inp2,outref,sz0,head_dim,sl1,sl2)
    if(all(outref==outp)) then; write(*,'(A)')'PASS'; else; write(*,'(A)')'FAIL'; end if
    start=omp_get_wtime(); do i=1,repeat; call concat(inp1,inp2,outp,sz0,head_dim,sl1,sl2); end do; stop=omp_get_wtime()
    avg=(stop-start)*1.0e6_real64/real(repeat,real64)
    write(*,'(A,F0.6,A)')'Average kernel execution time: ',avg,' (us)'
    write(*,'(A,F0.6,A)')'Average kernel throughput : ',size_bytes/(avg*1.0e-6_real64),' (GB/s)'
!$omp end target data
    deallocate(inp1,inp2,outp,outref)
  end do
end program
