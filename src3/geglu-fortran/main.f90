module geglu_module
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  implicit none
  integer(int64), parameter :: minstd_multiplier=16807_int64, minstd_modulus=2147483647_int64
!$omp declare target (gelu)
contains
  pure function gelu(x) result(value)
    real(real32),intent(in)::x
    real(real32)::value
    value=x*0.5_real32*(1.0_real32+erf(x*0.7071067811865475244_real32))
  end function gelu
  subroutine geglu_gpu(output,input,n,dim_last)
    integer,intent(in)::n,dim_last
    real(real32),intent(in)::input(0:)
    real(real32),intent(inout)::output(0:)
    integer::i,d
!$omp target teams distribute parallel do collapse(2) num_threads(160)
    do i=0,n-1
      do d=0,dim_last-1
        output(i*dim_last+d)=input((i*2)*dim_last+d)*gelu(input((i*2+1)*dim_last+d))
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine geglu_gpu
  subroutine geglu_reference(output,input,n,dim_last)
    integer,intent(in)::n,dim_last
    real(real32),intent(in)::input(0:)
    real(real32),intent(out)::output(0:)
    integer::i,d
    do i=0,n-1
      do d=0,dim_last-1
        output(i*dim_last+d)=input((i*2)*dim_last+d)*gelu(input((i*2+1)*dim_last+d))
      end do
    end do
  end subroutine geglu_reference
  subroutine fill_input(input,state)
    real(real32),intent(out)::input(0:)
    integer(int64),intent(inout)::state
    integer(int64)::i
    do i=0,size(input,kind=int64)-1
      state=modulo(minstd_multiplier*state,minstd_modulus)
      input(i)=-6.0_real32+12.0_real32*real(state,real32)/real(minstd_modulus,real32)
    end do
  end subroutine fill_input
end module geglu_module

program geglu
  use, intrinsic :: iso_fortran_env, only : int64, real32, real64
  use omp_lib
  use geglu_module
  implicit none
  integer,parameter::verify_batches(2)=[1,4],verify_shapes(3)=[128,256,512]
  integer,parameter::measure_batches(3)=[1,4,16],measure_shapes(2)=[4096,8192],measure_dims(3)=[1280,2560,5120]
  integer::argc,repeat,batch,shape,dim_last,n,i,j,k
  integer(int64)::nelems,state
  real(real32),allocatable::x_and_gate(:),output(:),output_ref(:)
  real(real64)::start_time,end_time
  real(real32)::elapsed_us
  character(len=64)::argument
  argc=command_argument_count()
  if(argc/=1)then;print '(a)','Usage: ./main <repeat>';stop 1;end if
  call get_command_argument(1,argument);read(argument,*)repeat
  state=123_int64
  do i=1,size(verify_batches)
    batch=verify_batches(i)
    do j=1,size(verify_shapes)
      shape=verify_shapes(j);dim_last=1280;nelems=int(batch,int64)*int(shape,int64)*int(dim_last,int64)*2_int64
      allocate(x_and_gate(0:nelems-1),output(0:nelems/2-1),output_ref(0:nelems/2-1))
      call fill_input(x_and_gate,state);call geglu_reference(output_ref,x_and_gate,batch*shape,dim_last)
!$omp target data map(to:x_and_gate) map(from:output)
      call geglu_gpu(output,x_and_gate,batch*shape,dim_last)
!$omp end target data
      if(all(abs(output-output_ref)<=1.0e-3_real32))then;print '(a)','PASS';else;print '(a)','FAIL';end if
      deallocate(x_and_gate,output,output_ref)
    end do
  end do
  do i=1,size(measure_batches)
    batch=measure_batches(i)
    do j=1,size(measure_shapes)
      shape=measure_shapes(j)
      do k=1,size(measure_dims)
        dim_last=measure_dims(k);nelems=int(batch,int64)*int(shape,int64)*int(dim_last,int64)*2_int64
        allocate(x_and_gate(0:nelems-1),output(0:nelems/2-1));call fill_input(x_and_gate,state)
!$omp target data map(to:x_and_gate) map(alloc:output)
        start_time=omp_get_wtime()
        do n=1,repeat;call geglu_gpu(output,x_and_gate,batch*shape,dim_last);end do
        end_time=omp_get_wtime()
        elapsed_us=real((end_time-start_time)*1.0e6_real64/real(repeat,real64),real32)
        print '(a,i0,a,i0,a,i0)','Batch size: ',batch,', sequence length: ',shape,', hidden dimension: ',dim_last
        print '(a,f0.6,a)','Average execution time of GeGLU kernel: ',elapsed_us,' (us)'
!$omp end target data
        deallocate(x_and_gate,output)
      end do
    end do
  end do
end program geglu
