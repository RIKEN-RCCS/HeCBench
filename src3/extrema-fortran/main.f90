program extrema
  use, intrinsic :: iso_fortran_env, only : int32, int64, real32, real64
  use, intrinsic :: iso_c_binding, only : c_int
  implicit none
  integer :: repeat, order
  integer(int64) :: total_time
  character(len=64) :: arg
  interface
    function c_rand() bind(C,name='rand') result(value)
      import c_int
      integer(c_int)::value
    end function c_rand
  end interface
  if(command_argument_count()/=1)then;write(*,'(a)')'Usage ./main <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)repeat
  if(repeat<=0)error stop 'repeat must be positive'
  total_time=0_int64
  order=1
  do while(order<=128)
    total_time=total_time+test_1d(1000000,order,.true.,repeat,'int',1)
    total_time=total_time+test_1d(1000000,order,.true.,repeat,'long',2)
    total_time=total_time+test_1d(1000000,order,.true.,repeat,'float',3)
    total_time=total_time+test_1d(1000000,order,.true.,repeat,'double',4)
    order=order*2
  end do
  order=1
  do while(order<=128)
    total_time=total_time+test_2d(1000,1000,order,.true.,1,repeat,'int',1)
    total_time=total_time+test_2d(1000,1000,order,.true.,1,repeat,'long',2)
    total_time=total_time+test_2d(1000,1000,order,.true.,1,repeat,'float',3)
    total_time=total_time+test_2d(1000,1000,order,.true.,1,repeat,'double',4)
    order=order*2
  end do
  order=1
  do while(order<=128)
    total_time=total_time+test_2d(1000,1000,order,.true.,0,repeat,'int',1)
    total_time=total_time+test_2d(1000,1000,order,.true.,0,repeat,'long',2)
    total_time=total_time+test_2d(1000,1000,order,.true.,0,repeat,'float',3)
    total_time=total_time+test_2d(1000,1000,order,.true.,0,repeat,'double',4)
    order=order*2
  end do
  write(*,'(/,a,/,a,f0.6,a,/,a)') '-----------------------------------------------', &
    'Total kernel execution time: ',real(total_time,real64)*1.0e-9_real64,' (s)','-----------------------------------------------'
contains
  integer(int64) function test_1d(length,order,clip,repeat,type_name,kind_case) result(time_ns)
    integer,intent(in)::length,order,repeat,kind_case
    logical,intent(in)::clip
    character(*),intent(in)::type_name
    integer(int32),allocatable::input_i32(:)
    integer(int64),allocatable::input_i64(:)
    real(real32),allocatable::input_sp(:)
    real(real64),allocatable::input_dp(:)
    logical,allocatable::cpu_result(:),gpu_result(:)
    integer::i,n,o,plus,minus
    integer(int64)::start_count,end_count,rate
    logical::temp,error
    allocate(input_i32(0:length-1),input_i64(0:length-1),input_sp(0:length-1),input_dp(0:length-1),cpu_result(0:length-1),gpu_result(0:length-1))
    do i=0,length-1
      select case(kind_case)
      case(1);input_i32(i)=modulo(c_rand(),int(length,c_int))
      case(2);input_i64(i)=int(modulo(c_rand(),int(length,c_int)),int64)
      case(3);input_sp(i)=real(modulo(c_rand(),int(length,c_int)),real32)
      case(4);input_dp(i)=real(modulo(c_rand(),int(length,c_int)),real64)
      end select
    end do
!$omp target data map(from:gpu_result(0:length-1))
!$omp target data if(kind_case==1) map(to:input_i32(0:length-1))
!$omp target data if(kind_case==2) map(to:input_i64(0:length-1))
!$omp target data if(kind_case==3) map(to:input_sp(0:length-1))
!$omp target data if(kind_case==4) map(to:input_dp(0:length-1))
    call system_clock(start_count,rate)
    do n=1,repeat
!$omp target teams distribute parallel do thread_limit(256) private(o,plus,minus,temp)
      do i=0,length-1
        temp=.true.
        do o=1,order
          plus=i+o;minus=i-o
          if(plus>=length)then;if(clip)then;plus=length-1;else;plus=plus-length;end if;end if
          if(minus<0)then;if(clip)then;minus=0;else;minus=minus+length;end if;end if
          select case(kind_case)
          case(1);temp=temp .and. input_i32(i)>input_i32(plus) .and. input_i32(i)>=input_i32(minus)
          case(2);temp=temp .and. input_i64(i)>input_i64(plus) .and. input_i64(i)>=input_i64(minus)
          case(3);temp=temp .and. input_sp(i)>input_sp(plus) .and. input_sp(i)>=input_sp(minus)
          case(4);temp=temp .and. input_dp(i)>input_dp(plus) .and. input_dp(i)>=input_dp(minus)
          end select
        end do
        gpu_result(i)=temp
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(end_count);time_ns=end_count-start_count
    write(*,'(a,a,a,i0,a,l1,a,f0.6,a)')'Average 1D kernel (type = ',trim(type_name),', order = ',order,', clip = ',clip,') execution time ',real(time_ns,real64)*1e-9_real64/real(repeat,real64),' (s)'
!$omp end target data
!$omp end target data
!$omp end target data
!$omp end target data
!$omp end target data
    do i=0,length-1
      temp=.true.
      do o=1,order
        plus=i+o;minus=i-o;call clip_both(clip,length,plus,minus)
        select case(kind_case)
        case(1);temp=temp .and. input_i32(i)>input_i32(plus) .and. input_i32(i)>=input_i32(minus)
        case(2);temp=temp .and. input_i64(i)>input_i64(plus) .and. input_i64(i)>=input_i64(minus)
        case(3);temp=temp .and. input_sp(i)>input_sp(plus) .and. input_sp(i)>=input_sp(minus)
        case(4);temp=temp .and. input_dp(i)>input_dp(plus) .and. input_dp(i)>=input_dp(minus)
        end select
      end do
      cpu_result(i)=temp
    end do
    error=any(cpu_result .neqv. gpu_result);if(error)write(*,'(a)')'1D test: FAILED'
    deallocate(input_i32,input_i64,input_sp,input_dp,cpu_result,gpu_result)
  end function test_1d
  integer(int64) function test_2d(length_x,length_y,order,clip,axis,repeat,type_name,kind_case) result(time_ns)
    integer,intent(in)::length_x,length_y,order,axis,repeat,kind_case
    logical,intent(in)::clip
    character(*),intent(in)::type_name
    integer::length,tx,ty,tid,n,o,plus,minus,i
    integer(int64)::start_count,end_count,rate
    integer(int32),allocatable::input_i32(:)
    integer(int64),allocatable::input_i64(:)
    real(real32),allocatable::input_sp(:)
    real(real64),allocatable::input_dp(:)
    logical,allocatable::cpu_result(:),gpu_result(:)
    logical::temp,error
    length=length_x*length_y;allocate(input_i32(0:length-1),input_i64(0:length-1),input_sp(0:length-1),input_dp(0:length-1),cpu_result(0:length-1),gpu_result(0:length-1))
    do i=0,length-1
      select case(kind_case)
      case(1);input_i32(i)=modulo(c_rand(),int(length,c_int))
      case(2);input_i64(i)=int(modulo(c_rand(),int(length,c_int)),int64)
      case(3);input_sp(i)=real(modulo(c_rand(),int(length,c_int)),real32)
      case(4);input_dp(i)=real(modulo(c_rand(),int(length,c_int)),real64)
      end select
    end do
!$omp target data map(from:gpu_result(0:length-1))
!$omp target data if(kind_case==1) map(to:input_i32(0:length-1))
!$omp target data if(kind_case==2) map(to:input_i64(0:length-1))
!$omp target data if(kind_case==3) map(to:input_sp(0:length-1))
!$omp target data if(kind_case==4) map(to:input_dp(0:length-1))
    call system_clock(start_count,rate)
    do n=1,repeat
!$omp target teams distribute parallel do collapse(2) thread_limit(256) private(tid,o,plus,minus,temp)
      do tx=0,length_y-1
        do ty=0,length_x-1
          tid=tx*length_x+ty;temp=.true.
          do o=1,order
            if(axis==0)then
              plus=tx+o;minus=tx-o
              if(plus>=length_y)then;if(clip)then;plus=length_y-1;else;plus=plus-length_y;end if;end if
              if(minus<0)then;if(clip)then;minus=0;else;minus=minus+length_y;end if;end if
              plus=plus*length_x+ty;minus=minus*length_x+ty
            else
              plus=ty+o;minus=ty-o
              if(plus>=length_x)then;if(clip)then;plus=length_x-1;else;plus=plus-length_x;end if;end if
              if(minus<0)then;if(clip)then;minus=0;else;minus=minus+length_x;end if;end if
              plus=tx*length_x+plus;minus=tx*length_x+minus
            end if
            select case(kind_case)
            case(1);temp=temp .and. input_i32(tid)>input_i32(plus) .and. input_i32(tid)>=input_i32(minus)
            case(2);temp=temp .and. input_i64(tid)>input_i64(plus) .and. input_i64(tid)>=input_i64(minus)
            case(3);temp=temp .and. input_sp(tid)>input_sp(plus) .and. input_sp(tid)>=input_sp(minus)
            case(4);temp=temp .and. input_dp(tid)>input_dp(plus) .and. input_dp(tid)>=input_dp(minus)
            end select
          end do
          gpu_result(tid)=temp
        end do
      end do
!$omp end target teams distribute parallel do
    end do
    call system_clock(end_count);time_ns=end_count-start_count
    write(*,'(a,a,a,i0,a,l1,a,i0,a,f0.6,a)')'Average 2D kernel (type = ',trim(type_name),', order = ',order,', clip = ',clip,', axis = ',axis,') execution time ',real(time_ns,real64)*1e-9_real64/real(repeat,real64),' (s)'
!$omp end target data
!$omp end target data
!$omp end target data
!$omp end target data
!$omp end target data
    do tx=0,length_y-1
      do ty=0,length_x-1
        tid=tx*length_x+ty;temp=.true.
        do o=1,order
          if(axis==0)then;plus=tx+o;minus=tx-o;call clip_both(clip,length_y,plus,minus);plus=plus*length_x+ty;minus=minus*length_x+ty
          else;plus=ty+o;minus=ty-o;call clip_both(clip,length_x,plus,minus);plus=tx*length_x+plus;minus=tx*length_x+minus;end if
          select case(kind_case)
          case(1);temp=temp .and. input_i32(tid)>input_i32(plus) .and. input_i32(tid)>=input_i32(minus)
          case(2);temp=temp .and. input_i64(tid)>input_i64(plus) .and. input_i64(tid)>=input_i64(minus)
          case(3);temp=temp .and. input_sp(tid)>input_sp(plus) .and. input_sp(tid)>=input_sp(minus)
          case(4);temp=temp .and. input_dp(tid)>input_dp(plus) .and. input_dp(tid)>=input_dp(minus)
          end select
        end do
        cpu_result(tid)=temp
      end do
    end do
    error=any(cpu_result .neqv. gpu_result);if(error)write(*,'(a)')'2D test: FAILED'
    deallocate(input_i32,input_i64,input_sp,input_dp,cpu_result,gpu_result)
  end function test_2d
  subroutine clip_plus(clip,n,plus)
    logical,intent(in)::clip;integer,intent(in)::n;integer,intent(inout)::plus
    if(plus>=n)plus=merge(n-1,plus-n,clip)
  end subroutine clip_plus
  subroutine clip_minus(clip,n,minus)
    logical,intent(in)::clip;integer,intent(in)::n;integer,intent(inout)::minus
    if(minus<0)minus=merge(0,minus+n,clip)
  end subroutine clip_minus
  subroutine clip_both(clip,n,plus,minus)
    logical,intent(in)::clip;integer,intent(in)::n;integer,intent(inout)::plus,minus
    call clip_plus(clip,n,plus);call clip_minus(clip,n,minus)
  end subroutine clip_both
end program extrema
