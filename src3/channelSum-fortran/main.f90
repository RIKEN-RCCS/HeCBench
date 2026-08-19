module channel_sum
  use iso_fortran_env, only : int32, int64, real64
  use iso_c_binding, only : c_int
  use omp_lib, only : omp_get_wtime
  implicit none
  integer, parameter :: num_threads = 256
  interface
    subroutine c_srand(seed) bind(C,name='srand')
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C,name='rand') result(value)
      import :: c_int
      integer(c_int) :: value
    end function c_rand
  end interface
contains

  subroutine channel_sum_nchw(n,c,hxw,x,sum,sumsq)
    integer, intent(in) :: n,c,hxw
    integer(int32), intent(in) :: x(0:)
    integer(int32), intent(out) :: sum(0:), sumsq(0:)
    integer :: channel, nn, hw, index
    integer(int32) :: m_val, v_val
    !$omp target teams distribute num_teams(c) private(m_val,v_val)
    do channel=0,c-1
      m_val=0_int32; v_val=0_int32
      !$omp parallel do collapse(2) reduction(+:m_val,v_val) num_threads(num_threads)
      do nn=0,n-1
        do hw=0,hxw-1
          index=(nn*c+channel)*hxw+hw
          m_val=m_val+x(index)
          v_val=v_val+x(index)*x(index)
        end do
      end do
      !$omp end parallel do
      sum(channel)=m_val
      sumsq(channel)=v_val
    end do
    !$omp end target teams distribute
  end subroutine channel_sum_nchw

  subroutine channel_sum_nhwc(n,c,hxw,x,sum,sumsq)
    integer, intent(in) :: n,c,hxw
    integer(int32), intent(in) :: x(0:)
    integer(int32), intent(out) :: sum(0:), sumsq(0:)
    integer :: channel, item, index
    integer(int32) :: m_val, v_val
    !$omp target teams distribute num_teams(c) private(m_val,v_val)
    do channel=0,c-1
      m_val=0_int32; v_val=0_int32
      !$omp parallel do reduction(+:m_val,v_val) num_threads(num_threads)
      do item=0,n*hxw-1
        index=item*c+channel
        m_val=m_val+x(index)
        v_val=v_val+x(index)*x(index)
      end do
      !$omp end parallel do
      sum(channel)=m_val
      sumsq(channel)=v_val
    end do
    !$omp end target teams distribute
  end subroutine channel_sum_nhwc

  subroutine compute_channel_sum_nchw(n,c,hxw,x,sum,sumsq,elapsed_ns,repeat)
    integer, intent(in) :: n,c,hxw,repeat
    integer(int32), intent(in) :: x(0:)
    integer(int32), intent(out) :: sum(0:), sumsq(0:)
    integer(int64), intent(out) :: elapsed_ns
    integer :: iteration
    real(real64) :: start_time,end_time
    start_time=omp_get_wtime()
    do iteration=0,repeat-1
      call channel_sum_nchw(n,c,hxw,x,sum,sumsq)
    end do
    end_time=omp_get_wtime()
    elapsed_ns=int((end_time-start_time)*1.0e9_real64,int64)
  end subroutine compute_channel_sum_nchw

  subroutine compute_channel_sum_nhwc(n,c,hxw,x,sum,sumsq,elapsed_ns,repeat)
    integer, intent(in) :: n,c,hxw,repeat
    integer(int32), intent(in) :: x(0:)
    integer(int32), intent(out) :: sum(0:), sumsq(0:)
    integer(int64), intent(out) :: elapsed_ns
    integer :: iteration
    real(real64) :: start_time,end_time
    start_time=omp_get_wtime()
    do iteration=0,repeat-1
      call channel_sum_nhwc(n,c,hxw,x,sum,sumsq)
    end do
    end_time=omp_get_wtime()
    elapsed_ns=int((end_time-start_time)*1.0e9_real64,int64)
  end subroutine compute_channel_sum_nhwc

  subroutine ref_nchw(n,c,hxw,x,sum,sumsq)
    integer, intent(in) :: n,c,hxw
    integer(int32), intent(in) :: x(0:)
    integer(int32), intent(out) :: sum(0:), sumsq(0:)
    integer :: channel,nn,hw,index
    integer(int32) :: m_val,v_val
    do channel=0,c-1
      m_val=0_int32; v_val=0_int32
      do nn=0,n-1
        do hw=0,hxw-1
          index=(nn*c+channel)*hxw+hw
          m_val=m_val+x(index); v_val=v_val+x(index)*x(index)
        end do
      end do
      sum(channel)=m_val; sumsq(channel)=v_val
    end do
  end subroutine ref_nchw

  subroutine ref_nhwc(n,c,hxw,x,sum,sumsq)
    integer, intent(in) :: n,c,hxw
    integer(int32), intent(in) :: x(0:)
    integer(int32), intent(out) :: sum(0:), sumsq(0:)
    integer :: channel,item,index
    integer(int32) :: m_val,v_val
    do channel=0,c-1
      m_val=0_int32; v_val=0_int32
      do item=0,n*hxw-1
        index=item*c+channel
        m_val=m_val+x(index); v_val=v_val+x(index)*x(index)
      end do
      sum(channel)=m_val; sumsq(channel)=v_val
    end do
  end subroutine ref_nhwc

  logical function check(size,device,host)
    integer, intent(in) :: size
    integer(int32), intent(in) :: device(0:),host(0:)
    integer :: item
    check=.true.
    do item=0,size-1
      if (abs(device(item)-host(item)) > 1_int32) then
        check=.false.; exit
      end if
    end do
  end function check
end module channel_sum

program main
  use iso_fortran_env, only : int32, int64, real64
  use channel_sum
  implicit none
  integer :: argc,w,h,repeat,n,c,numel,item,ios
  integer(int64) :: elapsed_ns
  integer(int32), allocatable :: h_x(:),h_sum(:),h_sumsq(:),r_sum(:)
  logical :: ok
  character(len=64) :: arg
  argc=command_argument_count()
  if (argc /= 3) then
    call get_command_argument(0,arg)
    print '(a,a,a)', 'Usage: ',trim(arg),' <width> <height> <repeat>'
    error stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) w; if(ios/=0) error stop 1
  call get_command_argument(2,arg); read(arg,*,iostat=ios) h; if(ios/=0) error stop 1
  call get_command_argument(3,arg); read(arg,*,iostat=ios) repeat; if(ios/=0) error stop 1
  n=1
  do while(n<=64)
    c=32
    do while(c<=512)
      print '(/,a,i0,a,i0,a,i0,a,i0,a)','(N=',n,' C=',c,' W=',w,' H=',h,')'
      numel=n*c*w*h
      allocate(h_x(0:numel-1),h_sum(0:c-1),h_sumsq(0:c-1),r_sum(0:c-1))
      call c_srand(int(numel,c_int))
      do item=0,numel-1
        h_x(item)=mod(c_rand(),256_c_int)
      end do
      !$omp target data map(to:h_x(0:numel-1)) map(from:h_sum(0:c-1),h_sumsq(0:c-1))
      call compute_channel_sum_nhwc(n,c,w*h,h_x,h_sum,h_sumsq,elapsed_ns,repeat)
      !$omp target update from(h_sum(0:c-1))
      call ref_nhwc(n,c,w*h,h_x,r_sum,h_sumsq)
      ok=check(c,h_sum,r_sum)
      print '(a,f0.6,a)','Average time of channel sum (nhwc): ',real(elapsed_ns,real64)*1.0e-6_real64/real(repeat,real64),' (ms)'
      if(ok) then
        print '(a)','Verification PASS for channel sum (nhwc)'
      else
        print '(a)','Verification FAIL for channel sum (nhwc)'
      end if
      call compute_channel_sum_nchw(n,c,w*h,h_x,h_sum,h_sumsq,elapsed_ns,repeat)
      !$omp target update from(h_sum(0:c-1))
      call ref_nchw(n,c,w*h,h_x,r_sum,h_sumsq)
      ok=check(c,h_sum,r_sum)
      print '(a,f0.6,a)','Average time of channel sum (nchw): ',real(elapsed_ns,real64)*1.0e-6_real64/real(repeat,real64),' (ms)'
      if(ok) then
        print '(a)','Verification PASS for channel sum (nchw)'
      else
        print '(a)','Verification FAIL for channel sum (nchw)'
      end if
      !$omp end target data
      deallocate(h_x,h_sum,h_sumsq,r_sum)
      c=c*4
    end do
    n=n*4
  end do
end program main
