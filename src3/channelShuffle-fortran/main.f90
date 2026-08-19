module channel_shuffle_kernels
  use iso_fortran_env, only : real32, real64, int64
  use omp_lib, only : omp_get_wtime
  implicit none
  integer, parameter :: num_threads = 256
contains

  subroutine channel_shuffle_nchw_kernel(n, g, k, hxw, x, y)
    integer, intent(in) :: n, g, k, hxw
    real(real32), intent(in) :: x(0:)
    real(real32), intent(out) :: y(0:)
    integer :: c, nn, s, channels
    channels = g*k
    !$omp target teams distribute parallel do collapse(3) num_threads(num_threads)
    do nn = 0, n-1
      do c = 0, channels-1
        do s = 0, hxw-1
          y((nn*channels+c)*hxw+s) = x((nn*channels+mod(c,g)*k+c/g)*hxw+s)
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine channel_shuffle_nchw_kernel

  subroutine channel_shuffle_nhwc_kernel(o, g, k, x, y)
    integer, intent(in) :: o, g, k
    real(real32), intent(in) :: x(0:)
    real(real32), intent(out) :: y(0:)
    integer :: oo, i, channels
    channels = g*k
    !$omp target teams distribute parallel do collapse(2) num_threads(num_threads)
    do oo = 0, o-1
      do i = 0, channels-1
        y(oo*channels+i) = x(oo*channels+mod(i,g)*k+i/g)
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine channel_shuffle_nhwc_kernel

  logical function channel_shuffle_nchw(x, n, c, g, numel, y, elapsed_ns, repeat)
    real(real32), intent(in) :: x(0:)
    real(real32), intent(out) :: y(0:)
    integer, intent(in) :: n, c, g, numel, repeat
    integer(int64), intent(out) :: elapsed_ns
    integer :: k, hxw, iteration
    real(real64) :: start_time, end_time
    if (mod(c,g) /= 0 .or. numel < n*c) then
      channel_shuffle_nchw = .false.
      return
    end if
    k = c/g; hxw = numel/(n*c)
    start_time = omp_get_wtime()
    do iteration = 0, repeat-1
      call channel_shuffle_nchw_kernel(n,g,k,hxw,x,y)
    end do
    end_time = omp_get_wtime()
    elapsed_ns = int((end_time-start_time)*1.0e9_real64,int64)
    channel_shuffle_nchw = .true.
  end function channel_shuffle_nchw

  logical function channel_shuffle_nhwc(x, n, c, g, numel, y, elapsed_ns, repeat)
    real(real32), intent(in) :: x(0:)
    real(real32), intent(out) :: y(0:)
    integer, intent(in) :: n, c, g, numel, repeat
    integer(int64), intent(out) :: elapsed_ns
    integer :: k, hxw, o, iteration
    real(real64) :: start_time, end_time
    if (mod(c,g) /= 0 .or. numel < n*c) then
      channel_shuffle_nhwc = .false.
      return
    end if
    k = c/g; hxw = numel/(n*c); o = n*hxw
    start_time = omp_get_wtime()
    do iteration = 0, repeat-1
      call channel_shuffle_nhwc_kernel(o,g,k,x,y)
    end do
    end_time = omp_get_wtime()
    elapsed_ns = int((end_time-start_time)*1.0e9_real64,int64)
    channel_shuffle_nhwc = .true.
  end function channel_shuffle_nhwc

  subroutine channel_shuffle_nchw_cpu(x, n, c, g, numel, y)
    real(real32), intent(in) :: x(0:)
    real(real32), intent(out) :: y(0:)
    integer, intent(in) :: n, c, g, numel
    integer :: k, hxw, nn, cc, s
    k=c/g; hxw=numel/(n*c)
    !$omp parallel do collapse(3)
    do nn=0,n-1
      do cc=0,c-1
        do s=0,hxw-1
          y((nn*c+cc)*hxw+s)=x((nn*c+mod(cc,g)*k+cc/g)*hxw+s)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine channel_shuffle_nchw_cpu

  subroutine channel_shuffle_nhwc_cpu(x, n, c, g, numel, y, repeat)
    real(real32), intent(in) :: x(0:)
    real(real32), intent(out) :: y(0:)
    integer, intent(in) :: n, c, g, numel, repeat
    integer :: k, hxw, o, oo, i, iteration
    k=c/g; hxw=numel/(n*c); o=n*hxw
    do iteration=0,repeat-1
      !$omp parallel do collapse(2)
      do oo=0,o-1
        do i=0,c-1
          y(oo*c+i)=x(oo*c+mod(i,g)*k+i/g)
        end do
      end do
      !$omp end parallel do
    end do
  end subroutine channel_shuffle_nhwc_cpu
end module channel_shuffle_kernels

program main
  use iso_fortran_env, only : real32, real64, int64
  use channel_shuffle_kernels
  implicit none
  integer :: argc, g, w, h, repeat, n, c, numel, element, ios
  integer(int64) :: elapsed_ns
  real(real32), allocatable :: h_x(:), h_y(:), h_y_ref(:)
  logical :: ok
  character(len=64) :: arg

  argc=command_argument_count()
  if (argc /= 4) then
    call get_command_argument(0,arg)
    print '(a,a,a)', 'Usage: ',trim(arg),' <group size> <width> <height> <repeat>'
    error stop 1
  end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) g; if (ios /= 0) error stop 1
  call get_command_argument(2,arg); read(arg,*,iostat=ios) w; if (ios /= 0) error stop 1
  call get_command_argument(3,arg); read(arg,*,iostat=ios) h; if (ios /= 0) error stop 1
  call get_command_argument(4,arg); read(arg,*,iostat=ios) repeat; if (ios /= 0) error stop 1

  n=1
  do while (n <= 64)
    c=32
    do while (c <= 512)
      print '(/,a,i0,a,i0,a,i0,a,i0,a)', '(N=',n,' C=',c,' W=',w,' H=',h,')'
      numel=n*c*w*h
      allocate(h_x(0:numel-1),h_y(0:numel-1),h_y_ref(0:numel-1))
      do element=0,numel-1
        h_x(element)=real(element,real32)/real(numel,real32)
      end do
      !$omp target data map(to:h_x(0:numel-1)) map(alloc:h_y(0:numel-1))
      ok=channel_shuffle_nhwc(h_x,n,c,g,numel,h_y,elapsed_ns,repeat)
      !$omp target update from(h_y(0:numel-1))
      call channel_shuffle_nhwc_cpu(h_x,n,c,g,numel,h_y_ref,repeat)
      if (any(h_y /= h_y_ref)) then
        print '(a)', 'Failed to pass channel shuffle (NHWC) check'
      else
        print '(a,f0.6,a)', 'Average time of channel shuffle (NHWC): ',real(elapsed_ns,real64)*1.0e-6_real64/real(repeat,real64),' (ms)'
      end if
      ok=channel_shuffle_nchw(h_x,n,c,g,numel,h_y,elapsed_ns,repeat)
      !$omp target update from(h_y(0:numel-1))
      call channel_shuffle_nchw_cpu(h_x,n,c,g,numel,h_y_ref)
      if (any(h_y /= h_y_ref)) then
        print '(a)', 'Failed to pass channel shuffle (NCHW) check'
      else
        print '(a,f0.6,a)', 'Average time of channel shuffle (NCHW): ',real(elapsed_ns,real64)*1.0e-6_real64/real(repeat,real64),' (ms)'
      end if
      !$omp end target data
      deallocate(h_x,h_y,h_y_ref)
      c=c*4
    end do
    n=n*4
  end do
end program main
