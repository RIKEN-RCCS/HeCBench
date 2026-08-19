program spm
  use iso_c_binding, only: c_int
  use iso_fortran_env, only: int8, real32, real64
  use omp_lib
  use spm_mod
  implicit none

  interface
    subroutine c_srand(seed) bind(C, name="srand")
      import :: c_int
      integer(c_int), value :: seed
    end subroutine c_srand
    function c_rand() bind(C, name="rand") result(r)
      import :: c_int
      integer(c_int) :: r
    end function c_rand
  end interface

  integer :: argc, v, repeat, data_size, vol_size, i, count_dev, count_host, max_diff
  type(int3) :: g_vol, f_vol
  integer, allocatable :: hist_d(:), hist_h(:)
  integer(int8), allocatable :: f_h(:), g_h(:), ivf_h(:), ivg_h(:)
  logical(kind=1), allocatable :: data_threshold_h(:)
  real(real32) :: M_h(0:15)
  real(real64) :: start_time, end_time
  character(len=64) :: arg

  argc = command_argument_count()
  if (argc /= 2) then
    print '(a)', 'Usage: ./main <dimension> <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) v
  call get_command_argument(2, arg); read(arg, *) repeat

  g_vol = int3(v, v, v)
  f_vol = int3(v, v, v)
  data_size = (g_vol%x + 1) * (g_vol%y + 1) * (g_vol%z + 5)
  vol_size = g_vol%x * g_vol%y * g_vol%z
  allocate(hist_d(0:65535), hist_h(0:65535), f_h(0:data_size-1), g_h(0:data_size-1), &
           ivf_h(0:vol_size-1), ivg_h(0:vol_size-1), data_threshold_h(0:vol_size-1))
  hist_d = 0
  hist_h = 0

  call c_srand(123_c_int)
  do i = 0, 15
    M_h(i) = real(c_rand(), real32) / 2147483647.0_real32
  end do
  do i = 0, data_size - 1
    f_h(i) = byte_value(mod(c_rand(), 256_c_int))
    g_h(i) = byte_value(mod(c_rand(), 256_c_int))
  end do

  !$omp target data map(to:M_h(0:15), g_h(0:data_size-1), f_h(0:data_size-1)) &
  !$omp& map(from:ivf_h(0:vol_size-1), ivg_h(0:vol_size-1), data_threshold_h(0:vol_size-1))
    start_time = omp_get_wtime()
    do i = 0, repeat - 1
      call spm_kernel(M_h, vol_size, g_h, f_h, g_vol, f_vol, ivf_h, ivg_h, data_threshold_h)
    end do
    end_time = omp_get_wtime()
    print '(a,f12.6,a)', 'Average kernel execution time: ', ((end_time-start_time)*1000.0_real64)/repeat, ' (ms)'
  !$omp end target data

  count_dev = 0
  do i = 0, vol_size - 1
    if (data_threshold_h(i)) then
      hist_d(u8(ivf_h(i)) + u8(ivg_h(i)) * 256) = hist_d(u8(ivf_h(i)) + u8(ivg_h(i)) * 256) + 1
      count_dev = count_dev + 1
    end if
  end do
  print '(a,i0)', 'Device count: ', count_dev

  count_host = 0
  call spm_reference(M_h, vol_size, g_h, f_h, g_vol, f_vol, ivf_h, ivg_h, data_threshold_h)
  do i = 0, vol_size - 1
    if (data_threshold_h(i)) then
      hist_h(u8(ivf_h(i)) + u8(ivg_h(i)) * 256) = hist_h(u8(ivf_h(i)) + u8(ivg_h(i)) * 256) + 1
      count_host = count_host + 1
    end if
  end do
  print '(a,i0)', 'Host count: ', count_host

  max_diff = maxval(abs(hist_h - hist_d))
  print '(a,i0)', 'Maximum difference ', max_diff
end program spm
