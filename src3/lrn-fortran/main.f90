module c_rng
  use iso_c_binding, only: c_int
  implicit none
  integer, parameter :: RAND_MAX_F = 2147483647
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
contains
  real function rand_unit()
    rand_unit = real(rand()) / real(RAND_MAX_F)
  end function rand_unit
end module c_rng

module lrn_kernels
  use iso_fortran_env, only: int64, real32
  implicit none
!$omp declare target(data_off)
contains
  pure integer(int64) function data_off(mb, c, d, h, w, stride_mb, hdim, wdim)
    integer(int64), intent(in) :: mb, c, d, h, w, stride_mb, hdim, wdim
    data_off = mb * stride_mb + c * hdim * wdim + h * wdim + w
  end function data_off

  subroutine lrn_fwd_kernel(src, dst, n, cdim, ddim, hdim, wdim, stride_mb, wg_cnt, wg_size, wk_size, size, alpha, beta, kval)
    integer(int64), intent(in) :: n, cdim, ddim, hdim, wdim, stride_mb, wg_cnt, wg_size, wk_size, size
    real(real32), intent(in) :: src(0:*)
    real(real32), intent(out) :: dst(0:*)
    real(real32), intent(in) :: alpha, beta, kval
    integer(int64) :: idx, mb, c, d, h, w, c_st, c_en, cc, off, half_size
    real(real32) :: sumv, omega, s
!$omp target teams distribute parallel do num_teams(wg_cnt) num_threads(wg_size) &
!$omp& map(to:src(0:wk_size-1)) map(from:dst(0:wk_size-1)) &
!$omp& private(mb,c,d,h,w,c_st,c_en,cc,off,half_size,sumv,omega,s)
    do idx = 0, wk_size-1
      mb = mod(idx / (cdim*ddim*hdim*wdim), n)
      c = mod(idx / (ddim*hdim*wdim), cdim)
      d = mod(idx / (hdim*wdim), ddim)
      h = mod(idx / wdim, hdim)
      w = mod(idx, wdim)
      half_size = (size - 1) / 2
      c_st = max(c - half_size, 0_int64)
      c_en = min(c + half_size + 1_int64, cdim)
      sumv = 0.0_real32
      do cc = c_st, c_en-1
        off = data_off(mb, cc, d, h, w, stride_mb, hdim, wdim)
        s = src(off)
        sumv = sumv + s*s
      end do
      omega = kval + alpha * sumv / real(size, real32)
      off = data_off(mb, c, d, h, w, stride_mb, hdim, wdim)
      dst(off) = src(off) * sqrt(1.0_real32 / (sqrt(omega) * omega))
    end do
!$omp end target teams distribute parallel do
  end subroutine lrn_fwd_kernel

  subroutine lrn_bwd_kernel(src, dst, diff_src, n, cdim, ddim, hdim, wdim, stride_mb, wg_cnt, wg_size, wk_size, size, alpha, beta, kval)
    integer(int64), intent(in) :: n, cdim, ddim, hdim, wdim, stride_mb, wg_cnt, wg_size, wk_size, size
    real(real32), intent(in) :: src(0:*), dst(0:*)
    real(real32), intent(inout) :: diff_src(0:*)
    real(real32), intent(in) :: alpha, beta, kval
    integer(int64) :: idx, mb, c, d, h, w, c_st, c_en, cc, c2_st, c2_en, c2, off, off2, half_size
    real(real32) :: sumv, omega, omega_in_beta, aval, bval, tmp, src_val
!$omp target teams distribute parallel do num_teams(wg_cnt) num_threads(wg_size) &
!$omp& map(to:src(0:wk_size-1),dst(0:wk_size-1),diff_src(0:wk_size-1)) &
!$omp& private(mb,c,d,h,w,c_st,c_en,cc,c2_st,c2_en,c2,off,off2,half_size, &
!$omp& sumv,omega,omega_in_beta,aval,bval,tmp,src_val)
    do idx = 0, wk_size-1
      mb = mod(idx / (cdim*ddim*hdim*wdim), n)
      c = mod(idx / (ddim*hdim*wdim), cdim)
      d = mod(idx / (hdim*wdim), ddim)
      h = mod(idx / wdim, hdim)
      w = mod(idx, wdim)
      half_size = (size - 1) / 2
      c_st = max(c - half_size, 0_int64)
      c_en = min(c + half_size + 1_int64, cdim)
      aval = 0.0_real32
      bval = 0.0_real32
      do cc = c_st, c_en-1
        c2_st = max(cc - half_size, 0_int64)
        c2_en = min(cc + half_size + 1_int64, cdim)
        sumv = 0.0_real32
        do c2 = c2_st, c2_en-1
          off2 = data_off(mb, c2, d, h, w, stride_mb, hdim, wdim)
          sumv = sumv + src(off2) * src(off2)
        end do
        omega = kval + alpha * sumv / real(size, real32)
        omega_in_beta = sqrt(1.0_real32 / (sqrt(omega) * omega))
        off = data_off(mb, cc, d, h, w, stride_mb, hdim, wdim)
        tmp = omega_in_beta * dst(off)
        if (cc == c) aval = tmp
        bval = bval + src(off) * tmp / omega
      end do
      off = data_off(mb, c, d, h, w, stride_mb, hdim, wdim)
      src_val = src(off)
      diff_src(off) = aval - bval * (2.0_real32 * alpha * beta * src_val / real(size, real32))
    end do
!$omp end target teams distribute parallel do
  end subroutine lrn_bwd_kernel
end module lrn_kernels

program lrn
  use iso_fortran_env, only: int64, real32, real64
  use omp_lib
  use c_rng
  use lrn_kernels
  implicit none
  integer :: repeat, iarg
  integer(int64) :: ndims, size, n, cdim, ddim, hdim, wdim, stride_mb, wk_size, wg_size, wg_cnt, i, iter
  character(len=64) :: arg
  real(real32), allocatable :: src(:), dst(:), diff_src(:)
  real(real32) :: alpha, beta, kval
  real(real64) :: start_time, elapsed, checksum, data_ingb, bandwidth

  if (command_argument_count() /= 1) then
    print '(a)', 'Usage: ./main <repeat>'
    stop 1
  end if
  call get_command_argument(1, arg)
  read(arg, *) repeat

  call forward()
  call backward()

contains
  subroutine forward()
    ndims = 5; size = 5; alpha = 0.000122_real32; beta = 0.750000_real32; kval = 1.000000_real32
    n = 6; cdim = 150; ddim = 100; hdim = 160; wdim = 160
    stride_mb = cdim*ddim*hdim*wdim
    wk_size = n*cdim*ddim*hdim*wdim
    allocate(src(0:wk_size-1), dst(0:wk_size-1))
    call srand(123)
    do i = 0, wk_size-1
      src(i) = rand_unit()
      dst(i) = 0.0_real32
    end do
!$omp target data map(to:src(0:wk_size-1)) map(from:dst(0:wk_size-1))
    print '(a)', 'Sweep the work-group sizes from 64 to 512'
    do wg_size = 64, 512, 64
      if (wg_size /= 64 .and. wg_size /= 128 .and. wg_size /= 256 .and. wg_size /= 512) cycle
      wg_cnt = (wk_size + wg_size - 1) / wg_size
      start_time = omp_get_wtime()
      do iter = 1, repeat
        call lrn_fwd_kernel(src, dst, n, cdim, ddim, hdim, wdim, stride_mb, wg_cnt, wg_size, wk_size, size, alpha, beta, kval)
      end do
      elapsed = omp_get_wtime() - start_time
      print '(a,f0.6,a)', 'Average execution time of lrn_fwd_kernel: ', elapsed / repeat, ' sec '
      data_ingb = (2.0_real64 * wk_size * 4.0_real64) / 1.0e9_real64
      bandwidth = data_ingb * repeat / elapsed
      print '(a,f0.6,a)', 'Kernel bandwidth: ', bandwidth, ' GB/s '
    end do
!$omp end target data
    checksum = sum(real(dst, real64)) / real(wk_size, real64)
    print '(a,f0.6)', 'Checksum: ', checksum
    deallocate(src, dst)
  end subroutine forward

  subroutine backward()
    ndims = 5; size = 5; alpha = 0.000122_real32; beta = 0.750000_real32; kval = 1.000000_real32
    n = 5; cdim = 150; ddim = 100; hdim = 160; wdim = 160
    stride_mb = cdim*ddim*hdim*wdim
    wk_size = n*cdim*ddim*hdim*wdim
    allocate(src(0:wk_size-1), dst(0:wk_size-1), diff_src(0:wk_size-1))
    call srand(123)
    do i = 0, wk_size-1
      src(i) = rand_unit()
      dst(i) = src(i)
      diff_src(i) = src(i)
    end do
!$omp target data map(to:src(0:wk_size-1),diff_src(0:wk_size-1)) &
!$omp& map(tofrom:dst(0:wk_size-1))
    print '(a)', 'Sweep the work-group sizes from 64 to 512'
    do wg_size = 64, 512, 64
      if (wg_size /= 64 .and. wg_size /= 128 .and. wg_size /= 256 .and. wg_size /= 512) cycle
      wg_cnt = (wk_size + wg_size - 1) / wg_size
      start_time = omp_get_wtime()
      do iter = 1, repeat
        call lrn_bwd_kernel(src, dst, diff_src, n, cdim, ddim, hdim, wdim, stride_mb, wg_cnt, wg_size, wk_size, size, alpha, beta, kval)
      end do
      elapsed = omp_get_wtime() - start_time
      print '(a,f0.6,a)', 'Average execution time of lrn_bwd_kernel: ', elapsed / repeat, ' sec '
      data_ingb = (3.0_real64 * wk_size * 4.0_real64) / 1.0e9_real64
      bandwidth = data_ingb * repeat / elapsed
      print '(a,f0.6,a)', 'Kernel bandwidth: ', bandwidth, ' GB/s '
    end do
!$omp end target data
    checksum = sum(real(dst, real64)) / real(wk_size, real64)
    print '(a,f0.6)', 'Checksum: ', checksum
    deallocate(src, dst, diff_src)
  end subroutine backward
end program lrn
