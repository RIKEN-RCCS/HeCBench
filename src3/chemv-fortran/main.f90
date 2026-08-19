module chemv_generated
  use iso_fortran_env, only : real32, real64, int64
  use omp_lib, only : omp_get_wtime, omp_get_team_num, omp_get_thread_num
  implicit none
  integer, parameter :: n=370, ldat=n, repeat_count=1000, at_size=n*ldat, x_size=n, y_size=n
  type :: complex_float
    real(real32) :: re
    real(real32) :: im
  end type complex_float
contains
  subroutine kernel0(at,x,y,alpha_im,alpha_re,beta_im,beta_re)
    type(complex_float), intent(in) :: at(0:at_size-1),x(0:x_size-1)
    type(complex_float), intent(inout) :: y(0:y_size-1)
    real(real32), intent(in) :: alpha_im,alpha_re,beta_im,beta_re
    integer :: b0,t0,c1,c3
    real(real32) :: p5r,p5i,p2r,p3i,p2i,p4i,p4r,p3r,p99r,p98i,p97i,p99i,p97r,p98r
    !$omp target teams num_teams(12) thread_limit(32)
    !$omp parallel private(b0,t0,c1,c3,p5r,p5i,p2r,p3i,p2i,p4i,p4r,p3r,p99r,p98i,p97i,p99i,p97r,p98r)
    b0=omp_get_team_num(); t0=omp_get_thread_num()
    do c1=0,min(368,32*b0+30),32
      if(32*b0+t0 <= 369 .and. c1 == 0) then
        p5r=y(32*b0+t0)%re*beta_re-y(32*b0+t0)%im*beta_im
        p5i=y(32*b0+t0)%im*beta_re+y(32*b0+t0)%re*beta_im
        y(32*b0+t0)%re=p5r; y(32*b0+t0)%im=p5i
        p2r=alpha_re*at(11872*b0+371*t0)%re
        p2i=alpha_im*at(11872*b0+371*t0)%re
        p3r=p2r*x(32*b0+t0)%re-p2i*x(32*b0+t0)%im
        p3i=p2i*x(32*b0+t0)%re+p2r*x(32*b0+t0)%im
        p4r=y(32*b0+t0)%re+p3r; p4i=y(32*b0+t0)%im+p3i
        y(32*b0+t0)%re=p4r; y(32*b0+t0)%im=p4i
      end if
      if(32*b0+t0 <= 369) then
        do c3=0,min(31,32*b0+t0-c1-1)
          p97r=alpha_re*at(32*b0+t0+370*c1+370*c3)%re-alpha_im*at(32*b0+t0+370*c1+370*c3)%im
          p97i=alpha_im*at(32*b0+t0+370*c1+370*c3)%re+alpha_re*at(32*b0+t0+370*c1+370*c3)%im
          p98r=p97r*x(c1+c3)%re-p97i*x(c1+c3)%im
          p98i=p97i*x(c1+c3)%re+p97r*x(c1+c3)%im
          p99r=y(32*b0+t0)%re+p98r; p99i=y(32*b0+t0)%im+p98i
          y(32*b0+t0)%re=p99r; y(32*b0+t0)%im=p99i
        end do
      end if
      !$omp barrier
    end do
    !$omp end parallel
    !$omp end target teams
  end subroutine kernel0

  subroutine kernel1(at,x,y,alpha_im,alpha_re)
    type(complex_float), intent(in) :: at(0:at_size-1),x(0:x_size-1)
    type(complex_float), intent(inout) :: y(0:y_size-1)
    real(real32), intent(in) :: alpha_im,alpha_re
    integer :: b0,t0,c1,c3
    real(real32) :: p96r,p96i,p94i,p95i,p94r,p95r
    !$omp target teams num_teams(12) thread_limit(32)
    !$omp parallel private(b0,t0,c1,c3,p96r,p96i,p94i,p95i,p94r,p95r)
    b0=omp_get_team_num(); t0=omp_get_thread_num()
    do c1=5888*b0,min(67712,5856*b0+6016),32
      do c3=max(0,5888*b0+184*t0-c1),min(31,5856*b0+183*t0-c1+368)
        p94r=alpha_re*at(5984*b0+187*t0+c1+c3+1)%re-alpha_im*(-at(5984*b0+187*t0+c1+c3+1)%im)
        p94i=alpha_im*at(5984*b0+187*t0+c1+c3+1)%re+alpha_re*(-at(5984*b0+187*t0+c1+c3+1)%im)
        p95r=p94r*x(-5856*b0-183*t0+c1+c3+1)%re-p94i*x(-5856*b0-183*t0+c1+c3+1)%im
        p95i=p94i*x(-5856*b0-183*t0+c1+c3+1)%re+p94r*x(-5856*b0-183*t0+c1+c3+1)%im
        p96r=y(32*b0+t0)%re+p95r; p96i=y(32*b0+t0)%im+p95i
        y(32*b0+t0)%re=p96r; y(32*b0+t0)%im=p96i
      end do
      !$omp barrier
    end do
    !$omp end parallel
    !$omp end target teams
  end subroutine kernel1

  subroutine chemv_cpu(alpha_re,alpha_im,beta_re,beta_im,at,x,y)
    real(real32), intent(in) :: alpha_re,alpha_im,beta_re,beta_im
    type(complex_float), intent(in) :: at(0:at_size-1),x(0:x_size-1)
    type(complex_float), intent(inout) :: y(0:y_size-1)
    integer :: i0,i1,i2,i3,jj
    real(real32) :: p5r,p5i,p2r,p3i,p2i,p4i,p4r,p3r,p99r,p96r,p98i,p96i,p94i,p95i,p94r,p95r,p97i,p99i,p97r,p98r
    do i0=0,n-1
      p5r=y(i0)%re*beta_re-y(i0)%im*beta_im; p5i=y(i0)%im*beta_re+y(i0)%re*beta_im
      y(i0)%re=p5r; y(i0)%im=p5i
    end do
    do i1=0,n-1
      p2r=alpha_re*at(i1*ldat+i1)%re; p2i=alpha_im*at(i1*ldat+i1)%re
      p3r=p2r*x(i1)%re-p2i*x(i1)%im; p3i=p2i*x(i1)%re+p2r*x(i1)%im
      p4r=y(i1)%re+p3r; p4i=y(i1)%im+p3i; y(i1)%re=p4r; y(i1)%im=p4i
    end do
    do i2=0,n-2
      do i3=0,n-2-i2
        jj=i3+i2+1
        p94r=alpha_re*at(i2*ldat+jj)%re-alpha_im*(-at(i2*ldat+jj)%im)
        p94i=alpha_im*at(i2*ldat+jj)%re+alpha_re*(-at(i2*ldat+jj)%im)
        p95r=p94r*x(jj)%re-p94i*x(jj)%im; p95i=p94i*x(jj)%re+p94r*x(jj)%im
        p96r=y(i2)%re+p95r; p96i=y(i2)%im+p95i; y(i2)%re=p96r; y(i2)%im=p96i
        p97r=alpha_re*at(i2*ldat+jj)%re-alpha_im*at(i2*ldat+jj)%im
        p97i=alpha_im*at(i2*ldat+jj)%re+alpha_re*at(i2*ldat+jj)%im
        p98r=p97r*x(i2)%re-p97i*x(i2)%im; p98i=p97i*x(i2)%re+p97r*x(i2)%im
        p99r=y(jj)%re+p98r; p99i=y(jj)%im+p98i; y(jj)%re=p99r; y(jj)%im=p99i
      end do
    end do
  end subroutine chemv_cpu

  subroutine chemv_gpu(alpha_re,alpha_im,beta_re,beta_im,at,x,y)
    real(real32), intent(in) :: alpha_re,alpha_im,beta_re,beta_im
    type(complex_float), intent(in) :: at(0:at_size-1),x(0:x_size-1)
    type(complex_float), intent(inout) :: y(0:y_size-1)
    integer :: iteration
    real(real64) :: start_time,end_time
    !$omp target data map(to:at(0:at_size-1),x(0:x_size-1)) map(tofrom:y(0:y_size-1))
    start_time=omp_get_wtime()
    do iteration=0,repeat_count-1
      call kernel0(at,x,y,alpha_im,alpha_re,beta_im,beta_re)
      call kernel1(at,x,y,alpha_im,alpha_re)
    end do
    end_time=omp_get_wtime()
    print '(a,f0.6,a)','Average execution time of chemv kernels: ',(end_time-start_time)*1.0e6_real64/real(repeat_count,real64),' (us)'
    !$omp end target data
  end subroutine chemv_gpu
end module chemv_generated

program main
  use iso_fortran_env, only : real32
  use chemv_generated
  implicit none
  type(complex_float) :: at(0:at_size-1),x(0:x_size-1),y_cpu(0:y_size-1),y_gpu(0:y_size-1)
  real(real32), parameter :: alpha_re=3.14_real32,alpha_im=1.59_real32,beta_re=2.71_real32,beta_im=8.28_real32
  integer :: i,j
  do i=0,n-1
    x(i)%re=real(i+5,real32); x(i)%im=real(i*2,real32)
    y_cpu(i)%re=real(i*3,real32); y_cpu(i)%im=real(i+7,real32)
    y_gpu(i)%re=real(i*3,real32); y_gpu(i)%im=real(i+7,real32)
    do j=0,ldat-1
      at(i*ldat+j)%re=real(i+j,real32); at(i*ldat+j)%im=real(i+3,real32)
    end do
  end do
  call chemv_cpu(alpha_re,alpha_im,beta_re,beta_im,at,x,y_cpu)
  call chemv_gpu(alpha_re,alpha_im,beta_re,beta_im,at,x,y_gpu)
  do i=0,n-1
    if(abs(y_cpu(i)%re-y_gpu(i)%re)>1.0e-3_real32 .or. abs(y_cpu(i)%im-y_gpu(i)%im)>1.0e-3_real32) then
      print '(i0,1x,f0.6,1x,f0.6)',i,y_cpu(i)%re,y_gpu(i)%re
      print '(a)','FAIL'
      error stop 1
    end if
  end do
  print '(a)','PASS'
end program main
