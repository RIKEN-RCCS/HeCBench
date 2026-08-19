module hypterm_kernels
  use iso_fortran_env, only: real64
  implicit none
!$omp declare target (d8x, d8y, d8z, dpx, dpy, dpz)
contains
  pure real(real64) function d8x(a, i, j, k)
    real(real64), intent(in) :: a(:, :, :)
    integer, intent(in) :: i, j, k
    d8x = 0.8_real64*(a(i+1,j,k)-a(i-1,j,k))-0.2_real64*(a(i+2,j,k)-a(i-2,j,k)) &
        + 0.038_real64*(a(i+3,j,k)-a(i-3,j,k))-0.0035_real64*(a(i+4,j,k)-a(i-4,j,k))
  end function
  pure real(real64) function d8y(a, i, j, k)
    real(real64), intent(in) :: a(:, :, :)
    integer, intent(in) :: i, j, k
    d8y = 0.8_real64*(a(i,j+1,k)-a(i,j-1,k))-0.2_real64*(a(i,j+2,k)-a(i,j-2,k)) &
        + 0.038_real64*(a(i,j+3,k)-a(i,j-3,k))-0.0035_real64*(a(i,j+4,k)-a(i,j-4,k))
  end function
  pure real(real64) function d8z(a, i, j, k)
    real(real64), intent(in) :: a(:, :, :)
    integer, intent(in) :: i, j, k
    d8z = 0.8_real64*(a(i,j,k+1)-a(i,j,k-1))-0.2_real64*(a(i,j,k+2)-a(i,j,k-2)) &
        + 0.038_real64*(a(i,j,k+3)-a(i,j,k-3))-0.0035_real64*(a(i,j,k+4)-a(i,j,k-4))
  end function
  pure real(real64) function dpx(a,b,i,j,k)
    real(real64), intent(in) :: a(:,:,:),b(:,:,:); integer,intent(in)::i,j,k
    dpx=0.8_real64*(a(i+1,j,k)*b(i+1,j,k)-a(i-1,j,k)*b(i-1,j,k))-0.2_real64*(a(i+2,j,k)*b(i+2,j,k)-a(i-2,j,k)*b(i-2,j,k))+0.038_real64*(a(i+3,j,k)*b(i+3,j,k)-a(i-3,j,k)*b(i-3,j,k))-0.0035_real64*(a(i+4,j,k)*b(i+4,j,k)-a(i-4,j,k)*b(i-4,j,k))
  end function
  pure real(real64) function dpy(a,b,i,j,k)
    real(real64), intent(in) :: a(:,:,:),b(:,:,:); integer,intent(in)::i,j,k
    dpy=0.8_real64*(a(i,j+1,k)*b(i,j+1,k)-a(i,j-1,k)*b(i,j-1,k))-0.2_real64*(a(i,j+2,k)*b(i,j+2,k)-a(i,j-2,k)*b(i,j-2,k))+0.038_real64*(a(i,j+3,k)*b(i,j+3,k)-a(i,j-3,k)*b(i,j-3,k))-0.0035_real64*(a(i,j+4,k)*b(i,j+4,k)-a(i,j-4,k)*b(i,j-4,k))
  end function
  pure real(real64) function dpz(a,b,i,j,k)
    real(real64), intent(in) :: a(:,:,:),b(:,:,:); integer,intent(in)::i,j,k
    dpz=0.8_real64*(a(i,j,k+1)*b(i,j,k+1)-a(i,j,k-1)*b(i,j,k-1))-0.2_real64*(a(i,j,k+2)*b(i,j,k+2)-a(i,j,k-2)*b(i,j,k-2))+0.038_real64*(a(i,j,k+3)*b(i,j,k+3)-a(i,j,k-3)*b(i,j,k-3))-0.0035_real64*(a(i,j,k+4)*b(i,j,k+4)-a(i,j,k-4)*b(i,j,k-4))
  end function
  subroutine hypterm_gpu(f0,f1,f2,f3,f4,c1,c2,c3,c4,q1,q2,q3,q4,dx0,dx1,dx2,n)
    integer, intent(in) :: n
    real(real64), intent(inout) :: f0(:,:,:),f1(:,:,:),f2(:,:,:),f3(:,:,:),f4(:,:,:)
    real(real64), intent(in) :: c1(:,:,:),c2(:,:,:),c3(:,:,:),c4(:,:,:),q1(:,:,:),q2(:,:,:),q3(:,:,:),q4(:,:,:)
    real(real64), intent(in) :: dx0,dx1,dx2
    integer :: i,j,k
!$omp target teams distribute parallel do collapse(3) thread_limit(256)
    do k=5,n-4
      do j=5,n-4
        do i=5,n-4
          f0(i,j,k) = -d8x(c1,i,j,k)*dx0-d8y(c2,i,j,k)*dx1-d8z(c3,i,j,k)*dx2
          f1(i,j,k) = -(dpx(c1,q1,i,j,k)+d8x(q4,i,j,k))*dx0-dpy(c1,q2,i,j,k)*dx1-dpz(c1,q3,i,j,k)*dx2
          f2(i,j,k) = -dpx(c2,q1,i,j,k)*dx0-(dpy(c2,q2,i,j,k)+d8y(q4,i,j,k))*dx1-dpz(c2,q3,i,j,k)*dx2
          f3(i,j,k) = -dpx(c3,q1,i,j,k)*dx0-dpy(c3,q2,i,j,k)*dx1-(dpz(c3,q3,i,j,k)+d8z(q4,i,j,k))*dx2
          f4(i,j,k) = -(dpx(c4,q1,i,j,k)+dpx(q4,q1,i,j,k))*dx0-(dpy(c4,q2,i,j,k)+dpy(q4,q2,i,j,k))*dx1-(dpz(c4,q3,i,j,k)+dpz(q4,q3,i,j,k))*dx2
        end do
      end do
    end do
!$omp end target teams distribute parallel do
  end subroutine
  subroutine hypterm_cpu(f0,f1,f2,f3,f4,c1,c2,c3,c4,q1,q2,q3,q4,dx0,dx1,dx2,n)
    integer, intent(in) :: n
    real(real64), intent(inout) :: f0(:,:,:),f1(:,:,:),f2(:,:,:),f3(:,:,:),f4(:,:,:)
    real(real64), intent(in) :: c1(:,:,:),c2(:,:,:),c3(:,:,:),c4(:,:,:),q1(:,:,:),q2(:,:,:),q3(:,:,:),q4(:,:,:)
    real(real64), intent(in) :: dx0,dx1,dx2
    integer :: i,j,k
    do k=5,n-4; do j=5,n-4; do i=5,n-4
      f0(i,j,k)=-d8x(c1,i,j,k)*dx0-d8y(c2,i,j,k)*dx1-d8z(c3,i,j,k)*dx2
      f1(i,j,k)=-(dpx(c1,q1,i,j,k)+d8x(q4,i,j,k))*dx0-dpy(c1,q2,i,j,k)*dx1-dpz(c1,q3,i,j,k)*dx2
      f2(i,j,k)=-dpx(c2,q1,i,j,k)*dx0-(dpy(c2,q2,i,j,k)+d8y(q4,i,j,k))*dx1-dpz(c2,q3,i,j,k)*dx2
      f3(i,j,k)=-dpx(c3,q1,i,j,k)*dx0-dpy(c3,q2,i,j,k)*dx1-(dpz(c3,q3,i,j,k)+d8z(q4,i,j,k))*dx2
      f4(i,j,k)=-(dpx(c4,q1,i,j,k)+dpx(q4,q1,i,j,k))*dx0-(dpy(c4,q2,i,j,k)+dpy(q4,q2,i,j,k))*dx1-(dpz(c4,q3,i,j,k)+dpz(q4,q3,i,j,k))*dx2
    end do; end do; end do
  end subroutine
end module
program hypterm
  use iso_fortran_env, only: real64
  use hypterm_kernels
  implicit none
  integer :: n,repeat,argc,r,seed_size
  integer, allocatable :: seed(:)
  real(real64), allocatable :: c1(:,:,:),c2(:,:,:),c3(:,:,:),c4(:,:,:),q1(:,:,:),q2(:,:,:),q3(:,:,:),q4(:,:,:),f0(:,:,:),f1(:,:,:),f2(:,:,:),f3(:,:,:),f4(:,:,:),g0(:,:,:),g1(:,:,:),g2(:,:,:),g3(:,:,:),g4(:,:,:)
  real(real64) :: start,finish,err
  character(32) :: arg
  argc=command_argument_count(); if(argc<1 .or. argc>2) error stop 'Usage: ./main <repeat> [side]'
  call get_command_argument(1,arg); read(arg,*) repeat; n=308
  if(argc==2) then; call get_command_argument(2,arg); read(arg,*) n; end if
  if(n<9) error stop 'side must be at least 9'
  allocate(c1(n,n,n),c2(n,n,n),c3(n,n,n),c4(n,n,n),q1(n,n,n),q2(n,n,n),q3(n,n,n),q4(n,n,n), &
           f0(n,n,n),f1(n,n,n),f2(n,n,n),f3(n,n,n),f4(n,n,n),g0(n,n,n),g1(n,n,n),g2(n,n,n),g3(n,n,n),g4(n,n,n))
  call random_seed(size=seed_size); allocate(seed(seed_size)); seed=12345; call random_seed(put=seed)
  call random_number(c1); call random_number(c2); call random_number(c3); call random_number(c4); call random_number(q1); call random_number(q2); call random_number(q3); call random_number(q4)
  c1=c1+0.02121_real64;c2=c2+0.02121_real64;c3=c3+0.02121_real64;c4=c4+0.02121_real64;q1=q1+0.02121_real64;q2=q2+0.02121_real64;q3=q3+0.02121_real64;q4=q4+0.02121_real64
  f0=0;f1=0;f2=0;f3=0;f4=0;g0=0;g1=0;g2=0;g3=0;g4=0
  call hypterm_cpu(g0,g1,g2,g3,g4,c1,c2,c3,c4,q1,q2,q3,q4,0.01_real64,0.02_real64,0.03_real64,n)
!$omp target data map(to:c1,c2,c3,c4,q1,q2,q3,q4) map(tofrom:f0,f1,f2,f3,f4)
  call cpu_time(start)
  do r=1,repeat; call hypterm_gpu(f0,f1,f2,f3,f4,c1,c2,c3,c4,q1,q2,q3,q4,0.01_real64,0.02_real64,0.03_real64,n); end do
  call cpu_time(finish)
!$omp end target data
  err=sqrt(sum((f0-g0)**2+ (f1-g1)**2 + (f2-g2)**2 + (f3-g3)**2 + (f4-g4)**2)/real(5*(n-8)**3,real64))
  write(*,'(A,F12.6,A)') 'Average kernel execution time: ',(finish-start)*1.e3_real64/repeat,' (ms)'
  write(*,'(A,ES12.4)') 'RMS Error : ',err
  if(err>1.e-10_real64) error stop 'FAIL'; write(*,'(A)') 'PASS'
end program
