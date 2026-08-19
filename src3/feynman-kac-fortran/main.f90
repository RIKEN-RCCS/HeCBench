program feynman_kac
  use, intrinsic :: iso_fortran_env, only : int32, int64, real64
  use omp_lib, only : omp_get_wtime
  implicit none
!$omp declare target (potential, uniform_01, evaluate_point)
  integer(int32) :: repeat,ni,nj,n_inside,seed,iteration,i,j,k,ios,trajectories
  real(real64) :: a,b,h,rth,err,begin_time,total_time,elapsed
  character(len=64)::argument
  if(command_argument_count()/=1)then;write(*,'(a)')'Usage: ./main <iterations>';stop 1;end if
  call get_command_argument(1,argument);read(argument,*,iostat=ios)repeat;if(ios/=0.or.repeat<=0)error stop 'iterations must be positive'
  a=2.0_real64;b=1.0_real64;h=0.001_real64;seed=123456789_int32;err=0.0_real64;n_inside=0;trajectories=1000_int32
  ni=1+ceiling(a/b)*(128-1);nj=128;rth=sqrt(2.0_real64*h)
  write(*,'(a)')'';write(*,'(a)')'FEYNMAN_KAC_2D:';write(*,'(a)')''
  write(*,'(a,i0)')'  X coordinate marked by ',ni;write(*,'(a,i0)')'  Y coordinate marked by ',nj
!$omp target data map(to:ni,nj,seed,trajectories,a,b,h,rth) map(from:err,n_inside)
  total_time=0.0_real64
  do iteration=1,repeat
!$omp target update to(err,n_inside)
    begin_time=omp_get_wtime()
!$omp target teams distribute parallel do collapse(2) thread_limit(256) reduction(+:err,n_inside)
    do j=0,nj-1
      do i=0,ni-1
        call evaluate_point(i,j,ni,nj,a,b,h,rth,trajectories,seed,err,n_inside)
      end do
    end do
!$omp end target teams distribute parallel do
    total_time=total_time+omp_get_wtime()-begin_time
  end do
!$omp end target data
  write(*,'(a,f0.6,a)')'Average kernel time: ',total_time/real(repeat,real64),' (s)'
  err=sqrt(err/real(n_inside,real64))
  write(*,'(a)')'';write(*,'(a,es14.6)')'  RMS absolute error in solution = ',err
  write(*,'(a)')'';write(*,'(a)')'FEYNMAN_KAC_2D:';write(*,'(a)')'  Normal end of execution.'
contains
  function potential(a,b,x,y) result(value)
    real(real64),intent(in)::a,b,x,y;real(real64)::value
    value=2.0_real64*((x/a/a)**2+(y/b/b)**2)+1.0_real64/a/a+1.0_real64/b/b
  end function potential
  function uniform_01(state) result(value)
    integer(int32),intent(inout)::state
    integer(int32)::quotient;real(real64)::value
    quotient=state/127773_int32;state=16807_int32*(state-quotient*127773_int32)-quotient*2836_int32
    if(state<0)state=state+2147483647_int32
    value=real(state,real64)*4.656612875e-10_real64
  end function uniform_01
  subroutine evaluate_point(i,j,ni,nj,a,b,h,rth,npaths,state,err,ninside)
    integer(int32),intent(in)::i,j,ni,nj,npaths
    integer(int32),intent(inout)::state,ninside
    real(real64),intent(in)::a,b,h,rth
    real(real64),intent(inout)::err
    integer(int32)::path;real(real64)::x,y,x1,x2,dx,dy,ut,us,vs,vh,w,we,wt,check,exact
    x=(real(nj-j,real64)*(-a)+real(j-1,real64)*a)/real(nj-1,real64)
    y=(real(ni-i,real64)*(-b)+real(i-1,real64)*b)/real(ni-1,real64)
    check=(x/a)**2+(y/b)**2
    if(1.0_real64<check)then
      exact=1.0_real64;wt=1.0_real64
    else
      ninside=ninside+1;exact=exp((x/a)**2+(y/b)**2-1.0_real64);wt=0.0_real64
      do path=1,npaths
        x1=x;x2=y;w=1.0_real64;check=0.0_real64
        do while(check<1.0_real64)
          ut=uniform_01(state)
          if(ut<0.5_real64)then;us=uniform_01(state)-0.5_real64;if(us<0.0_real64)then;dx=-rth;else;dx=rth;end if;else;dx=0.0_real64;end if
          ut=uniform_01(state)
          if(ut<0.5_real64)then;us=uniform_01(state)-0.5_real64;if(us<0.0_real64)then;dy=-rth;else;dy=rth;end if;else;dy=0.0_real64;end if
          vs=potential(a,b,x1,x2);x1=x1+dx;x2=x2+dy;vh=potential(a,b,x1,x2);we=(1.0_real64-h*vs)*w
          w=w-0.5_real64*h*(vh*we+vs*w);check=(x1/a)**2+(x2/b)**2
        end do
        wt=wt+w
      end do
      wt=wt/real(npaths,real64);err=err+(exact-wt)**2
    end if
  end subroutine evaluate_point
end program feynman_kac
