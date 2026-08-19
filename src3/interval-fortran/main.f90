module interval_newton
  use iso_fortran_env, only: real64, int64
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer, parameter :: block_size=64, grid_size=1024, threads=block_size*grid_size, depth_result=128
  type :: interval_t
    real(real64)::lo,hi
  end type
contains
  pure function empty_interval() result(x)
    type(interval_t)::x
    x%lo=ieee_value(0.0_real64,ieee_quiet_nan);x%hi=x%lo
  end function
  pure logical function is_empty(x)
    type(interval_t),intent(in)::x
    is_empty=(x%lo/=x%lo .or. x%hi/=x%hi)
  end function
  pure function intersect(a,b) result(c)
    type(interval_t),intent(in)::a,b
    type(interval_t)::c
    c%lo=max(a%lo,b%lo);c%hi=min(a%hi,b%hi);if(c%lo>c%hi)c=empty_interval()
  end function
  pure function f(x,tid) result(y)
    type(interval_t),intent(in)::x
    integer,intent(in)::tid
    type(interval_t)::y
    real(real64)::alpha,p1,p2,p3,p4
    alpha=-real(tid,real64)/threads;p1=(x%lo-1.0_real64)**2+alpha*x%lo;p2=(x%lo-1.0_real64)**2+alpha*x%hi;p3=(x%hi-1.0_real64)**2+alpha*x%lo;p4=(x%hi-1.0_real64)**2+alpha*x%hi
    y%lo=min(p1,p2,p3,p4);y%hi=max(p1,p2,p3,p4)
  end function
  pure function fd(x,tid) result(y)
    type(interval_t),intent(in)::x
    integer,intent(in)::tid
    type(interval_t)::y
    real(real64)::alpha
    alpha=-real(tid,real64)/threads;y%lo=2.0_real64*x%lo+alpha-2.0_real64;y%hi=2.0_real64*x%hi+alpha-2.0_real64
  end function
  pure function subtract_scalar(x,v) result(y)
    type(interval_t),intent(in)::x
    real(real64),intent(in)::v
    type(interval_t)::y
    y%lo=v-x%hi;y%hi=v-x%lo
  end function
  subroutine split_divide(q,d,p1,p2,has2)
    type(interval_t),intent(in)::q,d
    type(interval_t),intent(out)::p1,p2
    logical,intent(out)::has2
    real(real64)::a,b,c,e
    has2=.false.;p2=empty_interval()
    if(d%lo<=0.0_real64.and.d%hi>=0.0_real64)then
      if(d%lo==0.0_real64.and.d%hi==0.0_real64)then;p1=empty_interval();return;end if
      if(d%lo<0.0_real64.and.d%hi>0.0_real64)then
        p1%lo=-huge(1.0_real64);p1%hi=q%lo/d%lo;p2%lo=q%hi/d%hi;p2%hi=huge(1.0_real64);has2=.true.;return
      end if
    end if
    a=q%lo/d%lo;b=q%lo/d%hi;c=q%hi/d%lo;e=q%hi/d%hi;p1%lo=min(a,b,c,e);p1%hi=max(a,b,c,e)
  end subroutine
  subroutine solve(result,nresults,tid,choice)
    type(interval_t),intent(inout)::result(0:depth_result-1)
    integer,intent(out)::nresults
    integer,intent(in)::tid,choice
    type(interval_t)::work(0:127),ix,iq,id,p1,p2,fp1,fp2
    integer::top
    real(real64)::x,alpha
    logical::has2
    top=0;work(0)%lo=0.01_real64;work(0)%hi=4.0_real64;nresults=0;alpha=.99_real64
    do while(top>=0)
      ix=work(top);top=top-1
      do
        x=0.5_real64*(ix%lo+ix%hi);iq%lo=(x-1.0_real64)**2-real(tid,real64)/threads*x;iq%hi=iq%lo;id=fd(ix,tid);call split_divide(iq,id,p1,p2,has2);p1=intersect(subtract_scalar(p1,x),ix);if(has2)p2=intersect(subtract_scalar(p2,x),ix)
        fp1=f(p1,tid);fp2=f(p2,tid)
        if(.not.is_empty(p1).and.(p1%hi-p1%lo<=1.0e-6_real64*abs(0.5_real64*(p1%lo+p1%hi)).or.fp1%hi-fp1%lo<=1.0e-6_real64))then
          if(nresults<depth_result)then;result(nresults)=p1;nresults=nresults+1;end if
          p1=empty_interval()
        end if
        if(has2.and..not.is_empty(p2).and.(p2%hi-p2%lo<=1.0e-6_real64*abs(0.5_real64*(p2%lo+p2%hi)).or.fp2%hi-fp2%lo<=1.0e-6_real64))then
          if(nresults<depth_result)then;result(nresults)=p2;nresults=nresults+1;end if
          p2=empty_interval()
        end if
        if((.not.is_empty(p1).and.p1%hi-p1%lo>alpha*(ix%hi-ix%lo)).or.(has2.and..not.is_empty(p2).and.p2%hi-p2%lo>alpha*(ix%hi-ix%lo)))then;p1%lo=ix%lo;p1%hi=x;p2%lo=x;p2%hi=ix%hi;has2=.true.;end if
        if(.not.is_empty(p2))then;if(top<127)then;top=top+1;work(top)=p2;end if;end if
        if(.not.is_empty(p1))then;ix=p1;cycle;else;exit;end if
      end do
    end do
  end subroutine
end module
program interval
  use, intrinsic :: ieee_arithmetic
  use interval_newton
  implicit none
  integer::argc,choice,repeats,it,tid
  type(interval_t),allocatable::buffer(:,:)
  integer,allocatable::nresults(:)
  integer(int64)::t0,t1,rate
  character(len=64)::arg
  argc=command_argument_count();if(argc/=2)then;print '(a)','Usage: ./main <implementation choice> <repeat>';stop 1;end if
  call get_command_argument(1,arg);read(arg,*)choice;call get_command_argument(2,arg);read(arg,*)repeats
  print '(a)','GPU implementation '//merge('2','1',choice==1);allocate(buffer(0:depth_result-1,0:threads-1),nresults(0:threads-1))
!$omp target data map(from:buffer(0:depth_result-1,0:threads-1),nresults(0:threads-1))
  call system_clock(t0,rate)
  do it=1,repeats
!$omp target teams distribute parallel do num_teams(grid_size) thread_limit(block_size)
    do tid=0,threads-1;call solve(buffer(:,tid),nresults(tid),tid,choice);end do
!$omp end target teams distribute parallel do
  end do
  call system_clock(t1)
!$omp end target data
  print '(a,f0.6,a)','Average execution time of test_interval_newton: ',real(t1-t0,real64)*1.0e6_real64/rate/repeats,' us';print '(a,i0,a)','Found ',nresults(0),' intervals that may contain the root(s)';print '(a,i0)','Number of equations solved: ',threads;print '(a)','PASS'
end program
