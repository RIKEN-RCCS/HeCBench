module cobahh_update
  use iso_fortran_env, only: real32, real64, int8
  use iso_c_binding, only: c_int
  use omp_lib, only: omp_get_wtime
  implicit none
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
  !$omp declare target (timestep, update_one)
contains
  integer function timestep(t,dt)
    real(real32), intent(in) :: t,dt
    timestep=int((t+1.0e-3_real32*dt)/dt)
  end function timestep

  subroutine update_one(h,m,n,ge,v,gi,lastspike,dt,t,not_refract)
    real(real32), intent(inout) :: h,m,n,ge,v,gi
    real(real32), intent(in) :: lastspike,dt,t
    integer(int8), intent(out) :: not_refract
    real(real32) :: bh,bm,bn,bv,eh,em,en,eg,ev,ei
    real(real32), parameter :: l2=9.939082_real32,l3=-55.555556_real32,l4=0.00001_real32,l5=-200.0_real32
    real(real32), parameter :: l6=-0.02016_real32,l7=-0.000001_real32,l8=0.0_real32,l9=-250.0_real32,l10=0.00416_real32
    real(real32), parameter :: l11=0.02016_real32,l12=-0.01764_real32,l13=-0.000001_real32, &
      l14=0.000099_real32,l15=200.0_real32,l16=0.0112_real32
    real(real32), parameter :: l17=-0.002016_real32,l18=0.0_real32,l19=0.00048_real32, &
      l20=0.002016_real32,l21=132.901474_real32,l22=-25.0_real32
    real(real32), parameter :: l24=-3.0_real32,l25=-2700.0_real32,l26=5000.0_real32,l27=0.0_real32,l28=-400000000.0_real32
    real(real32), parameter :: l29=-50.0_real32,l30=-30000.0_real32,l31=100000.0_real32,l32=5000000000.0_real32
    real(real32) :: l23,l33
    not_refract=merge(1_int8,0_int8,timestep(t-lastspike,dt)>=timestep(0.003_real32,dt))
    eh=(-4.0_real32)/(0.001_real32+l4*exp(l5*v))-l2*exp(l3*v)
    bh=(l2*exp(l3*v))/eh; h=(-bh)+(bh+h)*exp(dt*eh)
    em=(((l11/(l7+l8*exp(l9*v)))+(l12/(l13+l14*exp(l15*v)))+l16/(l13+l14*exp(l15*v)))+0.32_real32*v/(l7+l8*exp(l9*v)))- &
       (l10/(l7+l8*exp(l9*v))+0.28_real32*v/(l13+l14*exp(l15*v)))
    bm=((l6/(l7+l8*exp(l9*v))+l10/(l7+l8*exp(l9*v)))-0.32_real32*v/(l7+l8*exp(l9*v)))/em
    m=(-bm)+(bm+m)*exp(dt*em)
    en=(l20/(l7+l18*exp(l5*v))+0.032_real32*v/(l7+l18*exp(l5*v)))- &
       (l19/(l7+l18*exp(l5*v))+l21*exp(l22*v))
    bn=((l17/(l7+l18*exp(l5*v))+l19/(l7+l18*exp(l5*v)))-0.032_real32*v/(l7+l18*exp(l5*v)))/en
    n=(-bn)+(bn+n)*exp(dt*en)
    l23=exp(-2000.0_real32*dt); ge=l23*ge
    ev=(l29+l30*(n*n*n*n))-((l31*(h*(m*m*m))+l32*ge)+l32*gi)
    bv=(l24+((l25*(n*n*n*n)+l26*(h*(m*m*m)))+l27*ge)+l28*gi)/ev
    v=(-bv)+(bv+v)*exp(dt*ev)
    l33=exp(-100.0_real32*dt); gi=l33*gi
  end subroutine update_one

  subroutine host_update(ge,gi,h,m,n,v,lastspike,dt,t,not_refract,count,iteration)
    integer, intent(in) :: count,iteration
    real(real32), intent(inout) :: ge(0:),gi(0:),h(0:),m(0:),n(0:),v(0:)
    real(real32), intent(in) :: lastspike(0:),dt(0:),t(0:)
    integer(int8), intent(out) :: not_refract(0:)
    integer :: iter,idx
    do iter=1,iteration
      do idx=0,count-1
        call update_one(h(idx),m(idx),n(idx),ge(idx),v(idx),gi(idx),lastspike(idx),dt(0),t(0),not_refract(idx))
      end do
    end do
  end subroutine host_update

  subroutine device_update(ge,gi,h,m,n,v,lastspike,dt,t,not_refract,count,iteration)
    real(real32), intent(inout) :: ge(0:),gi(0:),h(0:),m(0:),n(0:),v(0:)
    real(real32), intent(in) :: lastspike(0:),dt(0:),t(0:)
    integer(int8), intent(out) :: not_refract(0:)
    integer, intent(in) :: count,iteration
    integer :: iter,idx
    real(real32) :: dtime,now
    real(real64) :: start_time, stop_time, elapsed_us
    dtime=dt(0); now=t(0)
!$omp target data map(tofrom:h(0:count),m(0:count),n(0:count),ge(0:count),v(0:count),gi(0:count)) &
!$omp& map(to:lastspike(0:count)) map(from:not_refract(0:count))
    start_time = omp_get_wtime()
    do iter=1,iteration
      !$omp target teams distribute parallel do thread_limit(256)
      do idx=0,count-1
        call update_one(h(idx),m(idx),n(idx),ge(idx),v(idx),gi(idx),lastspike(idx),dtime,now,not_refract(idx))
      end do
      !$omp end target teams distribute parallel do
    end do
    stop_time = omp_get_wtime()
!$omp end target data
    elapsed_us = (stop_time - start_time) * 1.0e6_real64 / real(iteration, real64)
    write(*,'(A,F0.6,A)') 'Average kernel execution time ', elapsed_us, ' (us)'
  end subroutine device_update
end module cobahh_update

program main
  use iso_fortran_env, only: real32, real64, int8
  use cobahh_update
  implicit none
  integer :: count,iteration,i
  character(len=32) :: arg
  real(real32), allocatable :: hge(:),hgi(:),hh(:),hm(:),hn(:),hv(:),hlast(:),hdt(:),ht(:)
  real(real32), allocatable :: ge(:),gi(:),h(:),m(:),nn(:),v(:),last(:),dt(:),tm(:)
  integer(int8), allocatable :: hnr(:),nr(:)
  real(real64) :: rsme
  if(command_argument_count()/=2) then; print '(a)', 'Usage: ./main <neurons> <repeat>'; error stop 1; end if
  call get_command_argument(1,arg); read(arg,*) count; call get_command_argument(2,arg); read(arg,*) iteration
  allocate(hge(0:count-1),hgi(0:count-1),hh(0:count-1),hm(0:count-1),hn(0:count-1), &
           hv(0:count-1),hlast(0:count-1),hdt(0:0),ht(0:0),hnr(0:count-1))
  allocate(ge(0:count-1),gi(0:count-1),h(0:count-1),m(0:count-1),nn(0:count-1), &
           v(0:count-1),last(0:count-1),dt(0:0),tm(0:0),nr(0:count-1))
  call c_srand(2_c_int); print '(a)', 'initializing ... '
  do i=1,count-1
    hge(i)=0.15_real32+merge(0.1_real32,-0.1_real32,mod(c_rand(),2_c_int)==0); ge(i)=hge(i)
    hgi(i)=0.25_real32+merge(0.2_real32,-0.2_real32,mod(c_rand(),2_c_int)==0); gi(i)=hgi(i)
    hh(i)=0.35_real32+merge(0.3_real32,-0.3_real32,mod(c_rand(),2_c_int)==0); h(i)=hh(i)
    hm(i)=0.45_real32+merge(0.4_real32,-0.4_real32,mod(c_rand(),2_c_int)==0); m(i)=hm(i)
    hn(i)=0.55_real32+merge(0.5_real32,-0.5_real32,mod(c_rand(),2_c_int)==0); nn(i)=hn(i)
    hv(i)=0.65_real32+merge(0.6_real32,-0.6_real32,mod(c_rand(),2_c_int)==0); v(i)=hv(i)
    hlast(i)=1.0_real32/real(mod(c_rand(),1000_c_int)+1_c_int,real32); last(i)=hlast(i)
  end do
  hdt(0)=0.0001_real32; dt(0)=hdt(0); ht(0)=0.01_real32; tm(0)=ht(0); print '(a)', 'done.'
  call host_update(hge,hgi,hh,hm,hn,hv,hlast,hdt,ht,hnr,count,iteration)
  call device_update(ge,gi,h,m,nn,v,last,dt,tm,nr,count,iteration)
  rsme=0.0_real64
  do i=0,count-1
    rsme=rsme+real((ge(i)-hge(i))**2+(gi(i)-hgi(i))**2+(h(i)-hh(i))**2+(m(i)-hm(i))**2+(nn(i)-hn(i))**2+(v(i)-hv(i))**2,real64)
    rsme=rsme+real((nr(i)-hnr(i))*(nr(i)-hnr(i)),real64)
  end do
  print '(a,f0.6)', 'RSME = ',sqrt(rsme/real(count,real64))
end program main
