module bonds_mod
  use iso_fortran_env, only: real32, real64, int32, int64
  use iso_c_binding, only: c_int
  implicit none
  integer, parameter :: SIMPLE_INTEREST=0, COMPOUNDED_INTEREST=1, CONTINUOUS_INTEREST=2
  integer, parameter :: USE_EXACT_DAY=0, USE_SERIAL_NUMS=1, COMPUTE_AMOUNT=-1
  real(real32), parameter :: accuracy=1.0e-8_real32, ql_epsilon=1.0e-18_real32
  type :: date_t
    integer :: month, day, year, serial
  end type
  type :: bond_t
    type(date_t) :: start_date, maturity_date
    real(real32) :: rate
  end type
  type :: int_rate_t
    real(real32) :: rate, freq
    integer :: comp, day_counter
  end type
  type :: yield_curve_t
    real(real32) :: forward, compounding, frequency
    type(int_rate_t) :: int_rate
    type(date_t) :: ref_date, cal_date
    integer :: day_counter
  end type
  type :: coupon_t
    type(date_t) :: payment_date, accrual_start_date, accrual_end_date
    real(real32) :: amount
  end type
  type :: cashflows_t
    type(coupon_t) :: legs(0:8)
    type(int_rate_t) :: int_rate
    real(real32) :: nominal
    integer :: day_counter
  end type
  !$omp declare target (leap, month_length, month_offset, year_offset)
  !$omp declare target (initialize_date, day_count, year_fraction, advance_date, occurred)
  !$omp declare target (compound_factor, compound_dates, discount_factor, coupon_amount)
  !$omp declare target (next_cashflow, accrued, curve_discount, cashflow_npv, npv_yield)
  !$omp declare target (f_value, duration, close_enough, solve_yield, compute_one)
contains
  logical function leap(y)
    integer, intent(in) :: y
    ! The reference intentionally treats 1900 as a leap year (Excel convention).
    leap = (mod(y,4)==0 .and. (mod(y,100)/=0 .or. y==1900 .or. mod(y,400)==0))
  end function

  integer function month_length(m, is_leap)
    integer, intent(in) :: m
    logical, intent(in) :: is_leap
    integer, parameter :: regular(12)=[31,28,31,30,31,30,31,31,30,31,30,31]
    month_length=regular(m); if (m==2 .and. is_leap) month_length=29
  end function

  integer function month_offset(m, is_leap)
    integer, intent(in) :: m
    logical, intent(in) :: is_leap
    integer, parameter :: regular(13)=[0,31,59,90,120,151,181,212,243,273,304,334,365]
    integer, parameter :: leap_offsets(13)=[0,31,60,91,121,152,182,213,244,274,305,335,366]
    if (is_leap) then; month_offset=leap_offsets(m); else; month_offset=regular(m); end if
  end function

  integer function year_offset(y)
    integer, intent(in) :: y
    integer :: k
    year_offset=0
    do k=1900,y-1
      if (leap(k)) then; year_offset=year_offset+366; else; year_offset=year_offset+365; end if
    end do
  end function

  function initialize_date(d,m,y) result(x)
    integer, intent(in) :: d,m,y
    type(date_t) :: x
    x%day=d; x%month=m; x%year=y; x%serial=d+month_offset(m,leap(y))+year_offset(y)
  end function

  integer function day_count(d1,d2,dc)
    type(date_t), intent(in) :: d1,d2
    integer, intent(in) :: dc
    integer :: dd1,dd2,mm1,mm2,yy1,yy2
    if (dc==USE_EXACT_DAY) then
      dd1=d1%day; dd2=d2%day; mm1=d1%month; mm2=d2%month; yy1=d1%year; yy2=d2%year
      if (dd2==31 .and. dd1<30) then; dd2=1; mm2=mm2+1; end if
      day_count=360*(yy2-yy1)+30*(mm2-mm1-1)+max(0,30-dd1)+min(30,dd2)
    else
      day_count=d2%serial-d1%serial
    end if
  end function

  real(real32) function year_fraction(d1,d2,dc)
    type(date_t), intent(in) :: d1,d2
    integer, intent(in) :: dc
    year_fraction=real(day_count(d1,d2,dc),real32)/360.0_real32
  end function

  function advance_date(d,nmonths) result(out)
    type(date_t), intent(in) :: d
    integer, intent(in) :: nmonths
    type(date_t) :: out
    integer :: dd,mm,yy
    dd=d%day; mm=d%month+nmonths; yy=d%year
    do while(mm>12); mm=mm-12; yy=yy+1; end do
    do while(mm<1); mm=mm+12; yy=yy-1; end do
    dd=min(dd,month_length(mm,leap(yy))); out=initialize_date(dd,mm,yy)
  end function

  logical function occurred(curr,event)
    type(date_t), intent(in) :: curr,event
    occurred=event%serial>curr%serial
  end function

  real(real32) function compound_factor(r,t)
    type(int_rate_t), intent(in) :: r
    real(real32), intent(in) :: t
    select case(r%comp)
    case(SIMPLE_INTEREST); compound_factor=1.0_real32+r%rate*t
    case(COMPOUNDED_INTEREST); compound_factor=(1.0_real32+r%rate/r%freq)**(r%freq*t)
    case(CONTINUOUS_INTEREST); compound_factor=exp(r%rate*t)
    case default; compound_factor=0.0_real32
    end select
  end function

  real(real32) function compound_dates(r,d1,d2,dc)
    type(int_rate_t), intent(in) :: r
    type(date_t), intent(in) :: d1,d2
    integer, intent(in) :: dc
    compound_dates=compound_factor(r,year_fraction(d1,d2,dc))
  end function

  real(real32) function discount_factor(r,t)
    type(int_rate_t), intent(in) :: r
    real(real32), intent(in) :: t
    discount_factor=1.0_real32/compound_factor(r,t)
  end function

  real(real32) function coupon_amount(cf,i)
    type(cashflows_t), intent(in) :: cf
    integer, intent(in) :: i
    if (cf%legs(i)%amount==real(COMPUTE_AMOUNT,real32)) then
      coupon_amount=100.0_real32 * &
        (compound_dates(cf%int_rate,cf%legs(i)%accrual_start_date, &
        cf%legs(i)%accrual_end_date,cf%day_counter)-1.0_real32)
    else
      coupon_amount=cf%legs(i)%amount
    end if
  end function

  integer function next_cashflow(cf,cur,nlegs)
    type(cashflows_t), intent(in) :: cf
    type(date_t), intent(in) :: cur
    integer, intent(in) :: nlegs
    integer :: i
    next_cashflow=nlegs-1
    do i=0,nlegs-1
      if (.not.occurred(cf%legs(i)%payment_date,cur)) then; next_cashflow=i; return; end if
    end do
  end function

  real(real32) function accrued(cf,cur,mat,bond_num,nlegs)
    type(cashflows_t), intent(in) :: cf
    type(date_t), intent(in) :: cur,mat(0:)
    integer, intent(in) :: bond_num,nlegs
    integer :: i, first
    type(date_t) :: ending
    accrued=0.0_real32; first=next_cashflow(cf,cur,nlegs)
    do i=first,nlegs-1
      if (cur%serial>cf%legs(i)%accrual_start_date%serial .and. cur%serial<=mat(bond_num)%serial) then
        ending=cf%legs(i)%accrual_end_date; if (cur%serial<ending%serial) ending=cur
        accrued=accrued+100.0_real32*(compound_dates(cf%int_rate,cf%legs(i)%accrual_start_date,ending,cf%day_counter)-1.0_real32)
      end if
    end do
  end function

  real(real32) function curve_discount(curve,t)
    type(yield_curve_t), intent(in) :: curve
    type(date_t), intent(in) :: t
    type(int_rate_t) :: r
    r=curve%int_rate; r%rate=curve%forward; r%freq=curve%frequency; r%comp=nint(curve%compounding)
    curve_discount=discount_factor(r,year_fraction(curve%ref_date,t,curve%day_counter))
  end function

  real(real32) function cashflow_npv(cf,curve,cur,nlegs)
    type(cashflows_t), intent(in) :: cf
    type(yield_curve_t), intent(in) :: curve
    type(date_t), intent(in) :: cur
    integer, intent(in) :: nlegs
    integer :: i
    cashflow_npv=0.0_real32
    do i=0,nlegs-1
      if (.not.occurred(cf%legs(i)%payment_date,cur)) then
        cashflow_npv=cashflow_npv+coupon_amount(cf,i)* &
          curve_discount(curve,cf%legs(i)%payment_date)
      end if
    end do
    cashflow_npv=cashflow_npv/curve_discount(curve,cur)
  end function

  real(real32) function npv_yield(cf,y,cur,npv_date,nlegs)
    type(cashflows_t), intent(in) :: cf
    type(int_rate_t), intent(in) :: y
    type(date_t), intent(in) :: cur,npv_date
    integer, intent(in) :: nlegs
    integer :: i
    real(real32) :: discount
    type(date_t) :: last,coupon
    logical :: first
    npv_yield=0.0_real32; discount=1.0_real32; first=.true.
    do i=0,nlegs-1
      if (occurred(cf%legs(i)%payment_date,cur)) cycle
      coupon=cf%legs(i)%payment_date
      if (first) then
        first=.false.; if(i>0) then; last=advance_date(cf%legs(i)%payment_date,-6); else; last=cf%legs(i)%accrual_start_date; end if
        discount=discount*discount_factor(y,year_fraction(npv_date,coupon,y%day_counter))
      else
        discount=discount*discount_factor(y,year_fraction(last,coupon,y%day_counter))
      end if
      last=coupon; npv_yield=npv_yield+coupon_amount(cf,i)*discount
    end do
  end function

  real(real32) function f_value(cf,npv,dc,comp,freq,cur,npv_date,y,nlegs)
    type(cashflows_t), intent(in) :: cf
    real(real32), intent(in) :: npv,freq,y
    integer, intent(in) :: dc,comp,nlegs
    type(date_t), intent(in) :: cur,npv_date
    type(int_rate_t) :: r
    r%rate=y; r%comp=comp; r%freq=freq; r%day_counter=dc
    f_value=npv-npv_yield(cf,r,cur,npv_date,nlegs)
  end function

  real(real32) function duration(cf,dc,comp,freq,cur,npv_date,y,nlegs)
    type(cashflows_t), intent(in) :: cf
    integer, intent(in) :: dc,comp,nlegs
    real(real32), intent(in) :: freq,y
    type(date_t), intent(in) :: cur,npv_date
    type(int_rate_t) :: r
    integer :: i
    real(real32) :: p,dp,t,c,b
    r%rate=y; r%comp=comp; r%freq=freq; r%day_counter=dc; p=0.0_real32; dp=0.0_real32
    do i=0,nlegs-1
      if (.not.occurred(cf%legs(i)%payment_date,cur)) then
        t=year_fraction(npv_date,cf%legs(i)%payment_date,dc); c=coupon_amount(cf,i); b=discount_factor(r,t); p=p+c*b
        select case(comp)
        case(SIMPLE_INTEREST); dp=dp-c*b*b*t
        case(COMPOUNDED_INTEREST); dp=dp-c*t*b/(1.0_real32+y/freq)
        case(CONTINUOUS_INTEREST); dp=dp-c*b*t
        end select
      end if
    end do
    if(p==0.0_real32) then; duration=0.0_real32; else; duration=-dp/p; end if
  end function

  real(real32) function solve_yield(cf,npv,dc,comp,freq,cur,nlegs)
    type(cashflows_t), intent(in) :: cf
    real(real32), intent(in) :: npv,freq
    integer, intent(in) :: dc,comp,nlegs
    type(date_t), intent(in) :: cur
    real(real32) :: xl,xh,root,fl,fh,f,df,dx,dxold,step,eps
    integer :: it, flipflop, evaluations
    eps=max(accuracy,ql_epsilon); step=.005_real32; flipflop=-1; root=.05_real32
    fh=f_value(cf,npv,dc,comp,freq,cur,cur,root,nlegs)
    if (close_enough(fh,0.0_real32)) then
      solve_yield=root; return
    else if (close_enough(fh,0.0_real32)) then
      xl=root-step; fl=f_value(cf,npv,dc,comp,freq,cur,cur,xl,nlegs); xh=root
    else
      xl=root; fl=fh; xh=root+step; fh=f_value(cf,npv,dc,comp,freq,cur,cur,xh,nlegs)
    end if
    evaluations=2
    do while(evaluations<=100)
      if(fl*fh<=0.0_real32) then
        if(close_enough(fl,0.0_real32)) then; solve_yield=xl; return; end if
        if(close_enough(fh,0.0_real32)) then; solve_yield=xh; return; end if
        root=(xh+xl)/2.0_real32; exit
      end if
      if(abs(fl)<abs(fh)) then
        xl=xl+1.6_real32*(xl-xh); fl=f_value(cf,npv,dc,comp,freq,cur,cur,xl,nlegs)
      else if(abs(fl)>abs(fh)) then
        xh=xh+1.6_real32*(xh-xl); fh=f_value(cf,npv,dc,comp,freq,cur,cur,xh,nlegs)
      else if(flipflop==-1) then
        xl=xl+1.6_real32*(xl-xh); fl=f_value(cf,npv,dc,comp,freq,cur,cur,xl,nlegs); evaluations=evaluations+1; flipflop=1
      else
        xh=xh+1.6_real32*(xh-xl); fh=f_value(cf,npv,dc,comp,freq,cur,cur,xh,nlegs); flipflop=-1
      end if
      evaluations=evaluations+1
    end do
    if(evaluations>100) then; solve_yield=0.0_real32; return; end if
    if(fl<0.0_real32) then; else; root=xl; xl=xh; xh=root; end if
    dxold=xh-xl; dx=dxold; root=(xh+xl)/2.0_real32
    f=f_value(cf,npv,dc,comp,freq,cur,cur,root,nlegs); df=duration(cf,dc,comp,freq,cur,cur,root,nlegs); evaluations=evaluations+1
    do while(evaluations<=100)
      if((((root-xh)*df-f)*((root-xl)*df-f)>0.0_real32) .or. abs(2.0_real32*f)>abs(dxold*df)) then
        dxold=dx; dx=(xh-xl)/2.0_real32; root=xl+dx
      else
        dxold=dx; dx=f/df; root=root-dx
      end if
      if(abs(dx)<eps) then; solve_yield=root; return; end if
      f=f_value(cf,npv,dc,comp,freq,cur,cur,root,nlegs); df=duration(cf,dc,comp,freq,cur,cur,root,nlegs); evaluations=evaluations+1
      if(f<0.0_real32) then; xl=root; else; xh=root; end if
    end do
    solve_yield=root
  end function

  logical function close_enough(x,y)
    real(real32), intent(in) :: x,y
    real(real32) :: d,t
    d=abs(x-y); t=42.0_real32*ql_epsilon
    close_enough=(d<=t*abs(x) .and. d<=t*abs(y))
  end function

  subroutine compute_one(discount,curr,maturity,clean_input,bond,dirty,accr,clean,forward,i)
    type(yield_curve_t), intent(inout) :: discount(0:)
    type(date_t), intent(in) :: curr(0:), maturity(0:)
    type(bond_t), intent(in) :: bond(0:)
    real(real32), intent(in) :: clean_input(0:)
    real(real32), intent(out) :: dirty(0:),accr(0:),clean(0:),forward(0:)
    integer, intent(in) :: i
    integer :: nlegs,ncf,leg
    type(date_t) :: cfdate,cs,ce
    type(cashflows_t) :: cf
    ncf=0; cfdate=bond(i)%maturity_date
    do while(cfdate%serial>bond(i)%start_date%serial); ncf=ncf+1; cfdate=advance_date(cfdate,-6); end do
    nlegs=ncf+1; cf%int_rate%day_counter=USE_EXACT_DAY
    cf%int_rate%rate=bond(i)%rate; cf%int_rate%freq=1.0_real32
    cf%int_rate%comp=SIMPLE_INTEREST; cf%day_counter=USE_EXACT_DAY
    cf%nominal=100.0_real32
    cs=advance_date(bond(i)%maturity_date,-6*(nlegs-1)); ce=advance_date(cs,6)
    do leg=0,nlegs-2
      cf%legs(leg)%payment_date=ce; cf%legs(leg)%accrual_start_date=cs
      cf%legs(leg)%accrual_end_date=ce; cf%legs(leg)%amount=real(COMPUTE_AMOUNT,real32)
      cs=ce; ce=advance_date(ce,6)
    end do
    cf%legs(nlegs-1)%payment_date=bond(i)%maturity_date
    cf%legs(nlegs-1)%accrual_start_date=curr(i); cf%legs(nlegs-1)%accrual_end_date=curr(i)
    cf%legs(nlegs-1)%amount=100.0_real32
    forward(i)=solve_yield(cf,(clean_input(i)+accrued(cf,curr(i),maturity,i,nlegs))/100.0_real32, &
      USE_EXACT_DAY,COMPOUNDED_INTEREST,2.0_real32,curr(i),nlegs)
    discount(i)%forward=forward(i); dirty(i)=cashflow_npv(cf,discount(i),curr(i),nlegs)
    accr(i)=accrued(cf,curr(i),maturity,i,nlegs); clean(i)=dirty(i)-accr(i)
  end subroutine
  subroutine run_bonds(repeat)
    use omp_lib
    integer, intent(in) :: repeat
    integer, parameter :: nbonds=1000000
    type(yield_curve_t), allocatable :: discount(:),repo(:)
    type(date_t), allocatable :: curr(:), maturity(:)
    type(bond_t), allocatable :: bond(:)
    real(real32), allocatable :: clean_input(:),strike(:),dirty(:),accr(:),clean(:),forward(:)
    integer :: i,j, clock_rate,clock_start,clock_end
    integer(c_int) :: r
    type(date_t) :: issue,mat,today
    real(real64) :: host_start,host_end,total,ktime
    interface
      subroutine c_srand(seed) bind(C,name='srand')
        import c_int
        integer(c_int), value :: seed
      end subroutine
      function c_rand() bind(C,name='rand') result(v)
        import c_int
        integer(c_int) :: v
      end function
    end interface
    allocate(discount(0:nbonds-1),repo(0:nbonds-1),curr(0:nbonds-1),maturity(0:nbonds-1))
    allocate(bond(0:nbonds-1),clean_input(0:nbonds-1),strike(0:nbonds-1),dirty(0:nbonds-1))
    allocate(accr(0:nbonds-1),clean(0:nbonds-1),forward(0:nbonds-1))
    call c_srand(123_c_int)
    do i=0,nbonds-1
      r=c_rand(); issue=initialize_date(mod(r,28)+1,mod(c_rand(),12)+1,1999-mod(c_rand(),2))
      mat=initialize_date(mod(c_rand(),28)+1,mod(c_rand(),12)+1,2000+mod(c_rand(),2))
      today=initialize_date(mat%day-1,mat%month,mat%year)
      bond(i)%start_date=issue; bond(i)%maturity_date=mat
      bond(i)%rate=.08_real32+(real(c_rand(),real32)/2147483647.0_real32-.5_real32)*.1_real32
      discount(i)%ref_date=today; discount(i)%cal_date=today; discount(i)%forward=-.1_real32
      discount(i)%compounding=real(COMPOUNDED_INTEREST,real32); discount(i)%frequency=2.0_real32
      discount(i)%day_counter=USE_EXACT_DAY
      repo(i)%ref_date=today; repo(i)%cal_date=today; repo(i)%forward=.07_real32
      repo(i)%compounding=real(SIMPLE_INTEREST,real32); repo(i)%frequency=1.0_real32
      repo(i)%day_counter=USE_SERIAL_NUMS
      curr(i)=today; maturity(i)=mat; clean_input(i)=89.97693786_real32; strike(i)=91.5745_real32
    end do
    write(*,'(/,a,i0,a,/)') 'Number of Bonds: ',nbonds,''
    write(*,'(a,i0)') 'Inputs for bond with index ',nbonds/2
    write(*,'(a,i0,a,i0,a,i0)') 'Bond Issue Date: ',bond(nbonds/2)%start_date%month, &
      '-',bond(nbonds/2)%start_date%day,'-' ,bond(nbonds/2)%start_date%year
    host_start=omp_get_wtime(); ktime=0.0_real64
    do j=1,repeat
      call system_clock(clock_start,clock_rate)
      !$omp target data map(to:discount,repo,curr,maturity,clean_input,bond,strike) map(from:dirty,accr,clean,forward)
      !$omp target teams distribute parallel do thread_limit(256)
      do i=0,nbonds-1
        call compute_one(discount,curr,maturity,clean_input,bond,dirty,accr,clean,forward,i)
      end do
      !$omp end target teams distribute parallel do
      !$omp end target data
      call system_clock(clock_end)
      total=real(clock_end-clock_start,real64)/real(clock_rate,real64)*1.0e6_real64
      ktime=ktime+total
    end do
    host_end=omp_get_wtime()
    write(*,'(a)') 'Run on GPU'
    write(*,'(a,f12.6,a)') 'Average kernel execution time on GPU: ',ktime*1.0e-3_real64/real(repeat,real64),' (ms)'
    write(*,'(a,f12.6,a)') 'Average processing time on GPU: ',(host_end-host_start)*1.0e3_real64/real(repeat,real64),' (ms)'
    write(*,'(a,f14.6)') 'Sum of output dirty prices on GPU: ',sum(real(dirty,real64))
    write(*,'(a,f12.6)') 'Dirty Price: ',dirty(nbonds/2)
    write(*,'(a,f12.6)') 'Accrued Amount: ',accr(nbonds/2)
    write(*,'(a,f12.6)') 'Clean Price: ',clean(nbonds/2)
    write(*,'(a,f12.6)') 'Bond Forward Val: ',forward(nbonds/2)
    host_start=omp_get_wtime()
    do j=1,2
      do i=0,nbonds-1
        call compute_one(discount,curr,maturity,clean_input,bond,dirty,accr,clean,forward,i)
      end do
    end do
    host_end=omp_get_wtime()
    write(*,'(a)') 'Run on CPU'
    write(*,'(a,f12.6,a)') 'Average processing time on CPU: ',(host_end-host_start)*1.0e3_real64/2.0_real64,' (ms)'
    write(*,'(a,f14.6)') 'Sum of output dirty prices on CPU: ',sum(real(dirty,real64))
    write(*,'(a,f12.6)') 'Dirty Price: ',dirty(nbonds/2)
    write(*,'(a,f12.6)') 'Accrued Amount: ',accr(nbonds/2)
    write(*,'(a,f12.6)') 'Clean Price: ',clean(nbonds/2)
    write(*,'(a,f12.6)') 'Bond Forward Val: ',forward(nbonds/2)
    deallocate(discount,repo,curr,maturity,bond,clean_input,strike,dirty,accr,clean,forward)
  end subroutine
end module

program bonds
  use bonds_mod
  implicit none
  integer :: argc,repeat,ios
  character(len=64) :: arg
  argc=command_argument_count()
  if(argc/=1) then; write(*,'(a)') 'Usage: ./main <repeat>'; stop 1; end if
  call get_command_argument(1,arg); read(arg,*,iostat=ios) repeat
  if(ios/=0 .or. repeat<1) then; write(*,'(a)') 'Usage: ./main <repeat>'; stop 1; end if
  call run_bonds(repeat)
end program
