! NVIDIA American-option pricing benchmark (Longstaff-Schwartz), Fortran port.
! The storage and zero-based indexing deliberately match aop-omp/main.cpp.
module aop_kernels
  use, intrinsic :: iso_fortran_env, only : real64, int64
  use omp_lib
  implicit none
  integer, parameter :: dp = real64, slots = 12, threads_paths = 256, threads_beta = 128
  !$omp declare target (payoff_value, in_the_money, off_diag_norm, swap_value, assemble_r, svd_3x3)
contains

  pure real(dp) function payoff_value(s, k, put) result(value)
    real(dp), intent(in) :: s, k
    logical, intent(in) :: put
    if (put) then
      value = max(k - s, 0.0_dp)
    else
      value = max(s - k, 0.0_dp)
    end if
  end function payoff_value

  pure integer function in_the_money(s, k, put) result(value)
    real(dp), intent(in) :: s, k
    logical, intent(in) :: put
    if ((put .and. s < k) .or. ((.not. put) .and. s > k)) then
      value = 1
    else
      value = 0
    end if
  end function in_the_money

  pure real(dp) function off_diag_norm(a01, a02, a12) result(value)
    real(dp), intent(in) :: a01, a02, a12
    value = sqrt(2.0_dp * (a01*a01 + a02*a02 + a12*a12))
  end function off_diag_norm

  subroutine swap_value(x, y)
    real(dp), intent(inout) :: x, y
    real(dp) :: t
    t = x; x = y; y = t
  end subroutine swap_value

  subroutine assemble_r(m, sums, smem)
    integer, intent(in) :: m
    real(dp), intent(in) :: sums(0:3)
    real(dp), intent(inout) :: smem(0:slots-1)
    real(dp) :: x0,x1,x2,x0sq,x1sq,x2sq,sum1,sum2,sum3,sum4,mass,sigma,mu,v0,v0sq,beta
    real(dp) :: invv0, one_minus_beta, beta_div_v0, beta_div_v0sq,c1,c2,c3,c4,c5
    x0=smem(0); x1=smem(1); x2=smem(2); x0sq=x0*x0
    sum1=sums(0)-x0; sum2=sums(1)-x0sq; sum3=sums(2)-x0sq*x0; sum4=sums(3)-x0sq*x0sq
    mass=real(m,dp); sigma=mass-1.0_dp; mu=sqrt(mass); v0=-sigma/(1.0_dp+mu); v0sq=v0*v0
    beta=2.0_dp*v0sq/(sigma+v0sq); invv0=1.0_dp/v0; one_minus_beta=1.0_dp-beta; beta_div_v0=beta*invv0
    smem(0)=mu; smem(1)=one_minus_beta*x0-beta_div_v0*sum1; smem(2)=one_minus_beta*x0sq-beta_div_v0*sum2
    beta_div_v0sq=beta_div_v0*invv0; c1=beta_div_v0sq*sum1+beta_div_v0*x0; c2=beta_div_v0sq*sum2+beta_div_v0*x0sq
    x1sq=x1*x1; sum1=sum1-x1; sum2=sum2-x1sq; sum3=sum3-x1sq*x1; sum4=sum4-x1sq*x1sq
    x0=x1-c1; x0sq=x0*x0; sigma=sum2-2.0_dp*c1*sum1+(mass-2.0_dp)*c1*c1
    if (abs(sigma) < 1.0e-16_dp) then
      beta=0.0_dp
    else
      mu=sqrt(x0sq+sigma)
      if (x0 <= 0.0_dp) then; v0=x0-mu; else; v0=-sigma/(x0+mu); end if
      v0sq=v0*v0; beta=2.0_dp*v0sq/(sigma+v0sq)
    end if
    invv0=1.0_dp/v0; beta_div_v0=beta*invv0
    c3=(sum3-c1*sum2-c2*sum1+(mass-2.0_dp)*c1*c2)*beta_div_v0
    c4=(x1sq-c2)*beta_div_v0+c3*invv0; c5=c1*c4-c2; one_minus_beta=1.0_dp-beta
    smem(3)=one_minus_beta*x0-beta_div_v0*sigma; smem(4)=one_minus_beta*(x1sq-c2)-c3
    x2sq=x2*x2; sum1=sum1-x2; sum2=sum2-x2sq; sum3=sum3-x2sq*x2; sum4=sum4-x2sq*x2sq
    x0=x2sq-c4*x2+c5; sigma=sum4-2.0_dp*c4*sum3+(c4*c4+2.0_dp*c5)*sum2-2.0_dp*c4*c5*sum1+(mass-3.0_dp)*c5*c5
    if (abs(sigma) < 1.0e-12_dp) then
      beta=0.0_dp
    else
      mu=sqrt(x0*x0+sigma)
      if (x0 <= 0.0_dp) then; v0=x0-mu; else; v0=-sigma/(x0+mu); end if
      v0sq=v0*v0; beta=2.0_dp*v0sq/(sigma+v0sq)
    end if
    smem(5)=(1.0_dp-beta)*x0-(beta/v0)*sigma
  end subroutine assemble_r

  subroutine svd_3x3(m, sums, smem)
    integer, intent(in) :: m
    real(dp), intent(in) :: sums(0:3)
    real(dp), intent(inout) :: smem(0:slots-1)
    real(dp) :: r00,r01,r02,r11,r12,r22,a00,a01,a02,a11,a12,a22,c,s,tau,sgn,t
    real(dp) :: b00,b01,b02,b10,b11,b12,b20,b21,b22
    real(dp) :: v00,v01,v02,v10,v11,v12,v20,v21,v22,u00,u01,u02,u10,u11,u12,u20,u21,u22
    real(dp) :: is0,is1,is2
    integer :: iter
    call assemble_r(m,sums,smem)
    r00=smem(0); r01=smem(1); r02=smem(2); r11=smem(3); r12=smem(4); r22=smem(5)
    a00=r00*r00; a01=r00*r01; a02=r00*r02; a11=r01*r01+r11*r11; a12=r01*r02+r11*r12; a22=r02*r02+r12*r12+r22*r22
    v00=1.0_dp; v01=0.0_dp; v02=0.0_dp; v10=0.0_dp; v11=1.0_dp; v12=0.0_dp; v20=0.0_dp; v21=0.0_dp; v22=1.0_dp
    do iter=0,15
      if (off_diag_norm(a01,a02,a12) < 1.0e-12_dp) exit
      c=1.0_dp; s=0.0_dp
      if (a01 /= 0.0_dp) then
        tau=(a11-a00)/(2.0_dp*a01); if (tau < 0.0_dp) then; sgn=-1.0_dp; else; sgn=1.0_dp; end if
        t=sgn/(sgn*tau+sqrt(1.0_dp+tau*tau)); c=1.0_dp/sqrt(1.0_dp+t*t); s=t*c
      end if
      b00=c*a00-s*a01; b01=s*a00+c*a01; b10=c*a01-s*a11; b11=s*a01+c*a11; b02=a02
      a00=c*b00-s*b10; a01=c*b01-s*b11; a11=s*b01+c*b11; a02=c*b02-s*a12; a12=s*b02+c*a12
      b00=c*v00-s*v01; v01=s*v00+c*v01; v00=b00; b10=c*v10-s*v11; v11=s*v10+c*v11; v10=b10; b20=c*v20-s*v21; v21=s*v20+c*v21; v20=b20
      c=1.0_dp; s=0.0_dp
      if (a02 /= 0.0_dp) then
        tau=(a22-a00)/(2.0_dp*a02); if (tau < 0.0_dp) then; sgn=-1.0_dp; else; sgn=1.0_dp; end if
        t=sgn/(sgn*tau+sqrt(1.0_dp+tau*tau)); c=1.0_dp/sqrt(1.0_dp+t*t); s=t*c
      end if
      b00=c*a00-s*a02; b01=c*a01-s*a12; b02=s*a00+c*a02; b20=c*a02-s*a22; b22=s*a02+c*a22
      a00=c*b00-s*b20; a12=s*a01+c*a12; a02=c*b02-s*b22; a22=s*b02+c*b22; a01=b01
      b00=c*v00-s*v02; v02=s*v00+c*v02; v00=b00; b10=c*v10-s*v12; v12=s*v10+c*v12; v10=b10; b20=c*v20-s*v22; v22=s*v20+c*v22; v20=b20
      c=1.0_dp; s=0.0_dp
      if (a12 /= 0.0_dp) then
        tau=(a22-a11)/(2.0_dp*a12); if (tau < 0.0_dp) then; sgn=-1.0_dp; else; sgn=1.0_dp; end if
        t=sgn/(sgn*tau+sqrt(1.0_dp+tau*tau)); c=1.0_dp/sqrt(1.0_dp+t*t); s=t*c
      end if
      b02=s*a01+c*a02; b11=c*a11-s*a12; b12=s*a11+c*a12; b21=c*a12-s*a22; b22=s*a12+c*a22
      a01=c*a01-s*a02; a02=b02; a11=c*b11-s*b21; a12=c*b12-s*b22; a22=s*b12+c*b22
      b01=c*v01-s*v02; v02=s*v01+c*v02; v01=b01; b11=c*v11-s*v12; v12=s*v11+c*v12; v11=b11; b21=c*v21-s*v22; v22=s*v21+c*v22; v21=b21
    end do
    if (a00 < a11) then; call swap_value(a00,a11); call swap_value(v00,v01); call swap_value(v10,v11); call swap_value(v20,v21); end if
    if (a00 < a22) then; call swap_value(a00,a22); call swap_value(v00,v02); call swap_value(v10,v12); call swap_value(v20,v22); end if
    if (a11 < a22) then; call swap_value(a11,a22); call swap_value(v01,v02); call swap_value(v11,v12); call swap_value(v21,v22); end if
    if(abs(a00)<1.e-12_dp)then; is0=0.0_dp;else;is0=1.0_dp/a00;end if
    if(abs(a11)<1.e-12_dp)then; is1=0.0_dp;else;is1=1.0_dp/a11;end if
    if(abs(a22)<1.e-12_dp)then; is2=0.0_dp;else;is2=1.0_dp/a22;end if
    u00=v00*is0;u01=v01*is1;u02=v02*is2;u10=v10*is0;u11=v11*is1;u12=v12*is2;u20=v20*is0;u21=v21*is1;u22=v22*is2
    b00=u00*v00+u01*v01+u02*v02; b01=u00*v10+u01*v11+u02*v12; b02=u00*v20+u01*v21+u02*v22
    b11=u10*v10+u11*v11+u12*v12; b12=u10*v20+u11*v21+u12*v22; b22=u20*v20+u21*v21+u22*v22
    smem(6)=b00*r00+b01*r01+b02*r02; smem(7)=b01*r11+b02*r12; smem(8)=b02*r22
    smem(9)=b11*r11+b12*r12; smem(10)=b12*r22; smem(11)=b22*r22
  end subroutine svd_3x3

  subroutine generate_paths(num_timesteps,num_paths,k,put,dt,s0,r,sigma,samples,paths)
    integer,intent(in)::num_timesteps,num_paths
    real(dp),intent(in)::k,dt,s0,r,sigma
    logical,intent(in)::put
    real(dp),intent(in)::samples(0:)
    real(dp),intent(inout)::paths(0:)
    integer::path,timestep,offset
    real(dp)::s,drift,vol
    drift=(r-0.5_dp*sigma*sigma)*dt; vol=sigma*sqrt(dt)
    !$omp target teams distribute parallel do thread_limit(256) private(s,offset,timestep)
    do path=0,num_paths-1
      s=s0; offset=path
      do timestep=0,num_timesteps-2
        s=s*exp(drift+vol*samples(offset)); paths(offset)=s; offset=offset+num_paths
      end do
      s=s*exp(drift+vol*samples(offset)); paths(offset)=payoff_value(s,k,put)
    end do
    !$omp end target teams distribute parallel do
  end subroutine generate_paths

  subroutine prepare_svd(num_teams,num_paths,min_money,k,put,paths,out_money,svds)
    integer,intent(in)::num_teams,num_paths,min_money
    real(dp),intent(in)::k,paths(0:)
    logical,intent(in)::put
    integer,intent(inout)::out_money(0:)
    real(dp),intent(inout)::svds(0:)
    real(dp)::smem(0:slots-1),sums(0:3),s,x,xsq
    integer::bid,path,offset,m,inmoney,found_paths
    !$omp target teams distribute parallel do thread_limit(1) private(path,offset,m,inmoney,found_paths,smem,sums,s,x,xsq)
    do bid=0,num_teams-1
      offset=bid*num_paths; m=0; found_paths=0; sums=0.0_dp; smem=0.0_dp
      do path=0,num_paths-1
        s=paths(offset+path); inmoney=in_the_money(s,k,put)
        if(inmoney /= 0)then
          if(found_paths < 3) smem(found_paths)=s
          found_paths=found_paths+1; m=m+1; x=s; xsq=x*x
          sums(0)=sums(0)+x; sums(1)=sums(1)+xsq; sums(2)=sums(2)+xsq*x; sums(3)=sums(3)+xsq*xsq
        end if
      end do
      if(m < min_money)then
        out_money(bid)=1
      else
        out_money(bid)=0; call svd_3x3(m,sums,smem)
        svds(16*bid:16*bid+slots-1)=smem
      end if
    end do
    !$omp end target teams distribute parallel do
  end subroutine prepare_svd

  subroutine compute_beta(num_paths,k,put,svd,paths,cashflows,out_money,beta)
    integer,intent(in)::num_paths,out_money
    real(dp),intent(in)::k,svd(0:),paths(0:),cashflows(0:)
    logical,intent(in)::put
    ! C++ uses reduction(+:beta[:3]); keep the reduction domain to exactly
    ! these three coefficients rather than the complete temporary buffer.
    real(dp),intent(inout)::beta(0:2)
    integer::path,inmoney
    real(dp)::r00,r01,r02,r11,r12,r22,w00,w01,w02,w11,w12,w22,ir00,ir11,ir22,ir01,ir02,ir12,iw00,s,q1,q2,wi0,wi1,wi2,cf
    if(out_money == 0)then
      !$omp target teams distribute parallel do thread_limit(128) reduction(+:beta)
      do path=0,num_paths-1
        r00=svd(0);r01=svd(1);r02=svd(2);r11=svd(3);r12=svd(4);r22=svd(5);w00=svd(6);w01=svd(7);w02=svd(8);w11=svd(9);w12=svd(10);w22=svd(11)
        if(r00/=0.0_dp)then;ir00=1.0_dp/r00;else;ir00=0.0_dp;end if; if(r11/=0.0_dp)then;ir11=1.0_dp/r11;else;ir11=0.0_dp;end if; if(r22/=0.0_dp)then;ir22=1.0_dp/r22;else;ir22=0.0_dp;end if
        ir01=ir00*ir11*r01;ir02=ir00*ir22*r02;ir12=ir22*r12;iw00=w00*ir00;s=paths(path);inmoney=in_the_money(s,k,put)
        q1=ir11*s-ir01;q2=ir22*s*s-ir02-q1*ir12;wi0=iw00+w01*q1+w02*q2;wi1=w11*q1+w12*q2;wi2=w22*q2
        if(inmoney/=0)then;cf=cashflows(path);else;cf=0.0_dp;end if
        beta(0)=beta(0)+wi0*cf;beta(1)=beta(1)+wi1*cf;beta(2)=beta(2)+wi2*cf
      end do
      !$omp end target teams distribute parallel do
    end if
  end subroutine compute_beta

  subroutine update_cashflow(num_teams,num_paths,k,put,discount,beta,paths,out_money,cashflows)
    integer,intent(in)::num_teams,num_paths,out_money
    real(dp),intent(in)::k,discount,beta(0:2),paths(0:)
    logical,intent(in)::put
    real(dp),intent(inout)::cashflows(0:)
    integer::path
    real(dp)::old,s,pay,estimate
    !$omp target teams distribute parallel do num_teams(num_teams) thread_limit(128) private(old,s,pay,estimate)
    do path=0,num_paths-1
      old=discount*cashflows(path)
      if(out_money /= 0)then
        cashflows(path)=old
      else
        s=paths(path); pay=payoff_value(s,k,put); estimate=(beta(0)+beta(1)*s+beta(2)*s*s)*discount
        if(pay <= 1.0e-8_dp .or. pay <= estimate)pay=old
        cashflows(path)=pay
      end if
    end do
    !$omp end target teams distribute parallel do
  end subroutine update_cashflow

  subroutine compute_sum(num_paths,cashflows,discount,price)
    integer,intent(in)::num_paths
    real(dp),intent(in)::cashflows(0:),discount
    real(dp),intent(out)::price
    real(dp)::sum
    integer::path
    sum=0.0_dp
    !$omp target teams distribute parallel do thread_limit(128) map(tofrom:sum) reduction(+:sum)
    do path=0,num_paths-1; sum=sum+cashflows(path); end do
    !$omp end target teams distribute parallel do
    price=discount*sum/real(num_paths,dp)
  end subroutine compute_sum

  subroutine do_run(samples,num_timesteps,num_paths,k,put,dt,s0,r,sigma,paths,cashflows,svds,out_money,temp,price)
    integer,intent(in)::num_timesteps,num_paths
    real(dp),intent(inout)::samples(0:),paths(0:),cashflows(0:),svds(0:),temp(0:)
    integer,intent(inout)::out_money(0:)
    real(dp),intent(in)::k,dt,s0,r,sigma
    logical,intent(in)::put
    real(dp),intent(out)::price
    integer::i,timestep,grid_dim,update_grid
    real(dp)::discount,num_waves
    !$omp target update to(samples)
    call generate_paths(num_timesteps,num_paths,k,put,dt,s0,r,sigma,samples,paths)
    !$omp target teams distribute parallel do thread_limit(256)
    do i=0,num_timesteps-1;out_money(i)=0;end do
    !$omp end target teams distribute parallel do
    call prepare_svd(num_timesteps-1,num_paths,4,k,put,paths,out_money,svds)
    discount=exp(-r*dt);grid_dim=(num_paths+127)/128;num_waves=real(grid_dim*128,dp)/real(256*112,dp);update_grid=grid_dim
    if(num_waves < 10.0_dp .and. num_waves-real(int(num_waves),dp)<0.6_dp)update_grid=max(1,int(num_waves))*256*112/128
    do timestep=num_timesteps-2,0,-1
      call compute_beta(num_paths,k,put,svds(16*timestep:),paths(timestep*num_paths:),cashflows,out_money(timestep),temp)
      call update_cashflow(update_grid,num_paths,k,put,discount,temp,paths(timestep*num_paths:),out_money(timestep),cashflows)
    end do
    call compute_sum(num_paths,cashflows,discount,price)
  end subroutine do_run

  subroutine normal_samples(samples, state, saved, saved_available)
    real(dp),intent(out)::samples(0:)
    integer(int64),intent(inout)::state
    real(dp),intent(inout)::saved
    logical,intent(inout)::saved_available
    integer::i
    real(dp)::x,y,r2,mult,range,u
    ! This is libstdc++'s default_random_engine (minstd_rand0, seed 1) plus
    ! normal_distribution<double>'s Marsaglia polar method.  The OMP source
    ! constructs precisely those types and calls operator() one sample at a time.
    i=0
    do while(i <= ubound(samples,1))
      if(saved_available)then
        samples(i)=saved; saved_available=.false.; i=i+1
      else
        do
          range=2147483646.0_dp
          state=mod(16807_int64*state,2147483647_int64)
          u=real(state-1_int64,dp)
          state=mod(16807_int64*state,2147483647_int64)
          u=(u+real(state-1_int64,dp)*range)/(range*range)
          x=2.0_dp*u-1.0_dp
          state=mod(16807_int64*state,2147483647_int64)
          u=real(state-1_int64,dp)
          state=mod(16807_int64*state,2147483647_int64)
          u=(u+real(state-1_int64,dp)*range)/(range*range)
          y=2.0_dp*u-1.0_dp; r2=x*x+y*y
          if(r2 <= 1.0_dp .and. r2 /= 0.0_dp)exit
        end do
        mult=sqrt(-2.0_dp*log(r2)/r2)
        samples(i)=y*mult; saved=x*mult; saved_available=.true.; i=i+1
      end if
    end do
  end subroutine normal_samples

  function binomial_tree(num_timesteps,k,put,dt,s0,r,sigma) result(value)
    integer,intent(in)::num_timesteps
    real(dp),intent(in)::k,dt,s0,r,sigma
    logical,intent(in)::put
    real(dp)::value,u,d,a,p,x,expected
    real(dp),allocatable::tree(:)
    integer::t,i
    allocate(tree(0:num_timesteps));u=exp(sigma*sqrt(dt));d=exp(-sigma*sqrt(dt));a=exp(r*dt);p=(a-d)/(u-d);x=d**num_timesteps
    do t=0,num_timesteps;tree(t)=payoff_value(s0*x,k,put);x=x*u*u;end do
    do t=num_timesteps-1,0,-1
      x=d**t
      do i=0,t;expected=exp(-r*dt)*(p*tree(i+1)+(1.0_dp-p)*tree(i));tree(i)=max(payoff_value(s0*x,k,put),expected);x=x*u*u;end do
    end do
    value=tree(0);deallocate(tree)
  end function binomial_tree

  pure real(dp) function normcdf(x) result(value)
    real(dp),intent(in)::x
    value=(1.0_dp+erf(x/sqrt(2.0_dp)))/2.0_dp
  end function normcdf

  pure real(dp) function european_price(t,k,s0,r,sigma,put) result(value)
    real(dp),intent(in)::t,k,s0,r,sigma
    logical,intent(in)::put
    real(dp)::d1,d2
    d1=(log(s0/k)+(r+0.5_dp*sigma*sigma)*t)/(sigma*sqrt(t));d2=d1-sigma*sqrt(t)
    if(put)then;value=k*exp(-r*t)*normcdf(-d2)-s0*normcdf(-d1);else;value=s0*normcdf(d1)-k*exp(-r*t)*normcdf(d2);end if
  end function european_price

end module aop_kernels

program aop
  use, intrinsic :: iso_fortran_env, only : real64
  use aop_kernels
  implicit none
  integer,parameter::max_grid_size=2048
  integer::num_timesteps,num_paths,num_runs,argc,i,clock_start,clock_end,clock_rate,total_paths
  integer(int64)::rng_state
  real(dp)::t,k,s0,r,sigma,dt,h_price,reference,total_ms,saved_normal
  logical::put,saved_normal_available
  character(len=128)::arg,next
  real(dp),allocatable::samples(:),paths(:),svds(:),temp(:)
  integer,allocatable::out_money(:)
  num_timesteps=100;num_paths=32;num_runs=1;t=1.0_dp;k=4.0_dp;s0=3.6_dp;r=0.06_dp;sigma=0.20_dp;put=.true.
  argc=command_argument_count();i=1
  do while(i<=argc)
    call get_command_argument(i,arg)
    select case(trim(arg))
    case('-call');put=.false.
    case('-timesteps','-paths','-runs','-T','-S0','-K','-r','-sigma')
      if(i==argc)error stop 'Missing option value';i=i+1;call get_command_argument(i,next)
      select case(trim(arg))
      case('-timesteps');read(next,*)num_timesteps
      case('-paths');read(next,*)num_paths
      case('-runs');read(next,*)num_runs
      case('-T');read(next,*)t
      case('-S0');read(next,*)s0
      case('-K');read(next,*)k
      case('-r');read(next,*)r
      case('-sigma');read(next,*)sigma
      end select
    case default;write(*,'(A,A,A)')'Unknown option ',trim(arg),'. Aborting!!!';error stop 1
    end select
    i=i+1
  end do
  write(*,'(A)')'=============='
  write(*,'(A,I0)')'Num Timesteps         : ',num_timesteps;write(*,'(A,I0,A)')'Num Paths             : ',num_paths,'K';write(*,'(A,I0)')'Num Runs              : ',num_runs
  write(*,'(A,F0.6)')'T                     : ',t;write(*,'(A,F0.6)')'S0                    : ',s0;write(*,'(A,F0.6)')'K                     : ',k;write(*,'(A,F0.6)')'r                     : ',r;write(*,'(A,F0.6)')'sigma                 : ',sigma
  if(put)then;write(*,'(A)')'Option Type           : American Put';else;write(*,'(A)')'Option Type           : American Call';end if
  total_paths=num_paths*1024;dt=t/real(num_timesteps,dp);allocate(samples(0:num_timesteps*total_paths-1),paths(0:num_timesteps*total_paths-1),svds(0:16*num_timesteps-1),out_money(0:num_timesteps-1),temp(0:4*max_grid_size-1))
  total_ms=0.0_dp;rng_state=1_int64;saved_normal=0.0_dp;saved_normal_available=.false.
  !$omp target data map(alloc:samples,paths,svds,out_money,temp)
  do i=1,num_runs
    call normal_samples(samples,rng_state,saved_normal,saved_normal_available);call system_clock(clock_start,clock_rate)
    call do_run(samples,num_timesteps,total_paths,k,put,dt,s0,r,sigma,paths,paths((num_timesteps-1)*total_paths:),svds,out_money,temp,h_price)
    call system_clock(clock_end);total_ms=total_ms+1000.0_dp*real(clock_end-clock_start,dp)/real(clock_rate,dp)
  end do
  !$omp end target data
  write(*,'(A)')'=============='
  write(*,'(A,F0.8)')'GPU Longstaff-Schwartz: ',h_price
  reference=binomial_tree(num_timesteps,k,put,dt,s0,r,sigma);write(*,'(A,F0.8)')'Binonmial             : ',reference
  reference=european_price(t,k,s0,r,sigma,put);write(*,'(A,F0.8)')'European Price        : ',reference
  write(*,'(A)')'=============='
  write(*,'(A,F0.3,A)')'elapsed time for each run         : ',total_ms/real(num_runs,dp),'ms'
  write(*,'(A)')'=============='
  deallocate(samples,paths,svds,out_money,temp)
end program aop
