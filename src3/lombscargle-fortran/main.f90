program lombscargle
  use iso_fortran_env, only: real32, real64
  use omp_lib; implicit none
  integer,parameter::x_shape=1000,freqs_shape=100000
  integer::repeat,n,tid,j,i
  character(len=64)::arg
  real(real32),allocatable::x(:),y(:),f(:),p(:),p2(:)
  real(real32)::A,w,phi,y_dot,freq,xc,xs,cc,ss,cs,c,s,tau,ct,st,ct2,st2,cst
  real(real64)::start,elapsed
  logical::error
  if(command_argument_count()/=1)then; print '(a)','Usage: ./main <repeat>'; stop 1; end if
  call get_command_argument(1,arg); read(arg,*) repeat
  allocate(x(0:x_shape-1),y(0:x_shape-1),f(0:freqs_shape-1),p(0:freqs_shape-1),p2(0:freqs_shape-1))
  A=2.0_real32; w=1.0_real32; phi=1.57_real32; y_dot=2.0_real32/1.5_real32
  do i=0,x_shape-1; x(i)=0.01_real32+i*(31.4_real32-0.01_real32)/x_shape; y(i)=A*sin(w*x(i)+phi); end do
  do i=0,freqs_shape-1; f(i)=0.01_real32+i*(10.0_real32-0.01_real32)/freqs_shape; end do
!$omp target data map(to:x,y,f) map(from:p)
  start=omp_get_wtime(); do n=1,repeat
!$omp target teams distribute parallel do thread_limit(256) private(freq,xc,xs,cc,ss,cs,c,s,j,tau,ct,st,ct2,st2,cst)
    do tid=0,freqs_shape-1
      freq=f(tid); xc=0; xs=0; cc=0; ss=0; cs=0
      do j=0,x_shape-1; s=sin(freq*x(j)); c=cos(freq*x(j)); xc=xc+y(j)*c; xs=xs+y(j)*s; cc=cc+c*c; ss=ss+s*s; cs=cs+c*s; end do
      tau=atan2(2.0_real32*cs,cc-ss)/(2.0_real32*freq); st=sin(freq*tau); ct=cos(freq*tau); ct2=ct*ct; st2=st*st; cst=2.0_real32*ct*st
      p(tid)=0.5_real32*(((ct*xc+st*xs)**2/(ct2*cc+cst*cs+st2*ss))+((ct*xs-st*xc)**2/(ct2*ss-cst*cs+st2*cc)))*y_dot
    end do
!$omp end target teams distribute parallel do
  end do; elapsed=omp_get_wtime()-start
!$omp end target data
  print '(a,f12.6,a)','Average kernel execution time ',elapsed*1.0e6_real64/repeat,' (us)'
  do tid=0,freqs_shape-1; freq=f(tid); xc=0; xs=0; cc=0; ss=0; cs=0; do j=0,x_shape-1; s=sin(freq*x(j)); c=cos(freq*x(j)); xc=xc+y(j)*c; xs=xs+y(j)*s; cc=cc+c*c; ss=ss+s*s; cs=cs+c*s; end do; tau=atan2(2.0_real32*cs,cc-ss)/(2.0_real32*freq); st=sin(freq*tau); ct=cos(freq*tau); ct2=ct*ct; st2=st*st; cst=2.0_real32*ct*st; p2(tid)=0.5_real32*(((ct*xc+st*xs)**2/(ct2*cc+cst*cs+st2*ss))+((ct*xs-st*xc)**2/(ct2*ss-cst*cs+st2*cc)))*y_dot; end do
  error=any(abs(p-p2)>1.0e-1_real32); print '(a)',merge('FAIL','PASS',error)
end program
