module fresnel_module
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  real(real64), parameter :: sqrt_2pi = 2.506628274631000502415765284811045253006_real64
  real(real64), parameter :: cos01(21) = [ &
    4.200987560240514577713e-1_real64,-9.358785913634965235904e-2_real64,-7.642539415723373644927e-3_real64,4.958117751796130135544e-3_real64, &
    -9.750236036106120253456e-4_real64,1.075201474958704192865e-4_real64,-4.415344769301324238886e-6_real64,-7.861633919783064216022e-7_real64, &
    1.919240966215861471754e-7_real64,-2.175775608982741065385e-8_real64,1.296559541430849437217e-9_real64,2.207205095025162212169e-11_real64, &
    -1.479219615873704298874e-11_real64,1.821350127295808288614e-12_real64,-1.228919312990171362342e-13_real64,2.227139250593818235212e-15_real64, &
    5.734729405928016301596e-16_real64,-8.284965573075354177016e-17_real64,6.067422701530157308321e-18_real64,-1.994908519477689596319e-19_real64, &
    -1.173365630675305693390e-20_real64 ]
  real(real64), parameter :: cos13(25) = [ &
    2.098677278318224971989e-1_real64,-9.314234883154103266195e-2_real64,1.739905936938124979297e-2_real64,-2.454274824644285136137e-3_real64, &
    1.589872606981337312438e-4_real64,4.203943842506079780413e-5_real64,-2.018022256093216535093e-5_real64,5.125709636776428285284e-6_real64, &
    -9.601813551752718650057e-7_real64,1.373989484857155846826e-7_real64,-1.348105546577211255591e-8_real64,2.745868700337953872632e-10_real64, &
    2.401655517097260106976e-10_real64,-6.678059547527685587692e-11_real64,1.140562171732840809159e-11_real64,-1.401526517205212219089e-12_real64, &
    1.105498827380224475667e-13_real64,2.040731455126809208066e-16_real64,-1.946040679213045143184e-15_real64,4.151821375667161733612e-16_real64, &
    -5.642257647205149369594e-17_real64,5.266176626521504829010e-18_real64,-2.299025577897146333791e-19_real64,-2.952226367506641078731e-20_real64, &
    8.760405943193778149078e-21_real64 ]
  real(real64), parameter :: cos35(21) = [ &
    1.025703371090289562388e-1_real64,-2.569833023232301400495e-2_real64,3.160592981728234288078e-3_real64,-3.776110718882714758799e-4_real64, &
    4.325593433537248833341e-5_real64,-4.668447489229591855730e-6_real64,4.619254757356785108280e-7_real64,-3.970436510433553795244e-8_real64, &
    2.535664754977344448598e-9_real64,-2.108170964644819803367e-11_real64,-2.959172018518707683013e-11_real64,6.727219944906606516055e-12_real64, &
    -1.062829587519902899001e-12_real64,1.402071724705287701110e-13_real64,-1.619154679722651005075e-14_real64,1.651319588396970446858e-15_real64, &
    -1.461704569438083772889e-16_real64,1.053521559559583268504e-17_real64,-4.760946403462515858756e-19_real64,-1.803784084922403924313e-20_real64, &
    7.873130866418738207547e-21_real64 ]
  real(real64), parameter :: cos57(19) = [ &
    6.738667333400589274018e-2_real64,-1.128146832637904868638e-2_real64,9.408843234170404670278e-4_real64,-7.800074103496165011747e-5_real64, &
    6.409101169623350885527e-6_real64,-5.201350558247239981834e-7_real64,4.151668914650221476906e-8_real64,-3.242202015335530552721e-9_real64, &
    2.460339340900396789789e-10_real64,-1.796823324763304661865e-11_real64,1.244108496436438952425e-12_real64,-7.950417122987063540635e-14_real64, &
    4.419142625999150971878e-15_real64,-1.759082736751040110146e-16_real64,-1.307443936270786700760e-18_real64,1.362484141039320395814e-18_real64, &
    -2.055236564763877250559e-19_real64,2.329142055084791308691e-20_real64,-2.282438671525884861970e-21_real64 ]
  real(real64), parameter :: sin01(21) = [ &
    2.560134650043040830997e-1_real64,-1.993005146464943284549e-1_real64,4.025503636721387266117e-2_real64,-4.459600454502960250729e-3_real64, &
    6.447097305145147224459e-5_real64,7.544218493763717599380e-5_real64,-1.580422720690700333493e-5_real64,1.755845848573471891519e-6_real64, &
    -9.289769688468301734718e-8_real64,-5.624033192624251079833e-9_real64,1.854740406702369495830e-9_real64,-2.174644768724492443378e-10_real64, &
    1.392899828133395918767e-11_real64,-6.989216003725983789869e-14_real64,-9.959396121060010838331e-14_real64,1.312085140393647257714e-14_real64, &
    -9.240470383522792593305e-16_real64,2.472168944148817385152e-17_real64,2.834615576069400293894e-18_real64,-4.650983461314449088349e-19_real64, &
    3.544083040732391556797e-20_real64 ]
  real(real64), parameter :: sin13(27) = [ &
    3.470341566046115476477e-2_real64,-3.855580521778624043304e-2_real64,1.420604309383996764083e-2_real64,-4.037349972538938202143e-3_real64, &
    9.292478174580997778194e-4_real64,-1.742730601244797978044e-4_real64,2.563352976720387343201e-5_real64,-2.498437524746606551732e-6_real64, &
    -1.334367201897140224779e-8_real64,7.436854728157752667212e-8_real64,-2.059620371321272169176e-8_real64,3.753674773239250330547e-9_real64, &
    -5.052913010605479996432e-10_real64,4.580877371233042345794e-11_real64,-7.664740716178066564952e-13_real64,-7.200170736686941995387e-13_real64, &
    1.812701686438975518372e-13_real64,-2.799876487275995466163e-14_real64,3.048940815174731772007e-15_real64,-1.936754063718089166725e-16_real64, &
    -7.653673328908379651914e-18_real64,4.534308864750374603371e-18_real64,-8.011054486030591219007e-19_real64,9.374587915222218230337e-20_real64, &
    -7.144943099280650363024e-21_real64,1.105276695821552769144e-22_real64,6.989334213887669628647e-23_real64 ]
  real(real64), parameter :: sin35(23) = [ &
    3.684922395955255848372e-3_real64,-2.624595437764014386717e-3_real64,6.329162500611499391493e-4_real64,-1.258275676151483358569e-4_real64, &
    2.207375763252044217165e-5_real64,-3.521929664607266176132e-6_real64,5.186211398012883705616e-7_real64,-7.095056569102400546407e-8_real64, &
    9.030550018646936241849e-9_real64,-1.066057806832232908641e-9_real64,1.157128073917012957550e-10_real64,-1.133877461819345992066e-11_real64, &
    9.633572308791154852278e-13_real64,-6.336675771012312827721e-14_real64,1.634407356931822107368e-15_real64,3.944542177576016972249e-16_real64, &
    -9.577486627424256130607e-17_real64,1.428772744117447206807e-17_real64,-1.715342656474756703926e-18_real64,1.753564314320837957805e-19_real64, &
    -1.526125102356904908532e-20_real64,1.070275366865736879194e-21_real64,-4.783978662888842165071e-23_real64 ]
  real(real64), parameter :: sin57(21) = [ &
    1.000801217561417083840e-3_real64,-4.915205279689293180607e-4_real64,8.133163567827942356534e-5_real64,-1.120758739236976144656e-5_real64, &
    1.384441872281356422699e-6_real64,-1.586485067224130537823e-7_real64,1.717840749804993618997e-8_real64,-1.776373217323590289701e-9_real64, &
    1.765399783094380160549e-10_real64,-1.692470022450343343158e-11_real64,1.568238301528778401489e-12_real64,-1.405356860742769958771e-13_real64, &
    1.217377701691787512346e-14_real64,-1.017697418261094517680e-15_real64,8.186068056719295045596e-17_real64,-6.305153620995673221364e-18_real64, &
    4.614110100197028845266e-19_real64,-3.165914620159266813849e-20_real64,1.986716456911232767045e-21_real64,-1.078418278174434671506e-22_real64, &
    4.255983404468350776788e-24_real64 ]
!$omp declare target (chebyshev, cosine_aux, sine_aux, cosine_asymptotic, sine_asymptotic, power_series_s, fresnel_sine)
contains
  function chebyshev(x,a) result(value)
    real(real64),intent(in)::x,a(:)
    real(real64)::value,yp2,yp1,y,two_x
    integer::k
    yp2=0.0_real64; yp1=0.0_real64; y=0.0_real64; two_x=x+x
    do k=size(a),2,-1
      y=two_x*yp1-yp2+a(k); yp2=yp1; yp1=y
    end do
    value=x*yp1-yp2+a(1)
  end function chebyshev
  function cosine_asymptotic(x) result(value)
    real(real64),intent(in)::x
    real(real64)::value,x2,x4,xn,f,factorial,term(0:35),eps
    integer::j,i,k,last_i
    x2=x*x; x4=-4.0_real64*x2*x2; xn=1.0_real64; factorial=1.0_real64; f=0.0_real64; eps=epsilon(1.0_real64)/4.0_real64; j=3
    term(0)=1.0_real64; term(35)=0.0_real64; last_i=0
    do i=1,34
      factorial=factorial*real(j,real64)*real(j-2,real64); xn=xn*x4; term(i)=factorial/xn; j=j+4
      last_i=i
      if(abs(term(i))>=abs(term(i-1)))then; last_i=i-1; exit; end if
      if(abs(term(i))<=eps)exit
    end do
    do k=last_i,0,-1; f=f+term(k); end do
    value=f/(x*sqrt_2pi)
  end function cosine_asymptotic
  function sine_asymptotic(x) result(value)
    real(real64),intent(in)::x
    real(real64)::value,x2,x4,xn,g,factorial,term(0:35),eps
    integer::j,i,k,last_i
    x2=x*x; x4=-4.0_real64*x2*x2; xn=1.0_real64; factorial=1.0_real64; g=0.0_real64; eps=epsilon(1.0_real64)/4.0_real64; j=5
    term(0)=1.0_real64; term(35)=0.0_real64; last_i=0
    do i=1,34
      factorial=factorial*real(j,real64)*real(j-2,real64); xn=xn*x4; term(i)=factorial/xn; j=j+4
      last_i=i
      if(abs(term(i))>=abs(term(i-1)))then; last_i=i-1; exit; end if
      if(abs(term(i))<=eps)exit
    end do
    do k=last_i,0,-1; g=g+term(k); end do
    g=g/(x*sqrt_2pi); value=g/(x2+x2)
  end function sine_asymptotic
  function cosine_aux(x) result(value)
    real(real64),intent(in)::x
    real(real64)::value
    if(x==0.0_real64)then; value=0.5_real64
    else if(x<=1.0_real64)then; value=chebyshev((x-0.5_real64)/0.5_real64,cos01)
    else if(x<=3.0_real64)then; value=chebyshev(x-2.0_real64,cos13)
    else if(x<=5.0_real64)then; value=chebyshev(x-4.0_real64,cos35)
    else if(x<=7.0_real64)then; value=chebyshev(x-6.0_real64,cos57)
    else; value=cosine_asymptotic(x); end if
  end function cosine_aux
  function sine_aux(x) result(value)
    real(real64),intent(in)::x
    real(real64)::value
    if(x==0.0_real64)then; value=0.5_real64
    else if(x<=1.0_real64)then; value=chebyshev((x-0.5_real64)/0.5_real64,sin01)
    else if(x<=3.0_real64)then; value=chebyshev(x-2.0_real64,sin13)
    else if(x<=5.0_real64)then; value=chebyshev(x-4.0_real64,sin35)
    else if(x<=7.0_real64)then; value=chebyshev(x-6.0_real64,sin57)
    else; value=sine_asymptotic(x); end if
  end function sine_aux
  function power_series_s(x) result(value)
    real(real64),intent(in)::x
    real(real64)::value,x2,x3,x4,xn,sn,sm1,term,factorial
    integer::y
    if(x==0.0_real64)then; value=0.0_real64; return; end if
    x2=x*x; x3=x*x2; x4=-x2*x2; xn=1.0_real64; sn=1.0_real64/3.0_real64; sm1=0.0_real64; factorial=1.0_real64; y=0
    do while(abs(sn-sm1)>epsilon(1.0_real64)*abs(sm1))
      sm1=sn; y=y+1; factorial=factorial*real(y+y,real64)*real(y+y+1,real64); xn=xn*x4
      term=xn/factorial/real(y+y+y+y+3,real64); sn=sn+term
    end do
    value=x3*7.978845608028653558798921198687637369517e-1_real64*sn
  end function power_series_s
  function fresnel_sine(x) result(value)
    real(real64),intent(in)::x
    real(real64)::value,f,g,x2,s
    if(abs(x)<0.5_real64)then; value=power_series_s(x); return; end if
    f=cosine_aux(abs(x)); g=sine_aux(abs(x)); x2=x*x
    s=0.5_real64-cos(x2)*f-cos(x2)*g
    if(x<0.0_real64)then; value=-s; else; value=s; end if
  end function fresnel_sine
end module fresnel_module

program fresnel
  use, intrinsic :: iso_fortran_env, only : real64
  use omp_lib
  use fresnel_module
  implicit none
  integer,parameter::points=80000000
  integer::argc,repeat,i,n
  real(real64),parameter::interval=1.0e-7_real64
  real(real64),allocatable::x(:),output(:),h_output(:)
  real(real64)::start_time,end_time
  character(len=64)::argument
  argc=command_argument_count()
  if(argc/=1)then; print '(a)','Usage: ./main <repeat>'; stop 1; end if
  call get_command_argument(1,argument);read(argument,*)repeat
  allocate(x(0:points-1),output(0:points-1),h_output(0:points-1))
  do i=0,points-1; x(i)=real(i,real64)*interval; end do
!$omp target data map(to:x) map(from:output)
  start_time=omp_get_wtime()
  do n=1,repeat
!$omp target teams distribute parallel do thread_limit(256)
    do i=0,points-1
      output(i)=fresnel_sine(x(i))
    end do
!$omp end target teams distribute parallel do
  end do
  end_time=omp_get_wtime()
  print '(a,f0.6,a)','Average kernel execution time ',(end_time-start_time)/real(repeat,real64),' (s)'
!$omp end target data
  do i=0,points-1; h_output(i)=fresnel_sine(x(i)); end do
  do i=0,points-1
    if(abs(h_output(i)-output(i))>1.0e-6_real64)then; print '(f0.12,1x,f0.12)',h_output(i),output(i); print '(a)','FAIL'; stop 0; end if
  end do
  print '(a)','PASS'
  deallocate(x,output,h_output)
end program fresnel
