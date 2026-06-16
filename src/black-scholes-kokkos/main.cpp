#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

#define NUM_DIFF_SETTINGS 37
#define NUM_SAMPLES 50000000
#define CALL 0
#define PUT  1

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538f
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948f
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244008443621048490392848359376887f
#endif

// Error function constants
#define EF_tiny 1e-21f
#define EF_one  1.0f
#define EF_erx  8.45062911510467529297e-01f
#define EF_efx  1.28379167095512586316e-01f
#define EF_efx8 1.02703333676410069053e+00f
#define EF_pp0  1.28379167095512558561e-01f
#define EF_pp1 -3.25042107247001499370e-01f
#define EF_pp2 -2.84817495755985104766e-02f
#define EF_pp3 -5.77027029648944159157e-03f
#define EF_pp4 -2.37630166566501626084e-05f
#define EF_qq1  3.97917223959155352819e-01f
#define EF_qq2  6.50222499887672944485e-02f
#define EF_qq3  5.08130628187576562776e-03f
#define EF_qq4  1.32494738004321644526e-04f
#define EF_qq5 -3.96022827877536812320e-06f
#define EF_pa0 -2.36211856075265944077e-03f
#define EF_pa1  4.14856118683748331666e-01f
#define EF_pa2 -3.72207876035701323847e-01f
#define EF_pa3  3.18346619901161753674e-01f
#define EF_pa4 -1.10894694282396677476e-01f
#define EF_pa5  3.54783043256182359371e-02f
#define EF_pa6 -2.16637559486879084300e-03f
#define EF_qa1  1.06420880400844228286e-01f
#define EF_qa2  5.40397917702171048937e-01f
#define EF_qa3  7.18286544141962662868e-02f
#define EF_qa4  1.26171219808761642112e-01f
#define EF_qa5  1.36370839120290507362e-02f
#define EF_qa6  1.19844998467991074170e-02f
#define EF_ra0 -9.86494403484714822705e-03f
#define EF_ra1 -6.93858572707181764372e-01f
#define EF_ra2 -1.05586262253232909814e+01f
#define EF_ra3 -6.23753324503260060396e+01f
#define EF_ra4 -1.62396669462573470355e+02f
#define EF_ra5 -1.84605092906711035994e+02f
#define EF_ra6 -8.12874355063065934246e+01f
#define EF_ra7 -9.81432934416914548592e+00f
#define EF_sa1  1.96512716674392571292e+01f
#define EF_sa2  1.37657754143519042600e+02f
#define EF_sa3  4.34565877475229228821e+02f
#define EF_sa4  6.45387271733267880336e+02f
#define EF_sa5  4.29008140027567833386e+02f
#define EF_sa6  1.08635005541779435134e+02f
#define EF_sa7  6.57024977031928170135e+00f
#define EF_sa8 -6.04244152148580987438e-02f
#define EF_rb0 -9.86494292470009928597e-03f
#define EF_rb1 -7.99283237680523006574e-01f
#define EF_rb2 -1.77579549177547519889e+01f
#define EF_rb3 -1.60636384855821916062e+02f
#define EF_rb4 -6.37566443368389627722e+02f
#define EF_rb5 -1.02509513161107724954e+03f
#define EF_rb6 -4.83519191608651397019e+02f
#define EF_sb1  3.03380607434824582924e+01f
#define EF_sb2  3.25792512996573918826e+02f
#define EF_sb3  1.53672958608443695994e+03f
#define EF_sb4  3.19985821950859553908e+03f
#define EF_sb5  2.55305040643316442583e+03f
#define EF_sb6  4.74528541206955367215e+02f
#define EF_sb7 -2.24409524465858183362e+01f

struct optionInputStruct {
  int type; float strike,spot,q,r,t,vol,value,tol;
};
struct payoffStruct    { int type; float strike; };
struct yieldTermStruct { float timeYearFraction,forward; };
struct blackVolStruct  { float timeYearFraction,volatility; };
struct blackScholesMertStruct { float x0; yieldTermStruct dividendTS,riskFreeTS; blackVolStruct blackVolTS; };
struct normalDistStruct { float average,sigma,denominator,derNormalizationFactor,normalizationFactor; };
struct blackCalcStruct {
  float strike,forward,stdDev,discount,variance,d1,d2,alpha,beta,
        DalphaDd1,DbetaDd2,n_d1,cum_d1,n_d2,cum_d2,x,DxDs,DxDstrike;
};

KOKKOS_INLINE_FUNCTION float errorFunct(float x) {
  float R,S,P,Q,s,y,z,r,ax;
  ax = Kokkos::fabs(x);
  if(ax < 0.84375f) {
    if(ax < 3.7252902984e-09f) {
      if(ax < FLT_MIN*16.0f) return 0.125f*(8.0f*x+EF_efx8*x);
      return x + EF_efx*x;
    }
    z=x*x;
    r=EF_pp0+z*(EF_pp1+z*(EF_pp2+z*(EF_pp3+z*EF_pp4)));
    s=EF_one+z*(EF_qq1+z*(EF_qq2+z*(EF_qq3+z*(EF_qq4+z*EF_qq5))));
    y=r/s; return x+x*y;
  }
  if(ax < 1.25f) {
    s=ax-EF_one;
    P=EF_pa0+s*(EF_pa1+s*(EF_pa2+s*(EF_pa3+s*(EF_pa4+s*(EF_pa5+s*EF_pa6)))));
    Q=EF_one+s*(EF_qa1+s*(EF_qa2+s*(EF_qa3+s*(EF_qa4+s*(EF_qa5+s*EF_qa6)))));
    if(x>=0.0f) return EF_erx+P/Q; else return -EF_erx-P/Q;
  }
  if(ax>=6.0f) { if(x>=0.0f) return EF_one-EF_tiny; else return EF_tiny-EF_one; }
  s=EF_one/(ax*ax);
  if(ax < 2.85714285714285f) {
    R=EF_ra0+s*(EF_ra1+s*(EF_ra2+s*(EF_ra3+s*(EF_ra4+s*(EF_ra5+s*(EF_ra6+s*EF_ra7))))));
    S=EF_one+s*(EF_sa1+s*(EF_sa2+s*(EF_sa3+s*(EF_sa4+s*(EF_sa5+s*(EF_sa6+s*(EF_sa7+s*EF_sa8)))))));
  } else {
    R=EF_rb0+s*(EF_rb1+s*(EF_rb2+s*(EF_rb3+s*(EF_rb4+s*(EF_rb5+s*EF_rb6)))));
    S=EF_one+s*(EF_sb1+s*(EF_sb2+s*(EF_sb3+s*(EF_sb4+s*(EF_sb5+s*(EF_sb6+s*EF_sb7))))));
  }
  r=Kokkos::exp(-ax*ax-0.5625f+R/S);
  if(x>=0.0f) return EF_one-r/ax; else return r/ax-EF_one;
}

KOKKOS_INLINE_FUNCTION float cumNormDistOp(float z) {
  const float c = 0.5f*(1.0f+errorFunct(z*(float)M_SQRT_2));
  return c;
}

KOKKOS_INLINE_FUNCTION void initCumNormDist(normalDistStruct &nd) {
  nd.average=0.0f; nd.sigma=1.0f;
  nd.normalizationFactor=(float)(M_SQRT_2*(float)M_1_SQRTPI)/nd.sigma;
  nd.derNormalizationFactor=nd.sigma*nd.sigma;
  nd.denominator=2.0f*nd.derNormalizationFactor;
}

KOKKOS_INLINE_FUNCTION float gaussianFunctNormDist(normalDistStruct nd, float x) {
  float deltax=x-nd.average;
  float exponent=-(deltax*deltax)/nd.denominator;
  return exponent<=-690.0f ? 0.0f : nd.normalizationFactor*Kokkos::exp(exponent);
}

KOKKOS_INLINE_FUNCTION float cumNormDistDeriv(normalDistStruct nd, float x) {
  float xn=(x-nd.average)/nd.sigma;
  return gaussianFunctNormDist(nd,xn)/nd.sigma;
}

KOKKOS_INLINE_FUNCTION float interestRateCompoundFactor(float t, yieldTermStruct ys) {
  return Kokkos::exp(ys.forward*t);
}
KOKKOS_INLINE_FUNCTION float interestRateDiscountFactor(float t, yieldTermStruct ys) {
  return 1.0f/interestRateCompoundFactor(t,ys);
}
KOKKOS_INLINE_FUNCTION float getBlackVolBlackVar(blackVolStruct vs) {
  return vs.volatility*vs.volatility*vs.timeYearFraction;
}
KOKKOS_INLINE_FUNCTION float getDiscountOnDividendYield(float yf, yieldTermStruct dTS) {
  return interestRateDiscountFactor(yf,dTS);
}
KOKKOS_INLINE_FUNCTION float getDiscountOnRiskFreeRate(float yf, yieldTermStruct rTS) {
  return interestRateDiscountFactor(yf,rTS);
}

KOKKOS_INLINE_FUNCTION void initBlackCalcVars(blackCalcStruct &bc, payoffStruct payoff) {
  bc.d1=Kokkos::log(bc.forward/bc.strike)/bc.stdDev+0.5f*bc.stdDev;
  bc.d2=bc.d1-bc.stdDev;
  normalDistStruct nd; initCumNormDist(nd);
  bc.cum_d1=cumNormDistOp(bc.d1);
  bc.cum_d2=cumNormDistOp(bc.d2);
  bc.n_d1=cumNormDistDeriv(nd,bc.d1);
  bc.n_d2=cumNormDistDeriv(nd,bc.d2);
  bc.x=payoff.strike; bc.DxDstrike=1.0f; bc.DxDs=0.0f;
  if(payoff.type==CALL) {
    bc.alpha=bc.cum_d1; bc.DalphaDd1=bc.n_d1;
    bc.beta=-bc.cum_d2; bc.DbetaDd2=-bc.n_d2;
  } else {
    bc.alpha=-1.0f+bc.cum_d1; bc.DalphaDd1=bc.n_d1;
    bc.beta=1.0f-bc.cum_d2;   bc.DbetaDd2=-bc.n_d2;
  }
}

KOKKOS_INLINE_FUNCTION void initBlackCalculator(blackCalcStruct &bc, payoffStruct payoff,
    float forwardPrice, float stdDev, float riskFreeDiscount) {
  bc.strike=payoff.strike; bc.forward=forwardPrice;
  bc.stdDev=stdDev; bc.discount=riskFreeDiscount;
  bc.variance=stdDev*stdDev;
  initBlackCalcVars(bc,payoff);
}

KOKKOS_INLINE_FUNCTION float getResultVal(blackCalcStruct bc) {
  return bc.discount*(bc.forward*bc.alpha+bc.x*bc.beta);
}

int main(int argc, char **argv) {
  if (argc < 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int repeat = atoi(argv[1]);

  int numVals = NUM_SAMPLES;
  optionInputStruct *values = new optionInputStruct[numVals];

  // Initialize option data (37 different settings cycled)
  const optionInputStruct settings[NUM_DIFF_SETTINGS] = {
    {CALL,40,42,0.08f,0.04f,0.75f,0.35f,5.0975f,1e-4f},
    {CALL,100,90,0.1f,0.1f,0.1f,0.15f,0.0205f,1e-4f},
    {CALL,100,100,0.1f,0.1f,0.1f,0.15f,1.8734f,1e-4f},
    {CALL,100,110,0.1f,0.1f,0.1f,0.15f,9.9413f,1e-4f},
    {CALL,100,90,0.1f,0.1f,0.1f,0.25f,0.3150f,1e-4f},
    {CALL,100,100,0.1f,0.1f,0.1f,0.25f,3.1217f,1e-4f},
    {CALL,100,110,0.1f,0.1f,0.1f,0.25f,10.3556f,1e-4f},
    {CALL,100,90,0.1f,0.1f,0.1f,0.35f,0.9474f,1e-4f},
    {CALL,100,100,0.1f,0.1f,0.1f,0.35f,4.3693f,1e-4f},
    {CALL,100,110,0.1f,0.1f,0.1f,0.35f,11.1381f,1e-4f},
    {CALL,100,90,0.1f,0.1f,0.5f,0.15f,0.8069f,1e-4f},
    {CALL,100,100,0.1f,0.1f,0.5f,0.15f,4.0501f,1e-4f},
    {CALL,100,110,0.1f,0.1f,0.5f,0.15f,10.9584f,1e-4f},
    {CALL,100,90,0.1f,0.1f,0.5f,0.25f,2.7026f,1e-4f},
    {CALL,100,100,0.1f,0.1f,0.5f,0.25f,6.6997f,1e-4f},
    {CALL,100,110,0.1f,0.1f,0.5f,0.25f,12.7857f,1e-4f},
    {CALL,100,90,0.1f,0.1f,0.5f,0.35f,4.4145f,1e-4f},
    {CALL,100,100,0.1f,0.1f,0.5f,0.35f,8.8228f,1e-4f},
    {CALL,100,110,0.1f,0.1f,0.5f,0.35f,14.6433f,1e-4f},
    {PUT, 100,90,0.1f,0.1f,0.1f,0.15f,9.1280f,1e-4f},
    {PUT, 100,100,0.1f,0.1f,0.1f,0.15f,1.8734f,1e-4f},
    {PUT, 100,110,0.1f,0.1f,0.1f,0.15f,0.0408f,1e-4f},
    {PUT, 100,90,0.1f,0.1f,0.1f,0.25f,8.3521f,1e-4f},
    {PUT, 100,100,0.1f,0.1f,0.1f,0.25f,3.1217f,1e-4f},
    {PUT, 100,110,0.1f,0.1f,0.1f,0.25f,0.4556f,1e-4f},
    {PUT, 100,90,0.1f,0.1f,0.1f,0.35f,7.9562f,1e-4f},
    {PUT, 100,100,0.1f,0.1f,0.1f,0.35f,4.3693f,1e-4f},
    {PUT, 100,110,0.1f,0.1f,0.1f,0.35f,1.2376f,1e-4f},
    {PUT, 100,90,0.1f,0.1f,0.5f,0.15f,5.4252f,1e-4f},
    {PUT, 100,100,0.1f,0.1f,0.5f,0.15f,4.0501f,1e-4f},
    {PUT, 100,110,0.1f,0.1f,0.5f,0.15f,3.5130f,1e-4f},
    {PUT, 100,90,0.1f,0.1f,0.5f,0.25f,7.5580f,1e-4f},
    {PUT, 100,100,0.1f,0.1f,0.5f,0.25f,6.6997f,1e-4f},
    {PUT, 100,110,0.1f,0.1f,0.5f,0.25f,5.8593f,1e-4f},
    {PUT, 100,90,0.1f,0.1f,0.5f,0.35f,14.4452f,1e-4f},
    {PUT, 100,100,0.1f,0.1f,0.5f,0.35f,9.3679f,1e-4f},
    {PUT, 100,110,0.1f,0.1f,0.5f,0.35f,5.7963f,1e-4f},
  };
  for(int i=0;i<numVals;i++) values[i]=settings[i%NUM_DIFF_SETTINGS];

  float *outputVals=(float*)malloc(numVals*sizeof(float));
  printf("Number of options: %d\n\n", numVals);

  Kokkos::initialize(argc, argv);
  {
    using ViewOpts = Kokkos::View<optionInputStruct*>;
    using ViewF    = Kokkos::View<float*>;
    ViewOpts d_values("values", numVals);
    ViewF    d_output("output", numVals);
    {
      auto hv=Kokkos::create_mirror_view(d_values);
      for(int i=0;i<numVals;i++) hv(i)=values[i];
      Kokkos::deep_copy(d_values,hv);
    }

    auto t0=std::chrono::steady_clock::now();
    for(int iter=0;iter<repeat;iter++) {
      Kokkos::parallel_for("blackscholes", numVals, KOKKOS_LAMBDA(int optionNum) {
        const optionInputStruct opt = d_values(optionNum);
        payoffStruct currPayoff; currPayoff.type=opt.type; currPayoff.strike=opt.strike;
        yieldTermStruct qTS; qTS.timeYearFraction=opt.t; qTS.forward=opt.q;
        yieldTermStruct rTS; rTS.timeYearFraction=opt.t; rTS.forward=opt.r;
        blackVolStruct volTS; volTS.timeYearFraction=opt.t; volTS.volatility=opt.vol;
        blackScholesMertStruct stoch; stoch.x0=opt.spot;
        stoch.dividendTS=qTS; stoch.riskFreeTS=rTS; stoch.blackVolTS=volTS;

        float variance=getBlackVolBlackVar(stoch.blackVolTS);
        float dividendDiscount=getDiscountOnDividendYield(opt.t,stoch.dividendTS);
        float riskFreeDiscount=getDiscountOnRiskFreeRate(opt.t,stoch.riskFreeTS);
        float spot=stoch.x0;
        float forwardPrice=spot*dividendDiscount/riskFreeDiscount;

        blackCalcStruct bc;
        initBlackCalculator(bc,currPayoff,forwardPrice,Kokkos::sqrt(variance),riskFreeDiscount);
        d_output(optionNum)=getResultVal(bc);
      });
    }
    Kokkos::fence();
    auto t1=std::chrono::steady_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/repeat;
    printf("Average kernel execution time on GPU: %f (ms)\n", ms);

    auto ho=Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(ho,d_output);
    for(int i=0;i<numVals;i++) outputVals[i]=ho(i);
  }
  Kokkos::finalize();

  float totResult=0.0f;
  for(int i=0;i<numVals;i++) totResult+=outputVals[i];
  printf("Summation of output prices on GPU: %f\n", totResult);
  printf("Output price at index %d on GPU: %f\n", numVals/2, outputVals[numVals/2]);

  delete[] values;
  free(outputVals);
  return 0;
}
