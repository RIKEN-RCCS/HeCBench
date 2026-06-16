// Fresnel Sine Integral - Kokkos port
// Consolidates fresnel.cpp, sine.cpp, cosine.cpp, xchebyshev.cpp into one file.

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <chrono>

// ============================================================
//  xchebyshev.cpp  (Clenshaw recursion)
// ============================================================
KOKKOS_INLINE_FUNCTION
double xChebyshev_Tn_Series(double x, const double* a, int degree) {
  double yp2 = 0.0, yp1 = 0.0, y = 0.0;
  double two_x = x + x;
  if (degree < 0) return 0.0;
  for (int k = degree; k >= 1; k--, yp2 = yp1, yp1 = y)
    y = two_x * yp1 - yp2 + a[k];
  return x * yp1 - yp2 + a[0];
}

// ============================================================
//  cosine.cpp  (Fresnel Auxiliary Cosine Integral)
// ============================================================
KOKKOS_INLINE_FUNCTION
double xFresnel_Auxiliary_Cosine_Integral(double x);

KOKKOS_INLINE_FUNCTION
static double AuxCos_Chebyshev_0_1(double x) {
  static const double c[] = {
    +4.200987560240514577713e-1, -9.358785913634965235904e-2,
    -7.642539415723373644927e-3, +4.958117751796130135544e-3,
    -9.750236036106120253456e-4, +1.075201474958704192865e-4,
    -4.415344769301324238886e-6, -7.861633919783064216022e-7,
    +1.919240966215861471754e-7, -2.175775608982741065385e-8,
    +1.296559541430849437217e-9, +2.207205095025162212169e-11,
    -1.479219615873704298874e-11,+1.821350127295808288614e-12,
    -1.228919312990171362342e-13,+2.227139250593818235212e-15,
    +5.734729405928016301596e-16,-8.284965573075354177016e-17,
    +6.067422701530157308321e-18,-1.994908519477689596319e-19,
    -1.173365630675305693390e-20
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series((x - 0.5) / 0.5, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxCos_Chebyshev_1_3(double x) {
  static const double c[] = {
    +2.098677278318224971989e-1,-9.314234883154103266195e-2,
    +1.739905936938124979297e-2,-2.454274824644285136137e-3,
    +1.589872606981337312438e-4,+4.203943842506079780413e-5,
    -2.018022256093216535093e-5,+5.125709636776428285284e-6,
    -9.601813551752718650057e-7,+1.373989484857155846826e-7,
    -1.348105546577211255591e-8,+2.745868700337953872632e-10,
    +2.401655517097260106976e-10,-6.678059547527685587692e-11,
    +1.140562171732840809159e-11,-1.401526517205212219089e-12,
    +1.105498827380224475667e-13,+2.040731455126809208066e-16,
    -1.946040679213045143184e-15,+4.151821375667161733612e-16,
    -5.642257647205149369594e-17,+5.266176626521504829010e-18,
    -2.299025577897146333791e-19,-2.952226367506641078731e-20,
    +8.760405943193778149078e-21
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series(x - 2.0, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxCos_Chebyshev_3_5(double x) {
  static const double c[] = {
    +1.025703371090289562388e-1,-2.569833023232301400495e-2,
    +3.160592981728234288078e-3,-3.776110718882714758799e-4,
    +4.325593433537248833341e-5,-4.668447489229591855730e-6,
    +4.619254757356785108280e-7,-3.970436510433553795244e-8,
    +2.535664754977344448598e-9,-2.108170964644819803367e-11,
    -2.959172018518707683013e-11,+6.727219944906606516055e-12,
    -1.062829587519902899001e-12,+1.402071724705287701110e-13,
    -1.619154679722651005075e-14,+1.651319588396970446858e-15,
    -1.461704569438083772889e-16,+1.053521559559583268504e-17,
    -4.760946403462515858756e-19,-1.803784084922403924313e-20,
    +7.873130866418738207547e-21
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series(x - 4.0, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxCos_Chebyshev_5_7(double x) {
  static const double c[] = {
    +6.738667333400589274018e-2,-1.128146832637904868638e-2,
    +9.408843234170404670278e-4,-7.800074103496165011747e-5,
    +6.409101169623350885527e-6,-5.201350558247239981834e-7,
    +4.151668914650221476906e-8,-3.242202015335530552721e-9,
    +2.460339340900396789789e-10,-1.796823324763304661865e-11,
    +1.244108496436438952425e-12,-7.950417122987063540635e-14,
    +4.419142625999150971878e-15,-1.759082736751040110146e-16,
    -1.307443936270786700760e-18,+1.362484141039320395814e-18,
    -2.055236564763877250559e-19,+2.329142055084791308691e-20,
    -2.282438671525884861970e-21
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series(x - 6.0, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxCos_Asymptotic(double x) {
  const double sqrt_2pi = 2.506628274631000502415765284811045253006;
  const int NUM_ASYMPTOTIC_TERMS = 35;
  double x2 = x*x, x4 = -4.0*x2*x2, xn = 1.0, factorial = 1.0, f = 0.0;
  double term[36]; term[35] = 0.0;
  double epsilon = DBL_EPSILON / 4.0;
  int j = 3, i = 0;
  term[0] = 1.0;
  for (i = 1; i < NUM_ASYMPTOTIC_TERMS; i++) {
    factorial *= ((double)j*(double)(j-2)); xn *= x4;
    term[i] = factorial / xn; j += 4;
    if (fabs(term[i]) >= fabs(term[i-1])) { i--; break; }
    if (fabs(term[i]) <= epsilon) break;
  }
  for (; i >= 0; i--) f += term[i];
  return f / (x * sqrt_2pi);
}
KOKKOS_INLINE_FUNCTION
double xFresnel_Auxiliary_Cosine_Integral(double x) {
  if (x == 0.0) return 0.5;
  if (x <= 1.0) return AuxCos_Chebyshev_0_1(x);
  if (x <= 3.0) return AuxCos_Chebyshev_1_3(x);
  if (x <= 5.0) return AuxCos_Chebyshev_3_5(x);
  if (x <= 7.0) return AuxCos_Chebyshev_5_7(x);
  return AuxCos_Asymptotic(x);
}

// ============================================================
//  sine.cpp  (Fresnel Auxiliary Sine Integral)
// ============================================================
KOKKOS_INLINE_FUNCTION
double xFresnel_Auxiliary_Sine_Integral(double x);

KOKKOS_INLINE_FUNCTION
static double AuxSin_Chebyshev_0_1(double x) {
  static const double c[] = {
    +2.560134650043040830997e-1,-1.993005146464943284549e-1,
    +4.025503636721387266117e-2,-4.459600454502960250729e-3,
    +6.447097305145147224459e-5,+7.544218493763717599380e-5,
    -1.580422720690700333493e-5,+1.755845848573471891519e-6,
    -9.289769688468301734718e-8,-5.624033192624251079833e-9,
    +1.854740406702369495830e-9,-2.174644768724492443378e-10,
    +1.392899828133395918767e-11,-6.989216003725983789869e-14,
    -9.959396121060010838331e-14,+1.312085140393647257714e-14,
    -9.240470383522792593305e-16,+2.472168944148817385152e-17,
    +2.834615576069400293894e-18,-4.650983461314449088349e-19,
    +3.544083040732391556797e-20
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series((x - 0.5) / 0.5, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxSin_Chebyshev_1_3(double x) {
  static const double c[] = {
    +3.470341566046115476477e-2,-3.855580521778624043304e-2,
    +1.420604309383996764083e-2,-4.037349972538938202143e-3,
    +9.292478174580997778194e-4,-1.742730601244797978044e-4,
    +2.563352976720387343201e-5,-2.498437524746606551732e-6,
    -1.334367201897140224779e-8,+7.436854728157752667212e-8,
    -2.059620371321272169176e-8,+3.753674773239250330547e-9,
    -5.052913010605479996432e-10,+4.580877371233042345794e-11,
    -7.664740716178066564952e-13,-7.200170736686941995387e-13,
    +1.812701686438975518372e-13,-2.799876487275995466163e-14,
    +3.048940815174731772007e-15,-1.936754063718089166725e-16,
    -7.653673328908379651914e-18,+4.534308864750374603371e-18,
    -8.011054486030591219007e-19,+9.374587915222218230337e-20,
    -7.144943099280650363024e-21,+1.105276695821552769144e-22,
    +6.989334213887669628647e-23
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series(x - 2.0, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxSin_Chebyshev_3_5(double x) {
  static const double c[] = {
    +3.684922395955255848372e-3,-2.624595437764014386717e-3,
    +6.329162500611499391493e-4,-1.258275676151483358569e-4,
    +2.207375763252044217165e-5,-3.521929664607266176132e-6,
    +5.186211398012883705616e-7,-7.095056569102400546407e-8,
    +9.030550018646936241849e-9,-1.066057806832232908641e-9,
    +1.157128073917012957550e-10,-1.133877461819345992066e-11,
    +9.633572308791154852278e-13,-6.336675771012312827721e-14,
    +1.634407356931822107368e-15,+3.944542177576016972249e-16,
    -9.577486627424256130607e-17,+1.428772744117447206807e-17,
    -1.715342656474756703926e-18,+1.753564314320837957805e-19,
    -1.526125102356904908532e-20,+1.070275366865736879194e-21,
    -4.783978662888842165071e-23
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series(x - 4.0, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxSin_Chebyshev_5_7(double x) {
  static const double c[] = {
    +1.000801217561417083840e-3,-4.915205279689293180607e-4,
    +8.133163567827942356534e-5,-1.120758739236976144656e-5,
    +1.384441872281356422699e-6,-1.586485067224130537823e-7,
    +1.717840749804993618997e-8,-1.776373217323590289701e-9,
    +1.765399783094380160549e-10,-1.692470022450343343158e-11,
    +1.568238301528778401489e-12,-1.405356860742769958771e-13,
    +1.217377701691787512346e-14,-1.017697418261094517680e-15,
    +8.186068056719295045596e-17,-6.305153620995673221364e-18,
    +4.614110100197028845266e-19,-3.165914620159266813849e-20,
    +1.986716456911232767045e-21,-1.078418278174434671506e-22,
    +4.255983404468350776788e-24
  };
  const int degree = (int)(sizeof(c)/sizeof(double)) - 1;
  return xChebyshev_Tn_Series(x - 6.0, c, degree);
}
KOKKOS_INLINE_FUNCTION
static double AuxSin_Asymptotic(double x) {
  const double sqrt_2pi = 2.506628274631000502415765284811045253006;
  const int NUM_ASYMPTOTIC_TERMS = 35;
  double x2 = x*x, x4 = -4.0*x2*x2, xn = 1.0, factorial = 1.0, g = 0.0;
  double term[36]; term[35] = 0.0;
  double epsilon = DBL_EPSILON / 4.0;
  int j = 5, i = 0;
  term[0] = 1.0;
  for (i = 1; i < NUM_ASYMPTOTIC_TERMS; i++) {
    factorial *= ((double)j*(double)(j-2)); xn *= x4;
    term[i] = factorial / xn; j += 4;
    if (fabs(term[i]) >= fabs(term[i-1])) { i--; break; }
    if (fabs(term[i]) <= epsilon) break;
  }
  for (; i >= 0; i--) g += term[i];
  g /= (x * sqrt_2pi);
  return g / (x2 + x2);
}
KOKKOS_INLINE_FUNCTION
double xFresnel_Auxiliary_Sine_Integral(double x) {
  if (x == 0.0) return 0.5;
  if (x <= 1.0) return AuxSin_Chebyshev_0_1(x);
  if (x <= 3.0) return AuxSin_Chebyshev_1_3(x);
  if (x <= 5.0) return AuxSin_Chebyshev_3_5(x);
  if (x <= 7.0) return AuxSin_Chebyshev_5_7(x);
  return AuxSin_Asymptotic(x);
}

// ============================================================
//  fresnel.cpp  (Fresnel Sine Integral)
// ============================================================
KOKKOS_INLINE_FUNCTION
static double Power_Series_S(double x) {
  if (x == 0.0) return 0.0;
  const double sqrt_2_o_pi = 7.978845608028653558798921198687637369517e-1;
  double x2 = x*x, x3 = x*x2, x4 = -x2*x2;
  double xn = 1.0, Sn = 1.0/3.0, Sm1 = 0.0;
  double term, factorial = 1.0;
  int y = 0;
  while (fabs(Sn - Sm1) > DBL_EPSILON * fabs(Sm1)) {
    Sm1 = Sn;
    y++;
    factorial *= (double)(y+y) * (double)(y+y+1);
    xn *= x4;
    term = xn / factorial / (double)(y+y+y+y+3);
    Sn += term;
  }
  return x3 * sqrt_2_o_pi * Sn;
}

KOKKOS_INLINE_FUNCTION
double Fresnel_Sine_Integral(double x) {
  double f, g, x2, s;
  if (fabs(x) < 0.5) return Power_Series_S(x);
  f  = xFresnel_Auxiliary_Cosine_Integral(fabs(x));
  g  = xFresnel_Auxiliary_Sine_Integral(fabs(x));
  x2 = x * x;
  s  = 0.5 - cos(x2)*f - cos(x2)*g;   // note: original uses cos for both f and g terms
  return (x < 0.0) ? -s : s;
}

// ============================================================
//  Reference (CPU)
// ============================================================
static void reference(const double* input, double* output, int n) {
  for (int i = 0; i < n; i++)
    output[i] = Fresnel_Sine_Integral(input[i]);
}

// ============================================================
//  main
// ============================================================
int main(int argc, char* argv[]) {
  if (argc != 2) { printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int repeat = atoi(argv[1]);

  const double interval = 1e-7;
  const int    points   = (int)(8.0 / interval);
  const size_t sz       = (size_t)points;

  double* x      = (double*)malloc(sz * sizeof(double));
  double* output = (double*)malloc(sz * sizeof(double));
  double* h_ref  = (double*)malloc(sz * sizeof(double));
  for (int i = 0; i < points; i++) x[i] = (double)i * interval;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_x("x",      sz);
    Kokkos::View<double*> d_out("out",  sz);

    {
      auto h_x = Kokkos::create_mirror_view(d_x);
      for (int i = 0; i < points; i++) h_x(i) = x[i];
      Kokkos::deep_copy(d_x, h_x);
    }

    auto t_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(
        "fresnel",
        Kokkos::RangePolicy<>(0, points),
        KOKKOS_LAMBDA(int i) {
          d_out(i) = Fresnel_Sine_Integral(d_x(i));
        }
      );
      Kokkos::fence();
    }
    auto t_end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average kernel execution time %f (s)\n", (elapsed * 1e-9) / repeat);

    {
      auto h_out = Kokkos::create_mirror_view(d_out);
      Kokkos::deep_copy(h_out, d_out);
      for (int i = 0; i < points; i++) output[i] = h_out(i);
    }
  }
  Kokkos::finalize();

  reference(x, h_ref, points);
  bool ok = true;
  for (int i = 0; i < points; i++) {
    if (fabs(h_ref[i] - output[i]) > 1e-6) {
      printf("%lf %lf\n", h_ref[i], output[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(x); free(output); free(h_ref);
  return 0;
}
