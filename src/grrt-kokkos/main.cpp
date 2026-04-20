/***********************************************************************************
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  Kokkos port of the grrt-omp benchmark (General Relativistic Ray Tracing).
  Original code from https://github.com/hungyipu/Odyssey/
 ***********************************************************************************/

#include <Kokkos_Core.hpp>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

// ─── Constants ───────────────────────────────────────────────────────────────
#define GRRT_N        6
#define PI            3.14159265
#define C_G           6.67259e-08
#define C_c           2.99792458e+10
#define C_mSun        1.99e+33
#define C_rgeo        1.4774e+05
#define C_h           6.6260755e-27
#define C_kB          1.380658e-16
#define C_e           4.8032068e-10
#define C_me          9.1093897e-28
#define C_mp          1.6726231e-24
#define C_Jansky      1e-23
#define C_pc          3.086e18
#define C_sgrA_mbh    4.3e6
#define C_sgrA_d      8500

#define IMAGE_SIZE    1024
#define VarNUM        9
#define VarINNUM      4

// Indices into the thread-local Variables[] array
#define r0        Variables[0]
#define theta0    Variables[1]
#define a2        Variables[2]
#define Rhor      Variables[3]
#define Rmstable  Variables[4]
#define L         Variables[5]
#define kappa     Variables[6]
#define grid_x    Variables[7]
#define grid_y    Variables[8]

// Indices into the shared VariablesIn[] array
#define A_IN(vi)         vi[0]
#define INCLINATION_IN(vi) vi[1]
#define SIZE_IN(vi)      vi[2]
#define freq_obs_IN(vi)  vi[3]

// ─── K2 look-up table (constexpr for device access) ──────────────────────────
#define Te_min   0.1
#define Te_max   100.
#define Te_grids 50.

KOKKOS_INLINE_FUNCTION static double K2_tab_val(int idx) {
  constexpr double tab[] = {
    -10.747001, -9.362569, -8.141373, -7.061568, -6.104060,
    -5.252153,  -4.491244, -3.808555, -3.192909, -2.634534,
    -2.124893,  -1.656543, -1.223007, -0.818668, -0.438676,
    -0.078863,   0.264332,  0.593930,  0.912476,  1.222098,
     1.524560,   1.821311,  2.113537,  2.402193,  2.688050,
     2.971721,   3.253692,  3.534347,  3.813984,  4.092839,
     4.371092,   4.648884,  4.926323,  5.203493,  5.480457,
     5.757264,   6.033952,  6.310550,  6.587078,  6.863554,
     7.139990,   7.416395,  7.692778,  7.969143,  8.245495,
     8.521837,   8.798171,  9.074500,  9.350824,  9.627144
  };
  return tab[idx];
}

// ─── Device functions ─────────────────────────────────────────────────────────

KOKKOS_INLINE_FUNCTION static void geodesic(
    double* Variables, const double* VariablesIn,
    double* y, double* dydx)
{
  double r     = y[0];
  double theta = y[1];
  double pr    = y[4];
  double ptheta= y[5];

  double r2      = r * r;
  double twor    = 2.0 * r;
  double sintheta= Kokkos::sin(theta);
  double costheta= Kokkos::cos(theta);
  double cos2    = costheta * costheta;
  double sin2    = sintheta * sintheta;
  double sigma   = r2 + a2 * cos2;
  double delta   = r2 - twor + a2;
  double sd      = sigma * delta;
  double siginv  = 1.0 / sigma;
  double bot     = 1.0 / sd;

  if (sintheta < 1e-8) { sintheta = 1e-8; sin2 = 1e-16; }

  double Av = A_IN(VariablesIn);
  dydx[0] = -pr * delta * siginv;
  dydx[1] = -ptheta * siginv;
  dydx[2] = -(twor * Av + (sigma - twor) * L / sin2) * bot;
  dydx[3] = -(1.0 + (twor * (r2 + a2) - twor * Av * L) * bot);
  dydx[4] = -(((r - 1.0) * (-kappa) + twor * (r2 + a2) - 2.0 * Av * L) * bot
              - 2.0 * pr * pr * (r - 1.0) * siginv);
  dydx[5] = -sintheta * costheta * (L * L / (sin2 * sin2) - a2) * siginv;
}

KOKKOS_INLINE_FUNCTION static void rkstep(
    double* Variables, const double* VariablesIn,
    double* y, double* dydx, double h, double* yout, double* yerr)
{
  double ak[GRRT_N];
  double ytemp1[GRRT_N], ytemp2[GRRT_N], ytemp3[GRRT_N],
         ytemp4[GRRT_N], ytemp5[GRRT_N];

  for (int i = 0; i < GRRT_N; i++) {
    double hdydx = h * dydx[i];
    double yi    = y[i];
    ytemp1[i] = yi + 0.2 * hdydx;
    ytemp2[i] = yi + (3.0/40.0) * hdydx;
    ytemp3[i] = yi + 0.3 * hdydx;
    ytemp4[i] = yi - (11.0/54.0) * hdydx;
    ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
    yout[i]   = yi + (37.0/378.0) * hdydx;
    yerr[i]   = ((37.0/378.0) - (2825.0/27648.0)) * hdydx;
  }

  geodesic(Variables, VariablesIn, ytemp1, ak);
  for (int i = 0; i < GRRT_N; i++) {
    double yt = h * ak[i];
    ytemp2[i] += (9.0/40.0) * yt;
    ytemp3[i] -= 0.9 * yt;
    ytemp4[i] += 2.5 * yt;
    ytemp5[i] += (175.0/512.0) * yt;
  }

  geodesic(Variables, VariablesIn, ytemp2, ak);
  for (int i = 0; i < GRRT_N; i++) {
    double yt = h * ak[i];
    ytemp3[i] += 1.2 * yt;
    ytemp4[i] -= (70.0/27.0) * yt;
    ytemp5[i] += (575.0/13824.0) * yt;
    yout[i]   += (250.0/621.0) * yt;
    yerr[i]   += ((250.0/621.0) - (18575.0/48384.0)) * yt;
  }

  geodesic(Variables, VariablesIn, ytemp3, ak);
  for (int i = 0; i < GRRT_N; i++) {
    double yt = h * ak[i];
    ytemp4[i] += (35.0/27.0) * yt;
    ytemp5[i] += (44275.0/110592.0) * yt;
    yout[i]   += (125.0/594.0) * yt;
    yerr[i]   += ((125.0/594.0) - (13525.0/55296.0)) * yt;
  }

  geodesic(Variables, VariablesIn, ytemp4, ak);
  for (int i = 0; i < GRRT_N; i++) {
    double yt = h * ak[i];
    ytemp5[i] += (253.0/4096.0) * yt;
    yerr[i]   -= (277.0/14336.0) * yt;
  }

  geodesic(Variables, VariablesIn, ytemp5, ak);
  for (int i = 0; i < GRRT_N; i++) {
    double yt = h * ak[i];
    yout[i]  += (512.0/1771.0) * yt;
    yerr[i]  += ((512.0/1771.0) - 0.25) * yt;
  }
}

KOKKOS_INLINE_FUNCTION static double rk5(
    double* Variables, const double* VariablesIn,
    double* y, double* dydx,
    double htry, double escal, double* yscal, double* hdid)
{
  double hnext, errmax, h = htry, htemp;
  double yerr[GRRT_N], ytemp[GRRT_N];

  while (true) {
    rkstep(Variables, VariablesIn, y, dydx, h, ytemp, yerr);

    errmax = 0.0;
    for (int i = 0; i < GRRT_N; i++) {
      double temp = Kokkos::abs(yerr[i] / yscal[i]);
      if (temp > errmax) errmax = temp;
    }

    errmax *= escal;
    if (errmax <= 1.0) break;

    htemp = 0.9 * h / Kokkos::sqrt(Kokkos::sqrt(errmax));
    h *= 0.1;
    if (h >= 0.0) { if (htemp > h) h = htemp; }
    else          { if (htemp < h) h = htemp; }
  }

  if (errmax > 1.89e-4)
    hnext = 0.9 * h * Kokkos::pow(errmax, -0.2);
  else
    hnext = 5.0 * h;

  *hdid = h;
  for (int i = 0; i < GRRT_N; i++) y[i] = ytemp[i];
  return hnext;
}

KOKKOS_INLINE_FUNCTION static void initial(
    double* Variables, const double* VariablesIn,
    double* y0, double* ydot0)
{
  double alpha = grid_x;
  double beta  = grid_y;
  double Av    = A_IN(VariablesIn);

  double x = Kokkos::sqrt(r0*r0 + a2)*Kokkos::sin(theta0) - beta*Kokkos::cos(theta0);
  double y = alpha;
  double z = r0*Kokkos::cos(theta0) + beta*Kokkos::sin(theta0);
  double w = x*x + y*y + z*z - a2;

  y0[0] = Kokkos::sqrt((w + Kokkos::sqrt(w*w + (2.*Av*z)*(2.*Av*z)))/2.);
  y0[1] = Kokkos::acos(z / y0[0]);
  y0[2] = Kokkos::atan2(y, x);
  y0[3] = 0.0;

  double r     = y0[0];
  double theta = y0[1];
  double phi   = y0[2];
  double sigma = r*r + (Av*Kokkos::cos(theta))*(Av*Kokkos::cos(theta));
  double u     = Kokkos::sqrt(a2 + r*r);
  double v     = -Kokkos::sin(theta0)*Kokkos::cos(phi);
  double zdot  = -1.;

  double rdot0     = zdot*(-u*u*Kokkos::cos(theta0)*Kokkos::cos(theta) + r*u*v*Kokkos::sin(theta))/sigma;
  double thetadot0 = zdot*(Kokkos::cos(theta0)*r*Kokkos::sin(theta) + u*v*Kokkos::cos(theta))/sigma;
  double phidot0   = zdot*Kokkos::sin(theta0)*Kokkos::sin(phi)/(u*Kokkos::sin(theta));

  ydot0[0] = rdot0;
  ydot0[1] = thetadot0;
  ydot0[2] = phidot0;

  double sintheta = Kokkos::sin(theta);
  double sin2     = sintheta*sintheta;
  double r2       = r * r;
  double delta    = r2 - 2.0*r + a2;
  double s1       = sigma - 2.0*r;

  y0[4] = rdot0*sigma/delta;
  y0[5] = thetadot0*sigma;

  double energy2 = s1*(rdot0*rdot0/delta + thetadot0*thetadot0)
                 + delta*sin2*phidot0*phidot0;
  double energy  = Kokkos::sqrt(energy2);

  y0[4] = y0[4]/energy;
  y0[5] = y0[5]/energy;

  L     = ((sigma*delta*phidot0 - 2.0*Av*r*energy)*sin2/s1)/energy;
  kappa = y0[5]*y0[5] + a2*sin2 + L*L/sin2;
}

KOKKOS_INLINE_FUNCTION static float ISCO(const double* VariablesIn)
{
  double Av = A_IN(VariablesIn);
  double z1 = 1 + Kokkos::pow(1 - Av*Av, 1/3.0)
                  *(Kokkos::pow(1 + Av, 1/3.0) + Kokkos::pow(1 - Av, 1/3.0));
  double z2 = Kokkos::sqrt(3*Av*Av + z1*z1);
  return 3. + z2 - Kokkos::sqrt((3 - z1)*(3 + z1 + 2*z2));
}

KOKKOS_INLINE_FUNCTION static double K2_find(double Te)
{
  double d = Te_grids * (Kokkos::log(Te / Te_min) / Kokkos::log(Te_max / Te_min));
  int    i = (int)Kokkos::floor(d);
  return (1 - (double)(d-i)) * K2_tab_val(i) + (double)(d-i) * K2_tab_val(i+1);
}

KOKKOS_INLINE_FUNCTION static double K2(double Te)
{
  if (Te > 85.)    return 2.*Te*Te;
  if (Te < Te_min) return Kokkos::exp(-11.);
  return Kokkos::exp(K2_find(Te));
}

KOKKOS_INLINE_FUNCTION static double Jansky_Correction(
    const double* VariablesIn, double ima_width)
{
  double distance = C_sgrA_d * C_pc;
  double theta    = Kokkos::atan(ima_width * C_sgrA_mbh * C_rgeo / distance);
  double sz       = SIZE_IN(VariablesIn);
  double pix_str  = (theta/(sz/2.)) * (theta/(sz/2.));
  return pix_str / C_Jansky;
}

KOKKOS_INLINE_FUNCTION static double Luminosity_Correction(
    const double* VariablesIn, double ima_width)
{
  double distance = C_sgrA_d * C_pc;
  double theta    = Kokkos::atan(ima_width * C_sgrA_mbh * C_rgeo / distance);
  double sz       = SIZE_IN(VariablesIn);
  double pix_str  = (theta/(sz/2.)) * (theta/(sz/2.));
  return pix_str * distance * distance * 4. * PI * freq_obs_IN(VariablesIn);
}

KOKKOS_INLINE_FUNCTION static double task1fun_GetZ(
    double* Variables, const double* VariablesIn, double* y)
{
  double r1    = y[0];
  double Av    = A_IN(VariablesIn);
  double E_loc = -(r1*r1 + Av*Kokkos::sqrt(r1))
                 / (r1*Kokkos::sqrt(r1*r1 - 3.*r1 + 2.*Av*Kokkos::sqrt(r1)))
               + L / Kokkos::sqrt(r1)
                 / Kokkos::sqrt(r1*r1 - 3.*r1 + 2.*Av*Kokkos::sqrt(r1));
  return E_loc / (-1.0);
}

KOKKOS_INLINE_FUNCTION static double task2fun_GetZ(
    double* Variables, const double* VariablesIn, double* y)
{
  double r     = y[0];
  double theta = y[1];
  double pr    = y[4];
  double Av    = A_IN(VariablesIn);

  double r2       = r*r;
  double twor     = 2.0*r;
  double sintheta = Kokkos::sin(theta);
  double costheta = Kokkos::cos(theta);
  double cos2     = costheta*costheta;
  double sin2     = sintheta*sintheta;
  double sigma    = r2 + a2*cos2;
  double delta    = r2 - twor + a2;
  double ssig     = (r2+a2)*(r2+a2) - a2*delta*sin2;

  double gtt   = -(1. - 2.*r/sigma);
  double gtph  = -2.*Av*r*sin2/sigma;
  double grr   = sigma/delta;
  double gphph = ssig*sin2/sigma;

  double ut_k   = (r*r + Av*Kokkos::sqrt(r))
                  / (r*Kokkos::sqrt(r*r - 3.*r + 2.*Av*Kokkos::sqrt(r)));
  double ur_k   = 0.;
  double uphi_k = 1./(Kokkos::sqrt(r)*Kokkos::sqrt(r*r - 3.*r + 2.*Av*Kokkos::sqrt(r)));

  if (r < Rmstable) {
    double delta2  = r*r - 2.*r + a2;
    double lambda  = (Rmstable*Rmstable - 2.*Av*Kokkos::sqrt(Rmstable) + a2)
                     / (Kokkos::sqrt(Rmstable*Rmstable*Rmstable)
                        - 2.*Kokkos::sqrt(Rmstable) + Av);
    double gamma2  = Kokkos::sqrt(1 - 2./3./Rmstable);
    double h       = (2.*r - Av*lambda)/delta2;
    ut_k   = gamma2*(1. + 2./r*(1.+h));
    ur_k   = -Kokkos::sqrt(2./3./Rmstable)
              * Kokkos::sqrt(Kokkos::pow((Rmstable/r - 1.), 3.));
    uphi_k = gamma2/r/r*(lambda + Av*h);
  }

  double ut   = ut_k;
  double uphi = uphi_k;
  double ur   = ur_k;
  double omega = uphi/ut;
  double k0    = -(gtt + omega*omega*gphph + 2.*omega*gtph);
  ut           = Kokkos::sqrt((1. + grr*ur*ur) / k0);
  uphi         = omega*ut;

  double E_local = -ut + L*uphi + pr*ur;
  return E_local / (-1.0);
}

// ─── task1: redshift-only ray tracing ─────────────────────────────────────────
void task1(Kokkos::View<double*> Results,
           Kokkos::View<double*> varIn,
           int GridIdxX, int GridIdxY)
{
  Kokkos::parallel_for(
    "task1",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{50,100}),
    KOKKOS_LAMBDA(int y1, int x1)
    {
      int X1 = GridIdxX * 100 + x1;
      int Y1 = GridIdxY * 50  + y1;

      if (X1 >= IMAGE_SIZE || Y1 >= IMAGE_SIZE) return;

      // local copies of VariablesIn
      double VariablesIn_arr[VarINNUM];
      for (int i = 0; i < VarINNUM; i++) VariablesIn_arr[i] = varIn(i);
      const double* VariablesIn = VariablesIn_arr;

      double* ResultsPixel = Results.data() + 3*(IMAGE_SIZE*Y1 + X1);

      double Variables[VarNUM];
      r0       = 1000.0;
      theta0   = (PI/180.0) * INCLINATION_IN(VariablesIn);
      a2       = A_IN(VariablesIn) * A_IN(VariablesIn);
      Rhor     = 1.0 + Kokkos::sqrt(1.0 - a2) + 1e-5;
      Rmstable = ISCO(VariablesIn);

      double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
      double y[GRRT_N], dydx[GRRT_N], yscal[GRRT_N], ylaststep[GRRT_N];
      double Rdisk     = 50.;
      double ima_width = 55.;
      double s1  = ima_width;
      double s2  = 2.*ima_width / ((int)SIZE_IN(VariablesIn) + 1.);

      grid_x = -s1 + s2*(X1+1.);
      grid_y = -s1 + s2*(Y1+1.);

      initial(Variables, VariablesIn, y, dydx);

      ResultsPixel[0] = grid_x;
      ResultsPixel[1] = grid_y;
      ResultsPixel[2] = 0;

      while (true) {
        for (int i = 0; i < GRRT_N; i++) ylaststep[i] = y[i];
        geodesic(Variables, VariablesIn, y, dydx);
        for (int i = 0; i < GRRT_N; i++)
          yscal[i] = Kokkos::abs(y[i]) + Kokkos::abs(dydx[i]*htry) + 1.0e-3;
        hnext = rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

        if (y[0] < Rdisk && y[0] > Rmstable
            && (ylaststep[1] - PI/2.) * (y[1] - PI/2.) < 0.) {
          ResultsPixel[2] = 1.0 / task1fun_GetZ(Variables, VariablesIn, y);
          break;
        }
        if ((y[0] > r0) && (dydx[0] > 0)) break;
        if (y[0] < Rhor) break;
        htry = hnext;
      }
    });
  Kokkos::fence();
}

// ─── task2: radiative transfer ray tracing ────────────────────────────────────
void task2(Kokkos::View<double*> Results,
           Kokkos::View<double*> varIn,
           int GridIdxX, int GridIdxY)
{
  Kokkos::parallel_for(
    "task2",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{50,100}),
    KOKKOS_LAMBDA(int y1, int x1)
    {
      int X1 = GridIdxX * 100 + x1;
      int Y1 = GridIdxY * 50  + y1;

      if (X1 >= IMAGE_SIZE || Y1 >= IMAGE_SIZE) return;

      double VariablesIn_arr[VarINNUM];
      for (int i = 0; i < VarINNUM; i++) VariablesIn_arr[i] = varIn(i);
      const double* VariablesIn = VariablesIn_arr;

      double* ResultsPixel = Results.data() + 3*(IMAGE_SIZE*Y1 + X1);

      double Variables[VarNUM];
      r0       = 1000.0;
      theta0   = (PI/180.0) * INCLINATION_IN(VariablesIn);
      a2       = A_IN(VariablesIn) * A_IN(VariablesIn);
      Rhor     = 1.0 + Kokkos::sqrt(1.0 - a2) + 1e-5;
      Rmstable = ISCO(VariablesIn);

      double htry = 0.5, escal = 1e14, hdid = 0.0, hnext = 0.0;
      double y[GRRT_N], dydx[GRRT_N], yscal[GRRT_N];
      double Rdisk     = 500.;
      double ima_width = 10.;
      double s1  = ima_width;
      double s2  = 2.*ima_width / ((int)SIZE_IN(VariablesIn) + 1.);
      double Jy_corr = Jansky_Correction(VariablesIn, ima_width);
      double L_corr  = Luminosity_Correction(VariablesIn, ima_width);

      grid_x = -s1 + s2*(X1+1.);
      grid_y = -s1 + s2*(Y1+1.);

      initial(Variables, VariablesIn, y, dydx);

      ResultsPixel[0] = grid_x;
      ResultsPixel[1] = grid_y;
      ResultsPixel[2] = 0;

      double ds = 0., dtau = 0., dI = 0.;

      while (true) {
        geodesic(Variables, VariablesIn, y, dydx);
        for (int i = 0; i < GRRT_N; i++)
          yscal[i] = Kokkos::abs(y[i]) + Kokkos::abs(dydx[i]*htry) + 1.0e-3;
        hnext = rk5(Variables, VariablesIn, y, dydx, htry, escal, yscal, &hdid);

        double fobs = freq_obs_IN(VariablesIn);
        if ((y[0] > r0) && (dydx[0] > 0)) {
          ResultsPixel[2] = dI*fobs*fobs*fobs*L_corr;
          break;
        }
        if (y[0] < Rhor) {
          ResultsPixel[2] = dI*fobs*fobs*fobs*L_corr;
          break;
        }

        if (y[0] < Rdisk) {
          double zzz        = task2fun_GetZ(Variables, VariablesIn, y);
          double freq_local = fobs*zzz;

          double nth0 = 3e7;
          double zc   = y[0]*Kokkos::cos(y[1]);
          double rc   = y[0]*Kokkos::sin(y[1]);
          double nth  = nth0*Kokkos::exp(-zc*zc/2./rc/rc)*Kokkos::pow(y[0], -1.1);
          double Te   = 1.7e11*Kokkos::pow(y[0], -0.84);
          double b    = Kokkos::sqrt(8.*PI*0.1*nth*C_mp*C_c*C_c/6./y[0]);
          double vb   = C_e*b/2./PI/C_me/C_c;
          double theta_E = C_kB*Te/C_me/C_c/C_c;
          double v    = freq_local;
          double x    = 2.*v/3./vb/theta_E/theta_E;

          double K_value = K2(theta_E);
          double comp1   = 4.*PI*nth*C_e*C_e*v/Kokkos::sqrt(3.)/K_value/C_c;
          double comp2   = 4.0505/Kokkos::pow(x, 1./6.)
                           * (1.+0.4/Kokkos::pow(x,0.25)+0.5316/Kokkos::sqrt(x))
                           * Kokkos::exp(-1.8899*Kokkos::pow(x, 1./3.));
          double j_nu    = comp1*comp2;
          double B_nu    = 2.0*v*v*v*C_h/C_c/C_c
                           / (Kokkos::exp(C_h*v/C_kB/Te) - 1.0);

          ds   = htry;
          dtau = dtau + ds*C_sgrA_mbh*C_rgeo*j_nu/B_nu*zzz;
          dI   = dI   + ds*C_sgrA_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local
                        * Kokkos::exp(-dtau)*zzz;
        }
        htry = hnext;
      }
    });
  Kokkos::fence();
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    // Host-side VariablesIn buffer (4 doubles)
    Kokkos::View<double*> d_varIn("VariablesIn", VarINNUM);
    auto h_varIn = Kokkos::create_mirror_view(d_varIn);

    // Device Results buffer
    Kokkos::View<double*> d_Results("Results", IMAGE_SIZE * IMAGE_SIZE * 3);
    auto h_Results = Kokkos::create_mirror_view(d_Results);

    int BlockDimX = 100, BlockDimY = 1;
    int GridDimX  = 1,   GridDimY  = 50;
    int ImaDimX   = (int)ceil((double)IMAGE_SIZE / (BlockDimX * GridDimX));
    int ImaDimY   = (int)ceil((double)IMAGE_SIZE / (BlockDimY * GridDimY));

    // ── task1 ──
    h_varIn(0) = 0.;                         // A
    h_varIn(1) = acos(0.25)/PI*180.;         // INCLINATION
    h_varIn(2) = (double)IMAGE_SIZE;         // SIZE
    h_varIn(3) = 0.;                         // freq_obs (unused in task1)
    printf("task1: image size = %d x %d pixels\n", IMAGE_SIZE, IMAGE_SIZE);
    Kokkos::deep_copy(d_varIn, h_varIn);

    auto t1_start = std::chrono::steady_clock::now();
    for (int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++)
      for (int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++)
        task1(d_Results, d_varIn, GridIdxX, GridIdxY);
    auto t1_end = std::chrono::steady_clock::now();
    double t1_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     t1_end - t1_start).count() * 1e-9f;
    printf("Total kernel execution time (task1) %f (s)\n", t1_ms);

    Kokkos::deep_copy(h_Results, d_Results);
    FILE* fp = fopen("Output_task1.txt","w");
    if (fp) {
      fprintf(fp,"###output data:(alpha,  beta,  redshift)\n");
      for (int j = 0; j < IMAGE_SIZE; j++)
        for (int i = 0; i < IMAGE_SIZE; i++) {
          fprintf(fp,"%f\t",(float)h_Results(3*(IMAGE_SIZE*j+i)+0));
          fprintf(fp,"%f\t",(float)h_Results(3*(IMAGE_SIZE*j+i)+1));
          fprintf(fp,"%f\n",(float)h_Results(3*(IMAGE_SIZE*j+i)+2));
        }
      fclose(fp);
    }

    // ── task2 ──
    h_varIn(0) = 0.;
    h_varIn(1) = 45.;
    h_varIn(2) = (double)IMAGE_SIZE;
    h_varIn(3) = 340e9;
    printf("task2: image size = %d x %d pixels\n", IMAGE_SIZE, IMAGE_SIZE);
    Kokkos::deep_copy(d_varIn, h_varIn);

    auto t2_start = std::chrono::steady_clock::now();
    for (int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++)
      for (int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++)
        task2(d_Results, d_varIn, GridIdxX, GridIdxY);
    auto t2_end = std::chrono::steady_clock::now();
    double t2_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     t2_end - t2_start).count() * 1e-9f;
    printf("Total kernel execution time (task2) %f (s)\n", t2_ms);

    Kokkos::deep_copy(h_Results, d_Results);
    fp = fopen("Output_task2.txt","w");
    if (fp) {
      fprintf(fp,"###output data:(alpha,  beta, Luminosity (erg/sec))\n");
      for (int j = 0; j < IMAGE_SIZE; j++)
        for (int i = 0; i < IMAGE_SIZE; i++) {
          fprintf(fp,"%f\t",(float)h_Results(3*(IMAGE_SIZE*j+i)+0));
          fprintf(fp,"%f\t",(float)h_Results(3*(IMAGE_SIZE*j+i)+1));
          fprintf(fp,"%f\n",(float)h_Results(3*(IMAGE_SIZE*j+i)+2));
        }
      fclose(fp);
    }
  }
  Kokkos::finalize();
  return 0;
}
