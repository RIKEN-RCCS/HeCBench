// RSBench Kokkos port - event-based simulation
// Original C code ported to C++ with Kokkos for GPU offload.

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <stdint.h>
#include <chrono>
#include <string>

#define PI 3.14159265359

typedef enum { SMALL, LARGE, XL, XXL } HM_size;
#define HISTORY_BASED 1
#define EVENT_BASED   2
#define STARTING_SEED      1070ULL
#define INITIALIZATION_SEED 42ULL

// ─── Data structures ──────────────────────────────────────────────────────
struct RSComplex { double r, i; };

struct Input {
  int nthreads, n_nuclides, lookups;
  HM_size HM;
  int avg_n_poles, avg_n_windows, numL, doppler, particles, simulation_method, kernel_id;
};

struct Pole {
  RSComplex MP_EA, MP_RT, MP_RA, MP_RF;
  short int l_value;
};

struct Window {
  double T, A, F;
  int start, end;
};

// ─── RSComplex math ───────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION RSComplex c_add(RSComplex A, RSComplex B) { return {A.r+B.r, A.i+B.i}; }
KOKKOS_INLINE_FUNCTION RSComplex c_sub(RSComplex A, RSComplex B) { return {A.r-B.r, A.i-B.i}; }
KOKKOS_INLINE_FUNCTION RSComplex c_mul(RSComplex A, RSComplex B) {
  return {A.r*B.r - A.i*B.i, A.r*B.i + A.i*B.r};
}
KOKKOS_INLINE_FUNCTION RSComplex c_div(RSComplex A, RSComplex B) {
  double d = B.r*B.r + B.i*B.i;
  return {(A.r*B.r + A.i*B.i)/d, (A.i*B.r - A.r*B.i)/d};
}
KOKKOS_INLINE_FUNCTION double c_abs(RSComplex A) { return sqrt(A.r*A.r + A.i*A.i); }

// ─── LCG random ──────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION double LCG_random_double(uint64_t *seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}
KOKKOS_INLINE_FUNCTION uint64_t LCG_random_int(uint64_t *seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}
KOKKOS_INLINE_FUNCTION uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
  const uint64_t m = 9223372036854775808ULL;
  uint64_t a = 2806196910506780709ULL, c = 1ULL;
  n %= m;
  uint64_t a_new = 1, c_new = 0;
  while (n > 0) {
    if (n & 1) { a_new *= a; c_new = c_new * a + c; }
    c *= (a + 1); a *= a; n >>= 1;
  }
  return (a_new * seed + c_new) % m;
}

// ─── Fast exp / cexp ─────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION double fast_exp(double x) {
  x = 1.0 + x * 0.000244140625;
  x*=x;x*=x;x*=x;x*=x; x*=x;x*=x;x*=x;x*=x; x*=x;x*=x;x*=x;x*=x;
  return x;
}
KOKKOS_INLINE_FUNCTION RSComplex fast_cexp(RSComplex z) {
  double t1 = fast_exp(z.r);
  RSComplex t4 = {cos(z.i), sin(z.i)};
  RSComplex t5 = {t1, 0};
  return c_mul(t5, t4);
}

// ─── Faddeeva (fast_nuclear_W) ───────────────────────────────────────────
KOKKOS_INLINE_FUNCTION RSComplex fast_nuclear_W(RSComplex Z) {
  if (c_abs(Z) < 6.0) {
    RSComplex prefactor = {0, 8.124330e+01};
    double an[10]  = {2.758402e-01,2.245740e-01,1.594149e-01,9.866577e-02,5.324414e-02,
                      2.505215e-02,1.027747e-02,3.676164e-03,1.146494e-03,3.117570e-04};
    double neg1n[10] = {-1,1,-1,1,-1,1,-1,1,-1,1};
    double denL[10] = {9.869604e+00,3.947842e+01,8.882644e+01,1.579137e+02,2.467401e+02,
                       3.553058e+02,4.836106e+02,6.316547e+02,7.994380e+02,9.869604e+02};
    RSComplex t1={0,12}, t2={12,0}, i_c={0,1}, one={1,0};
    RSComplex W = c_div(c_mul(i_c, c_sub(one, fast_cexp(c_mul(t1,Z)))), c_mul(t2,Z));
    RSComplex sum={0,0};
    for (int n=0; n<10; n++) {
      RSComplex t3={neg1n[n],0};
      RSComplex top = c_sub(c_mul(t3, fast_cexp(c_mul(t1,Z))), one);
      RSComplex t4={denL[n],0}, t5={144,0};
      RSComplex bot = c_sub(t4, c_mul(t5, c_mul(Z,Z)));
      RSComplex t6={an[n],0};
      sum = c_add(sum, c_mul(t6, c_div(top,bot)));
    }
    return c_add(W, c_mul(prefactor, c_mul(Z,sum)));
  } else {
    RSComplex a={0.512424224754768462984202823134979415014943561548661637413182,0};
    RSComplex b={0.275255128608410950901357962647054304017026259671664935783653,0};
    RSComplex cc={0.051765358792987823963876628425793170829107067780337219430904,0};
    RSComplex d={2.724744871391589049098642037352945695982973740328335064216346,0};
    RSComplex i_c={0,1};
    RSComplex Z2 = c_mul(Z,Z);
    return c_mul(c_mul(Z,i_c),
                 c_add(c_div(a,c_sub(Z2,b)), c_div(cc,c_sub(Z2,d))));
  }
}

// ─── calculate_sig_T ─────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void calculate_sig_T(
    int nuc, double E, int numL, const double *pseudo_K0RS,
    RSComplex *sigTfactors)
{
  for (int i = 0; i < 4; i++) {
    double phi = pseudo_K0RS[nuc * numL + i] * sqrt(E);
    if      (i==1) phi -= -atan(phi);
    else if (i==2) phi -= atan(3.0*phi/(3.0-phi*phi));
    else if (i==3) phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));
    phi *= 2.0;
    sigTfactors[i].r = cos(phi);
    sigTfactors[i].i = -sin(phi);
  }
}

// ─── calculate_micro_xs (0K) ─────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void calculate_micro_xs(
    double *micro_xs, int nuc, double E, int numL,
    const int *n_windows, const double *pseudo_K0RS,
    const Window *windows, const Pole *poles,
    int max_num_windows, int max_num_poles)
{
  double spacing = 1.0 / n_windows[nuc];
  int window = (int)(E / spacing);
  if (window == n_windows[nuc]) window--;

  RSComplex sigTfactors[4];
  calculate_sig_T(nuc, E, numL, pseudo_K0RS, sigTfactors);

  Window w = windows[nuc * max_num_windows + window];
  double sigT = E * w.T, sigA = E * w.A, sigF = E * w.F;

  for (int i = w.start; i < w.end; i++) {
    Pole pole = poles[nuc * max_num_poles + i];
    RSComplex t1={0,1}, t2={sqrt(E),0};
    RSComplex PSIIKI = c_div(t1, c_sub(pole.MP_EA, t2));
    RSComplex E_c={E,0};
    RSComplex CDUM = c_div(PSIIKI, E_c);
    sigT += c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])).r;
    sigA += c_mul(pole.MP_RA, CDUM).r;
    sigF += c_mul(pole.MP_RF, CDUM).r;
  }
  micro_xs[0]=sigT; micro_xs[1]=sigA;
  micro_xs[2]=sigF; micro_xs[3]=sigT-sigA;
}

// ─── calculate_micro_xs_doppler ──────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void calculate_micro_xs_doppler(
    double *micro_xs, int nuc, double E, int numL,
    const int *n_windows, const double *pseudo_K0RS,
    const Window *windows, const Pole *poles,
    int max_num_windows, int max_num_poles)
{
  double spacing = 1.0 / n_windows[nuc];
  int window = (int)(E / spacing);
  if (window == n_windows[nuc]) window--;

  RSComplex sigTfactors[4];
  calculate_sig_T(nuc, E, numL, pseudo_K0RS, sigTfactors);

  Window w = windows[nuc * max_num_windows + window];
  double sigT = E * w.T, sigA = E * w.A, sigF = E * w.F;
  double dopp = 0.5;

  for (int i = w.start; i < w.end; i++) {
    Pole pole = poles[nuc * max_num_poles + i];
    RSComplex E_c={E,0}, dopp_c={dopp,0};
    RSComplex Z = c_mul(c_sub(E_c, pole.MP_EA), dopp_c);
    RSComplex faddeeva = fast_nuclear_W(Z);
    sigT += c_mul(pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value])).r;
    sigA += c_mul(pole.MP_RA, faddeeva).r;
    sigF += c_mul(pole.MP_RF, faddeeva).r;
  }
  micro_xs[0]=sigT; micro_xs[1]=sigA;
  micro_xs[2]=sigF; micro_xs[3]=sigT-sigA;
}

// ─── calculate_macro_xs ──────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION void calculate_macro_xs(
    double *macro_xs, int mat, double E, int numL, int doppler,
    const int *num_nucs, const int *mats, int max_num_nucs,
    const double *concs, const int *n_windows,
    const double *pseudo_K0RS, const Window *windows,
    const Pole *poles, int max_num_windows, int max_num_poles)
{
  for (int i=0;i<4;i++) macro_xs[i]=0;
  for (int i=0; i<num_nucs[mat]; i++) {
    double micro_xs[4];
    int nuc = mats[mat * max_num_nucs + i];
    if (doppler == 1)
      calculate_micro_xs_doppler(micro_xs, nuc, E, numL, n_windows, pseudo_K0RS,
                                 windows, poles, max_num_windows, max_num_poles);
    else
      calculate_micro_xs(micro_xs, nuc, E, numL, n_windows, pseudo_K0RS,
                         windows, poles, max_num_windows, max_num_poles);
    for (int j=0;j<4;j++) macro_xs[j] += micro_xs[j] * concs[mat * max_num_nucs + i];
  }
}

// ─── pick_mat ─────────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION int pick_mat(uint64_t *seed) {
  static const double dist[12] = {
    0.140, 0.052, 0.275, 0.134, 0.154, 0.064,
    0.066, 0.055, 0.008, 0.015, 0.025, 0.013
  };
  double roll = LCG_random_double(seed);
  for (int i=0; i<12; i++) {
    double running = 0;
    for (int j=i; j>0; j--) running += dist[j];
    if (roll < running) return i;
  }
  return 0;
}

// ─── Simulation data (host side) ─────────────────────────────────────────
struct SimulationData {
  int *n_poles;       unsigned long length_n_poles;
  int *n_windows;     unsigned long length_n_windows;
  Pole   *poles;      unsigned long length_poles;
  Window *windows;    unsigned long length_windows;
  double *pseudo_K0RS;unsigned long length_pseudo_K0RS;
  int *num_nucs;      unsigned long length_num_nucs;
  int *mats;          unsigned long length_mats;
  double *concs;      unsigned long length_concs;
  int max_num_nucs, max_num_poles, max_num_windows;
  double *p_energy_samples; unsigned long length_p_energy_samples;
  int    *mat_samples;      unsigned long length_mat_samples;
};

// ─── Initialization helpers (CPU only) ───────────────────────────────────
static int *generate_n_poles(const Input &input, uint64_t *seed) {
  int *R = (int*)malloc(input.n_nuclides*sizeof(int));
  for(int i=0;i<input.n_nuclides;i++) R[i]=1;
  int total=input.avg_n_poles*input.n_nuclides;
  for(int i=0;i<total-input.n_nuclides;i++) R[LCG_random_int(seed)%input.n_nuclides]++;
  return R;
}
static int *generate_n_windows(const Input &input, uint64_t *seed) {
  int *R=(int*)malloc(input.n_nuclides*sizeof(int));
  for(int i=0;i<input.n_nuclides;i++) R[i]=1;
  int total=input.avg_n_windows*input.n_nuclides;
  for(int i=0;i<total-input.n_nuclides;i++) R[LCG_random_int(seed)%input.n_nuclides]++;
  return R;
}
static Pole *generate_poles(const Input &input, int *n_poles, uint64_t *seed, int *max_num_poles) {
  double f=152.5; RSComplex f_c={f,0};
  int max=-1;
  for(int i=0;i<input.n_nuclides;i++) if(n_poles[i]>max) max=n_poles[i];
  *max_num_poles=max;
  Pole *R=(Pole*)malloc(input.n_nuclides*max*sizeof(Pole));
  for(int i=0;i<input.n_nuclides;i++)
    for(int j=0;j<n_poles[i];j++){
      RSComplex t1={LCG_random_double(seed),LCG_random_double(seed)};
      R[i*max+j].MP_EA=c_mul(f_c,t1);
      R[i*max+j].MP_RT={f*LCG_random_double(seed),LCG_random_double(seed)};
      R[i*max+j].MP_RA={f*LCG_random_double(seed),LCG_random_double(seed)};
      R[i*max+j].MP_RF={f*LCG_random_double(seed),LCG_random_double(seed)};
      R[i*max+j].l_value=(short)(LCG_random_int(seed)%input.numL);
    }
  return R;
}
static Window *generate_window_params(const Input &input, int *n_windows, int *n_poles,
                                       uint64_t *seed, int *max_num_windows) {
  int max=-1;
  for(int i=0;i<input.n_nuclides;i++) if(n_windows[i]>max) max=n_windows[i];
  *max_num_windows=max;
  Window *R=(Window*)malloc(input.n_nuclides*max*sizeof(Window));
  for(int i=0;i<input.n_nuclides;i++){
    int sp=n_poles[i]/n_windows[i];
    int rem=n_poles[i]-sp*n_windows[i];
    int ctr=0;
    for(int j=0;j<n_windows[i];j++){
      R[i*max+j].T=LCG_random_double(seed);
      R[i*max+j].A=LCG_random_double(seed);
      R[i*max+j].F=LCG_random_double(seed);
      R[i*max+j].start=ctr;
      R[i*max+j].end=ctr+sp-1;
      ctr+=sp;
      if(j<rem){ctr++;R[i*max+j].end++;}
    }
  }
  return R;
}
static double *generate_pseudo_K0RS(const Input &input, uint64_t *seed) {
  double *R=(double*)malloc(input.n_nuclides*input.numL*sizeof(double));
  for(int i=0;i<input.n_nuclides*input.numL;i++) R[i]=LCG_random_double(seed);
  return R;
}

// Material loading (from material.c)
static int *load_num_nucs(const Input &input) {
  int *n=(int*)malloc(12*sizeof(int));
  n[0]=(input.n_nuclides==68)?34:321;
  n[1]=5;n[2]=4;n[3]=4;n[4]=27;n[5]=21;
  n[6]=21;n[7]=21;n[8]=21;n[9]=21;n[10]=9;n[11]=9;
  return n;
}
static int *load_mats(const Input &input, int *num_nucs, int *max_num_nucs, unsigned long *length_mats) {
  int nmat=12, mx=0;
  for(int m=0;m<nmat;m++) if(num_nucs[m]>mx) mx=num_nucs[m];
  *max_num_nucs=mx;
  int *mats=(int*)calloc(nmat*mx,sizeof(int));
  *length_mats=(unsigned long)(nmat*mx);
  int m0s[]={58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
  int m0l[321]; memcpy(m0l,m0s,34*sizeof(int));
  for(int i=0;i<321-34;i++) m0l[34+i]=68+i;
  int m1[]={63,64,65,66,67};
  int m2[]={24,41,4,5};
  int m3[]={24,41,4,5};
  int m4[]={19,20,21,22,35,36,37,38,39,25,27,28,29,30,31,32,26,49,50,51,11,12,13,14,6,16,17};
  int m5[]={24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
  int m6[]={24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
  int m7[]={24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
  int m8[]={24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
  int m9[]={24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
  int m10[]={24,41,4,5,63,64,65,66,67};
  int m11[]={24,41,4,5,63,64,65,66,67};
  if(input.n_nuclides==68) memcpy(mats,m0s,num_nucs[0]*sizeof(int));
  else                      memcpy(mats,m0l,num_nucs[0]*sizeof(int));
  memcpy(mats+mx*1,  m1,  num_nucs[1]*sizeof(int));
  memcpy(mats+mx*2,  m2,  num_nucs[2]*sizeof(int));
  memcpy(mats+mx*3,  m3,  num_nucs[3]*sizeof(int));
  memcpy(mats+mx*4,  m4,  num_nucs[4]*sizeof(int));
  memcpy(mats+mx*5,  m5,  num_nucs[5]*sizeof(int));
  memcpy(mats+mx*6,  m6,  num_nucs[6]*sizeof(int));
  memcpy(mats+mx*7,  m7,  num_nucs[7]*sizeof(int));
  memcpy(mats+mx*8,  m8,  num_nucs[8]*sizeof(int));
  memcpy(mats+mx*9,  m9,  num_nucs[9]*sizeof(int));
  memcpy(mats+mx*10,m10, num_nucs[10]*sizeof(int));
  memcpy(mats+mx*11,m11, num_nucs[11]*sizeof(int));
  return mats;
}
static double *load_concs(int *num_nucs, uint64_t *seed, int max_num_nucs) {
  double *c=(double*)malloc(12*max_num_nucs*sizeof(double));
  for(int i=0;i<12;i++) for(int j=0;j<num_nucs[i];j++) c[i*max_num_nucs+j]=LCG_random_double(seed);
  return c;
}

static SimulationData initialize_simulation(const Input &input) {
  uint64_t seed = INITIALIZATION_SEED;
  SimulationData SD; memset(&SD,0,sizeof(SD));

  SD.num_nucs = load_num_nucs(input);
  SD.length_num_nucs = 12;
  SD.mats = load_mats(input, SD.num_nucs, &SD.max_num_nucs, &SD.length_mats);
  SD.concs = load_concs(SD.num_nucs, &seed, SD.max_num_nucs);
  SD.length_concs = 12 * SD.max_num_nucs;

  SD.n_poles = generate_n_poles(input, &seed);
  SD.length_n_poles = input.n_nuclides;
  SD.n_windows = generate_n_windows(input, &seed);
  SD.length_n_windows = input.n_nuclides;
  SD.poles = generate_poles(input, SD.n_poles, &seed, &SD.max_num_poles);
  SD.length_poles = input.n_nuclides * SD.max_num_poles;
  SD.windows = generate_window_params(input, SD.n_windows, SD.n_poles, &seed, &SD.max_num_windows);
  SD.length_windows = input.n_nuclides * SD.max_num_windows;
  SD.pseudo_K0RS = generate_pseudo_K0RS(input, &seed);
  SD.length_pseudo_K0RS = input.n_nuclides * input.numL;
  return SD;
}

// ─── I/O helpers ──────────────────────────────────────────────────────────
static void border_print() { printf("================================================================================\n"); }
static void center_print(const char *s, int w) {
  int l=(int)strlen(s); for(int i=0;i<=(w-l)/2;i++) fputs(" ",stdout); puts(s);
}
static void fancy_int(int a) {
  if(a<1000) printf("%d\n",a);
  else if(a<1000000) printf("%d,%03d\n",a/1000,a%1000);
  else if(a<1000000000) printf("%d,%03d,%03d\n",a/1000000,(a%1000000)/1000,a%1000);
  else printf("%d,%03d,%03d,%03d\n",a/1000000000,(a%1000000000)/1000000,(a%1000000)/1000,a%1000);
}

static Input read_CLI(int argc, char *argv[]) {
  Input in;
  in.simulation_method = EVENT_BASED;
  in.nthreads    = 1;
  in.n_nuclides  = 355;
  in.particles   = 300000;
  in.lookups     = 34;
  in.HM          = LARGE;
  in.avg_n_poles = 1000;
  in.avg_n_windows = 100;
  in.numL        = 4;
  in.doppler     = 1;
  in.kernel_id   = 0;
  int def_l=1, def_p=1;
  for(int i=1;i<argc;i++){
    char *arg=argv[i];
    if(!strcmp(arg,"-m")){
      if(++i<argc){
        if(!strcmp(argv[i],"history")) in.simulation_method=HISTORY_BASED;
        else if(!strcmp(argv[i],"event")){
          in.simulation_method=EVENT_BASED;
          if(def_l&&def_p){in.lookups*=in.particles;in.particles=0;}
        }
      }
    } else if(!strcmp(arg,"-l")){if(++i<argc){in.lookups=atoi(argv[i]);def_l=0;}}
    else if(!strcmp(arg,"-p")){if(++i<argc){in.particles=atoi(argv[i]);def_p=0;}}
    else if(!strcmp(arg,"-n")){if(++i<argc) in.n_nuclides=atoi(argv[i]);}
    else if(!strcmp(arg,"-s")){
      if(++i<argc){
        if(!strcmp(argv[i],"small")) in.HM=SMALL;
        else in.HM=LARGE;
      }
    } else if(!strcmp(arg,"-d")){in.doppler=0;}
    else if(!strcmp(arg,"-W")){if(++i<argc) in.avg_n_windows=atoi(argv[i]);}
    else if(!strcmp(arg,"-P")){if(++i<argc) in.avg_n_poles=atoi(argv[i]);}
    else if(!strcmp(arg,"-k")){if(++i<argc) in.kernel_id=atoi(argv[i]);}
  }
  if(in.HM==SMALL) in.n_nuclides=68;
  return in;
}

static void print_input_summary(const Input &in) {
  printf("Programming Model:           Kokkos\n");
  printf("Simulation Method:           Event Based\n");
  printf("Materials:                   12\n");
  printf("H-M Benchmark Size:          %s\n", (in.HM==SMALL)?"Small":"Large");
  printf("Temperature Dependence:      %s\n", (in.doppler==1)?"ON":"OFF");
  printf("Total Nuclides:              %d\n", in.n_nuclides);
  printf("Avg Poles per Nuclide:       "); fancy_int(in.avg_n_poles);
  printf("Avg Windows per Nuclide:     "); fancy_int(in.avg_n_windows);
  printf("Particles:                   "); fancy_int(in.particles);
  printf("XS Lookups per Particle:     "); fancy_int(in.lookups);
  int lookups = in.lookups * in.particles;
  printf("Total XS Lookups:            "); fancy_int(lookups);
}

static int validate_and_print_results(const Input &in, double runtime,
                                       unsigned long vhash, double kernel_time)
{
  int lookups = in.lookups * in.particles;
  printf("Runtime:               %.3f seconds\n", runtime);
  printf("Lookups:               "); fancy_int(lookups);
  printf("Lookups/s:             "); fancy_int((int)((double)lookups/runtime));
  printf("Kernel Lookups/s:      "); fancy_int((int)((double)lookups/kernel_time));

  unsigned long large_ref = 351485, small_ref = 879693;
  unsigned long ref = (in.HM==LARGE) ? large_ref : small_ref;
  int invalid = (vhash != ref);
  if(invalid) printf("Verification checksum: %lu (WARNING - INVALID!)\n", vhash);
  else        printf("Verification checksum: %lu (Valid)\n", vhash);
  return invalid;
}

// ─── main ─────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  int version = 12;
  Input input = read_CLI(argc, argv);
  border_print(); center_print("INPUT SUMMARY",79); border_print();
  print_input_summary(input);
  border_print(); center_print("INITIALIZATION",79); border_print();

  SimulationData SD = initialize_simulation(input);
  printf("Initialization Complete.\n");

  border_print(); center_print("SIMULATION",79); border_print();

  // Normalize event-based: if particles > 0, total lookups = lookups * particles
  if (input.simulation_method == EVENT_BASED && input.particles > 0) {
    input.lookups   *= input.particles;
    input.particles  = 0;
  }

  if (input.simulation_method == HISTORY_BASED) {
    printf("History-based not implemented in Kokkos port. Use -m event.\n");
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    unsigned long verification = 0;
    double kernel_time = 0.0;

    // Create device views
    Kokkos::View<int*>    d_n_poles  ("n_poles",   SD.length_n_poles);
    Kokkos::View<int*>    d_n_windows("n_windows", SD.length_n_windows);
    Kokkos::View<Pole*>   d_poles    ("poles",     SD.length_poles);
    Kokkos::View<Window*> d_windows  ("windows",   SD.length_windows);
    Kokkos::View<double*> d_pseudo   ("pseudo_K0", SD.length_pseudo_K0RS);
    Kokkos::View<int*>    d_num_nucs ("num_nucs",  SD.length_num_nucs);
    Kokkos::View<int*>    d_mats     ("mats",      SD.length_mats);
    Kokkos::View<double*> d_concs    ("concs",     SD.length_concs);

    // Mirror and deep_copy
    {
      auto h = Kokkos::create_mirror_view(d_n_poles);
      for(unsigned long i=0;i<SD.length_n_poles;i++) h(i)=SD.n_poles[i];
      Kokkos::deep_copy(d_n_poles, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_n_windows);
      for(unsigned long i=0;i<SD.length_n_windows;i++) h(i)=SD.n_windows[i];
      Kokkos::deep_copy(d_n_windows, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_poles);
      for(unsigned long i=0;i<SD.length_poles;i++) h(i)=SD.poles[i];
      Kokkos::deep_copy(d_poles, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_windows);
      for(unsigned long i=0;i<SD.length_windows;i++) h(i)=SD.windows[i];
      Kokkos::deep_copy(d_windows, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_pseudo);
      for(unsigned long i=0;i<SD.length_pseudo_K0RS;i++) h(i)=SD.pseudo_K0RS[i];
      Kokkos::deep_copy(d_pseudo, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_num_nucs);
      for(unsigned long i=0;i<SD.length_num_nucs;i++) h(i)=SD.num_nucs[i];
      Kokkos::deep_copy(d_num_nucs, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_mats);
      for(unsigned long i=0;i<SD.length_mats;i++) h(i)=SD.mats[i];
      Kokkos::deep_copy(d_mats, h);
    }
    {
      auto h = Kokkos::create_mirror_view(d_concs);
      for(unsigned long i=0;i<SD.length_concs;i++) h(i)=SD.concs[i];
      Kokkos::deep_copy(d_concs, h);
    }

    // Capture scalar parameters for lambda
    const int lookups       = input.lookups;
    const int numL          = input.numL;
    const int doppler       = input.doppler;
    const int max_num_nucs  = SD.max_num_nucs;
    const int max_num_poles = SD.max_num_poles;
    const int max_num_windows = SD.max_num_windows;

    printf("Beginning event based simulation ...\n");

    auto wall_start = std::chrono::steady_clock::now();

    Kokkos::parallel_reduce(
      lookups,
      KOKKOS_LAMBDA(int i, unsigned long &local_ver) {
        uint64_t seed = STARTING_SEED;
        seed = fast_forward_LCG(seed, (uint64_t)(2*i));

        double E   = LCG_random_double(&seed);
        int    mat = pick_mat(&seed);

        double macro_xs[4] = {0,0,0,0};
        calculate_macro_xs(macro_xs, mat, E, numL, doppler,
                           d_num_nucs.data(), d_mats.data(), max_num_nucs,
                           d_concs.data(), d_n_windows.data(),
                           d_pseudo.data(), d_windows.data(), d_poles.data(),
                           max_num_windows, max_num_poles);

        double maxv = -DBL_MAX; int max_idx = 0;
        for (int x=0; x<4; x++) if (macro_xs[x] > maxv) { maxv=macro_xs[x]; max_idx=x; }
        local_ver += (unsigned long)(max_idx + 1);
      },
      verification);

    Kokkos::fence();
    auto wall_end = std::chrono::steady_clock::now();
    kernel_time = std::chrono::duration<double>(wall_end - wall_start).count();
    printf("Kernel execution time: %.2f seconds.\n", kernel_time);

    verification %= 999983;

    border_print(); center_print("RESULTS",79); border_print();
    validate_and_print_results(input, kernel_time, verification, kernel_time);
    border_print();
  }
  Kokkos::finalize();

  // Free host data
  free(SD.n_poles); free(SD.n_windows); free(SD.poles);
  free(SD.windows); free(SD.pseudo_K0RS); free(SD.num_nucs);
  free(SD.mats); free(SD.concs);
  return 0;
}
