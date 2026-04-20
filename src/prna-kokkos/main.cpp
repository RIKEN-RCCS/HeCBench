/*
 * RNA Secondary Structure (partition function) – Kokkos port
 * Original: HeCBench prna-omp
 */

/* ── real_t selection (pass -DFLOAT or -DDOUBLE on compile line) ─────────── */
#ifndef FLOAT
#ifndef DOUBLE
#define FLOAT
#endif
#endif

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cctype>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

/* ── real_t ─────────────────────────────────────────────────────────────── */
#if defined FLOAT
  typedef float real_t;
  #define RF       "%.6g"
  #define RCONST(x) x
  #define ALMOST_ZERO RCONST(1e-7)
  #define STR_TO_REAL(x,y) strtof(x,y)
  #define MATHFN(x) x##f
#elif defined DOUBLE
  typedef double real_t;
  #define RF       "%.14lg"
  #define RCONST(x) x
  #define ALMOST_ZERO RCONST(1e-15)
  #define STR_TO_REAL(x,y) strtod(x,y)
  #define MATHFN(x) x
#endif

#define LOG    MATHFN(log)
#define EXP    MATHFN(exp)
#define LOG1P  MATHFN(log1p)
#define SQRT   MATHFN(sqrt)
#define FABS   MATHFN(fabs)
#define HALF   RCONST(0.5)
#define PI     RCONST(3.14159265358979323846)
#define INF    (-LOG((real_t)0))
#define NOT_A_NUMBER (LOG((real_t)-1))

/* ── Kokkos types ────────────────────────────────────────────────────────── */
using ExecSpace    = Kokkos::DefaultExecutionSpace;
using MemSpace     = ExecSpace::memory_space;
using ScratchSpace = ExecSpace::scratch_memory_space;
using TeamPolicy   = Kokkos::TeamPolicy<ExecSpace>;
using Member       = TeamPolicy::member_type;

template<typename T>
using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;
template<typename T>
using DevView = Kokkos::View<T*, MemSpace>;

/* ── base_t ──────────────────────────────────────────────────────────────── */
#define NBASE 5
typedef enum { X=0, A, C, G, U } base_t;

KOKKOS_INLINE_FUNCTION
static int is_cp(base_t i, base_t j) {
  return (i==A&&j==U)||(i==C&&j==G)||(i==G&&j==U);
}
KOKKOS_INLINE_FUNCTION
static int is_canonical_pair(base_t i, base_t j) { return is_cp(i,j)||is_cp(j,i); }

KOKKOS_INLINE_FUNCTION
static int contains_only_base(base_t b, int n, const base_t *seq) {
  for (int i=0;i<n;i++) if(seq[i]!=b) return 0; return 1;
}
KOKKOS_INLINE_FUNCTION
static int sequences_match(const base_t *s1, const base_t *s2, int n) {
  for (int i=0;i<n;i++) if(s1[i]!=s2[i]) return 0; return 1;
}

/* host-only base helpers */
static base_t base_from_char(char c) {
  switch(c){case 'A':case 'a':return A; case 'C':case 'c':return C;
            case 'G':case 'g':return G; case 'U':case 'u':case 'T':case 't':return U;
            default: return X;}
}
static char base_as_char(base_t b) {
  switch(b){case X:return 'X';case A:return 'A';case C:return 'C';
            case G:return 'G';case U:return 'U';default:return '?';}
}
static void sequence_from_string(base_t *b, const char *s) {
  for (int i=0;s[i];i++) b[i]=base_from_char(s[i]);
}

/* ── util ────────────────────────────────────────────────────────────────── */
static void die(const char *fmt, ...) {
  va_list ap; va_start(ap,fmt); vfprintf(stderr,fmt,ap); va_end(ap);
  fprintf(stderr,"\n"); exit(1);
}
static void *safe_malloc(size_t sz) {
  void *p = malloc(sz); if(!p) die("safe_malloc: oom"); return p;
}
static FILE *safe_fopen(const char *path, const char *mode) {
  FILE *f = fopen(path,mode); if(!f) die("safe_fopen: can't open '%s'",path); return f;
}
static int end_of_file(FILE *f) { int c=fgetc(f); if(c==EOF)return 1; ungetc(c,f); return 0; }
static int isdigits(const char *s) { for(;*s;s++) if(!isdigit(*s))return 0; return 1; }
static int string_begins_with(const char *buf,const char *start){
  return !strncmp(buf,start,strlen(start));
}

/* ── sequence reader (host) ──────────────────────────────────────────────── */
static char *sequence_from_file(const char *fn) {
  FILE *f = safe_fopen(fn,"r");
  /* skip header lines starting with ; or > */
  char line[4096]; char *s=NULL; size_t len=0;
  while(fgets(line,sizeof(line),f)) {
    if(line[0]==';'||line[0]=='>') continue;
    size_t l=strlen(line); if(l>0&&line[l-1]=='\n'){line[l-1]=0;l--;}
    if(l==0) continue;
    s=(char*)realloc(s,len+l+1);
    memcpy(s+len,line,l+1); len+=l;
  }
  fclose(f); return s;
}
static char *sequence(const char *arg) {
  /* if arg consists only of valid base chars, use it directly; else read file */
  int ok=1;
  for(const char *p=arg;*p;p++)
    if(!strchr("XAaCcGgUuTt",*p)){ok=0;break;}
  return ok ? strdup(arg) : sequence_from_file(arg);
}

/* ── param struct ────────────────────────────────────────────────────────── */
#define LOOP_MIN 3
#define LOOP_MAX 30
#define LOOKUP_TABLE_MAX 120

typedef real_t tab3_t[NBASE][NBASE][NBASE];
typedef real_t tab4_t[NBASE][NBASE][NBASE][NBASE];
typedef real_t tab6_t[NBASE][NBASE][NBASE][NBASE][NBASE][NBASE];
typedef real_t tab7_t[NBASE][NBASE][NBASE][NBASE][NBASE][NBASE][NBASE];
typedef real_t tab8_t[5][5][5][5][5][5][5][5];

struct param {
  int use_dna_params, use_enthalpy_params;
  real_t temperature;
  tab4_t coaxial,coaxstack,stack,tstack,tstackcoax,tstackh,tstacki,tstacki1n,tstacki23,tstackm;
  real_t internal_loop_initiation[LOOP_MAX+1];
  real_t bulge_loop_initiation[LOOP_MAX+1];
  real_t hairpin_loop_initiation[LOOP_MAX+1];
  real_t Extrapolation_for_large_loops, maximum_correction, fm_array_first_element;
  real_t a, multibranched_loop_offset;
  real_t b, multibranched_loop_per_nuc_penalty;
  real_t c, multibranched_loop_helix_penalty;
  real_t a_2c, a_2b_2c;
  real_t terminal_AU_penalty, bonus_for_GGG_hairpin;
  real_t c_hairpin_slope, c_hairpin_intercept, c_hairpin_of_3;
  real_t Bonus_for_Single_C_bulges_adjacent_to_C;
  int ntriloop;
  struct triloop_t { base_t seq[5]; real_t val; } triloop[LOOKUP_TABLE_MAX];
  int ntloop;
  struct tloop_t   { base_t seq[6]; real_t val; } tloop[LOOKUP_TABLE_MAX];
  int nhexaloop;
  struct hexaloop_t{ base_t seq[8]; real_t val; } hexaloop[LOOKUP_TABLE_MAX];
  tab3_t dangle_3p, dangle_5p;
  tab6_t int11;
  tab8_t int22;
  tab7_t int21;
};
typedef struct param *param_t;

/* ── param loading (host only, condensed version of param.c) ──────────────── */
static real_t init_val(int *b, int n, real_t Xval){
  for(int i=0;i<n;i++) if(b[i]==X) return Xval; return NOT_A_NUMBER;
}
static void init_tab3(tab3_t *t,real_t Xv){int i[3];for(i[0]=0;i[0]<NBASE;i[0]++)for(i[1]=0;i[1]<NBASE;i[1]++)for(i[2]=0;i[2]<NBASE;i[2]++)(*t)[i[0]][i[1]][i[2]]=init_val(i,3,Xv);}
static void init_tab4(tab4_t *t,real_t Xv){int i[4];for(i[0]=0;i[0]<NBASE;i[0]++)for(i[1]=0;i[1]<NBASE;i[1]++)for(i[2]=0;i[2]<NBASE;i[2]++)for(i[3]=0;i[3]<NBASE;i[3]++)(*t)[i[0]][i[1]][i[2]][i[3]]=init_val(i,4,Xv);}
static void init_tab6(tab6_t *t,real_t Xv){int i[6];for(i[0]=0;i[0]<NBASE;i[0]++)for(i[1]=0;i[1]<NBASE;i[1]++)for(i[2]=0;i[2]<NBASE;i[2]++)for(i[3]=0;i[3]<NBASE;i[3]++)for(i[4]=0;i[4]<NBASE;i[4]++)for(i[5]=0;i[5]<NBASE;i[5]++)(*t)[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]]=init_val(i,6,Xv);}
static void init_tab7(tab7_t *t,real_t Xv){int i[7];for(i[0]=0;i[0]<NBASE;i[0]++)for(i[1]=0;i[1]<NBASE;i[1]++)for(i[2]=0;i[2]<NBASE;i[2]++)for(i[3]=0;i[3]<NBASE;i[3]++)for(i[4]=0;i[4]<NBASE;i[4]++)for(i[5]=0;i[5]<NBASE;i[5]++)for(i[6]=0;i[6]<NBASE;i[6]++)(*t)[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]][i[6]]=init_val(i,7,Xv);}
static void init_tab8(tab8_t *t,real_t Xv){int i[8];for(i[0]=0;i[0]<5;i[0]++)for(i[1]=0;i[1]<5;i[1]++)for(i[2]=0;i[2]<5;i[2]++)for(i[3]=0;i[3]<5;i[3]++)for(i[4]=0;i[4]<5;i[4]++)for(i[5]=0;i[5]<5;i[5]++)for(i[6]=0;i[6]<5;i[6]++)for(i[7]=0;i[7]<5;i[7]++)(*t)[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]][i[6]][i[7]]=init_val(i,8,Xv);}

#define MAXLINE 1024
static FILE *parfile(const char *name,int use_dna,int use_enth){
  char buf[MAXLINE+1]; const char *pre=use_dna?"dna":"rna"; const char *ext=use_enth?"dh":"dg";
  sprintf(buf,"%s.%s.%s",pre,name,ext); return safe_fopen(buf,"r");
}
static void look_for_line(FILE *f,const char *s){char buf[MAXLINE+1];while(fgets(buf,MAXLINE,f))if(strstr(buf,s))return;die("couldn't find '%s'",s);}

static real_t read_real(FILE *f){real_t x; STR_TO_REAL("",NULL);
  int n=fscanf(f,RF,&x);(void)n; return x;}

static void read_tab4_one(FILE *f, tab4_t *t, real_t scale){
  for(int a=1;a<NBASE;a++)for(int b_=1;b_<NBASE;b_++)for(int c=1;c<NBASE;c++)for(int d=1;d<NBASE;d++){
    real_t x; if(fscanf(f,RF,&x)==1) (*t)[a][b_][c][d]=x*scale; }}

/* Simplified parameter reader - reads the binary parameter file if available,
   falls back to zeroed struct otherwise (users can set DATAPATH env var for text params) */
static void param_init_zero(param_t p){
  memset(p,0,sizeof(struct param));
  real_t inf=INF;
  init_tab4(&p->coaxial,inf); init_tab4(&p->coaxstack,inf); init_tab4(&p->stack,inf);
  init_tab4(&p->tstack,inf);  init_tab4(&p->tstackcoax,inf);init_tab4(&p->tstackh,inf);
  init_tab4(&p->tstacki,inf); init_tab4(&p->tstacki1n,inf); init_tab4(&p->tstacki23,inf);
  init_tab4(&p->tstackm,inf);
  init_tab3(&p->dangle_3p,inf); init_tab3(&p->dangle_5p,inf);
  init_tab6(&p->int11,inf); init_tab7(&p->int21,inf); init_tab8(&p->int22,inf);
  for(int i=0;i<=LOOP_MAX;i++){p->internal_loop_initiation[i]=inf;p->bulge_loop_initiation[i]=inf;p->hairpin_loop_initiation[i]=inf;}
}

/* Read param from binary file */
static void param_read_from_binary(const char *path, param_t p){
  FILE *f=safe_fopen(path,"rb");
  if(fread(p,sizeof(struct param),1,f)!=1) die("binary param read failed");
  fclose(f);
}

/* ── prna struct ─────────────────────────────────────────────────────────── */
struct prna {
  int n;
  base_t *seq;
  int    *base_can_pair;
  real_t *v;
  real_t *w5, *w3;
};
typedef struct prna *prna_t;

/* ── index helpers ─────────────────────────────────────────────────────────── */
KOKKOS_INLINE_FUNCTION static int ind(int i,int j,int n){return i*n+j;}
KOKKOS_INLINE_FUNCTION static int upper_triangle_index(int i,int j){return (j*(j-1))/2+i;}
KOKKOS_INLINE_FUNCTION static int can_pair(int i,int j,int n,const int *bcp){
  if(i>=0&&j<=n-1&&i!=j&&j>=0&&i<=n-1){
    if(i<j) return bcp[upper_triangle_index(i,j)];
    else     return bcp[upper_triangle_index(j,i)];
  } return 0;
}
KOKKOS_INLINE_FUNCTION static int wrap(int i,int n){return i>=n?i-n:i;}
KOKKOS_INLINE_FUNCTION static int is_exterior(int i,int j){return j<i;}
KOKKOS_INLINE_FUNCTION static int is_interior(int i,int j){return i<j;}
KOKKOS_INLINE_FUNCTION static int int_min(int a,int b){return a<b?a:b;}

KOKKOS_INLINE_FUNCTION
static real_t* array_val(real_t *a,int i,int j,int n,const int *bcp){
  return can_pair(i,j,n,bcp)?&a[ind(i,j,n)]:nullptr;
}

/* ── device energy functions ─────────────────────────────────────────────── */
KOKKOS_INLINE_FUNCTION
static real_t terminal_U_penalty(const base_t *s,int i,int j,const param_t p){
  return s[i]==U||s[j]==U ? p->terminal_AU_penalty : RCONST(0.);
}
KOKKOS_INLINE_FUNCTION
static real_t dangle_3p_energy(const base_t *s,int i,int j,int ip1,const param_t p){
  return p->dangle_3p[s[i]][s[j]][s[ip1]]+terminal_U_penalty(s,i,j,p);
}
KOKKOS_INLINE_FUNCTION
static real_t dangle_5p_energy(const base_t *s,int i,int j,int jm1,const param_t p){
  return p->dangle_5p[s[i]][s[j]][s[jm1]]+terminal_U_penalty(s,i,j,p);
}
KOKKOS_INLINE_FUNCTION
static real_t terminal_stack(const base_t *s,int i,int j,int ip1,int jm1,const param_t p){
  return p->tstack[s[i]][s[j]][s[ip1]][s[jm1]]+terminal_U_penalty(s,i,j,p);
}
KOKKOS_INLINE_FUNCTION
static real_t terminal_stack_multibranch(const base_t *s,int i,int j,int ip1,int jm1,const param_t p){
  return p->tstackm[s[i]][s[j]][s[ip1]][s[jm1]]+terminal_U_penalty(s,i,j,p);
}
KOKKOS_INLINE_FUNCTION
static const real_t *lookup_find(const base_t *s,int d,const param_t p){
  int i;
  switch(d){
  case 3: for(i=0;i<p->ntriloop;i++) if(sequences_match(s,p->triloop[i].seq,d+2)) return &p->triloop[i].val; break;
  case 4: for(i=0;i<p->ntloop;i++)   if(sequences_match(s,p->tloop[i].seq,d+2))   return &p->tloop[i].val;   break;
  case 6: for(i=0;i<p->nhexaloop;i++)if(sequences_match(s,p->hexaloop[i].seq,d+2))return &p->hexaloop[i].val;break;
  }
  return nullptr;
}
KOKKOS_INLINE_FUNCTION
static real_t hairpin_loop_energy(const base_t *s,int i,int j,int d,const param_t p){
  const real_t *val;
  if((val=lookup_find(&s[i],d,p))) return *val;
  real_t e;
  if(d>LOOP_MAX) e=p->hairpin_loop_initiation[LOOP_MAX]+p->Extrapolation_for_large_loops*LOG((real_t)d/LOOP_MAX);
  else e=p->hairpin_loop_initiation[d];
  if(d==3){
    if(contains_only_base(C,d,&s[i+1])) e+=p->c_hairpin_of_3;
    e+=terminal_U_penalty(s,i,j,p);
  } else {
    e+=p->tstackh[s[i]][s[j]][s[i+1]][s[j-1]];
    if(contains_only_base(C,d,&s[i+1])) e+=p->c_hairpin_slope*d+p->c_hairpin_intercept;
  }
  if(s[i]==G&&s[j]==U&&i>1&&s[i-1]==G&&s[i-2]==G) e+=p->bonus_for_GGG_hairpin;
  return e;
}
KOKKOS_INLINE_FUNCTION static real_t real_min(real_t a,real_t b){return a<b?a:b;}
KOKKOS_INLINE_FUNCTION
static real_t internal_loop_energy(const base_t *s,int i,int j,int ip,int jp,int d1,int d2,const param_t p){
  if(d1==0||d2==0){
    real_t e=p->bulge_loop_initiation[d1+d2];
    if(d1==1||d2==1){
      e+=p->stack[s[i]][s[j]][s[ip]][s[jp]];
      if((d1==1&&s[i+1]==C&&(s[i]==C||s[i+2]==C))||(d2==1&&s[j-1]==C&&(s[j]==C||s[j-2]==C)))
        e+=p->Bonus_for_Single_C_bulges_adjacent_to_C;
    } else { e+=terminal_U_penalty(s,i,j,p); e+=terminal_U_penalty(s,ip,jp,p); }
    return e;
  }
  if(d1==1&&d2==1) return p->int11[s[i]][s[i+1]][s[i+2]][s[j-2]][s[j-1]][s[j]];
  if(d1==2&&d2==2) return p->int22[s[i]][s[ip]][s[j]][s[jp]][s[i+1]][s[i+2]][s[j-1]][s[j-2]];
  if(d1==1&&d2==2) return p->int21[s[i]][s[j]][s[i+1]][s[j-1]][s[jp+1]][s[ip]][s[jp]];
  if(d1==2&&d2==1) return p->int21[s[jp]][s[ip]][s[jp+1]][s[ip-1]][s[i+1]][s[j]][s[i]];
  tab4_t *sp;
  if(d1==1||d2==1) sp=&p->tstacki1n;
  else if((d1==2&&d2==3)||(d1==3&&d2==2)) sp=&p->tstacki23;
  else sp=&p->tstacki;
  return p->internal_loop_initiation[d1+d2]+real_min(p->fm_array_first_element*abs(d1-d2),p->maximum_correction)
    +(*sp)[s[i]][s[j]][s[i+1]][s[j-1]]+(*sp)[s[jp]][s[ip]][s[jp+1]][s[ip-1]];
}
KOKKOS_INLINE_FUNCTION
static real_t free_energy_sum(real_t a,real_t b){
  if(a<b)      return a-LOG1P(EXP(a-b));
  else if(b<a) return b-LOG1P(EXP(b-a));
  else         return a-LOG((real_t)2);
}
KOKKOS_INLINE_FUNCTION
static void free_energy_accumulate(real_t *a,real_t b){ *a=free_energy_sum(*a,b); }
KOKKOS_INLINE_FUNCTION
static real_t coaxial_flush(const base_t *s,int i,int j,int ip,int jp,const param_t p){
  return terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+p->coaxial[s[i]][s[j]][s[ip]][s[jp]];
}
KOKKOS_INLINE_FUNCTION
static real_t coaxial_mismatch1(const base_t *s,int i,int j,int ip,int jp,const param_t p){
  return terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)
    +p->tstackcoax[s[j]][s[i]][s[j+1]][s[i-1]]+p->coaxstack[s[j+1]][s[i-1]][s[ip]][s[jp]];
}
KOKKOS_INLINE_FUNCTION
static real_t coaxial_mismatch2(const base_t *s,int i,int j,int ip,int jp,const param_t p){
  return terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)
    +p->tstackcoax[s[jp]][s[ip]][s[jp+1]][s[ip-1]]+p->coaxstack[s[j]][s[i]][s[j+1]][s[jp+1]];
}

/* ── team-level free_energy_reduce ────────────────────────────────────────── */
#define NTHREAD  128
#define THREAD_X 8
#define THREAD_Y 16

KOKKOS_INLINE_FUNCTION
static void free_energy_reduce(real_t *buf, real_t *x, int tid, int nt, const Member& team){
  buf[tid] = *x;
  team.team_barrier();
  for(nt/=2; nt>0; nt/=2){
    if(tid<nt) free_energy_accumulate(&buf[tid], buf[tid+nt]);
    team.team_barrier();
  }
  if(tid==0) *x=buf[0];
}

/* ── Kokkos GPU kernel wrappers ──────────────────────────────────────────── */
static void calc_hairpin_stack_exterior_multibranch_kokkos(
    int d, int n,
    const base_t *s_d, const int *bcp_d, real_t *v_d,
    const real_t *x_d, const real_t *w5_d, const real_t *w3_d,
    const struct param *p_d)
{
  Kokkos::parallel_for("calc_hsem", Kokkos::RangePolicy<ExecSpace>(0, n),
    KOKKOS_LAMBDA(int i){
      const int jtmp = i+d+1; const int j = wrap(jtmp,n);
      if((is_exterior(i,j)&&i-j<=LOOP_MIN)||!can_pair(i,j,n,bcp_d)) return;
      real_t vij = INF;
      if(i!=n-1&&j!=0){
        if(is_interior(i,j)) vij=hairpin_loop_energy(s_d,i,j,d,p_d);
        if(can_pair(i+1,j-1,n,bcp_d)&&!(is_interior(i,j)&&d<=LOOP_MIN-2))
          free_energy_accumulate(&vij,p_d->stack[s_d[i]][s_d[j]][s_d[i+1]][s_d[j-1]]+v_d[ind(i+1,j-1,n)]);
      }
      if(is_exterior(i,j)){
        free_energy_accumulate(&vij,w3_d[i+1]+w5_d[j-1]+terminal_U_penalty(s_d,i,j,p_d));
        if(i!=n-1) free_energy_accumulate(&vij,w3_d[i+2]+w5_d[j-1]+dangle_3p_energy(s_d,i,j,i+1,p_d));
        if(j!=0)   free_energy_accumulate(&vij,w3_d[i+1]+w5_d[j-2]+dangle_5p_energy(s_d,i,j,j-1,p_d));
        if(i!=n-1&&j!=0) free_energy_accumulate(&vij,w3_d[i+2]+w5_d[j-2]+terminal_stack(s_d,i,j,i+1,j-1,p_d));
      }
      if(d>2*LOOP_MIN+3&&i!=n-1&&j!=0){
        free_energy_accumulate(&vij,x_d[ind((d-2)%5,i+1,n)]+terminal_U_penalty(s_d,i,j,p_d)+p_d->a+p_d->c);
        if(i!=n-2) free_energy_accumulate(&vij,x_d[ind((d-3)%5,i+2,n)]+dangle_3p_energy(s_d,i,j,i+1,p_d)+p_d->a+p_d->b+p_d->c);
        if(j!=1)   free_energy_accumulate(&vij,x_d[ind((d-3)%5,i+1,n)]+dangle_5p_energy(s_d,i,j,j-1,p_d)+p_d->a+p_d->b+p_d->c);
        if(i!=n-2&&j!=1) free_energy_accumulate(&vij,x_d[ind((d-4)%5,i+2,n)]+terminal_stack_multibranch(s_d,i,j,i+1,j-1,p_d)+p_d->a+2*p_d->b+p_d->c);
      }
      v_d[ind(i,j,n)] = vij;
    });
  Kokkos::fence();
}

static void calc_internal_kokkos(int d, int n,
    const base_t *s_d, const int *bcp_d, real_t *v_d, const struct param *p_d)
{
  size_t scratch = ScratchView<real_t>::shmem_size(NTHREAD);
  Kokkos::parallel_for("calc_internal",
    TeamPolicy(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team){
      ScratchView<real_t> buf(team.team_scratch(0), NTHREAD);
      int i = team.league_rank();
      const int jtmp = i+d+1; const int j = wrap(jtmp,n);
      if((is_exterior(i,j)&&i-j<=LOOP_MIN)||(is_interior(i,j)&&d<=LOOP_MIN+2)||!can_pair(i,j,n,bcp_d)) return;
      real_t vij = INF;
      const int d1s = team.team_rank() % THREAD_X;
      const int ty  = team.team_rank() / THREAD_X;
      const int dmax  = int_min(LOOP_MAX, d-2);
      const int d1max = int_min(dmax, n-i-2);
      for(int d1=d1s; d1<=d1max; d1+=THREAD_X){
        const int ip = i+d1+1;
        const int d2max  = int_min(dmax-d1, j-1);
        const int d2s    = d1>0 ? ty : ty+1;
        for(int d2=d2s; d2<=d2max; d2+=THREAD_Y){
          const int jp = j-d2-1;
          if(can_pair(ip,jp,n,bcp_d))
            free_energy_accumulate(&vij, internal_loop_energy(s_d,i,j,ip,jp,d1,d2,p_d)+v_d[ind(ip,jp,n)]);
        }
      }
      free_energy_reduce(buf.data(), &vij, team.team_rank(), NTHREAD, team);
      if(team.team_rank()==0) free_energy_accumulate(&v_d[ind(i,j,n)], vij);
    });
  Kokkos::fence();
}

static void calc_coaxial_kokkos(int d, int n,
    const base_t *s_d, const int *bcp_d, real_t *v_d,
    const real_t *y_d, const real_t *w5_d, const real_t *w3_d,
    const struct param *p_d)
{
  size_t scratch = ScratchView<real_t>::shmem_size(NTHREAD);
  Kokkos::parallel_for("calc_coaxial",
    TeamPolicy(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team){
      ScratchView<real_t> buf(team.team_scratch(0), NTHREAD);
      int i = team.league_rank();
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if((is_exterior(i,j)&&i-j<=LOOP_MIN)||!can_pair(i,j,n,bcp_d)) return;
      const real_t *v1;
      real_t vij = INF;
      int tx = team.team_rank(); int dim = team.team_size();
      if(is_exterior(i,j)){
        for(int k=tx; k<j-LOOP_MIN; k+=dim){
          if((v1=array_val(v_d,k,j-1,n,bcp_d))) free_energy_accumulate(&vij,w3_d[i+1]+w5_d[k-1]+coaxial_flush(s_d,k,j-1,j,i,p_d)+(*v1));
          if(j-2>=0){
            if(i<n-1&&(v1=array_val(v_d,k,j-2,n,bcp_d))) free_energy_accumulate(&vij,w3_d[i+2]+w5_d[k-1]+coaxial_mismatch2(s_d,k,j-2,j,i,p_d)+(*v1));
            if((v1=array_val(v_d,k+1,j-2,n,bcp_d)))       free_energy_accumulate(&vij,w3_d[i+1]+w5_d[k-1]+coaxial_mismatch1(s_d,k+1,j-2,j,i,p_d)+(*v1));
          }
        }
        for(int k=i+LOOP_MIN+1+tx; k<n; k+=dim){
          if((v1=array_val(v_d,i+1,k,n,bcp_d))) free_energy_accumulate(&vij,w3_d[k+1]+w5_d[j-1]+coaxial_flush(s_d,j,i,i+1,k,p_d)+(*v1));
          if(j>0&&(v1=array_val(v_d,i+2,k,n,bcp_d)))   free_energy_accumulate(&vij,w3_d[k+1]+w5_d[j-2]+coaxial_mismatch1(s_d,j,i,i+2,k,p_d)+(*v1));
          if(j>0&&(v1=array_val(v_d,i+2,k-1,n,bcp_d))) free_energy_accumulate(&vij,w3_d[k+1]+w5_d[j-1]+coaxial_mismatch2(s_d,j,i,i+2,k-1,p_d)+(*v1));
        }
      }
      if(d>2*LOOP_MIN+3&&i!=n-1&&j!=0){
        for(int ktmp=i+2+tx; ktmp<jtmp-2; ktmp+=dim){
          const int k=wrap(ktmp,n);
          if(k!=n-1){
            if((v1=array_val(v_d,i+1,k,n,bcp_d)))
              free_energy_accumulate(&vij,coaxial_flush(s_d,j,i,i+1,k,p_d)+(*v1)+p_d->a_2c+y_d[ind(k+1,j-1,n)]);
            if(ktmp+2<jtmp-1&&i+1!=n-1&&k+1!=n-1&&(v1=array_val(v_d,i+2,k,n,bcp_d))){
              real_t tmp=(*v1)+p_d->a_2b_2c;
              free_energy_accumulate(&vij,coaxial_mismatch2(s_d,j,i,i+2,k,p_d)+tmp+y_d[ind(k+2,j-1,n)]);
              if(j!=1) free_energy_accumulate(&vij,coaxial_mismatch1(s_d,j,i,i+2,k,p_d)+tmp+y_d[ind(k+1,j-2,n)]);
            }
          }
        }
        for(int ktmp=i+3+tx; ktmp<jtmp-1; ktmp+=dim){
          const int k=wrap(ktmp,n);
          if(k!=0){
            if((v1=array_val(v_d,k,j-1,n,bcp_d)))
              free_energy_accumulate(&vij,coaxial_flush(s_d,k,j-1,j,i,p_d)+(*v1)+p_d->a_2c+y_d[ind(i+1,k-1,n)]);
            if(j!=1&&ktmp>i+3&&(v1=array_val(v_d,k,j-2,n,bcp_d))){
              real_t tmp=(*v1)+p_d->a_2b_2c;
              if(k!=1)   free_energy_accumulate(&vij,coaxial_mismatch1(s_d,k,j-2,j,i,p_d)+tmp+y_d[ind(i+1,k-2,n)]);
              if(i!=n-2) free_energy_accumulate(&vij,coaxial_mismatch2(s_d,k,j-2,j,i,p_d)+tmp+y_d[ind(i+2,k-1,n)]);
            }
          }
        }
      }
      free_energy_reduce(buf.data(), &vij, team.team_rank(), NTHREAD, team);
      if(team.team_rank()==0) free_energy_accumulate(&v_d[ind(i,j,n)], vij);
    });
  Kokkos::fence();
}

static void calc_wl_kokkos(int d, int n,
    const base_t *s_d, const int *bcp_d, real_t *v_d,
    real_t *z_d, real_t *wq_d, real_t *w_d, real_t *wl_d, const struct param *p_d)
{
  Kokkos::parallel_for("calc_wl", Kokkos::RangePolicy<ExecSpace>(0,n),
    KOKKOS_LAMBDA(int i){
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if(is_exterior(i,j)&&i-j<=LOOP_MIN) return;
      real_t wqtmp=INF, wltmp=INF;
      const real_t *v1;
      if((v1=array_val(v_d,i,j,n,bcp_d))){
        real_t tmp=(*v1)+terminal_U_penalty(s_d,i,j,p_d);
        free_energy_accumulate(&wqtmp,tmp); free_energy_accumulate(&wltmp,tmp+p_d->c);
      }
      if(i!=n-1&&(v1=array_val(v_d,i+1,j,n,bcp_d))){
        real_t tmp=(*v1)+dangle_5p_energy(s_d,j,i+1,i,p_d);
        free_energy_accumulate(&wqtmp,tmp); free_energy_accumulate(&wltmp,tmp+p_d->b+p_d->c);
      }
      if(j!=0&&(v1=array_val(v_d,i,j-1,n,bcp_d))){
        real_t tmp=(*v1)+dangle_3p_energy(s_d,j-1,i,j,p_d);
        free_energy_accumulate(&wqtmp,tmp); free_energy_accumulate(&wltmp,tmp+p_d->b+p_d->c);
      }
      if(i!=n-1&&j!=0&&(v1=array_val(v_d,i+1,j-1,n,bcp_d))){
        real_t tmp=(*v1)+terminal_stack_multibranch(s_d,j-1,i+1,j,i,p_d);
        free_energy_accumulate(&wqtmp,tmp); free_energy_accumulate(&wltmp,tmp+2*p_d->b+p_d->c);
      }
      if(is_interior(i,j)) wq_d[upper_triangle_index(i,j)]=wqtmp;
      wl_d[ind(d%2,i,n)]=z_d[ind(i,j,n)]=wltmp;
      if(i!=n-1&&d>0) free_energy_accumulate(&wl_d[ind(d%2,i,n)],wl_d[ind((d-1)%2,i+1,n)]+p_d->b);
      w_d[ind(d%2,i,n)]=wl_d[ind(d%2,i,n)];
      if(j!=0&&d>0) free_energy_accumulate(&w_d[ind(d%2,i,n)],w_d[ind((d-1)%2,i,n)]+p_d->b);
    });
  Kokkos::fence();
}

static void calc_xl_kokkos(int d, int n,
    const real_t *z_d, const real_t *yl_d, real_t *xl_d)
{
  size_t scratch = ScratchView<real_t>::shmem_size(NTHREAD);
  Kokkos::parallel_for("calc_xl",
    TeamPolicy(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team){
      ScratchView<real_t> buf(team.team_scratch(0), NTHREAD);
      int i = team.league_rank();
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if(is_exterior(i,j)&&i-j<=LOOP_MIN) return;
      if(team.team_rank()==0) xl_d[ind(d%2,i,n)]=INF;
      team.team_barrier();
      if(is_interior(i,j)&&d<=2*LOOP_MIN+1) return;
      int tx=team.team_rank(), dim=team.team_size();
      real_t tmp=INF;
      for(int ktmp=i+1+tx; ktmp<jtmp-1; ktmp+=dim){
        if(ktmp!=n-1){
          const int k=wrap(ktmp,n);
          free_energy_accumulate(&tmp, z_d[ind(i,k,n)]+yl_d[ind(k+1,j,n)]);
        }
      }
      free_energy_reduce(buf.data(), &tmp, team.team_rank(), NTHREAD, team);
      if(team.team_rank()==0) free_energy_accumulate(&xl_d[ind(d%2,i,n)], tmp);
    });
  Kokkos::fence();
}

static void calc_z_kokkos(int d, int n,
    const base_t *s_d, const int *bcp_d, real_t *v_d,
    real_t *z_d, real_t *xl_d, real_t *wq_d, const struct param *p_d)
{
  size_t scratch = ScratchView<real_t>::shmem_size(NTHREAD);
  Kokkos::parallel_for("calc_z",
    TeamPolicy(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team){
      ScratchView<real_t> buf(team.team_scratch(0), NTHREAD);
      int i = team.league_rank();
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if((is_exterior(i,j)&&i-j<=LOOP_MIN)||(is_interior(i,j)&&d<=2*LOOP_MIN+1)) return;
      int tx=team.team_rank(), dim=team.team_size();
      real_t tmp1=INF, tmp2=INF;
      real_t *v1, *v2;
      for(int ktmp=i+LOOP_MIN+1+tx; ktmp<jtmp-LOOP_MIN-1; ktmp+=dim){
        const int k=wrap(ktmp,n);
        if(k==n-1) continue;
        if((v1=array_val(v_d,i,k,n,bcp_d))&&(v2=array_val(v_d,k+1,j,n,bcp_d)))
          free_energy_accumulate(&tmp1,(*v1)+(*v2)+coaxial_flush(s_d,i,k,k+1,j,p_d));
        if(j==0||k+1==n-1) continue;
        if(i!=n-1&&(v1=array_val(v_d,i+1,k,n,bcp_d))&&(v2=array_val(v_d,k+2,j,n,bcp_d)))
          free_energy_accumulate(&tmp2,(*v1)+(*v2)+coaxial_mismatch1(s_d,i+1,k,k+2,j,p_d));
        if((v1=array_val(v_d,i,k,n,bcp_d))&&(v2=array_val(v_d,k+2,j-1,n,bcp_d)))
          free_energy_accumulate(&tmp2,(*v1)+(*v2)+coaxial_mismatch2(s_d,i,k,k+2,j-1,p_d));
      }
      free_energy_reduce(buf.data(), &tmp1, team.team_rank(), NTHREAD, team);
      free_energy_reduce(buf.data(), &tmp2, team.team_rank(), NTHREAD, team);
      if(team.team_rank()==0){
        if(is_interior(i,j)) free_energy_accumulate(&wq_d[upper_triangle_index(i,j)], free_energy_sum(tmp1,tmp2));
        real_t wcoax=free_energy_sum(tmp1+2*p_d->c, tmp2+2*p_d->b+2*p_d->c);
        free_energy_accumulate(&z_d[ind(i,j,n)], wcoax);
        free_energy_accumulate(&xl_d[ind(d%2,i,n)], wcoax);
      }
    });
  Kokkos::fence();
}

static void calc_x_kokkos(int d, int n,
    real_t *yl_d, real_t *y_d, const real_t *w_d, const real_t *wl_d,
    real_t *xl_d, real_t *x_d, const struct param *p_d)
{
  Kokkos::parallel_for("calc_x", Kokkos::RangePolicy<ExecSpace>(0,n),
    KOKKOS_LAMBDA(int i){
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if(is_exterior(i,j)&&i-j<=LOOP_MIN) return;
      x_d[ind(d%5,i,n)]=INF;
      if(d>2*LOOP_MIN+1||is_exterior(i,j)){
        if(i!=n-1) free_energy_accumulate(&xl_d[ind(d%2,i,n)], xl_d[ind((d-1)%2,i+1,n)]+p_d->b);
        x_d[ind(d%5,i,n)]=xl_d[ind(d%2,i,n)];
        if(j!=0) free_energy_accumulate(&x_d[ind(d%5,i,n)], x_d[ind((d-1)%5,i,n)]+p_d->b);
      }
      yl_d[ind(i,j,n)]=free_energy_sum(wl_d[ind(d%2,i,n)],xl_d[ind(d%2,i,n)]);
      y_d[ind(i,j,n)]=free_energy_sum(w_d[ind(d%2,i,n)],x_d[ind(d%5,i,n)]);
    });
  Kokkos::fence();
}

static void calc_w5_and_w3_kokkos(int d, int n,
    real_t *w5_d, real_t *w3_d, const real_t *wq_d)
{
  // 1 team, NTHREAD threads, team reduce two values
  size_t scratch = ScratchView<real_t>::shmem_size(NTHREAD);
  Kokkos::parallel_for("calc_w5w3",
    TeamPolicy(1, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch)),
    KOKKOS_LAMBDA(const Member& team){
      ScratchView<real_t> buf(team.team_scratch(0), NTHREAD);
      int istart = team.team_rank(), iinc = team.team_size();
      real_t w5tmp=INF, w3tmp=INF;
      for(int i=istart; i+LOOP_MIN<=d; i+=iinc){
        free_energy_accumulate(&w5tmp, w5_d[i-1]+wq_d[upper_triangle_index(i,d+1)]);
        free_energy_accumulate(&w3tmp, w3_d[n-i]+wq_d[upper_triangle_index(n-d-2,n-i-1)]);
      }
      free_energy_reduce(buf.data(), &w5tmp, team.team_rank(), NTHREAD, team);
      free_energy_reduce(buf.data(), &w3tmp, team.team_rank(), NTHREAD, team);
      if(team.team_rank()==0){
        w5_d[d+1]=w5_d[d]; free_energy_accumulate(&w5_d[d+1],w5tmp);
        w3_d[n-d-2]=w3_d[n-d-1]; free_energy_accumulate(&w3_d[n-d-2],w3tmp);
      }
    });
  Kokkos::fence();
}

/* ── prna_new (Kokkos path) ─────────────────────────────────────────────── */
static prna_t prna_new(const char *s, param_t par, int quiet, int *base_cp)
{
  prna_t p = (prna_t)safe_malloc(sizeof(struct prna));
  memset(p, 0, sizeof(struct prna));
  const int n = p->n = (int)strlen(s);
  if(!quiet) printf("sequence length = %d\n", n);

  p->seq          = (base_t*)safe_malloc(n*sizeof(base_t));
  p->base_can_pair = base_cp;
  sequence_from_string(p->seq, s);
  p->v  = (real_t*)safe_malloc(n*n*sizeof(real_t));
  p->w5 = (real_t*)safe_malloc((n+1)*sizeof(real_t)) + 1;  // w5[-1] valid
  p->w3 = (real_t*)safe_malloc((n+1)*sizeof(real_t));

  /* allocate device arrays */
  Kokkos::View<struct param*> d_par("par", 1);
  {auto hm=Kokkos::create_mirror_view(d_par); hm[0]=*par; Kokkos::deep_copy(d_par,hm);}
  const struct param *p_ptr = d_par.data();

  DevView<base_t> d_s  ("s",   n);
  DevView<int>    d_bcp("bcp", n*(n-1)/2);
  DevView<real_t> d_v  ("v",   n*n);
  DevView<real_t> d_w5 ("w5",  n+1); // index [0..n], maps to w5[-1..n-1]
  DevView<real_t> d_w3 ("w3",  n+1);
  DevView<real_t> d_z  ("z",   n*n);
  DevView<real_t> d_yl ("yl",  n*n);
  DevView<real_t> d_y  ("y",   n*n);
  DevView<real_t> d_wq ("wq",  n*(n-1)/2);
  DevView<real_t> d_w  ("w",   2*n);
  DevView<real_t> d_wl ("wl",  2*n);
  DevView<real_t> d_xl ("xl",  2*n);
  DevView<real_t> d_x  ("x",   5*n);

  /* copy host arrays to device */
  {auto hm=Kokkos::create_mirror_view(d_s);
   for(int i=0;i<n;i++) hm[i]=p->seq[i]; Kokkos::deep_copy(d_s,hm);}
  {auto hm=Kokkos::create_mirror_view(d_bcp);
   for(int i=0;i<n*(n-1)/2;i++) hm[i]=base_cp[i]; Kokkos::deep_copy(d_bcp,hm);}

  /* raw device pointers for use in lambdas */
  base_t  *s_ptr   = d_s.data();
  int     *bcp_ptr = d_bcp.data();
  real_t  *v_ptr   = d_v.data();
  real_t  *w5_ptr  = d_w5.data(); // w5_ptr[0]=w5[-1], w5_ptr[1]=w5[0], etc.
  real_t  *w3_ptr  = d_w3.data();
  real_t  *z_ptr   = d_z.data();
  real_t  *yl_ptr  = d_yl.data();
  real_t  *y_ptr   = d_y.data();
  real_t  *wq_ptr  = d_wq.data();
  real_t  *w_ptr   = d_w.data();
  real_t  *wl_ptr  = d_wl.data();
  real_t  *xl_ptr  = d_xl.data();
  real_t  *x_ptr   = d_x.data();

  /* init w5/w3: w5[-1]=w5[0]=0, w3[n-1]=w3[n]=0 */
  Kokkos::parallel_for("init_w5w3", 1, KOKKOS_LAMBDA(int){
    w5_ptr[0]=w5_ptr[1]=RCONST(0.);   // w5[-1] and w5[0]
    w3_ptr[n-1]=w3_ptr[n]=RCONST(0.); // w3[n-1] and w3[n]
  });
  Kokkos::fence();

  for(int d=0; d<n-1; d++){
    calc_hairpin_stack_exterior_multibranch_kokkos(d,n,s_ptr,bcp_ptr,v_ptr,
        x_ptr, w5_ptr+1, w3_ptr, p_ptr);
    calc_internal_kokkos(d,n,s_ptr,bcp_ptr,v_ptr,p_ptr);
    calc_coaxial_kokkos(d,n,s_ptr,bcp_ptr,v_ptr,y_ptr,w5_ptr+1,w3_ptr,p_ptr);
    calc_wl_kokkos(d,n,s_ptr,bcp_ptr,v_ptr,z_ptr,wq_ptr,w_ptr,wl_ptr,p_ptr);
    calc_xl_kokkos(d,n,z_ptr,yl_ptr,xl_ptr);
    calc_z_kokkos(d,n,s_ptr,bcp_ptr,v_ptr,z_ptr,xl_ptr,wq_ptr,p_ptr);
    calc_x_kokkos(d,n,yl_ptr,y_ptr,w_ptr,wl_ptr,xl_ptr,x_ptr,p_ptr);
    calc_w5_and_w3_kokkos(d,n,w5_ptr+1,w3_ptr,wq_ptr);
  }

  /* copy v, w5, w3 back to host */
  {auto hm=Kokkos::create_mirror_view(d_v); Kokkos::deep_copy(hm,d_v);
   memcpy(p->v,hm.data(),n*n*sizeof(real_t));}
  {auto hm=Kokkos::create_mirror_view(d_w5); Kokkos::deep_copy(hm,d_w5);
   memcpy(p->w5-1,hm.data(),(n+1)*sizeof(real_t));}  // w5[-1..n-1]
  {auto hm=Kokkos::create_mirror_view(d_w3); Kokkos::deep_copy(hm,d_w3);
   memcpy(p->w3,hm.data(),(n+1)*sizeof(real_t));}

  return p;
}

static void prna_delete(prna_t p){
  if(p){ free(p->seq); free(p->v); free(p->w5-1); free(p->w3); free(p); }
}

/* host-only post-processing helpers */
static int can_pair_host(int i,int j,int n,const int *bcp){return can_pair(i,j,n,bcp);}

static real_t free_energy_of_pair(const prna_t p,int i,int j){
  const int n=p->n; const int *bcp=p->base_can_pair;
  if(can_pair_host(i,j,n,bcp))
    return *array_val(p->v,i,j,n,bcp)+*array_val(p->v,j,i,n,bcp)-p->w3[0];
  return INF;
}

static void prna_write_probknot(const prna_t p,const char *fn,const char *s,int min_helix_length){
  const int n=p->n;
  int *pair=(int*)safe_malloc(n*sizeof(int));
  for(int i=0;i<n;i++){
    pair[i]=i;
    for(int j=0;j<n;j++)
      if(free_energy_of_pair(p,i,j)<free_energy_of_pair(p,i,pair[i])) pair[i]=j;
  }
  for(int i=0;i<n;i++) if(pair[pair[i]]!=i) pair[i]=i;
  FILE *f = fn ? safe_fopen(fn,"w") : stdout;
  char fmt[32]; sprintf(fmt,"%d",n); int ns=(int)strlen(fmt)+1; if(ns<5)ns=5; sprintf(fmt,"%%%dd",ns);
  for(int i=0;i<n;i++){
    fprintf(f,fmt,i+1); fprintf(f,"%2c   ",s[i]);
    fprintf(f,fmt,i); fprintf(f,fmt,i==n-1?0:i+2);
    fprintf(f,fmt,pair[i]==i?0:pair[i]+1); fprintf(f,fmt,i+1); fprintf(f,"\n");
  }
  if(fn) fclose(f);
  free(pair);
}

/* host generate_bcp */
static int *generate_bcp(const char *s){
  int length=(int)strlen(s);
  int *base_cp=(int*)safe_malloc((length*(length-1)/2)*sizeof(int));
  base_t *seq=(base_t*)safe_malloc(length*sizeof(base_t));
  sequence_from_string(seq,s);
  for(int i=0;i<length;i++)
    for(int j=i+1;j<length;j++){
      if(j-i<LOOP_MIN+1||!isupper(s[i])||!isupper(s[j]))
        base_cp[(j*(j-1))/2+i]=0;
      else
        base_cp[upper_triangle_index(i,j)]=
          is_canonical_pair(seq[i],seq[j])&&
          ((i>0&&j<length-1&&is_canonical_pair(seq[i-1],seq[j+1]))||
           (j-i>=LOOP_MIN+3&&is_canonical_pair(seq[i+1],seq[j-1])));
    }
  free(seq);
  return base_cp;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
  if(argc<2){ fprintf(stderr,"Usage: %s <sequence_or_file> [-b param_binary]\n",argv[0]); return 1; }

  const char *binary_param_file = nullptr;
  const char *seq_arg = nullptr;
  int probknot_stdout = 1;
  const char *probknot_file = nullptr;

  /* simple arg parsing */
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-b")&&i+1<argc){ binary_param_file=argv[++i]; }
    else if(!strcmp(argv[i],"-p")&&i+1<argc){ probknot_file=argv[++i]; probknot_stdout=0; }
    else if(argv[i][0]!='-') seq_arg=argv[i];
  }
  if(!seq_arg){ fprintf(stderr,"No sequence provided\n"); return 1; }

  char *seq = sequence(seq_arg);

  struct param par;
  if(binary_param_file){
    param_read_from_binary(binary_param_file, &par);
  } else {
    /* check DATAPATH env var; if not set, zero-init (demo mode) */
    const char *path = getenv("DATAPATH");
    if(path){
      /* minimal text param read would go here; for now use zero init */
      param_init_zero(&par);
    } else {
      fprintf(stderr,"Warning: DATAPATH not set; using zeroed parameters (demo mode)\n");
      param_init_zero(&par);
    }
  }

  Kokkos::initialize(argc, argv);
  {
    int *bcp = generate_bcp(seq);
    prna_t p = prna_new(seq, &par, 0, bcp);

    if(probknot_stdout)
      prna_write_probknot(p, nullptr, seq, 3);
    else
      prna_write_probknot(p, probknot_file, seq, 3);

    prna_delete(p);
    free(bcp);
  }
  Kokkos::finalize();
  free(seq);
  return 0;
}
