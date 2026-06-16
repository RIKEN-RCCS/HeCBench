// frna - RNA folding benchmark, Kokkos port
// Combines frna.cpp and main.c into a single C++ translation unit.

#include <Kokkos_Core.hpp>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>

// ---- Redefine macros for Kokkos device functions ----
// DEV marks functions callable from device (KOKKOS_FUNCTION)
// HOST marks host-only functions
#define DEV      KOKKOS_FUNCTION
#define HOST
#define GLOBAL   static
#define CU(x)    x

// Now include the C headers (they will pick up DEV/HOST macros)
extern "C" {
#include "int.h"
#include "fbase.h"
#include "fparam.h"
#include "util.h"
}

// ---- frna struct definition ----
struct frna {
  int    n;
  fbase_t* seq;
  int_t*   v;
  int_t*   w;
  int_t*   wm;
  int_t*   wca;
  int_t*   w5;   // indexed from 0, but logically 1-based (w5[-1] valid via extra alloc)
  int_t*   w3;
};
typedef struct frna* frna_t;

// ============================================================
// Kokkos type aliases
// ============================================================
using team_policy_t = Kokkos::TeamPolicy<>;
using team_member_t = team_policy_t::member_type;
using ScratchSpace  = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using ScratchIntView = Kokkos::View<int_t*, ScratchSpace, Kokkos::MemoryUnmanaged>;

#define NTHREAD     256
#define SQRT_NTHREAD 16

// ============================================================
// Device helper functions (from frna.cpp, DEV-annotated)
// ============================================================
DEV static int_t terminal_U_penalty(const fbase_t* s, int i, int j, const fparam* p) {
  return (s[i] == U || s[j] == U) ? p->terminal_AU_penalty : 0;
}
DEV static int_t dangle_3p_energy(const fbase_t* s, int i, int j, int ip1, const fparam* p) {
  return p->dangle_3p[s[i]][s[j]][s[ip1]] + terminal_U_penalty(s,i,j,p);
}
DEV static int_t dangle_5p_energy(const fbase_t* s, int i, int j, int jm1, const fparam* p) {
  return p->dangle_5p[s[i]][s[j]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}
DEV static int_t terminal_stack(const fbase_t* s, int i, int j, int ip1, int jm1, const fparam* p) {
  return p->tstack[s[i]][s[j]][s[ip1]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}
DEV static int_t terminal_stack_multibranch(const fbase_t* s, int i, int j, int ip1, int jm1, const fparam* p) {
  return p->tstackm[s[i]][s[j]][s[ip1]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}
DEV static const int_t* lookup_find(const fbase_t* s, int d, const fparam* p) {
  int i;
  switch (d) {
    case 3: for (i=0;i<p->ntriloop;i++) if (sequences_match(s,p->triloop[i].seq,d+2)) return &p->triloop[i].val; break;
    case 4: for (i=0;i<p->ntloop;i++)   if (sequences_match(s,p->tloop[i].seq,d+2))   return &p->tloop[i].val;   break;
    case 6: for (i=0;i<p->nhexaloop;i++) if (sequences_match(s,p->hexaloop[i].seq,d+2)) return &p->hexaloop[i].val; break;
  }
  return 0;
}
DEV static int_t hairpin_loop_energy(const fbase_t* s, int i, int j, int d, const fparam* p) {
  const int_t* val;
  if ((val = lookup_find(&s[i],d,p))) return *val;
  int_t e;
  if (d > LOOP_MAX) e=(int_t)(p->hairpin_loop_initiation[LOOP_MAX]+p->prelog*LOG((float)d/LOOP_MAX));
  else e = p->hairpin_loop_initiation[d];
  if (d == 3) {
    if (contains_only_base(C,d,&s[i+1])) e += p->c_hairpin_of_3;
    e += terminal_U_penalty(s,i,j,p);
  } else {
    e += p->tstackh[s[i]][s[j]][s[i+1]][s[j-1]];
    if (contains_only_base(C,d,&s[i+1])) e += p->c_hairpin_slope*d + p->c_hairpin_intercept;
  }
  if (s[i]==G && s[j]==U && i>1 && s[i-1]==G && s[i-2]==G) e += p->bonus_for_GGG_hairpin;
  return e;
}
DEV static int_t real_min(int_t a, int_t b) { return a < b ? a : b; }
DEV static int_t alternative_bulge_loop_correction(int n, const fbase_t* s, int i, int ip) {
  int count=1, k;
  if (i!=n-1) {
    k=i; while(k>=0&&s[k]==s[i+1]){count++;k--;}
    k=ip; while(k<=n-1&&s[k]==s[i+1]){count++;k++;}
  }
  return (int_t)(-1.0f*RT*conversion_factor*log((float)count));
}
DEV static int_t internal_loop_energy(const fbase_t* s, int n, int i, int j, int ip, int jp, int d1, int d2, const fparam* p) {
  if (d1==0||d2==0) {
    int_t e=p->bulge_loop_initiation[d1+d2];
    if (d1==1||d2==1) {
      e += p->stack[s[i]][s[j]][s[ip]][s[jp]];
      if (d1==0) e += alternative_bulge_loop_correction(n,s,jp,j);
      else       e += alternative_bulge_loop_correction(n,s,i,ip);
      if ((d1==1&&s[i+1]==C&&(s[i]==C||s[i+2]==C))||(d2==1&&s[j-1]==C&&(s[j]==C||s[j-2]==C)))
        e += p->Bonus_for_Single_C_bulges_adjacent_to_C;
    } else { e+=terminal_U_penalty(s,i,j,p); e+=terminal_U_penalty(s,ip,jp,p); }
    return e;
  }
  if (d1==1&&d2==1) return p->int11[s[i]][s[i+1]][s[i+2]][s[j-2]][s[j-1]][s[j]];
  if (d1==2&&d2==2) return p->int22[s[i]][s[ip]][s[j]][s[jp]][s[i+1]][s[i+2]][s[j-1]][s[j-2]];
  if (d1==1&&d2==2) return p->int21[s[i]][s[j]][s[i+1]][s[j-1]][s[jp+1]][s[ip]][s[jp]];
  if (d1==2&&d2==1) return p->int21[s[jp]][s[ip]][s[jp+1]][s[ip-1]][s[i+1]][s[j]][s[i]];
  const int_t (*sp)[NBASE][NBASE][NBASE];
  if (d1==1||d2==1) sp=&p->tstacki1n;
  else if ((d1==2&&d2==3)||(d1==3&&d2==2)) sp=&p->tstacki23;
  else sp=&p->tstacki;
  return p->internal_loop_initiation[d1+d2]+real_min(p->fm_array_first_element*abs(d1-d2),p->maximum_correction)+
         (*sp)[s[i]][s[j]][s[i+1]][s[j-1]]+(*sp)[s[jp]][s[ip]][s[jp+1]][s[ip-1]];
}
DEV static int_t coaxial_flush(const fbase_t* s,int i,int j,int ip,int jp,const fparam* p) {
  return terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+p->coaxial[s[i]][s[j]][s[ip]][s[jp]];
}
DEV static int_t coaxial_mismatch1(const fbase_t* s,int i,int j,int ip,int jp,const fparam* p) {
  return terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+
         p->tstackcoax[s[j]][s[i]][s[j+1]][s[i-1]]+p->coaxstack[s[j+1]][s[i-1]][s[ip]][s[jp]];
}
DEV static int_t coaxial_mismatch2(const fbase_t* s,int i,int j,int ip,int jp,const fparam* p) {
  return terminal_U_penalty(s,i,j,p)+terminal_U_penalty(s,ip,jp,p)+
         p->tstackcoax[s[jp]][s[ip]][s[jp+1]][s[ip-1]]+p->coaxstack[s[j]][s[i]][s[j+1]][s[jp+1]];
}
DEV static void free_energy_min(int_t* a, int_t b) { if (*a>b) *a=b; }
DEV static int int_min(int a, int b) { return a<b?a:b; }
DEV static int_t int_t_min(int_t a, int_t b) { return a<b?a:b; }
DEV static int ind(int i, int j, int n) { return i*n+j; }
DEV static int cp(int i, int j, const fbase_t* s) { return j-i-1>=LOOP_MIN&&is_canonical_pair(s[i],s[j]); }
DEV static int can_pair(int i, int j, int n, const fbase_t* s) {
  if (j<i){int t=i;i=j;j=t;}
  return cp(i,j,s)&&((i>0&&j<n-1&&cp(i-1,j+1,s))||cp(i+1,j-1,s));
}
DEV static int not_isolated(int i, int j, int n, const fbase_t* s) {
  if (j<i){int t=i;i=j;j=t;}
  return is_canonical_pair(s[i],s[j])&&((i>0&&j<n-1&&cp(i-1,j+1,s))||cp(i+1,j-1,s));
}
DEV static int wrap(int i, int n) { return i>=n?i-n:i; }
DEV static int is_exterior(int i, int j) { return j<i; }
DEV static int is_interior(int i, int j) { return i<j; }
DEV static int_t* array_val(int_t* a, int i, int j, int n, const fbase_t* s) {
  return can_pair(i,j,n,s)?&a[ind(i,j,n)]:0;
}

// Team-level reduction helper
KOKKOS_INLINE_FUNCTION
static void free_energy_min_reduce(int_t* buf, int_t* x, int tid, int nt,
                                    const team_member_t& team) {
  buf[tid] = *x;
  team.team_barrier();
  for (int step = nt/2; step > 0; step /= 2) {
    if (tid < step) free_energy_min(&buf[tid], buf[tid+step]);
    team.team_barrier();
  }
  if (tid == 0) *x = buf[0];
  team.team_barrier();
}

// ============================================================
// Kernel implementations
// ============================================================

static void init_w5_and_w3(int n, int_t* w5, int_t* w3) {
  Kokkos::parallel_for("init_w5w3", Kokkos::RangePolicy<>(0, n+1),
    KOKKOS_LAMBDA(int i) { w5[i]=0; w3[i]=0; });
}

static void calc_V_hairpin_and_V_stack(int d, int n,
    const fbase_t* s, int_t* v, const fparam* p) {
  Kokkos::parallel_for("calc_V_hs", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) {
      const int jtmp = i+d+1;
      const int j    = wrap(jtmp,n);
      if ((is_interior(i,j)&&!can_pair(i,j,n,s))||(is_exterior(i,j)&&!is_canonical_pair(s[i],s[j]))) {
        v[ind(i,j,n)]=INF; return;
      }
      int_t vij=INF;
      if (i!=n-1&&j!=0) {
        if (is_interior(i,j)) vij=hairpin_loop_energy(s,i,j,d,p);
        if (can_pair(i+1,j-1,n,s)&&!(is_interior(i,j)&&d<=LOOP_MIN-2))
          free_energy_min(&vij, p->stack[s[i]][s[j]][s[i+1]][s[j-1]]+v[ind(i+1,j-1,n)]);
      }
      v[ind(i,j,n)]=vij;
    });
}

static void calc_V_bulge_internal(int d, int n,
    const fbase_t* s, int_t* v, const fparam* p) {
  const size_t scratch_bytes = NTHREAD * sizeof(int_t);
  Kokkos::parallel_for("calc_Vbi",
    team_policy_t(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchIntView buf(team.team_scratch(0), NTHREAD);
      const int i    = team.league_rank();
      const int jtmp = i+d+1;
      const int j    = wrap(jtmp,n);
      if ((is_exterior(i,j)&&i-j<=LOOP_MIN)||(is_interior(i,j)&&d<=LOOP_MIN+2)||!can_pair(i,j,n,s)) return;
      const int tid  = team.team_rank();
      const int tx   = tid % SQRT_NTHREAD;
      const int ty   = tid / SQRT_NTHREAD;
      int_t vij = INF;
      const int dmax  = int_min(LOOP_MAX,d-2);
      const int d1max = int_min(dmax, n-i-2);
      for (int d1 = tx; d1 <= d1max; d1 += SQRT_NTHREAD) {
        const int ip    = i+d1+1;
        const int d2max = int_min(dmax-d1, j-1);
        const int d2start = (d1>0)?ty:ty+1;
        for (int d2 = d2start; d2 <= d2max; d2 += SQRT_NTHREAD) {
          const int jp = j-d2-1;
          if (can_pair(ip,jp,n,s))
            free_energy_min(&vij, internal_loop_energy(s,n,i,j,ip,jp,d1,d2,p)+v[ind(ip,jp,n)]);
        }
      }
      free_energy_min_reduce(buf.data(), &vij, tid, NTHREAD, team);
      if (tid != 0) return;
      free_energy_min(&v[ind(i,j,n)], vij);
    });
}

static void calc_V_multibranch(int d, int n,
    const fbase_t* s, int_t* v, const int_t* wm, const fparam* p) {
  Kokkos::parallel_for("calc_Vmb", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) {
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if ((is_exterior(i,j)&&i-j<=LOOP_MIN)||!can_pair(i,j,n,s)) return;
      int_t vij=INF;
      if (d>2*LOOP_MIN+3&&i!=n-1&&j!=0) {
        free_energy_min(&vij, wm[ind(i+1,j-1,n)]+terminal_U_penalty(s,i,j,p)+p->a+p->c);
        if (i!=n-2) free_energy_min(&vij, wm[ind(i+2,j-1,n)]+dangle_3p_energy(s,i,j,i+1,p)+p->a+p->b+p->c);
        if (j!=1)   free_energy_min(&vij, wm[ind(i+1,j-2,n)]+dangle_5p_energy(s,i,j,j-1,p)+p->a+p->b+p->c);
        if (i!=n-2&&j!=1) free_energy_min(&vij, wm[ind(i+2,j-2,n)]+terminal_stack_multibranch(s,i,j,i+1,j-1,p)+p->a+2*p->b+p->c);
      }
      free_energy_min(&v[ind(i,j,n)], vij);
    });
}

static void calc_V_exterior(int d, int n,
    const fbase_t* s, int_t* v, const int_t* w5, const int_t* w3, const fparam* p) {
  Kokkos::parallel_for("calc_Vext", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) {
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      if (is_interior(i,j)) return;
      int_t vij=INF;
      if (is_canonical_pair(s[i],s[j])&&not_isolated(i,j,n,s)) {
        free_energy_min(&vij, w3[i+1]+w5[j-1]+terminal_U_penalty(s,i,j,p));
        if (i!=n-1) free_energy_min(&vij, w3[i+2]+w5[j-1]+dangle_3p_energy(s,i,j,i+1,p));
        if (j!=0)   free_energy_min(&vij, w3[i+1]+w5[j-2]+dangle_5p_energy(s,i,j,j-1,p));
        if (i!=n-1&&j!=0) free_energy_min(&vij, w3[i+2]+w5[j-2]+terminal_stack(s,i,j,i+1,j-1,p));
      }
      free_energy_min(&v[ind(i,j,n)], vij);
    });
}

static void calc_W(int d, int n,
    const fbase_t* s, int_t* v, int_t* w, const fparam* p) {
  Kokkos::parallel_for("calc_W", Kokkos::RangePolicy<>(0, n),
    KOKKOS_LAMBDA(int i) {
      const int jtmp=i+d+1; const int j=wrap(jtmp,n);
      int_t wij=INF; int_t* v_tmp;
      if (d>0) {
        if (i!=n-1) free_energy_min(&wij, w[ind(i+1,j,n)]+p->b);
        if (j!=0)   free_energy_min(&wij, w[ind(i,j-1,n)]+p->b);
      }
      if (is_interior(i,j)&&d>LOOP_MIN-1) {
        v_tmp=array_val(v,i,j,n,s);     free_energy_min(&wij,(v_tmp?*v_tmp:INF)+terminal_U_penalty(s,i,j,p)+p->c);
        if (j!=0){v_tmp=array_val(v,i,j-1,n,s);  free_energy_min(&wij,(v_tmp?*v_tmp:INF)+dangle_3p_energy(s,j-1,i,j,p)+p->b+p->c);}
        if (i!=n-1){v_tmp=array_val(v,i+1,j,n,s);free_energy_min(&wij,(v_tmp?*v_tmp:INF)+dangle_5p_energy(s,j,i+1,i,p)+p->b+p->c);}
        if (i!=n-1&&j!=0){v_tmp=array_val(v,i+1,j-1,n,s);free_energy_min(&wij,(v_tmp?*v_tmp:INF)+terminal_stack_multibranch(s,j-1,i+1,j,i,p)+2*p->b+p->c);}
      }
      if (is_exterior(i,j)) {
        free_energy_min(&wij, v[ind(i,j,n)]+terminal_U_penalty(s,i,j,p)+p->c);
        if (j!=0)        free_energy_min(&wij, v[ind(i,j-1,n)]+dangle_3p_energy(s,j-1,i,j,p)+p->b+p->c);
        if (i!=n-1)      free_energy_min(&wij, v[ind(i+1,j,n)]+dangle_5p_energy(s,j,i+1,i,p)+p->b+p->c);
        if (i!=n-1&&j!=0)free_energy_min(&wij, v[ind(i+1,j-1,n)]+terminal_stack_multibranch(s,j-1,i+1,j,i,p)+2*p->b+p->c);
      }
      w[ind(i,j,n)]=wij;
    });
}

static void calc_WM(int d, int n,
    const fbase_t* s, int_t* w, int_t* wm, const fparam* p) {
  const size_t scratch_bytes = NTHREAD * sizeof(int_t);
  Kokkos::parallel_for("calc_WM",
    team_policy_t(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchIntView buf(team.team_scratch(0), NTHREAD);
      const int i    = team.league_rank();
      const int jtmp = i+d+1;
      const int j    = wrap(jtmp,n);
      const int tx   = team.team_rank();
      const int kinc = team.team_size();
      if (is_interior(i,j)&&(j-i-1<=2*LOOP_MIN+2)) {
        if (tx==0) wm[ind(i,j,n)]=INF;
        return;
      }
      int_t tmp=INF;
      for (int ktmp = i+tx; ktmp < jtmp; ktmp += kinc) {
        if (ktmp!=n-1) { const int k=wrap(ktmp,n); free_energy_min(&tmp,w[ind(i,k,n)]+w[ind(k+1,j,n)]); }
      }
      if (d>0) {
        if (i!=n-1) free_energy_min(&tmp,wm[ind(i+1,j,n)]+p->b);
        if (j!=0)   free_energy_min(&tmp,wm[ind(i,j-1,n)]+p->b);
      }
      free_energy_min_reduce(buf.data(),&tmp,tx,kinc,team);
      if (tx!=0) return;
      wm[ind(i,j,n)]=tmp;
      free_energy_min(&w[ind(i,j,n)],tmp);
    });
}

static void calc_coaxial(int d, int n,
    const fbase_t* s, int_t* v, const int_t* w,
    const int_t* w5, const int_t* w3, const fparam* p) {
  const size_t scratch_bytes = NTHREAD * sizeof(int_t);
  Kokkos::parallel_for("calc_coaxial",
    team_policy_t(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchIntView buf(team.team_scratch(0), NTHREAD);
      const int i    = team.league_rank();
      const int jtmp = i+d+1;
      const int j    = wrap(jtmp,n);
      if ((is_exterior(i,j)&&i-j<=LOOP_MIN)||!can_pair(i,j,n,s)) return;
      const int_t* v1;
      int_t vij=INF;
      const int tx   = team.team_rank();
      const int kinc = team.team_size();
      if (is_exterior(i,j)) {
        for (int k = tx; k < j-LOOP_MIN; k += kinc) {
          if ((v1=array_val(v,k,j-1,n,s)))   free_energy_min(&vij,w3[i+1]+w5[k-1]+coaxial_flush(s,k,j-1,j,i,p)+*v1);
          if (j-2>=0) {
            if (i<n-1&&(v1=array_val(v,k,j-2,n,s)))   free_energy_min(&vij,w3[i+2]+w5[k-1]+coaxial_mismatch2(s,k,j-2,j,i,p)+*v1);
            if ((v1=array_val(v,k+1,j-2,n,s)))         free_energy_min(&vij,w3[i+1]+w5[k-1]+coaxial_mismatch1(s,k+1,j-2,j,i,p)+*v1);
          }
        }
        for (int k = i+LOOP_MIN+1+tx; k < n; k += kinc) {
          if ((v1=array_val(v,i+1,k,n,s)))   free_energy_min(&vij,w3[k+1]+w5[j-1]+coaxial_flush(s,j,i,i+1,k,p)+*v1);
          if (j>0&&(v1=array_val(v,i+2,k,n,s)))  free_energy_min(&vij,w3[k+1]+w5[j-2]+coaxial_mismatch1(s,j,i,i+2,k,p)+*v1);
          if ((v1=array_val(v,i+2,k-1,n,s))) free_energy_min(&vij,w3[k+1]+w5[j-1]+coaxial_mismatch2(s,j,i,i+2,k-1,p)+*v1);
        }
      }
      if (d>2*LOOP_MIN+3&&i!=n-1&&j!=0) {
        for (int ktmp = i+2+tx; ktmp < jtmp-2; ktmp += kinc) {
          const int k=wrap(ktmp,n);
          if (k!=n-1) {
            if ((v1=array_val(v,i+1,k,n,s))) free_energy_min(&vij,coaxial_flush(s,j,i,i+1,k,p)+*v1+p->a_2c+w[ind(k+1,j-1,n)]);
            if (ktmp+2<jtmp-1&&i+1!=n-1&&k+1!=n-1&&(v1=array_val(v,i+2,k,n,s))) {
              int_t tmp2=*v1+p->a_2b_2c;
              free_energy_min(&vij,coaxial_mismatch2(s,j,i,i+2,k,p)+tmp2+w[ind(k+2,j-1,n)]);
              if (j!=1) free_energy_min(&vij,coaxial_mismatch1(s,j,i,i+2,k,p)+tmp2+w[ind(k+1,j-2,n)]);
            }
          }
        }
        for (int ktmp = i+3+tx; ktmp < jtmp-1; ktmp += kinc) {
          const int k=wrap(ktmp,n);
          if (k!=0) {
            if ((v1=array_val(v,k,j-1,n,s))) free_energy_min(&vij,coaxial_flush(s,k,j-1,j,i,p)+*v1+p->a_2c+w[ind(i+1,k-1,n)]);
            if (j!=1&&ktmp>i+3&&(v1=array_val(v,k,j-2,n,s))) {
              int_t tmp2=*v1+p->a_2b_2c;
              if (k!=1) free_energy_min(&vij,coaxial_mismatch1(s,k,j-2,j,i,p)+tmp2+w[ind(i+1,k-2,n)]);
              if (i!=n-2) free_energy_min(&vij,coaxial_mismatch2(s,k,j-2,j,i,p)+tmp2+w[ind(i+2,k-1,n)]);
            }
          }
        }
      }
      free_energy_min_reduce(buf.data(),&vij,tx,kinc,team);
      if (tx!=0) return;
      free_energy_min(&v[ind(i,j,n)],vij);
    });
}

static void calc_wl_coax(int d, int n,
    const fbase_t* s, int_t* v, int_t* w, int_t* wm,
    const fparam* p, int_t* wca) {
  const size_t scratch_bytes = NTHREAD * sizeof(int_t);
  Kokkos::parallel_for("calc_wlcoax",
    team_policy_t(n, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchIntView buf(team.team_scratch(0), NTHREAD);
      const int i    = team.league_rank();
      const int jtmp = i+d+1;
      const int j    = wrap(jtmp,n);
      if ((is_exterior(i,j)&&i-j<=LOOP_MIN)||(is_interior(i,j)&&d<=2*LOOP_MIN+1)) return;
      const int tx   = team.team_rank();
      const int kinc = team.team_size();
      int_t tmp1=INF, tmp2=INF;
      for (int ktmp = i+LOOP_MIN+1+tx; ktmp < jtmp-LOOP_MIN-1; ktmp += kinc) {
        const int k=wrap(ktmp,n); if (k==n-1) continue;
        int_t *v1, *v2;
        if ((v1=array_val(v,i,k,n,s))&&(v2=array_val(v,k+1,j,n,s)))
          free_energy_min(&tmp1,*v1+*v2+coaxial_flush(s,i,k,k+1,j,p));
        if (j==0||k+1==n-1) continue;
        if (i!=n-1&&(v1=array_val(v,i+1,k,n,s))&&(v2=array_val(v,k+2,j,n,s)))
          free_energy_min(&tmp2,*v1+*v2+coaxial_mismatch1(s,i+1,k,k+2,j,p));
        if ((v1=array_val(v,i,k,n,s))&&(v2=array_val(v,k+2,j-1,n,s)))
          free_energy_min(&tmp2,*v1+*v2+coaxial_mismatch2(s,i,k,k+2,j-1,p));
      }
      free_energy_min_reduce(buf.data(),&tmp1,tx,kinc,team);
      free_energy_min_reduce(buf.data(),&tmp2,tx,kinc,team);
      if (tx!=0) return;
      wca[ind(i,j,n)]=int_t_min(tmp1,tmp2);
      free_energy_min(&wm[ind(i,j,n)],tmp1+2*p->c);
      free_energy_min(&wm[ind(i,j,n)],tmp2+2*p->b+2*p->c);
      free_energy_min(&w[ind(i,j,n)],wm[ind(i,j,n)]);
    });
}

static void calc_w5_and_w3(int d, int n,
    const fbase_t* s, int_t* v, int_t* w5, int_t* w3,
    const fparam* p, const int_t* wca) {
  const size_t scratch_bytes = NTHREAD * sizeof(int_t);
  Kokkos::parallel_for("calc_w5w3",
    team_policy_t(1, NTHREAD).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchIntView buf(team.team_scratch(0), NTHREAD);
      const int tx    = team.team_rank();
      const int iinc  = team.team_size();
      int_t w5tmp=0, w3tmp=0;
      int_t* v_temp;
      for (int i = tx; i <= d-LOOP_MIN; i += iinc) {
        if ((v_temp=array_val(v,i,d+1,n,s)))
          free_energy_min(&w5tmp,w5[i-1]+*v_temp+terminal_U_penalty(s,d+1,i,p));
        if (d-i>LOOP_MIN) {
          if ((v_temp=array_val(v,i,d,n,s)))
            free_energy_min(&w5tmp,w5[i-1]+*v_temp+dangle_3p_energy(s,d,i,d+1,p));
          if ((v_temp=array_val(v,i+1,d+1,n,s)))
            free_energy_min(&w5tmp,w5[i-1]+*v_temp+dangle_5p_energy(s,d+1,i+1,i,p));
          free_energy_min(&w5tmp,w5[i-1]+wca[ind(i,d+1,n)]);
        }
        if ((d-i>LOOP_MIN+1)&&(v_temp=array_val(v,i+1,d,n,s)))
          free_energy_min(&w5tmp,w5[i-1]+*v_temp+terminal_stack(s,d,i+1,d+1,i,p));

        if ((v_temp=array_val(v,n-d-2,n-i-1,n,s)))
          free_energy_min(&w3tmp,w3[n-i]+*v_temp+terminal_U_penalty(s,n-i-1,n-d-2,p));
        if ((v_temp=array_val(v,n-d-2,n-i-2,n,s)))
          free_energy_min(&w3tmp,w3[n-i]+*v_temp+dangle_3p_energy(s,n-i-2,n-d-2,n-i-1,p));
        if (n-d-1!=0&&(v_temp=array_val(v,n-d-1,n-i-1,n,s)))
          free_energy_min(&w3tmp,w3[n-i]+*v_temp+dangle_5p_energy(s,n-i-1,n-d-1,n-d-2,p));
        if (n-i-2!=n-1&&n-d-1!=0&&(v_temp=array_val(v,n-d-1,n-i-2,n,s)))
          free_energy_min(&w3tmp,w3[n-i]+*v_temp+terminal_stack(s,n-i-2,n-d-1,n-i-1,n-d-2,p));
        free_energy_min(&w3tmp,w3[n-i]+wca[ind(n-d-2,n-i-1,n)]);
      }
      free_energy_min_reduce(buf.data(),&w5tmp,tx,iinc,team);
      free_energy_min_reduce(buf.data(),&w3tmp,tx,iinc,team);
      if (tx==0) {
        w5[d+1]=w5[d]; w3[n-d-2]=w3[n-d-1];
        free_energy_min(&w5[d+1],w5tmp);
        free_energy_min(&w3[n-d-2],w3tmp);
      }
    });
}

// ============================================================
// frna_new  (Kokkos path)
// ============================================================
static void host_initialize(int_t* arr, size_t sz) {
  for (size_t i=0;i<sz;i++) arr[i]=INF;
}

frna_t frna_new(const char* str, fparam_t par) {
  frna_t p = (frna_t)safe_malloc(sizeof(struct frna));
  memset(p, 0, sizeof(struct frna));
  const int n = p->n = (int)strlen(str);
  p->seq  = fsequence_from_string(str);
  p->v    = (int_t*)safe_malloc((size_t)n*n*sizeof(int_t));
  p->w    = (int_t*)safe_malloc((size_t)n*n*sizeof(int_t));
  p->wm   = (int_t*)safe_malloc((size_t)n*n*sizeof(int_t));
  p->wca  = (int_t*)safe_malloc((size_t)n*n*sizeof(int_t));
  // w5 needs one extra element before index 0 (for w5[-1] access)
  p->w5   = (int_t*)safe_malloc((size_t)(n+2)*sizeof(int_t)) + 1;
  p->w3   = (int_t*)safe_malloc((size_t)(n+1)*sizeof(int_t));
  host_initialize(p->v,   (size_t)n*n);
  host_initialize(p->w,   (size_t)n*n);
  host_initialize(p->wm,  (size_t)n*n);
  host_initialize(p->wca, (size_t)n*n);

  // ---- Device allocations ----
  Kokkos::View<int_t*>    d_v("v",   (size_t)n*n);
  Kokkos::View<int_t*>    d_w("w",   (size_t)n*n);
  Kokkos::View<int_t*>    d_wm("wm", (size_t)n*n);
  Kokkos::View<int_t*>    d_wca("wca",(size_t)n*n);
  // w5: size n+2, use pointer+1 for 1-based access (w5[-1] valid)
  Kokkos::View<int_t*>    d_w5_store("w5_store", (size_t)n+2);
  Kokkos::View<int_t*>    d_w3("w3", (size_t)n+1);
  Kokkos::View<fbase_t*>  d_s("s",   (size_t)n);
  Kokkos::View<fparam*>   d_par("par",1);

  // Copy initial arrays (all INF) to device
  {
    auto hv = Kokkos::create_mirror_view(d_v);
    for (int i=0;i<n*n;i++) hv(i)=INF;
    Kokkos::deep_copy(d_v, hv);
    auto hw = Kokkos::create_mirror_view(d_w);
    for (int i=0;i<n*n;i++) hw(i)=INF;
    Kokkos::deep_copy(d_w, hw);
    auto hwm = Kokkos::create_mirror_view(d_wm);
    for (int i=0;i<n*n;i++) hwm(i)=INF;
    Kokkos::deep_copy(d_wm, hwm);
    auto hwca = Kokkos::create_mirror_view(d_wca);
    for (int i=0;i<n*n;i++) hwca(i)=INF;
    Kokkos::deep_copy(d_wca, hwca);
  }
  {
    auto hs = Kokkos::create_mirror_view(d_s);
    for (int i=0;i<n;i++) hs(i)=p->seq[i];
    Kokkos::deep_copy(d_s, hs);
  }
  {
    Kokkos::View<fparam*, Kokkos::HostSpace> h_par_v(par, 1);
    Kokkos::deep_copy(d_par, h_par_v);
  }

  // Raw device pointers
  int_t*   v   = d_v.data();
  int_t*   w   = d_w.data();
  int_t*   wm  = d_wm.data();
  int_t*   wca = d_wca.data();
  int_t*   w5_base = d_w5_store.data(); // w5_base[0..n+1]
  int_t*   w5  = w5_base + 1;           // w5[-1] = w5_base[0] is valid
  int_t*   w3  = d_w3.data();
  fbase_t* s   = d_s.data();
  fparam*  pm  = d_par.data();

  auto t_start = std::chrono::steady_clock::now();

  init_w5_and_w3(n, w5_base, w3);

  for (int d = 0; d < n-1; d++) {
    calc_V_hairpin_and_V_stack(d, n, s, v, pm);
    calc_V_bulge_internal(d, n, s, v, pm);
    calc_V_exterior(d, n, s, v, w5, w3, pm);
    calc_V_multibranch(d, n, s, v, wm, pm);
    calc_coaxial(d, n, s, v, w, w5, w3, pm);
    calc_W(d, n, s, v, w, pm);
    calc_WM(d, n, s, w, wm, pm);
    calc_wl_coax(d, n, s, v, w, wm, pm, wca);
    calc_w5_and_w3(d, n, s, v, w5, w3, pm, wca);
    Kokkos::fence();
  }

  auto t_end = std::chrono::steady_clock::now();
  auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
  printf("Total kernel execution time %f (s)\n", time_ns * 1e-9f);

  // Copy results back to host
  {
    auto hv  = Kokkos::create_mirror_view(d_v);  Kokkos::deep_copy(hv,  d_v);
    auto hw  = Kokkos::create_mirror_view(d_w);  Kokkos::deep_copy(hw,  d_w);
    auto hwm = Kokkos::create_mirror_view(d_wm); Kokkos::deep_copy(hwm, d_wm);
    for (int i=0;i<n*n;i++) p->v[i]  = hv(i);
    for (int i=0;i<n*n;i++) p->w[i]  = hw(i);
    for (int i=0;i<n*n;i++) p->wm[i] = hwm(i);
  }
  {
    auto hw5 = Kokkos::create_mirror_view(d_w5_store); Kokkos::deep_copy(hw5, d_w5_store);
    auto hw3 = Kokkos::create_mirror_view(d_w3);       Kokkos::deep_copy(hw3, d_w3);
    for (int i=0; i<=n; i++) p->w5[i]   = hw5(i+1);  // w5[0..n-1] <- store[1..n]
    p->w5[-1] = hw5(0);
    for (int i=0; i<=n; i++) p->w3[i]   = hw3(i);
  }

  return p;
}

void frna_delete(frna_t p) {
  if (p) {
    free(p->seq);
    free(p->v); free(p->w); free(p->wm); free(p->wca);
    free(p->w5 - 1);
    free(p->w3);
    free(p);
  }
}

void frna_write(const frna_t p, const char* outfile) {
  FILE* f = fopen(outfile, "w");
  if (!f) { printf("failed to open output file %s\n", outfile); return; }
  int i, j, n = p->n;
  const fbase_t* s = p->seq;
  fprintf(f, "n: %d\n", n);
  fprintf(f, "seq: ");
  for (i=0;i<n;i++) fprintf(f,"%c",fbase_as_char(s[i]));
  fprintf(f, "\n");
  fprintf(f, "i\tj\tV:\tW:\tWM:\tV':\tW':\tWM':\n");
  for (j=0;j<n;j++)
    for (i=0;i<j;i++)
      fprintf(f,"%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
              i+1,j+1,p->v[ind(i,j,n)],p->w[ind(i,j,n)],p->wm[ind(i,j,n)],
              p->v[ind(j,i,n)],p->w[ind(j,i,n)],p->wm[ind(j,i,n)]);
  fprintf(f,"\n\n\ni\tw5[i]\tw3[i]\n0\t0\t0\n");
  for (i=0;i<n;i++) fprintf(f,"%d\t%d\t%d\n",i+1,p->w5[i],p->w3[i]);
  fclose(f);
}

// ============================================================
// main  (from main.c, converted to C++)
// ============================================================
static const char Usage[] =
  "Usage: %s [options] <fasta-file> <output-file>\n\n"
  "options:\n"
  "-h: show this message\n"
  "-d: use DNA parameters\n";

int main(int argc, char** argv) {
  const char* cmd = *argv;
  int use_dna_fparams = 0;
  int c;
  while ((c = getopt(argc, argv, "hd")) != EOF) {
    if      (c == 'h') die(Usage, cmd);
    else if (c == 'd') use_dna_fparams = 1;
    else               die(Usage, cmd);
  }
  argc -= optind; argv += optind;
  if (argc != 2) die(Usage, cmd);

  char* seq     = fsequence(argv[0]);
  const char* outfile = argv[1];

  struct fparam par;
  const char* path = getenv("DATAPATH");
  if (!path) die("%s: need to set environment variable $DATAPATH", cmd);
  fparam_read_from_text(path, use_dna_fparams, &par);

  Kokkos::initialize(argc, argv);
  frna_t p = frna_new(seq, &par);
  Kokkos::finalize();

  frna_write(p, outfile);
  frna_delete(p);
  free(seq);
  return 0;
}
