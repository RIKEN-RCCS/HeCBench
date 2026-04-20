// minisweep-kokkos/main.cpp
// Port of minisweep-omp to Kokkos.
// All device functions are annotated KOKKOS_INLINE_FUNCTION.
// The sweep wavefront kernel is launched via Kokkos::parallel_for.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Atomic.hpp>

// ──────────────────────────────────────────────────────────────────────────────
// Definitions (from utils.h)
// ──────────────────────────────────────────────────────────────────────────────
#define P double

enum { NDIM = 3 };
enum { NOCTANT = 8 };
enum { DIR_UP = +1, DIR_DN = -1 };
enum { DIR_HI = +1, DIR_LO = -1 };

#ifndef NU_VALUE
enum { NU = 4 };
#else
enum { NU = NU_VALUE };
#endif

#ifndef NM_VALUE
enum { NM = 4 };
#else
enum { NM = NM_VALUE };
#endif

typedef struct { int ncell_x, ncell_y, ncell_z, ne, nm, na; } Dimensions;

typedef struct {
  int nblock_z_;
  int nproc_x_;
  int nproc_y_;
  int nblock_octant_;
  int noctant_per_block_;
} StepScheduler;

typedef struct {
  int  block_z;
  int  octant;
  int  is_active;
} StepInfo;

typedef struct { StepInfo stepinfo[NOCTANT]; } StepInfoAll;

typedef struct {
  int           nblock_z;
  int           nblock_octant;
  int           noctant_per_block;
  Dimensions    dims;
  Dimensions    dims_b;
  StepScheduler stepscheduler;
} Sweeper;

// Argument parsing helpers
typedef struct {
  int    argc;
  char **argv_unconsumed;
  char  *argstring;
} Arguments;

#define A_FROM_M_ADDR(dims_na, im, ia, octant) \
  ((ia) + (dims_na)*((im) + NM*(octant)))
#define M_FROM_A_ADDR(dims_na, im, ia, octant) \
  ((im) + NM*((ia) + (dims_na)*(octant)))

// ──────────────────────────────────────────────────────────────────────────────
// Device helper functions
// ──────────────────────────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION int Dir_x(int octant) { return octant & (1<<0) ? DIR_DN : DIR_UP; }
KOKKOS_INLINE_FUNCTION int Dir_y(int octant) { return octant & (1<<1) ? DIR_DN : DIR_UP; }
KOKKOS_INLINE_FUNCTION int Dir_z(int octant) { return octant & (1<<2) ? DIR_DN : DIR_UP; }

KOKKOS_INLINE_FUNCTION
int Quantities_scalefactor_space_acceldir(int ix_g, int iy_g, int iz_g)
{
  int result = 0;
#ifndef RELAXED_TESTING
  const int im = 134456, ia = 8121, ic = 28411;
  result = ( (result+(ix_g+2))*ia + ic ) % im;
  result = ( (result+(iy_g+2))*ia + ic ) % im;
  result = ( (result+(iz_g+2))*ia + ic ) % im;
  result = ( (result+(ix_g+3*iy_g+7*iz_g+2))*ia + ic ) % im;
  result = ix_g+3*iy_g+7*iz_g+2;
  result = result & ((1<<2)-1);
#endif
  result = 1 << result;
  return result;
}

KOKKOS_INLINE_FUNCTION
P Quantities_init_face_acceldir(int ia, int ie, int iu,
                                int scalefactor_space, int octant)
{
  return ( (P)(1+ia) )
       * ( (P)(1 << (ia & ((1<<3)-1))) )
       * ( (P)scalefactor_space )
       * ( (P)(1 << (((ie*1366+150889)%714025) & ((1<<2)-1))) )
       * ( (P)(1 << (((iu*741 +60037) %312500) & ((1<<2)-1))) )
       * ( (P)1 + octant );
}

KOKKOS_INLINE_FUNCTION
void Quantities_solve_acceldir(P *__restrict vs_local,
                               Dimensions dims,
                               P *__restrict facexy,
                               P *__restrict facexz,
                               P *__restrict faceyz,
                               int ix, int iy, int iz,
                               int ix_g, int iy_g, int iz_g,
                               int ie, int ia,
                               int octant, int octant_in_block,
                               int noctant_per_block)
{
  const int dir_x = Dir_x(octant);
  const int dir_y = Dir_y(octant);
  const int dir_z = Dir_z(octant);

  const P scalefactor_octant    = 1 + octant;
  const P scalefactor_octant_r  = ((P)1) / scalefactor_octant;
  const P scalefactor_space     = (P)Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);
  const P scalefactor_space_r   = ((P)1) / scalefactor_space;
  const P scalefactor_space_x_r = ((P)1) /
    Quantities_scalefactor_space_acceldir(ix_g - dir_x, iy_g,       iz_g);
  const P scalefactor_space_y_r = ((P)1) /
    Quantities_scalefactor_space_acceldir(ix_g,       iy_g - dir_y, iz_g);
  const P scalefactor_space_z_r = ((P)1) /
    Quantities_scalefactor_space_acceldir(ix_g,       iy_g,         iz_g - dir_z);

  for (int iu = 0; iu < NU; ++iu) {
    int vs_idx = ia + dims.na*(iu + NU*(ie + dims.ne*(ix + dims.ncell_x*(iy + dims.ncell_y*(octant + NOCTANT*0)))));

    const P result =
      ( vs_local[vs_idx] * scalefactor_space_r
        + ( facexy[ia + dims.na*(iu + NU*(ie + dims.ne*(ix + dims.ncell_x*(iy + dims.ncell_y*(octant + NOCTANT*0)))))]
              * (P)(1./(P)2) * scalefactor_space_z_r
          + facexz[ia + dims.na*(iu + NU*(ie + dims.ne*(ix + dims.ncell_x*(iz + dims.ncell_z*(octant + NOCTANT*0)))))]
              * (P)(1./(P)4) * scalefactor_space_y_r
          + faceyz[ia + dims.na*(iu + NU*(ie + dims.ne*(iy + dims.ncell_y*(iz + dims.ncell_z*(octant + NOCTANT*0)))))]
              * ((P)(1./(P)4) - (P)(1./(1<<(ia & ((1<<3)-1))))) * scalefactor_space_x_r
          ) * scalefactor_octant_r
      ) * scalefactor_space;

    vs_local[vs_idx] = result;
    const P result_scaled = result * scalefactor_octant;

    facexy[ia + dims.na*(iu + NU*(ie + dims.ne*(ix + dims.ncell_x*(iy + dims.ncell_y*(octant + NOCTANT*0)))))]
      = result_scaled;
    facexz[ia + dims.na*(iu + NU*(ie + dims.ne*(ix + dims.ncell_x*(iz + dims.ncell_z*(octant + NOCTANT*0)))))]
      = result_scaled;
    faceyz[ia + dims.na*(iu + NU*(ie + dims.ne*(iy + dims.ncell_y*(iz + dims.ncell_z*(octant + NOCTANT*0)))))]
      = result_scaled;
  }
}

KOKKOS_INLINE_FUNCTION
void Sweeper_sweep_cell_acceldir(
    const Dimensions &dims,
    int wavefront, int octant,
    int ix, int iy,
    int ix_g, int iy_g, int iz_g,
    int dir_x, int dir_y, int dir_z,
    P *__restrict facexy,
    P *__restrict facexz,
    P *__restrict faceyz,
    const P *__restrict a_from_m,
    const P *__restrict m_from_a,
    const P *__restrict vi,
    P *__restrict vo,
    P *__restrict vs_local,
    int octant_in_block, int noctant_per_block, int ie)
{
  const int dims_ncell_x = dims.ncell_x;
  const int dims_ncell_y = dims.ncell_y;
  const int dims_ncell_z = dims.ncell_z;
  const int dims_na = dims.na;
  const int dims_nm = dims.nm;
  const int dims_ne = dims.ne;

  const int ixwav = dir_x == DIR_UP ? ix : (dims_ncell_x-1) - ix;
  const int iywav = dir_y == DIR_UP ? iy : (dims_ncell_y-1) - iy;
  const int izwav = wavefront - ixwav - iywav;
  const int iz    = dir_z == DIR_UP ? izwav : (dims_ncell_z-1) - izwav;

  if (iz >= 0 && iz < dims_ncell_z) {
    // Transform state: moments → angles
    for (int iu = 0; iu < NU; ++iu)
    for (int ia = 0; ia < dims_na; ++ia) {
      P result = (P)0;
      for (int im = 0; im < dims_nm; ++im) {
        result += a_from_m[A_FROM_M_ADDR(dims_na, im, ia, octant)]
                * vi[im + dims.nm*(iu + NU*(ix + dims_ncell_x*(iy + dims_ncell_y*(ie + dims_ne*(iz + dims_ncell_z*0)))))];
      }
      vs_local[ia + dims.na*(iu + NU*(ie + dims_ne*(ix + dims_ncell_x*(iy + dims_ncell_y*(octant + NOCTANT*0)))))] = result;
    }

    // Solve (updates vs_local and faces)
    for (int ia = 0; ia < dims_na; ++ia) {
      Quantities_solve_acceldir(vs_local, dims, facexy, facexz, faceyz,
                                ix, iy, iz, ix_g, iy_g, iz_g,
                                ie, ia, octant, octant_in_block, noctant_per_block);
    }

    // Transform state: angles → moments, accumulate into vo (atomic)
    for (int iu = 0; iu < NU; ++iu)
    for (int im = 0; im < dims_nm; ++im) {
      P result = (P)0;
      for (int ia = 0; ia < dims_na; ++ia) {
        result += m_from_a[M_FROM_A_ADDR(dims_na, im, ia, octant)]
                * vs_local[ia + dims.na*(iu + NU*(ie + dims_ne*(ix + dims_ncell_x*(iy + dims_ncell_y*(octant + NOCTANT*0)))))];
      }
      const int vo_idx = im + dims.nm*(iu + NU*(ix + dims_ncell_x*(iy + dims_ncell_y*(ie + dims_ne*(iz + dims_ncell_z*0)))));
      Kokkos::atomic_add(&vo[vo_idx], result);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Host-only utility functions
// ──────────────────────────────────────────────────────────────────────────────
static inline int Dir_x_host(int octant) { return octant & (1<<0) ? DIR_DN : DIR_UP; }
static inline int Dir_y_host(int octant) { return octant & (1<<1) ? DIR_DN : DIR_UP; }
static inline int Dir_z_host(int octant) { return octant & (1<<2) ? DIR_DN : DIR_UP; }

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

static double Quantities_flops_per_solve(const Dimensions &) { return 3. + 3. * NDIM; }

static size_t Dimensions_size_state(const Dimensions &dims, int nu) {
  return (size_t)dims.ncell_x * dims.ncell_y * dims.ncell_z * dims.ne * dims.nm * nu;
}
static size_t Dimensions_size_state_angles(const Dimensions &dims, int nu) {
  return (size_t)dims.ncell_x * dims.ncell_y * dims.ncell_z * dims.ne * dims.na * nu * NOCTANT;
}
static int Dimensions_size_facexy(const Dimensions &dims, int nu, int noctant_per_block) {
  return dims.na * nu * dims.ne * dims.ncell_x * dims.ncell_y * noctant_per_block;
}
static int Dimensions_size_facexz(const Dimensions &dims, int nu, int noctant_per_block) {
  return dims.na * nu * dims.ne * dims.ncell_x * dims.ncell_z * noctant_per_block;
}
static int Dimensions_size_faceyz(const Dimensions &dims, int nu, int noctant_per_block) {
  return dims.na * nu * dims.ne * dims.ncell_y * dims.ncell_z * noctant_per_block;
}

static void initialize_input_state(P *vi, const Dimensions &dims, int nu) {
  for (int iz = 0; iz < dims.ncell_z; ++iz)
  for (int iy = 0; iy < dims.ncell_y; ++iy)
  for (int ix = 0; ix < dims.ncell_x; ++ix)
  for (int ie = 0; ie < dims.ne; ++ie)
  for (int im = 0; im < dims.nm; ++im)
  for (int iu = 0; iu < nu; ++iu) {
    const int scalefactor_space = Quantities_scalefactor_space_acceldir(ix, iy, iz);
    vi[im + dims.nm*(iu + nu*(ix + dims.ncell_x*(iy + dims.ncell_y*(ie + dims.ne*(iz + dims.ncell_z*0)))))]
      = ( (P)(1+im) )
        * ( (P)(1 << (im & ((1<<3)-1))) )
        * ( (P)scalefactor_space )
        * ( (P)(1 << (((ie*1366+150889)%714025) & ((1<<2)-1))) )
        * ( (P)(1 << (((iu*741+60037)%312500) & ((1<<2)-1))) );
  }
}

// ── StepScheduler helpers ────────────────────────────────────────────────────
static int StepScheduler_nblock(const StepScheduler *s) {
  return s->nblock_z_;
}
static int StepScheduler_nstep(const StepScheduler *s) {
  const int nblock = StepScheduler_nblock(s);
  switch (s->nblock_octant_) {
    case 1: return 8 * nblock + s->nproc_x_ - 1 + s->nproc_y_ - 1
                              + (s->nproc_y_-1) + (s->nproc_x_-1)
                              + (s->nproc_y_-1) + (s->nproc_x_-1)
                              + (s->nproc_y_-1) + (s->nproc_x_-1)
                              + s->nproc_y_ - 1;
    case 8: return nblock;
    default:
      printf("Error: unknown nblock_octant %d\n", s->nblock_octant_);
      return 0;
  }
}

static StepInfo StepScheduler_stepinfo(const StepScheduler *s,
                                       int step, int octant_in_block,
                                       int proc_x, int proc_y)
{
  assert(octant_in_block >= 0 && octant_in_block * s->nblock_octant_ < NOCTANT);

  const int nproc_x  = s->nproc_x_;
  const int nproc_y  = s->nproc_y_;
  const int nblock   = StepScheduler_nblock(s);
  const int nstep    = StepScheduler_nstep(s);
  const int nopb     = s->noctant_per_block_;

  const int octant_selector[NOCTANT] = {0,4,2,6,3,7,1,5};
  const int is_folded_x = nopb >= 2;
  const int is_folded_y = nopb >= 4;
  const int is_folded_z = nopb >= 8;

  const int folded_proc_x = (is_folded_x && (octant_in_block & (1<<0)))
                          ? (nproc_x - 1 - proc_x) : proc_x;
  const int folded_proc_y = (is_folded_y && (octant_in_block & (1<<1)))
                          ? (nproc_y - 1 - proc_y) : proc_y;

  int wave = step, octant_key = 0, step_base = 0;

  step_base += nblock;
  if (step >= (step_base + folded_proc_x + folded_proc_y) && !is_folded_z) {
    wave = step - step_base; octant_key = 1;
  }
  step_base += nblock;
  if (step >= (step_base + folded_proc_x + folded_proc_y) && !is_folded_y) {
    wave = step - (step_base + (nproc_y-1)); octant_key = 2;
  }
  step_base += nblock + (nproc_y-1);
  if (step >= (step_base + (nproc_y-1-folded_proc_y) + folded_proc_x) && !is_folded_y) {
    wave = step - step_base; octant_key = 3;
  }
  step_base += nblock;
  if (step >= (step_base + (nproc_y-1-folded_proc_y) + folded_proc_x) && !is_folded_x) {
    wave = step - (step_base + (nproc_x-1)); octant_key = 4;
  }
  step_base += nblock + (nproc_x-1);
  if (step >= (step_base + (nproc_y-1-folded_proc_y) + (nproc_x-1-folded_proc_x)) && !is_folded_x) {
    wave = step - step_base; octant_key = 5;
  }
  step_base += nblock;
  if (step >= (step_base + (nproc_y-1-folded_proc_y) + (nproc_x-1-folded_proc_x)) && !is_folded_x) {
    wave = step - (step_base + (nproc_y-1)); octant_key = 6;
  }
  step_base += nblock + (nproc_y-1);
  if (step >= (step_base + folded_proc_y + (nproc_x-1-folded_proc_x)) && !is_folded_x) {
    wave = step - step_base; octant_key = 7;
  }

  const int folded_octant = octant_selector[octant_key];
  const int octant = folded_octant + octant_in_block;

  const int dir_x   = Dir_x_host(folded_octant);
  const int dir_y   = Dir_y_host(folded_octant);
  const int dir_z   = Dir_z_host(folded_octant);
  const int start_x = dir_x == DIR_UP ? 0 : (nproc_x-1);
  const int start_y = dir_y == DIR_UP ? 0 : (nproc_y-1);
  const int start_z = dir_z == DIR_UP ? 0 : (nblock-1);

  const int folded_block = (wave - (start_x + folded_proc_x*dir_x)
                                 - (start_y + folded_proc_y*dir_y)
                                 - start_z) / dir_z;
  const int block = (is_folded_z && (octant_in_block & (1<<2)))
                  ? (nblock-1 - folded_block) : folded_block;

  StepInfo si;
  si.is_active = block  >= 0 && block  < nblock &&
                 step   >= 0 && step   < nstep  &&
                 proc_x >= 0 && proc_x < nproc_x &&
                 proc_y >= 0 && proc_y < nproc_y;
  si.block_z = si.is_active ? block : 0;
  si.octant  = octant;
  return si;
}

// Argument parsing
static int Arguments_exists(const Arguments *args, const char *name) {
  for (int i = 0; i < args->argc; ++i)
    if (args->argv_unconsumed[i] && strcmp(args->argv_unconsumed[i], name) == 0)
      return 1;
  return 0;
}
static int Arguments_consume_int_(Arguments *args, const char *name) {
  for (int i = 0; i < args->argc; ++i) {
    if (args->argv_unconsumed[i] && strcmp(args->argv_unconsumed[i], name) == 0) {
      args->argv_unconsumed[i] = NULL; ++i;
      assert(i < args->argc);
      int v = atoi(args->argv_unconsumed[i]);
      args->argv_unconsumed[i] = NULL;
      return v;
    }
  }
  return 0;
}
static int Arguments_consume_int_or_default(Arguments *args, const char *name, int def) {
  return Arguments_exists(args, name) ? Arguments_consume_int_(args, name) : def;
}
static void Arguments_destroy(Arguments *args) {
  free(args->argv_unconsumed);
  if (args->argstring) free(args->argstring);
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
  Arguments args;
  memset(&args, 0, sizeof(Arguments));
  args.argc = argc;
  args.argv_unconsumed = (char **)malloc(argc * sizeof(char *));
  for (int i = 0; i < argc; ++i) args.argv_unconsumed[i] = argv[i];

  Dimensions dims_g, dims;
  Sweeper sweeper;
  memset(&sweeper, 0, sizeof(Sweeper));
  int niterations = 0;

  dims_g.ncell_x = Arguments_consume_int_or_default(&args, "--ncell_x",  5);
  dims_g.ncell_y = Arguments_consume_int_or_default(&args, "--ncell_y",  5);
  dims_g.ncell_z = Arguments_consume_int_or_default(&args, "--ncell_z",  5);
  dims_g.ne      = Arguments_consume_int_or_default(&args, "--ne",      30);
  dims_g.na      = Arguments_consume_int_or_default(&args, "--na",      33);
  niterations    = Arguments_consume_int_or_default(&args, "--niterations", 1);
  dims_g.nm      = NM;

  if (dims_g.ncell_x <= 0 || dims_g.ncell_y <= 0 || dims_g.ncell_z <= 0 ||
      dims_g.ne <= 0 || dims_g.nm <= 0 || dims_g.na <= 0 || niterations < 1) {
    printf("Invalid arguments\n"); Arguments_destroy(&args); return -1;
  }

  dims = dims_g;

  // ── Allocate host arrays ──────────────────────────────────────────────────
  const int a_from_m_size = dims.nm * dims.na * NOCTANT;
  P *a_from_m = (P *)malloc(a_from_m_size * sizeof(P));
  P *m_from_a = (P *)malloc(a_from_m_size * sizeof(P));

  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int im = 0; im < dims.nm; ++im)
  for (int ia = 0; ia < dims.na; ++ia)
    a_from_m[A_FROM_M_ADDR(dims.na, im, ia, octant)] = (P)0;

  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int i = 0; i < dims.na; ++i) {
    const int quot = (i+1) / dims.nm;
    const int rem  = (i+1) % dims.nm;
    a_from_m[A_FROM_M_ADDR(dims.na, dims.nm-1, i, octant)] += quot;
    if (rem != 0) {
      a_from_m[A_FROM_M_ADDR(dims.na, 0,   i, octant)] += (P)-1;
      a_from_m[A_FROM_M_ADDR(dims.na, rem, i, octant)] += (P)1;
    }
  }
  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int im = 0; im < dims.nm-2; ++im)
  for (int ia = 0; ia < dims.na; ++ia) {
    const int rv = 21 + (im + dims.nm*ia) % 17;
    a_from_m[A_FROM_M_ADDR(dims.na, im,   ia, octant)] += -rv;
    a_from_m[A_FROM_M_ADDR(dims.na, im+1, ia, octant)] += 2*rv;
    a_from_m[A_FROM_M_ADDR(dims.na, im+2, ia, octant)] += -rv;
  }

  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int im = 0; im < dims.nm; ++im)
  for (int ia = 0; ia < dims.na; ++ia)
    m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)] = (P)0;

  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int i = 0; i < dims.nm; ++i) {
    const int quot = (i+1) / dims.na;
    const int rem  = (i+1) % dims.na;
    m_from_a[M_FROM_A_ADDR(dims.na, i, dims.na-1, octant)] += quot;
    if (rem != 0) {
      m_from_a[M_FROM_A_ADDR(dims.na, i, 0,   octant)] += (P)-1;
      m_from_a[M_FROM_A_ADDR(dims.na, i, rem, octant)] += (P)1;
    }
  }
  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int im = 0; im < dims.nm; ++im)
  for (int ia = 0; ia < dims.na-2; ++ia) {
    const int rv = 37 + (im + dims.nm*ia) % 19;
    m_from_a[M_FROM_A_ADDR(dims.na, im, ia,   octant)] += -rv;
    m_from_a[M_FROM_A_ADDR(dims.na, im, ia+1, octant)] += 2*rv;
    m_from_a[M_FROM_A_ADDR(dims.na, im, ia+2, octant)] += -rv;
  }
  for (int octant = 0; octant < NOCTANT; ++octant)
  for (int im = 0; im < dims.nm; ++im)
  for (int ia = 0; ia < dims.na; ++ia) {
    m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)] /= NOCTANT;
    m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)] /= 1 << (ia & ((1<<3)-1));
  }

  const int v_size = (int)Dimensions_size_state(dims, NU);
  P *h_vi = (P *)malloc(v_size * sizeof(P));
  P *h_vo = (P *)malloc(v_size * sizeof(P));
  initialize_input_state(h_vi, dims, NU);
  memset(h_vo, 0, v_size * sizeof(P));

  // ── Sweeper setup ─────────────────────────────────────────────────────────
  sweeper.nblock_z          = 1;
  sweeper.noctant_per_block = NOCTANT;
  sweeper.nblock_octant     = 1;
  sweeper.dims   = dims;
  sweeper.dims_b = dims;
  sweeper.dims_b.ncell_z = dims.ncell_z / sweeper.nblock_z;

  sweeper.stepscheduler.nblock_z_          = sweeper.nblock_z;
  sweeper.stepscheduler.nproc_x_           = 1;
  sweeper.stepscheduler.nproc_y_           = 1;
  sweeper.stepscheduler.nblock_octant_     = sweeper.nblock_octant;
  sweeper.stepscheduler.noctant_per_block_ = NOCTANT / sweeper.nblock_octant;

  const int noctant_per_block = sweeper.noctant_per_block;
  const int facexy_size = Dimensions_size_facexy(sweeper.dims_b, NU, noctant_per_block);
  const int facexz_size = Dimensions_size_facexz(sweeper.dims_b, NU, noctant_per_block);
  const int faceyz_size = Dimensions_size_faceyz(sweeper.dims_b, NU, noctant_per_block);
  const int vslocal_size = dims.na * NU * dims.ne * NOCTANT * dims.ncell_x * dims.ncell_y;

  // ── Upload to device ──────────────────────────────────────────────────────
  Kokkos::View<P*> d_a_from_m("a_from_m", a_from_m_size);
  Kokkos::View<P*> d_m_from_a("m_from_a", a_from_m_size);
  Kokkos::View<P*> d_vi("vi", v_size);
  Kokkos::View<P*> d_vo("vo", v_size);
  Kokkos::View<P*> d_facexy("facexy", facexy_size);
  Kokkos::View<P*> d_facexz("facexz", facexz_size);
  Kokkos::View<P*> d_faceyz("faceyz", faceyz_size);
  Kokkos::View<P*> d_vslocal("vslocal", vslocal_size);

  auto h_a_from_m = Kokkos::create_mirror_view(d_a_from_m);
  auto h_m_from_a = Kokkos::create_mirror_view(d_m_from_a);
  auto h_vi_view  = Kokkos::create_mirror_view(d_vi);
  std::memcpy(h_a_from_m.data(), a_from_m, a_from_m_size * sizeof(P));
  std::memcpy(h_m_from_a.data(), m_from_a, a_from_m_size * sizeof(P));
  std::memcpy(h_vi_view.data(),  h_vi,     v_size * sizeof(P));
  Kokkos::deep_copy(d_a_from_m, h_a_from_m);
  Kokkos::deep_copy(d_m_from_a, h_m_from_a);
  Kokkos::deep_copy(d_vi, h_vi_view);

  double time = 0, ktime = 0;
  const double t1 = get_time();
  double k_start = 0, k_end = 0;

  for (int iteration = 0; iteration < niterations; ++iteration) {
    const int nstep = StepScheduler_nstep(&sweeper.stepscheduler);

    for (int step = 0; step < nstep; ++step) {
      Dimensions dims_b  = sweeper.dims_b;
      int dims_b_ncell_x = dims_b.ncell_x;
      int dims_b_ncell_y = dims_b.ncell_y;
      int dims_b_ncell_z = dims_b.ncell_z;
      int dims_ncell_z   = dims.ncell_z;
      int dims_b_ne      = dims_b.ne;
      int dims_b_na      = dims_b.na;
      int v_b_size = dims_b.ncell_x * dims_b.ncell_y * dims_b.ncell_z * dims_b.ne * dims_b.nm * NU;

      StepInfoAll stepinfoall;
      for (int oib = 0; oib < noctant_per_block; ++oib)
        stepinfoall.stepinfo[oib] = StepScheduler_stepinfo(&sweeper.stepscheduler,
                                    step, oib, 0, 0);

      const int ix_base = 0, iy_base = 0;
      const int num_wavefronts = dims_b_ncell_z + dims_b_ncell_y + dims_b_ncell_x - 2;
      const bool is_first_step = (step == 0);
      const bool is_last_step  = (step == nstep - 1);

      if (is_first_step) {
        k_start = get_time();
        // Initialize vo to zero on device
        Kokkos::deep_copy(d_vo, 0.0);

        // Initialize facexy
        Kokkos::parallel_for("init_facexy",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
              {NOCTANT, dims_b_ncell_y, dims_b_ncell_x}),
          KOKKOS_LAMBDA(int octant, int iy, int ix) {
            const int dir_z = Dir_z(octant);
            const int iz    = dir_z == DIR_UP ? -1 : dims_b_ncell_z;
            const int ix_g  = ix + ix_base;
            const int iy_g  = iy + iy_base;
            const int iz_g  = iz + (dir_z == DIR_UP ? 0 : dims_ncell_z - dims_b_ncell_z);
            const int sf    = Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);
            for (int ie = 0; ie < dims_b_ne; ++ie)
            for (int iu = 0; iu < NU; ++iu)
            for (int ia = 0; ia < dims_b_na; ++ia) {
              d_facexy[ia + dims_b_na*(iu + NU*(ie + dims_b_ne*(ix + dims_b_ncell_x*(iy + dims_b_ncell_y*(octant + NOCTANT*0)))))]
                = Quantities_init_face_acceldir(ia, ie, iu, sf, octant);
            }
          });

        // Initialize facexz
        Kokkos::parallel_for("init_facexz",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
              {NOCTANT, dims_b_ncell_z, dims_b_ncell_x}),
          KOKKOS_LAMBDA(int octant, int iz, int ix) {
            const int dir_y = Dir_y(octant);
            const int iy    = dir_y == DIR_UP ? -1 : dims_b_ncell_y;
            const int ix_g  = ix + ix_base;
            const int iy_g  = iy + iy_base;
            const int iz_g  = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;
            const int sf    = Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);
            for (int ie = 0; ie < dims_b_ne; ++ie)
            for (int iu = 0; iu < NU; ++iu)
            for (int ia = 0; ia < dims_b_na; ++ia) {
              d_facexz[ia + dims_b_na*(iu + NU*(ie + dims_b_ne*(ix + dims_b_ncell_x*(iz + dims_b_ncell_z*(octant + NOCTANT*0)))))]
                = Quantities_init_face_acceldir(ia, ie, iu, sf, octant);
            }
          });

        // Initialize faceyz
        Kokkos::parallel_for("init_faceyz",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
              {NOCTANT, dims_b_ncell_z, dims_b_ncell_y}),
          KOKKOS_LAMBDA(int octant, int iz, int iy) {
            const int dir_x = Dir_x(octant);
            const int ix    = dir_x == DIR_UP ? -1 : dims_b_ncell_x;
            const int ix_g  = ix + ix_base;
            const int iy_g  = iy + iy_base;
            const int iz_g  = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;
            const int sf    = Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);
            for (int ie = 0; ie < dims_b_ne; ++ie)
            for (int iu = 0; iu < NU; ++iu)
            for (int ia = 0; ia < dims_b_na; ++ia) {
              d_faceyz[ia + dims_b_na*(iu + NU*(ie + dims_b_ne*(iy + dims_b_ncell_y*(iz + dims_b_ncell_z*(octant + NOCTANT*0)))))]
                = Quantities_init_face_acceldir(ia, ie, iu, sf, octant);
            }
          });
        Kokkos::fence();
      }

      // ── Sweep wavefront kernel ────────────────────────────────────────────
      // For each (ie, octant) pair, iterate over wavefronts serially.
      // Within a wavefront, iterate over (iy, ix) in parallel.
      for (int wavefront = 0; wavefront < num_wavefronts; ++wavefront) {
        Kokkos::parallel_for("sweep",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
              {dims_b_ne, NOCTANT, dims_b_ncell_y}),
          KOKKOS_LAMBDA(int ie, int octant, int iywav) {
            if (!stepinfoall.stepinfo[octant].is_active) return;

            const int dir_x = Dir_x(octant);
            const int dir_y = Dir_y(octant);
            const int dir_z = Dir_z(octant);
            const int v_offset = stepinfoall.stepinfo[octant].block_z * v_b_size;

            for (int ixwav = 0; ixwav < dims_b_ncell_x; ++ixwav) {
              const int ix = dir_x == DIR_UP ? ixwav : dims_b_ncell_x - 1 - ixwav;
              const int iy = dir_y == DIR_UP ? iywav : dims_b_ncell_y - 1 - iywav;
              const int ix_g = ix + ix_base;
              const int iy_g = iy + iy_base;
              const int izwav = wavefront - ixwav - iywav;
              const int iz    = dir_z == DIR_UP ? izwav : (dims_b_ncell_z-1) - izwav;
              const int iz_g  = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;

              Sweeper_sweep_cell_acceldir(
                  dims_b, wavefront, octant,
                  ix, iy, ix_g, iy_g, iz_g,
                  dir_x, dir_y, dir_z,
                  d_facexy.data(), d_facexz.data(), d_faceyz.data(),
                  d_a_from_m.data(), d_m_from_a.data(),
                  d_vi.data() + v_offset, d_vo.data() + v_offset,
                  d_vslocal.data(),
                  octant, noctant_per_block, ie);
            }
          });
        Kokkos::fence();
      }

      if (is_last_step) {
        k_end = get_time();
        ktime += k_end - k_start;
      }
    } // step

    // Swap vi and vo on device
    auto tmp = d_vo;
    d_vo = d_vi;
    d_vi = tmp;
  }

  const double t2 = get_time();
  time = t2 - t1;

  // Copy final result back to host for verification
  // After loop, d_vi holds the "output" (due to the swap)
  auto h_vo_view = Kokkos::create_mirror_view(d_vi);
  auto h_vi_view2 = Kokkos::create_mirror_view(d_vo);
  Kokkos::deep_copy(h_vo_view,  d_vi);
  Kokkos::deep_copy(h_vi_view2, d_vo);
  std::memcpy(h_vo, h_vo_view.data(),  v_size * sizeof(P));
  std::memcpy(h_vi, h_vi_view2.data(), v_size * sizeof(P));

  P normsq = (P)0, normsqdiff = (P)0;
  for (size_t i = 0; i < Dimensions_size_state(dims, NU); i++) {
    normsq     += h_vo[i] * h_vo[i];
    normsqdiff += (h_vi[i] - h_vo[i]) * (h_vi[i] - h_vo[i]);
  }
  double flops = niterations * (
    Dimensions_size_state(dims, NU) * NOCTANT * 2. * dims.na +
    Dimensions_size_state_angles(dims, NU) * Quantities_flops_per_solve(dims) +
    Dimensions_size_state(dims, NU) * NOCTANT * 2. * dims.na);

  double floprate_h = (time  <= 0) ? 0 : flops / (time  * 1e-6) / 1e9;
  double floprate_d = (ktime <= 0) ? 0 : flops / (ktime * 1e-6) / 1e9;

  printf("Normsq result: %.8e  diff: %.3e  verify: %s  host time: %.3f (s) kernel time: %.3f (s)\n",
         normsq, normsqdiff,
         normsqdiff == (P)0 ? "PASS" : "FAIL",
         time * 1e-6, ktime * 1e-6);
  printf("GF/s (host): %.3f\nGF/s (device): %.3f\n", floprate_h, floprate_d);

  free(h_vi); free(h_vo); free(a_from_m); free(m_from_a);
  Arguments_destroy(&args);
  } // Kokkos scope
  Kokkos::finalize();
  return 0;
}
