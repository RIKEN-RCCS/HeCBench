/*
 * metropolis – Ising model Exchange Monte Carlo (trueke)
 * Kokkos port from the OMP offload version.
 *
 * Original: Cristobal A. Navarro, Wei Huang (2015)
 * License: GPLv3
 *
 * Usage: ./main -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <inttypes.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>

// ============================================================
// Lattice index macros
// ============================================================
#define C(x,y,z,L)       ((z)*(L)*(L)+(y)*(L)+(x))
#define sC(x,y,z,Lx,Ly)  ((z+1)*(Ly)*(Lx)+(y+1)*(Lx)+(x+1))

// Block dimensions (same defaults as OMP utils.h)
#define BX  16
#define BY  8
#define BZ  4
#define BLOCKSIZE1D 256

#define sLx (BX+2)
#define sLy (BY+2)
#define sLz (BZ+2)
#define SVOLUME ((sLx)*(sLy)*(sLz))   // 18*10*6 = 1080

#define BLOCK_STEPS 1
#define EPSILON 0.00000000001f
#define INV_UINT_MAX 2.3283064e-10f

// ============================================================
// PCG PRNG – device functions
// ============================================================
KOKKOS_INLINE_FUNCTION
void gpu_pcg32_srandom_r(uint64_t *state, uint64_t *inc,
                         uint64_t initstate, uint64_t initseq)
{
  *state = 0U;
  *inc   = (initseq << 1u) | 1u;
  uint64_t old = *state;
  *state = old * 6364136223846793005ULL + *inc;
  uint32_t xs  = ((old >> 18u) ^ old) >> 27u;
  uint32_t rot = old >> 59u;
  (void)((xs >> rot) | (xs << ((-rot) & 31)));  // discard first
  *state += initstate;
  old = *state;
  *state = old * 6364136223846793005ULL + *inc;
  xs  = ((old >> 18u) ^ old) >> 27u;
  rot = old >> 59u;
  (void)((xs >> rot) | (xs << ((-rot) & 31)));  // discard second
}

KOKKOS_INLINE_FUNCTION
uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc)
{
  uint64_t old = *state;
  *state = old * 6364136223846793005ULL + *inc;
  uint32_t xs  = ((old >> 18u) ^ old) >> 27u;
  uint32_t rot = old >> 59u;
  return (xs >> rot) | (xs << ((-rot) & 31));
}

KOKKOS_INLINE_FUNCTION
float gpu_rand01(uint64_t *state, uint64_t *inc)
{
  return (float)gpu_pcg32_random_r(state, inc) * INV_UINT_MAX;
}

// Murmur hash 64-bit (host + device)
KOKKOS_INLINE_FUNCTION
uint64_t mmhash64(const void *key, int len, unsigned int seed)
{
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;
  uint64_t h = (uint64_t)seed ^ ((uint64_t)len * m);
  const uint64_t *data = (const uint64_t *)key;
  const uint64_t *end  = data + (len / 8);
  while (data != end) {
    uint64_t k = *data++;
    k *= m; k ^= k >> r; k *= m;
    h ^= k; h *= m;
  }
  const unsigned char *data2 = (const unsigned char*)data;
  switch (len & 7) {
    case 7: h ^= (uint64_t)data2[6] << 48; [[fallthrough]];
    case 6: h ^= (uint64_t)data2[5] << 40; [[fallthrough]];
    case 5: h ^= (uint64_t)data2[4] << 32; [[fallthrough]];
    case 4: h ^= (uint64_t)data2[3] << 24; [[fallthrough]];
    case 3: h ^= (uint64_t)data2[2] << 16; [[fallthrough]];
    case 2: h ^= (uint64_t)data2[1] << 8;  [[fallthrough]];
    case 1: h ^= (uint64_t)data2[0]; h *= m; break;
    default: break;
  }
  h ^= h >> r; h *= m; h ^= h >> r;
  return h;
}

// ============================================================
// Host-side PCG (for host parallel-tempering logic)
// ============================================================
static inline void h_pcg32_srandom_r(uint64_t *state, uint64_t *inc,
                                      uint64_t initstate, uint64_t initseq)
{
  *state = 0U; *inc = (initseq << 1u) | 1u;
  gpu_pcg32_random_r(state, inc); *state += initstate;
  gpu_pcg32_random_r(state, inc);
}
static inline uint32_t h_pcg32_random_r(uint64_t *s, uint64_t *i)
{ return gpu_pcg32_random_r(s, i); }
static inline double gpu_rand01_host(uint64_t *s, uint64_t *i)
{ return (double)h_pcg32_random_r(s, i) * INV_UINT_MAX; }

// ============================================================
// Fragmented index
// ============================================================
struct findex_t { int f; int i; };

// ============================================================
// Heap (inline, for host use)
// ============================================================
struct hnode_t { float data; findex_t coord; };
struct minHeap_t { int size; hnode_t *elem; };

static minHeap_t initMinHeap() { return {0, nullptr}; }
static void hswap(hnode_t *a, hnode_t *b) { hnode_t t=*a; *a=*b; *b=t; }
static void heapify(minHeap_t *hp, int i) {
  int s = (2*i+1 < hp->size && hp->elem[2*i+1].data < hp->elem[i].data) ? 2*i+1 : i;
  if (2*i+2 < hp->size && hp->elem[2*i+2].data < hp->elem[s].data) s = 2*i+2;
  if (s != i) { hswap(&hp->elem[i], &hp->elem[s]); heapify(hp, s); }
}
static void insertNode(minHeap_t *hp, float data, findex_t frag) {
  hp->elem = (hnode_t*)realloc(hp->elem, (hp->size+1)*sizeof(hnode_t));
  hnode_t nd; nd.data = data; nd.coord = frag;
  int i = hp->size++;
  while (i && nd.data < hp->elem[i/2].data) { hp->elem[i] = hp->elem[i/2]; i = i/2; }
  hp->elem[i] = nd;
}
static hnode_t popRoot(minHeap_t *hp) {
  hnode_t out = hp->elem[0];
  hp->elem[0] = hp->elem[--(hp->size)];
  hp->elem = (hnode_t*)realloc(hp->elem, hp->size*sizeof(hnode_t));
  heapify(hp, 0);
  return out;
}

// ============================================================
// Host utility functions (from utils.cpp / utils.h)
// ============================================================
static double rtclock() {
  struct timeval tp; gettimeofday(&tp, nullptr);
  return tp.tv_sec + tp.tv_usec * 1.0e-6;
}

static void reset_array(float *a, int n, float val) {
  for (int i = 0; i < n; i++) a[i] = val;
}

static int floatcomp(const void *a, const void *b) {
  float x = *(float*)a, y = *(float*)b;
  return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static findex_t fgetleft(findex_t f, int ar) {
  findex_t out = f; out.i--;
  if (out.i < 0) { out.f--; out.i = ar-1; }
  if (out.f < 0) out = {-1,-1};
  return out;
}
static void fgoleft(findex_t *f, int ar) { *f = fgetleft(*f, ar); }

static void newtemp(float *aT, int *ar, int *R, findex_t l) {
  findex_t left = fgetleft(l, *ar);
  aT[*ar] = (aT[l.i] + aT[left.i]) / 2.0f;
  (*ar)++; (*R)++;
}
static void rebuild_temps(float *aT, int R, int ar) {
  float *flat = (float*)malloc(R * sizeof(float));
  for (int j = 0; j < ar; j++) flat[j] = aT[j];
  qsort(flat, R, sizeof(float), floatcomp);
  for (int j = 0; j < ar; j++) aT[j] = flat[j];
  free(flat);
}
static void insert_temps(float *aavex, float *aT, int *R, int *ar, int ains) {
  minHeap_t hp = initMinHeap();
  for (int j = *ar-1; j > 0; j--) insertNode(&hp, aavex[j], {0, j});
  for (int i = 0; i < ains && hp.size > 0; i++) {
    hnode_t nd = popRoot(&hp);
    newtemp(aT, ar, R, nd.coord);
  }
  free(hp.elem);
}
static void rebuild_indices(findex_t *arts, findex_t *atrs, int ar) {
  for (int j = 0; j < ar; j++) arts[j] = atrs[j] = {0, j};
}

static void printarrayfrag(float *a, int m, const char *name) {
  printf("%s = [", name);
  for (int j = 0; j < m; j++) printf("%f, ", a[j]);
  printf("]\n");
}
static void printindexarrayfrag(float *a, findex_t *ind, int m, const char *name) {
  printf("%s = [", name);
  for (int j = 0; j < m; j++) printf("%f, ", a[ind[j].i]);
  printf("]\n");
}

// ============================================================
// Kokkos kernel: PRNG setup (one state per spin)
// ============================================================
void kernel_gpupcg_setup(Kokkos::View<uint64_t*> pcga,
                         Kokkos::View<uint64_t*> pcgb,
                         int N, uint64_t seed, uint64_t seq,
                         int offset)
{
  Kokkos::parallel_for(
    "gpupcg_setup",
    Kokkos::RangePolicy<>(offset, offset + N),
    KOKKOS_LAMBDA(int x) {
      int lx = x - offset;
      uint64_t tseed = (uint64_t)lx + seed;
      uint64_t hseed = mmhash64(&tseed, sizeof(uint64_t), 17);
      uint64_t hseq  = mmhash64(&seq,   sizeof(uint64_t), 47);
      gpu_pcg32_srandom_r(&pcga(x), &pcgb(x), hseed, hseq);
    });
}

// ============================================================
// Kokkos kernel: reset lattice to random ±1 (one state per spin)
// ============================================================
void kernel_reset_random(Kokkos::View<int*> s, int N,
                         Kokkos::View<uint64_t*> pcga,
                         Kokkos::View<uint64_t*> pcgb)
{
  Kokkos::parallel_for(
    "reset_random",
    Kokkos::RangePolicy<>(0, N),
    KOKKOS_LAMBDA(int x) {
      uint64_t ls = pcga(x), li = pcgb(x);
      float v = (float)(int)(gpu_rand01(&ls, &li) + 0.5f);
      s(x) = 1 - 2*(int)v;
      pcga(x) = ls; pcgb(x) = li;
    });
}

template <typename T>
void kernel_reset_val(Kokkos::View<T*> a, int N, T val) {
  Kokkos::parallel_for("reset_val", N, KOKKOS_LAMBDA(int i){ a(i) = val; });
}

// ============================================================
// Kokkos kernel: energy reduction
// ============================================================
void kernel_redenergy(Kokkos::View<int*> s, int L,
                      Kokkos::View<int*> dH, float h,
                      float *out)
{
  float energy = 0.f;
  Kokkos::parallel_reduce(
    "redenergy",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{L,L,L}),
    KOKKOS_LAMBDA(int z, int y, int x, float &acc) {
      int id = C(x, y, z, L);
      int xp = (x+1 >= L) ? 0 : x+1;
      int yp = (y+1 >= L) ? 0 : y+1;
      int zp = (z+1 >= L) ? 0 : z+1;
      float sum = -(float)(s(id) * ((float)(s(C(xp,y,z,L)) +
                                             s(C(x,yp,z,L)) +
                                             s(C(x,y,zp,L))) + h*dH(id)));
      acc += sum;
    },
    energy);
  *out = energy;
}

// ============================================================
// Kokkos kernel: Metropolis update (flat checkerboard, one state per spin)
// Equivalent checkerboard decomposition: alt=0 updates spins where
// (x+y+z)%2==0, alt=1 updates spins where (x+y+z)%2==1.
// Each spin reads its 6 periodic neighbors directly from global memory.
// ============================================================
void kernel_metropolis(int N, int L,
                       Kokkos::View<int*>       s,
                       Kokkos::View<const int*> H,
                       float h, float B,
                       Kokkos::View<uint64_t*>  state,
                       Kokkos::View<uint64_t*>  inc,
                       int alt)
{
  Kokkos::parallel_for(
    "metropolis",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{L,L,L}),
    KOKKOS_LAMBDA(int z, int y, int x) {
      if (((x + y + z) & 1) != alt) return;

      int id = C(x, y, z, L);

      int xp = (x + 1 >= L) ? 0 : x + 1;
      int xm = (x - 1 <  0) ? L-1 : x - 1;
      int yp = (y + 1 >= L) ? 0 : y + 1;
      int ym = (y - 1 <  0) ? L-1 : y - 1;
      int zp = (z + 1 >= L) ? 0 : z + 1;
      int zm = (z - 1 <  0) ? L-1 : z - 1;

      float dh = (float)(s(id) * (
          (float)(s(C(xm,y,z,L)) + s(C(xp,y,z,L)) +
                  s(C(x,ym,z,L)) + s(C(x,yp,z,L)) +
                  s(C(x,y,zm,L)) + s(C(x,y,zp,L))) + h * H(id)));

      uint64_t ls = state(id), li = inc(id);
      int c = (int)(dh < EPSILON) |
              (int)(gpu_rand01(&ls, &li) < expf(dh * B));
      s(id) *= (1 - 2*c);
      state(id) = ls;
      inc(id)   = li;
    });
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
  int   L       = 32;
  int   R       = 1;
  int   atrials = 1;
  int   ains    = 1;
  int   apts    = 1;
  int   ams     = 1;
  uint64_t seed = 2;
  float TR      = 0.1f;
  float dT      = 0.1f;
  float h       = 0.1f;

  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-l") && i+2 < argc)
      { L = atoi(argv[i+1]); R = atoi(argv[i+2]); }
    else if (!strcmp(argv[i], "-t") && i+2 < argc)
      { TR = atof(argv[i+1]); dT = atof(argv[i+2]); }
    else if (!strcmp(argv[i], "-h") && i+1 < argc)
      { h = atof(argv[i+1]); }
    else if (!strcmp(argv[i], "-a") && i+4 < argc) {
      atrials = atoi(argv[i+1]); ains  = atoi(argv[i+2]);
      apts    = atoi(argv[i+3]); ams   = atoi(argv[i+4]);
    }
    else if (!strcmp(argv[i], "-z") && i+1 < argc)
      { seed = (uint64_t)atol(argv[i+1]); }
  }
  if ((L % 32) != 0) {
    fprintf(stderr, "L must be a multiple of 32\n"); return 1;
  }

  const int N    = L * L * L;
  const int Ra   = R + (atrials * ains);
  const int ar0  = R;
  const int rpool = Ra;

  uint64_t hpcgs, hpcgi;
  h_pcg32_srandom_r(&hpcgs, &hpcgi, seed, 1);
  seed = h_pcg32_random_r(&hpcgs, &hpcgi);

  float *T      = (float*)malloc(Ra * sizeof(float));
  float *aex    = (float*)calloc(rpool, sizeof(float));
  float *aavex  = (float*)calloc(rpool, sizeof(float));
  float *aexE   = (float*)malloc(rpool * sizeof(float));
  findex_t *arts = (findex_t*)malloc(rpool * sizeof(findex_t));
  findex_t *atrs = (findex_t*)malloc(rpool * sizeof(findex_t));
  float *aT      = (float*)malloc(rpool * sizeof(float));
  int  *h_dH    = (int*)malloc(N * sizeof(int));
  int  *h_mdlat = (int*)malloc(N * rpool * sizeof(int));

  printf("\tparameters:{\n");
  printf("\t\tL:                            %d\n", L);
  printf("\t\tvolume:                       %d\n", N);
  printf("\t\t[TR,dT]:                      [%f, %f]\n", TR, dT);
  printf("\t\t[atrials, ains, apts, ams]:   [%d, %d, %d, %d]\n",
         atrials, ains, apts, ams);
  printf("\t\tmag_field h:                  %f\n", h);
  printf("\t\treplicas:                     %d\n", R);
  printf("\t\tseed:                         %lu\n", seed);
  printf("\t}\n");

  Kokkos::initialize(argc, argv);
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = ExecSpace::memory_space;

    Kokkos::View<int*>      d_dH  ("dH",    N);
    Kokkos::View<int*>      d_mdlat("mdlat", N * rpool);
    // One PRNG state per spin (changed from N/4 to N for flat kernel)
    Kokkos::View<uint64_t*> d_pcga("pcga",  N * rpool);
    Kokkos::View<uint64_t*> d_pcgb("pcgb",  N * rpool);

    // Views for single-replica slices (subviews)
    int ar = ar0;

    // Initialize PRNG states on device (N states per replica)
    for (int k = 0; k < rpool; k++) {
      kernel_gpupcg_setup(d_pcga, d_pcgb, N,
                          seed + (uint64_t)(N * k), (uint64_t)k,
                          k * N);
    }

    // Initialize temperatures
    for (int i = 0; i < R; i++) T[i] = TR - (R-1-i) * dT;
    int count = 0;
    for (int j = 0; j < ar; j++) {
      arts[j] = atrs[j] = {0, j};
      aT[j] = TR - (float)(R-1-count) * dT;
      aex[j] = 0; count++;
    }

    FILE *fw = fopen("trials.dat", "w");
    fprintf(fw, "trial  av  min max\n");

    double total_ktime = 0.0;
    double start = rtclock();

    for (int trial = 0; trial < atrials; trial++) {
      printf("[trial %d of %d]\n", trial+1, atrials); fflush(stdout);

      // Reset dH with random ±1
      kernel_reset_random(d_dH, N, d_pcga, d_pcgb);

      reset_array(aex,   rpool, 0.0f);
      reset_array(aavex, rpool, 0.0f);

      seed = h_pcg32_random_r(&hpcgs, &hpcgi);

      for (int k = 0; k < ar; k++) {
        // reset lattice to +1
        auto mdlat_k = Kokkos::subview(d_mdlat,
            Kokkos::make_pair(k*N, (k+1)*N));
        kernel_reset_val<int>(mdlat_k, N, 1);
        kernel_gpupcg_setup(d_pcga, d_pcgb, N,
                            seed + (uint64_t)(N * k), (uint64_t)k,
                            k * N);
      }

      // Parallel tempering loop
      for (int p = 0; p < apts; p++) {
        double k_start = rtclock();

        // Metropolis sweeps
        for (int i = 0; i < ams; i++) {
          for (int k = 0; k < ar; k++) {
            auto mdlat_k = Kokkos::subview(d_mdlat,
                Kokkos::make_pair(k*N, (k+1)*N));
            auto pcga_k  = Kokkos::subview(d_pcga,
                Kokkos::make_pair(k*N, (k+1)*N));
            auto pcgb_k  = Kokkos::subview(d_pcgb,
                Kokkos::make_pair(k*N, (k+1)*N));

            float B = -2.0f / aT[atrs[k].i];
            kernel_metropolis(N, L, mdlat_k,
                              Kokkos::View<const int*>(d_dH),
                              h, B, pcga_k, pcgb_k, 0);
            kernel_metropolis(N, L, mdlat_k,
                              Kokkos::View<const int*>(d_dH),
                              h, B, pcga_k, pcgb_k, 1);
          }
        }
        Kokkos::fence();
        double k_end = rtclock();
        total_ktime += k_end - k_start;

        // Compute energies
        for (int k = 0; k < ar; k++) {
          auto mdlat_k = Kokkos::subview(d_mdlat,
              Kokkos::make_pair(k*N, (k+1)*N));
          kernel_redenergy(mdlat_k, L, d_dH, h, &aexE[k]);
        }

        // Exchange phase
        findex_t fnow = {0, ar-1};
        for (int k = R-1; k > 0; k--) {
          if ((k % 2) == (p % 2)) { fgoleft(&fnow, ar); continue; }
          findex_t fleft = fgetleft(fnow, ar);

          double delta = (1.0f/aT[fnow.i] - 1.0f/aT[fleft.i]) *
                         (aexE[arts[fleft.i].i] - aexE[arts[fnow.i].i]);

          double randme = gpu_rand01_host(&hpcgs, &hpcgi);
          if (delta < 0.0 || randme < exp(-delta)) {
            findex_t t1 = arts[fnow.i], t2 = arts[fleft.i];
            findex_t taux = atrs[t1.i], raux = arts[fnow.i];
            arts[fnow.i] = arts[fleft.i]; arts[fleft.i] = raux;
            atrs[t1.i]   = atrs[t2.i];   atrs[t2.i]   = taux;
            aex[fnow.i] += 1.0f;
          }
          fgoleft(&fnow, ar);
        }
        printf("\rpt........%d%%", 100*(p+1)/apts); fflush(stdout);
      }

      double avex = 0;
      for (int k = 1; k < ar; k++) avex += aavex[k] = 2.0*aex[k]/(double)apts;
      avex /= (double)(R-1);

      double minex = 1, maxex = 0;
      for (int k = 1; k < ar; k++) {
        if (aavex[k] < minex) minex = aavex[k];
        if (aavex[k] > maxex) maxex = aavex[k];
      }

      fprintf(fw, "%d %f  %f  %f\n", trial, avex, minex, maxex);
      fflush(fw);
      printf(" [<avg>=%.3f <min>=%.3f <max>=%.3f]\n\n", avex, minex, maxex);
      printarrayfrag(aex, ar, "aex");
      printarrayfrag(aavex, ar, "aavex");
      printindexarrayfrag(aexE, arts, ar, "aexE");

      insert_temps(aavex, aT, &R, &ar, ains);
      rebuild_temps(aT, R, ar);
      rebuild_indices(arts, atrs, ar);
    }

    double end = rtclock();
    printf("Total trial time %.2f secs\n", end - start);
    printf("Total kernel time (metropolis simulation) %.2f secs\n", total_ktime);
    fclose(fw);
  }
  Kokkos::finalize();

  free(T); free(aex); free(aavex); free(aexE);
  free(arts); free(atrs); free(aT);
  free(h_dH); free(h_mdlat);
  return 0;
}
