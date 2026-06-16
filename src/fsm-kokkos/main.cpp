/*
 FSM_GA - Finite-State Machine Genetic Algorithm, Kokkos port.
 Original: Texas State University (Martin Burtscher)
*/

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

// ---- Parameters (from ../fsm-cuda/parameters.h) ----
#define REPEAT   10
#define SEED     1234
#define FSMSIZE  8
#define TABSIZE  32768
#define POPCNT   1024
#define POPSIZE  256
#define CUTOFF   1

// ---- Device utility: LCG random ----
KOKKOS_INLINE_FUNCTION
unsigned int LCG_random(unsigned int* seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}
KOKKOS_INLINE_FUNCTION
void LCG_random_init(unsigned int* seed) {
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
}

// ---- FSMKernel ----
void FSMKernel(
    const int            length,
    const unsigned short* data,
    int*                  best,
    unsigned int*         rndstate,
    unsigned char*        bfsm,
    unsigned char*        same,
    int*                  smax,
    int*                  sbest,
    int*                  oldmax)
{
  using team_policy_t  = Kokkos::TeamPolicy<>;
  using team_member_t  = team_policy_t::member_type;
  using ScratchSpace   = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  // Scratch: next[FSMSIZE*2*POPSIZE] unsigned chars
  using ScratchUCView  = Kokkos::View<unsigned char*, ScratchSpace, Kokkos::MemoryUnmanaged>;

  const size_t scratch_bytes = (size_t)FSMSIZE * 2 * POPSIZE * sizeof(unsigned char);

  Kokkos::parallel_for(
    "FSMKernel",
    team_policy_t(POPCNT, POPSIZE)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const team_member_t& team) {
      ScratchUCView next_scratch(team.team_scratch(0), (size_t)FSMSIZE*2*POPSIZE);

      const int lid = team.team_rank();
      const int bid = team.league_rank();

      unsigned char* fsm = &next_scratch[lid * (FSMSIZE * 2)];

      // Thread 0 initialises per-block variables
      if (lid == 0) { oldmax[bid] = 0; same[bid] = 0; }
      team.team_barrier();

      const int id = lid + bid * POPSIZE;
      rndstate[id] = (unsigned int)SEED ^ (unsigned int)id;
      LCG_random_init(&rndstate[id]);

      // Initial FSM population
      for (int i = 0; i < FSMSIZE*2; i++)
        fsm[i] = (unsigned char)(LCG_random(rndstate+id) & (FSMSIZE-1));

      // Thread-local state table (per-thread, goes to device local memory)
      unsigned char state[TABSIZE];

      do {
        memset(state, 0, TABSIZE);
        int misses = 0;

        for (int i = 0; i < length; i++) {
          int d   = (int)data[i];
          int pc  = (d >> 1) & (TABSIZE-1);
          int bit = d & 1;
          int s   = (int)state[pc];
          misses += bit ^ (s & 1);
          state[pc] = fsm[s+s+bit];
        }

        // Determine best FSM within block
        if (lid == 0) {
          Kokkos::atomic_fetch_add(&best[2], 1);
          smax[bid]  = 0;
          sbest[bid] = 0;
        }
        team.team_barrier();

        // Atomic max for smax[bid]
        {
          int hits = length - misses;
          int old_smax = smax[bid];
          while (old_smax < hits) {
            int assumed = old_smax;
            old_smax = Kokkos::atomic_compare_exchange(&smax[bid], assumed, hits);
            if (old_smax == assumed) break;
          }
        }
        team.team_barrier();

        if (length - misses == smax[bid]) {
          int old_sb = sbest[bid];
          while (old_sb < lid) {
            int assumed = old_sb;
            old_sb = Kokkos::atomic_compare_exchange(&sbest[bid], assumed, lid);
            if (old_sb == assumed) break;
          }
        }
        team.team_barrier();

        int bit_flag = 0;
        if (sbest[bid] == lid) {
          same[bid]++;
          if (oldmax[bid] < smax[bid]) { oldmax[bid] = smax[bid]; same[bid] = 0; }
        } else {
          if ((LCG_random(rndstate+id) & 7) == 0) bit_flag = 1;
        }
        team.team_barrier();

        if (bit_flag) {
          for (int i = 0; i < FSMSIZE*2; i++) {
            unsigned int rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
            fsm[i] = (unsigned char)((next_scratch[i + sbest[bid]*(FSMSIZE*2)] ^ (unsigned char)rnd) & (FSMSIZE-1));
          }
        } else {
          for (int i = 0; i < FSMSIZE*2; i++) {
            unsigned int rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
            fsm[i] = (unsigned char)((fsm[i] & rnd) | (next_scratch[i + sbest[bid]*(FSMSIZE*2)] & ~(unsigned char)rnd));
          }
        }
      } while (same[bid] < CUTOFF);

      // Record best result globally using 64-bit packed compare-and-swap
      if (sbest[bid] == lid) {
        unsigned long long myresult = (unsigned long long)(length - misses);
        myresult = (myresult << 32) + (unsigned long long)bid;
        volatile unsigned long long* best_ull = (volatile unsigned long long*)best;
        unsigned long long current = *best_ull;
        while (myresult > current) {
          unsigned long long old =
              Kokkos::atomic_compare_exchange((unsigned long long*)best, current, myresult);
          if (old == current) break;
          current = old;
        }
        for (int i = 0; i < FSMSIZE*2; i++)
          bfsm[bid*(FSMSIZE*2)+i] = fsm[i];
      }
    }
  );
}

// ---- MaxKernel ----
void MaxKernel(int* best, const unsigned char* bfsm) {
  Kokkos::parallel_for("MaxKernel", Kokkos::RangePolicy<>(0, 1),
    KOKKOS_LAMBDA(int) {
      int id = best[0];
      for (int i = 0; i < FSMSIZE*2; i++)
        best[i+3] = (int)bfsm[id*(FSMSIZE*2)+i];
    });
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[]) {
  if (argc != 2) { fprintf(stderr, "usage: %s trace_length\n", argv[0]); exit(-1); }
  int length = atoi(argv[1]);

  assert(sizeof(unsigned short) == 2);
  assert(0 < length);
  assert((FSMSIZE & (FSMSIZE-1)) == 0);
  assert((TABSIZE & (TABSIZE-1)) == 0);
  assert((0 < FSMSIZE) && (FSMSIZE <= 256));
  assert((0 < TABSIZE) && (TABSIZE <= 32768));
  assert(0 < POPCNT);
  assert((0 < POPSIZE) && (POPSIZE <= 1024));
  assert(0 < CUTOFF);

  printf("%d\t#kernel execution times\n", REPEAT);
  printf("%d\t#fsm size\n",    FSMSIZE);
  printf("%d\t#entries\n",     length);
  printf("%d\t#tab size\n",    TABSIZE);
  printf("%d\t#blocks\n",      POPCNT);
  printf("%d\t#threads\n",     POPSIZE);
  printf("%d\t#cutoff\n",      CUTOFF);

  unsigned short* data = (unsigned short*)malloc(sizeof(unsigned short)*length);
  srand(123);
  for (int i = 0; i < length; i++) data[i] = (unsigned short)rand();

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<unsigned short*> d_data("data",   length);
    Kokkos::View<int*>            d_best("best",   FSMSIZE*2+3);
    Kokkos::View<unsigned int*>   d_rnd("rnd",     POPCNT*POPSIZE);
    Kokkos::View<unsigned char*>  d_bfsm("bfsm",   POPCNT*FSMSIZE*2);
    Kokkos::View<unsigned char*>  d_same("same",   POPCNT);
    Kokkos::View<int*>            d_smax("smax",   POPCNT);
    Kokkos::View<int*>            d_sbest("sbest",  POPCNT);
    Kokkos::View<int*>            d_oldmax("oldmax",POPCNT);

    {
      auto h_data = Kokkos::create_mirror_view(d_data);
      for (int i=0;i<length;i++) h_data(i)=data[i];
      Kokkos::deep_copy(d_data, h_data);
    }

    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);

    for (int iter = 0; iter < REPEAT; iter++) {
      // Reset best array
      Kokkos::parallel_for("init_best", Kokkos::RangePolicy<>(0, FSMSIZE*2+3),
        KOKKOS_LAMBDA(int i) { d_best(i) = 0; });
      Kokkos::fence();

      FSMKernel(length, d_data.data(), d_best.data(),
                d_rnd.data(), d_bfsm.data(),
                d_same.data(), d_smax.data(), d_sbest.data(), d_oldmax.data());
      Kokkos::fence();

      MaxKernel(d_best.data(), d_bfsm.data());
      Kokkos::fence();
    }

    gettimeofday(&endtime, NULL);
    double runtime = endtime.tv_sec + endtime.tv_usec/1000000.0
                   - starttime.tv_sec - starttime.tv_usec/1000000.0;
    printf("%.6lf\t#runtime [s]\n", runtime / REPEAT);

    // Copy results back
    int best[FSMSIZE*2+3];
    {
      auto h_best = Kokkos::create_mirror_view(d_best);
      Kokkos::deep_copy(h_best, d_best);
      for (int i=0;i<FSMSIZE*2+3;i++) best[i]=h_best(i);
    }

    int besthits    = best[1];
    int generations = best[2];
    printf("%.6lf\t#throughput [Gtr/s]\n",
           0.000000001 * POPSIZE * generations * length / (runtime / REPEAT));

    // CPU evaluation: saturating up/down counter
    unsigned char state[TABSIZE], fsm[FSMSIZE*2];
    int i,j,d,s,bit,pc,misses;
    for (i=0;i<FSMSIZE;i++) { fsm[i*2]=i?i-1:0; fsm[i*2+1]=(i<FSMSIZE-1)?i+1:FSMSIZE-1; }
    memset(state,0,TABSIZE); misses=0;
    for (i=0;i<length;i++) {
      d=(int)data[i]; pc=(d>>1)&(TABSIZE-1); bit=d&1;
      s=(int)state[pc]; misses+=bit^(((s+s)/FSMSIZE)&1); state[pc]=fsm[s+s+bit];
    }
    printf("%d\t#sudcnt hits\n", length-misses);
    printf("%d\t#GAfsm hits\n",  besthits);
    printf("%.3lf%%\t#sudcnt hits\n", 100.0*(length-misses)/length);
    printf("%.3lf%%\t#GAfsm hits\n\n", 100.0*besthits/length);

    // Verify GA result
    int trans[FSMSIZE][2];
    for (i=0;i<FSMSIZE;i++) for (j=0;j<2;j++) trans[i][j]=0;
    for (i=0;i<FSMSIZE*2;i++) fsm[i]=(unsigned char)best[i+3];
    memset(state,0,TABSIZE); misses=0;
    for (i=0;i<length;i++) {
      d=(int)data[i]; pc=(d>>1)&(TABSIZE-1); bit=d&1;
      s=(int)state[pc]; trans[s][bit]++; misses+=bit^(s&1);
      state[pc]=(unsigned char)fsm[s+s+bit];
    }
    bool ok = ((length-misses)==besthits);
    printf("%s\n", ok?"PASS":"FAIL");
  }
  Kokkos::finalize();
  free(data);
  return 0;
}
