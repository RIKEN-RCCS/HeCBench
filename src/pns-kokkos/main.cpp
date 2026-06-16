/*
 * Petri Net Simulation – Kokkos port
 * Original: Copyright (c) 2007 The Board of Trustees of the University of Illinois
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

// ─── types ────────────────────────────────────────────────────────────────────
typedef signed   int int32;
typedef unsigned int uint32;

using ExecSpace    = Kokkos::DefaultExecutionSpace;
using MemSpace     = ExecSpace::memory_space;
using ScratchSpace = ExecSpace::scratch_memory_space;
using TeamPolicy   = Kokkos::TeamPolicy<ExecSpace>;
using Member       = TeamPolicy::member_type;

template<typename T>
using ScratchView = Kokkos::View<T*, ScratchSpace, Kokkos::MemoryUnmanaged>;

// ─── Mersenne Twister constants ───────────────────────────────────────────────
#define MERS_N   624
#define MERS_M   397
#define MERS_R   31
#define MERS_U   11
#define MERS_S   7
#define MERS_T   15
#define MERS_L   18
#define MERS_A   0x9908B0DFu
#define MERS_B   0x9D2C5680u
#define MERS_C   0xEFC60000u
#define LOWER_MASK ((1Lu << MERS_R) - 1)
#define UPPER_MASK (0xFFFFFFFFu << MERS_R)

#define BLOCK_SIZE      256
#define BLOCK_SIZE_BITS   8
#define MAX_DEVICE_MEM  750000000

// ─── device Mersenne Twister ──────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
void RandomInit(uint32* mt, uint32 seed, int tid, const Member& team)
{
  if (tid == 0) {
    mt[0] = seed & 0xffffffffUL;
    for (int i = 1; i < MERS_N; i++)
      mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
  }
  team.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void BRandom(uint32* mt, int tid, const Member& team)
{
  uint32 y;
  int thdx;

  // step 1: indices 0 .. MERS_N-MERS_M-1  (0..226)
  if (tid < MERS_N - MERS_M) {
    y = (mt[tid] & UPPER_MASK) | (mt[tid+1] & LOWER_MASK);
    y = mt[tid + MERS_M] ^ (y >> 1) ^ ((y & 1) ? MERS_A : 0u);
  }
  team.team_barrier();
  if (tid < MERS_N - MERS_M)
    mt[tid] = y;
  team.team_barrier();

  // step 2: indices MERS_N-MERS_M .. 2*(MERS_N-MERS_M)-1  (227..453)
  thdx = tid + (MERS_N - MERS_M);
  if (tid < MERS_N - MERS_M) {
    y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
    y = mt[tid] ^ (y >> 1) ^ ((y & 1) ? MERS_A : 0u);
  }
  team.team_barrier();
  if (tid < MERS_N - MERS_M)
    mt[thdx] = y;
  team.team_barrier();

  // step 3: indices 2*(MERS_N-MERS_M) .. MERS_N-2  (454..622)
  thdx += (MERS_N - MERS_M);
  if (thdx < MERS_N - 1) {
    y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
    y = mt[tid + (MERS_N - MERS_M)] ^ (y >> 1) ^ ((y & 1) ? MERS_A : 0u);
  }
  team.team_barrier();
  if (thdx < MERS_N - 1)
    mt[thdx] = y;
  team.team_barrier();

  // step 4: index MERS_N-1  (623)
  if (tid == 0) {
    y = (mt[MERS_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[MERS_N-1] = mt[MERS_M-1] ^ (y >> 1) ^ ((y & 1) ? MERS_A : 0u);
  }
  team.team_barrier();

  // Tempering (all threads independently temper their own mt[tid])
  // Note: this uses the implicit y computed by tid 0 for tid 0.
  // Each thread applies tempering to mt[tid] just as in original code.
  // (The original code does this to the local y, not mt[tid]; we follow suit.)
  // The tempered value is NOT stored back – this matches the original which
  // uses BRandom's side effect (the updated mt[]) directly in run_trajectory.
  (void)y;  // tempering result unused beyond this point (mt[] is updated)
}

// ─── fire_transition ──────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
void fire_transition(char* g_places, int* conflict_array,
                     int tr, int tc, int step, int N,
                     int thd_thrd, int tid, const Member& team)
{
  int val1, val2, val3, to_update = 0;
  int mark1, mark2;

  if (tid < thd_thrd) {
    val1 = (tr == 0) ? (N+N)-1 : tr-1;
    val2 = (tr & 1)  ? (tc == N-1 ? 0 : tc+1) : tc;
    val3 = (tr == (N+N)-1) ? 0 : tr+1;
    mark1 = g_places[val1*N + val2];
    mark2 = g_places[tr*N   + tc];
    if (mark1 > 0 && mark2 > 0) {
      to_update = 1;
      conflict_array[tr*N + tc] = step;
    }
  }
  team.team_barrier();

  if (to_update) {
    to_update = ((step & 1) == (tr & 1)) ||
      ((conflict_array[val1*N + val2] != step) &&
       (conflict_array[val3*N + ((val2 == 0) ? N-1 : val2-1)] != step));
  }

  if (to_update) {
    g_places[val1*N + val2] = (char)(mark1 - 1);
    g_places[tr*N   + tc]   = (char)(mark2 - 1);
  }
  team.team_barrier();
  if (to_update) {
    g_places[val3*N + val2]++;
    g_places[tr*N + (tc == N-1 ? 0 : tc+1)]++;
  }
  team.team_barrier();
}

// ─── initialize_grid ─────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
void initialize_grid(uint32* mt, int* g_places, int nsquare2,
                     uint32 seed, int tid, int team_id, const Member& team)
{
  int loop_num = nsquare2 >> (BLOCK_SIZE_BITS + 2);
  for (int i = 0; i < loop_num; i++)
    g_places[tid + (i << BLOCK_SIZE_BITS)] = 0x01010101;
  if (tid < (nsquare2 >> 2) - (loop_num << BLOCK_SIZE_BITS))
    g_places[tid + (loop_num << BLOCK_SIZE_BITS)] = 0x01010101;

  RandomInit(mt, (uint32)(team_id) + seed, tid, team);
}

// ─── run_trajectory ───────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
void run_trajectory(uint32* mt, int* g_places, int n, int max_steps,
                    int tid, const Member& team)
{
  int step = 0, nsquare2 = (n+n)*n, val;

  while (step < max_steps) {
    BRandom(mt, tid, team);

    val = mt[tid] % nsquare2;
    fire_transition((char*)g_places, g_places + (nsquare2 >> 2),
                    val/n, val%n, step+7, n, BLOCK_SIZE, tid, team);

    val = mt[tid + BLOCK_SIZE] % nsquare2;
    fire_transition((char*)g_places, g_places + (nsquare2 >> 2),
                    val/n, val%n, step+11, n, BLOCK_SIZE, tid, team);

    if (tid < MERS_N - (BLOCK_SIZE << 1))
      val = mt[tid + (BLOCK_SIZE << 1)] % nsquare2;
    fire_transition((char*)g_places, g_places + (nsquare2 >> 2),
                    val/n, val%n, step+13, n, MERS_N - (BLOCK_SIZE<<1), tid, team);

    step += MERS_N >> 1;
  }
}

// ─── compute_reward_stat ──────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
void compute_reward_stat(uint32* mt, int* g_places,
                          float* g_vars, int* g_maxs,
                          int nsquare2, int tid, int team_id, const Member& team)
{
  float sum = 0;
  int   mx  = 0, data, temp;
  int loop_num = nsquare2 >> (BLOCK_SIZE_BITS + 2);

  for (int i = 0; i <= loop_num - 1; i++) {
    data = g_places[tid + (i << BLOCK_SIZE_BITS)];
    temp = data & 0xFF;         sum += (float)(temp*temp); if (temp>mx) mx=temp;
    temp = (data>>8)  & 0xFF;   sum += (float)(temp*temp); if (temp>mx) mx=temp;
    temp = (data>>16) & 0xFF;   sum += (float)(temp*temp); if (temp>mx) mx=temp;
    temp = (data>>24) & 0xFF;   sum += (float)(temp*temp); if (temp>mx) mx=temp;
  }
  int tail = (nsquare2 >> 2) & 0xFF;
  if (tid <= tail - 1) {
    data = g_places[tid + loop_num * BLOCK_SIZE];
    temp = data & 0xFF;         sum += (float)(temp*temp); if (temp>mx) mx=temp;
    temp = (data>>8)  & 0xFF;   sum += (float)(temp*temp); if (temp>mx) mx=temp;
    temp = (data>>16) & 0xFF;   sum += (float)(temp*temp); if (temp>mx) mx=temp;
    temp = (data>>24) & 0xFF;   sum += (float)(temp*temp); if (temp>mx) mx=temp;
  }

  // store partial sum & max in mt[] re-used as reduction buffer
  ((float*)mt)[tid]        = sum;
  mt[tid + BLOCK_SIZE]     = (uint32)mx;
  team.team_barrier();

  for (int i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
    if (tid < i) {
      ((float*)mt)[tid] += ((float*)mt)[tid + i];
      if (mt[tid + BLOCK_SIZE] < mt[tid + i + BLOCK_SIZE])
        mt[tid + BLOCK_SIZE] = mt[tid + i + BLOCK_SIZE];
    }
    team.team_barrier();
  }

  if (tid == 0) {
    g_vars[team_id] = ((float*)mt)[0] / nsquare2 - 1.0f;
    g_maxs[team_id] = (int)mt[BLOCK_SIZE];
  }
}

// ─── PetrinetKernel ───────────────────────────────────────────────────────────
// Called from within a TeamPolicy lambda; each team simulates one trajectory.
KOKKOS_INLINE_FUNCTION
void PetrinetKernel(uint32* mt, int* g_s,
                     float* g_v, int* g_m,
                     int n, int s, uint32 seed,
                     int tid, int team_id, const Member& team)
{
  int nsquare2 = n * n * 2;
  // each team's slice of g_s: ((nsquare2/4) + nsquare2) ints
  int per_team = (nsquare2 >> 2) + nsquare2;
  int* g_places = g_s + (long long)team_id * per_team;

  initialize_grid(mt, g_places, nsquare2, seed, tid, team_id, team);
  run_trajectory(mt, g_places, n, s, tid, team);
  compute_reward_stat(mt, g_places, g_v, g_m, nsquare2, tid, team_id, team);
}

// ─── host helpers ─────────────────────────────────────────────────────────────
static long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
}

static float h_vars_result[4];
static void compute_statistics(float* h_vars, int* h_maxs, int T) {
  double sum=0, sv=0, sm=0, smv=0;
  for(int i=0;i<T;i++){
    sum  += h_vars[i]; sv  += (double)h_vars[i]*h_vars[i];
    sm   += h_maxs[i]; smv += (double)h_maxs[i]*h_maxs[i];
  }
  h_vars_result[0] = (float)(sum/T);
  h_vars_result[1] = (float)(sv/T - (double)(h_vars_result[0])*(h_vars_result[0]));
  h_vars_result[2] = (float)(sm/T);
  h_vars_result[3] = (float)(smv/T - (double)(h_vars_result[2])*(h_vars_result[2]));
}

// ─── PetrinetOnDevice (Kokkos) ────────────────────────────────────────────────
static void PetrinetOnDevice(int N, int S, int T, float* h_vars, int* h_maxs)
{
  int nsquare2 = N * N * 2;
  int per_team = (nsquare2 >> 2) + nsquare2;  // ints per team's g_places slice
  long long unit_size = per_team * (long long)sizeof(int) + sizeof(float) + sizeof(int);
  int block_num = (int)(MAX_DEVICE_MEM / unit_size);
  if (block_num < 1) block_num = 1;
  printf("Number of thread blocks: %d\n", block_num);

  // Device allocations
  Kokkos::View<int*>   d_places("places", (long long)block_num * per_team);
  Kokkos::View<float*> d_vars  ("vars",   block_num);
  Kokkos::View<int*>   d_maxs  ("maxs",   block_num);

  // Scratch memory size: mt[MERS_N] per team
  size_t scratch_sz = ScratchView<uint32>::shmem_size(MERS_N);

  int* h_maxs_ptr = h_maxs;
  float* h_vars_ptr = h_vars;
  int n=N, s=S;

  for (int offset = 0; offset < T; offset += block_num) {
    int batch = (offset + block_num <= T) ? block_num : T - offset;
    uint32 seed = (uint32)(5489 * (offset + 1));

    int* places_raw = d_places.data();
    float* vars_raw = d_vars.data();
    int*   maxs_raw = d_maxs.data();

    auto t0 = get_time();
    Kokkos::parallel_for(
      "petri_kernel",
      TeamPolicy(batch, BLOCK_SIZE).set_scratch_size(0, Kokkos::PerTeam(scratch_sz)),
      KOKKOS_LAMBDA(const Member& team) {
        ScratchView<uint32> mt(team.team_scratch(0), MERS_N);
        int tid     = team.team_rank();
        int team_id = team.league_rank();
        PetrinetKernel(mt.data(), places_raw, vars_raw, maxs_raw,
                       n, s, seed, tid, team_id, team);
      }
    );
    Kokkos::fence();
    auto t1 = get_time();
    (void)t0; (void)t1;

    // copy results back
    {
      auto hv = Kokkos::create_mirror_view(d_vars);
      auto hm = Kokkos::create_mirror_view(d_maxs);
      Kokkos::deep_copy(hv, d_vars);
      Kokkos::deep_copy(hm, d_maxs);
      memcpy(h_vars_ptr, hv.data(), batch * sizeof(float));
      memcpy(h_maxs_ptr, hm.data(), batch * sizeof(int));
    }
    h_vars_ptr += batch;
    h_maxs_ptr += batch;
  }
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
  if (argc < 4) {
    printf("Usage: %s N S T\n", argv[0]);
    printf("  N: grid size (2N x 2N places)\n");
    printf("  S: max steps per trajectory\n");
    printf("  T: number of trajectories\n");
    return -1;
  }
  int N = atoi(argv[1]);
  int S = atoi(argv[2]);
  int T = atoi(argv[3]);
  if (N < 1 || S < 1 || T < 1) return -1;

  float* h_vars = (float*)malloc(T * sizeof(float));
  int*   h_maxs = (int*)  malloc(T * sizeof(int));

  Kokkos::initialize(argc, argv);
  {
    auto t0 = get_time();
    PetrinetOnDevice(N, S, T, h_vars, h_maxs);
    auto t1 = get_time();
    printf("Total device execution time: %.2f s\n", (t1 - t0) / 1e6f);
  }
  Kokkos::finalize();

  compute_statistics(h_vars, h_maxs, T);
  printf("petri N=%d S=%d T=%d\n", N, S, T);
  printf("mean_vars: %f    var_vars: %f\n", h_vars_result[0], h_vars_result[1]);
  printf("mean_maxs: %f    var_maxs: %f\n", h_vars_result[2], h_vars_result[3]);

  free(h_vars);
  free(h_maxs);
  return 0;
}
