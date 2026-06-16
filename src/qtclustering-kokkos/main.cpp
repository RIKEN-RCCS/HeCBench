/*
 * Kokkos port of QTC Clustering (Rodinia suite).
 * Original: qtclustering-omp/
 *
 * Key translation:
 *   omp target teams num_teams(N) thread_limit(T)  → Kokkos::TeamPolicy(N, T)
 *   omp_get_team_num()    → team.league_rank()
 *   omp_get_thread_num()  → team.team_rank()
 *   omp_get_num_threads() → team.team_size()
 *   #pragma omp barrier   → team.team_barrier()
 *   team-scoped arrays    → Kokkos scratch memory (level 0)
 */

#include <Kokkos_Core.hpp>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <fstream>
#include <set>
#include <map>

// ============================================================
// Tuning parameters (from tuningParameters.h)
// ============================================================
#define THREADSPERBLOCK     64
#define SM_COUNT            24
#define OVR_SBSCR_FACTOR    24
#define GPU_MIN_SATURATION_FACTOR 32
#define INVALID_POINT_MARKER -42

#ifdef MIN
#undef MIN
#endif
#define MIN(_X, _Y) (((_X) < (_Y)) ? (_X) : (_Y))
#ifdef MAX
#undef MAX
#endif
#define MAX(_X, _Y) (((_X) > (_Y)) ? (_X) : (_Y))

// ============================================================
// Data generation (from libdata.cpp / libdata.h)
// ============================================================
#define MAX_WIDTH  20.0f
#define MAX_HEIGHT 20.0f

static inline float frand_local(void) { return (float)random() / RAND_MAX; }

float *generate_synthetic_data(float **rslt_mtrx, int **indr_mtrx,
                                int *max_degree, float threshold, int N,
                                int /*matrix_type_mask*/)
{
  float *points = (float *)malloc(2 * N * sizeof(float));
  float min_dim = MAX_WIDTH < MAX_HEIGHT ? MAX_WIDTH : MAX_HEIGHT;
  int count = 0;
  while (count < N) {
    float cntr_x = frand_local() * MAX_WIDTH;
    float cntr_y = frand_local() * MAX_HEIGHT;
    float R = frand_local() * min_dim / 2;
    int group_cnt = random() % (N / 30);
    if (group_cnt > N - count) group_cnt = N - count;
    while (group_cnt > 0) {
      float sign = (frand_local() < 0.5f) ? -1.0f : 1.0f;
      float r = frand_local() * R;
      float dx = (2.0f * frand_local() - 1.0f) * r;
      float dy = sqrtf(r * r - dx * dx) * sign;
      float x = cntr_x + dx, y = cntr_y + dy;
      if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
      points[2 * count]     = x;
      points[2 * count + 1] = y;
      count++; group_cnt--;
    }
  }

  float threshold_sq = threshold * threshold;
  int D = 0;
  for (int i = 0; i < N; i++) {
    int delta = 0;
    for (int j = 0; j < N; j++) {
      if (j == i) continue;
      float dx = points[2*i] - points[2*j];
      float dy = points[2*i+1] - points[2*j+1];
      if (dx*dx + dy*dy < threshold_sq) delta++;
    }
    if (delta > D) D = delta;
  }

  float *dist_mtrx  = (float *)malloc(sizeof(float) * N * D);
  int   *index_mtrx = (int *)  malloc(sizeof(int)   * N * D);

  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) { dist_mtrx[i*D+j]  = FLT_MAX; index_mtrx[i*D+j] = -1; }

  for (int i = 0; i < N; i++) {
    int delta = 0;
    for (int j = 0; j < N; j++) {
      if (j == i) continue;
      float dx = points[2*i] - points[2*j];
      float dy = points[2*i+1] - points[2*j+1];
      float dist_sq = dx*dx + dy*dy;
      if (dist_sq < threshold_sq) {
        dist_mtrx [i*D + delta] = sqrtf(dist_sq);
        index_mtrx[i*D + delta] = j;
        delta++;
      }
    }
  }
  *max_degree = D;
  *rslt_mtrx  = dist_mtrx;
  *indr_mtrx  = index_mtrx;
  return points;
}

// ============================================================
// Kokkos view types
// ============================================================
using DevExec  = Kokkos::DefaultExecutionSpace;
using TeamPol  = Kokkos::TeamPolicy<>;
using MemberT  = TeamPol::member_type;
using ScratchF = Kokkos::View<float*, DevExec::scratch_memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ScratchI = Kokkos::View<int*,   DevExec::scratch_memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// ============================================================
// generate_candidate_cluster_compact_storage (Kokkos TeamPolicy version)
//
// Maps the OMP team-parallel function to Kokkos:
//   - dist_array/point_index_array come from team scratch memory
//   - #pragma omp barrier  → team.team_barrier()
//   - omp_get_thread_num() → team.team_rank()
//   - omp_get_num_threads()→ team.team_size()
//   - shared scalars (flag,cnt,latest_point) are maintained consistently
//     by all threads (each thread writes the same value after each barrier)
// ============================================================
KOKKOS_INLINE_FUNCTION
int generate_candidate_cluster_compact_storage(
    const MemberT &team,
    ScratchF       dist_array,
    ScratchI       point_index_array,
    const int      seed_point,
    const int      degree,
    char          *Ai_mask,
    const float   *compact_storage_dist_matrix,
    const char    *clustered_pnts_mask,
    const int     *indr_mtrx,
    float         *dist_to_clust,
    const int      point_count,
    const int      N0,
    const int      max_degree,
    int           *candidate_cluster,
    const float    threshold)
{
  int tid            = team.team_rank();
  int curThreadCount = team.team_size();

  // Clear Ai_mask
  for (int i = 0; i + tid < N0; i += curThreadCount)
    Ai_mask[i + tid] = 0;
  // Clear dist_to_clust
  for (int i = 0; i + tid < max_degree; i += curThreadCount)
    dist_to_clust[i + tid] = 0.0f;

  bool flag        = true;
  int  cnt         = 1;
  int  latest_point = seed_point;

  if (tid == 0) {
    if (candidate_cluster != nullptr) candidate_cluster[0] = seed_point;
    Ai_mask[seed_point] = 1;
  }
  team.team_barrier();

  int seed_p_off = seed_point * max_degree;

  // Prefetch 12 candidate points per thread
  int cand_pnt_0  = -1, cand_pnt_1  = -1, cand_pnt_2  = -1, cand_pnt_3  = -1;
  int cand_pnt_4  = -1, cand_pnt_5  = -1, cand_pnt_6  = -1, cand_pnt_7  = -1;
  int cand_pnt_8  = -1, cand_pnt_9  = -1, cand_pnt_10 = -1, cand_pnt_11 = -1;

#define FETCH_POINT_K(_CAND_PNT_, _I_) \
  do { \
    int _tmp_index_ = (_I_) * curThreadCount + tid; \
    if (_tmp_index_ >= max_degree) break; \
    _CAND_PNT_ = indr_mtrx[seed_p_off + _tmp_index_]; \
  } while (0)

  FETCH_POINT_K(cand_pnt_0,  0);
  FETCH_POINT_K(cand_pnt_1,  1);
  FETCH_POINT_K(cand_pnt_2,  2);
  FETCH_POINT_K(cand_pnt_3,  3);
  FETCH_POINT_K(cand_pnt_4,  4);
  FETCH_POINT_K(cand_pnt_5,  5);
  FETCH_POINT_K(cand_pnt_6,  6);
  FETCH_POINT_K(cand_pnt_7,  7);
  FETCH_POINT_K(cand_pnt_8,  8);
  FETCH_POINT_K(cand_pnt_9,  9);
  FETCH_POINT_K(cand_pnt_10, 10);
  FETCH_POINT_K(cand_pnt_11, 11);

  team.team_barrier();

#define COMPUTE_DIAMETER_K(_CAND_PNT_, _CURR_DIST_, _I_) \
  do { \
    if ((_CAND_PNT_) < 0) break; \
    if ((_CAND_PNT_) == seed_point) break; \
    int _tmp_idx_ = (_I_) * curThreadCount + tid; \
    float _cd_ = dist_to_clust[_tmp_idx_]; \
    if ((_cd_ > threshold) || (0 != Ai_mask[(_CAND_PNT_)]) || \
        (0 != clustered_pnts_mask[(_CAND_PNT_)])) { \
      _CAND_PNT_ = seed_point; break; \
    } \
    float _dist_new_ = threshold + 1.0f; \
    for (int _j_ = last_index_checked; _j_ < max_degree; _j_++) { \
      int _tp_ = indr_mtrx[latest_p_off + _j_]; \
      if ((_tp_ > (_CAND_PNT_)) || (_tp_ < 0)) { last_index_checked = _j_; break; } \
      if (_tp_ == (_CAND_PNT_)) { _dist_new_ = compact_storage_dist_matrix[latest_p_off + _j_]; break; } \
    } \
    float _diam_; \
    if (_dist_new_ > _cd_) { _diam_ = _dist_new_; dist_to_clust[_tmp_idx_] = _diam_; } \
    else                   { _diam_ = _cd_; } \
    (_CURR_DIST_) = _diam_; \
    if (_diam_ < min_dist) { min_dist = _diam_; point_index = (_CAND_PNT_); } \
  } while (0)

  while (cnt < point_count && flag) {
    int   point_index = -1;
    float min_dist    = 3.0f * threshold;
    int   last_index_checked = 0;
    int   latest_p_off = latest_point * max_degree;
    float curr0=0,curr1=0,curr2=0,curr3=0,curr4=0,curr5=0;
    float curr6=0,curr7=0,curr8=0,curr9=0,curr10=0,curr11=0;
    float curr_i = 0;

    COMPUTE_DIAMETER_K(cand_pnt_0,  curr0,  0);
    COMPUTE_DIAMETER_K(cand_pnt_1,  curr1,  1);
    COMPUTE_DIAMETER_K(cand_pnt_2,  curr2,  2);
    COMPUTE_DIAMETER_K(cand_pnt_3,  curr3,  3);
    COMPUTE_DIAMETER_K(cand_pnt_4,  curr4,  4);
    COMPUTE_DIAMETER_K(cand_pnt_5,  curr5,  5);
    COMPUTE_DIAMETER_K(cand_pnt_6,  curr6,  6);
    COMPUTE_DIAMETER_K(cand_pnt_7,  curr7,  7);
    COMPUTE_DIAMETER_K(cand_pnt_8,  curr8,  8);
    COMPUTE_DIAMETER_K(cand_pnt_9,  curr9,  9);
    COMPUTE_DIAMETER_K(cand_pnt_10, curr10, 10);
    COMPUTE_DIAMETER_K(cand_pnt_11, curr11, 11);

    team.team_barrier();

    for (int i = 12; i * curThreadCount + tid < max_degree; i++) {
      int cand_pnt_i = -1;
      FETCH_POINT_K(cand_pnt_i, i);
      COMPUTE_DIAMETER_K(cand_pnt_i, curr_i, i);
    }
    team.team_barrier();

    dist_array[tid]        = min_dist;
    point_index_array[tid] = point_index;
    team.team_barrier();

    if (tid == 0) {
      for (int j = 1; j < curThreadCount; j++) {
        float d = dist_array[j];
        if ((d < min_dist) || (d == min_dist && point_index_array[j] < point_index_array[0])) {
          min_dist = d;
          point_index_array[0] = point_index_array[j];
        }
      }
      if (min_dist > threshold) point_index_array[0] = -1;
    }
    team.team_barrier();

    int min_G_index = point_index_array[0];

    if (min_G_index >= 0) {
      if (tid == 0) {
        Ai_mask[min_G_index] = 1;
        if (candidate_cluster != nullptr) candidate_cluster[cnt] = min_G_index;
      }
      latest_point = min_G_index;
      cnt++;
    } else {
      flag = false;
    }
    team.team_barrier();
  }
  team.team_barrier();

  return cnt;
}

#undef FETCH_POINT_K
#undef COMPUTE_DIAMETER_K

// ============================================================
// calculate_participants
// ============================================================
void calculate_participants(int point_count, int node_count, int cwrank,
                             int *thread_block_count,
                             int *total_thread_block_count,
                             int *active_node_count)
{
  int ac_nd_cnt = node_count;
  if (point_count <= (node_count - 1) * SM_COUNT * GPU_MIN_SATURATION_FACTOR) {
    int K = SM_COUNT * GPU_MIN_SATURATION_FACTOR;
    ac_nd_cnt = (point_count + K - 1) / K;
  }
  int thr_blc_cnt, total_thr_blc_cnt;
  if (point_count >= ac_nd_cnt * SM_COUNT * OVR_SBSCR_FACTOR) {
    thr_blc_cnt = SM_COUNT * OVR_SBSCR_FACTOR;
    total_thr_blc_cnt = thr_blc_cnt * ac_nd_cnt;
  } else {
    thr_blc_cnt = point_count / ac_nd_cnt;
    if (cwrank < point_count % ac_nd_cnt) thr_blc_cnt++;
    total_thr_blc_cnt = point_count;
  }
  *active_node_count          = ac_nd_cnt;
  *thread_block_count         = thr_blc_cnt;
  *total_thread_block_count   = total_thr_blc_cnt;
  (void)total_thr_blc_cnt;
}

// ============================================================
// QTC main function
// ============================================================
void QTC(int point_count_in, float threshold, bool be_verbose)
{
  using namespace std;

  int max_degree, thread_block_count, total_thread_block_count, active_node_count;
  int cwrank = 0, node_count = 1, tpb, max_card, iter = 0;
  unsigned long int max_point_count;
  int point_count = point_count_in;

  float *pnts = generate_synthetic_data(
      /* dist_source */ nullptr,
      /* indr_mtrx   */ nullptr,
      &max_degree, threshold, point_count, 0);

  // Re-generate properly
  float *dist_source    = nullptr;
  int   *indr_mtrx_host = nullptr;
  free(pnts);
  pnts = generate_synthetic_data(&dist_source, &indr_mtrx_host,
                                  &max_degree, threshold, point_count, 0);

  assert(max_degree > 0);
  unsigned long int dst_matrix_elems = (unsigned long int)point_count * max_degree;

  calculate_participants(point_count, node_count, cwrank,
                          &thread_block_count, &total_thread_block_count, &active_node_count);

  // Allocate host arrays
  int   *ungrpd_pnts_indr_host = (int *)  malloc(sizeof(int)   * point_count);
  int   *cardnl                = (int *)  malloc(sizeof(int)   * thread_block_count * 2);
  int   *result                = (int *)  malloc(sizeof(int)   * point_count);
  int   *degrees               = (int *)  malloc(sizeof(int)   * point_count);
  char  *Ai_mask               = (char *) malloc(sizeof(char)  * thread_block_count * point_count);
  float *dist_to_clust         = (float *)malloc(sizeof(float) * thread_block_count * max_degree);
  char  *clustered_pnts_mask   = (char *) malloc(sizeof(char)  * point_count);

  for (int i = 0; i < point_count; i++) ungrpd_pnts_indr_host[i] = i;

  // ---- Copy host data to Kokkos device views ----
  using ViewF  = Kokkos::View<float*>;
  using ViewI  = Kokkos::View<int*>;
  using ViewC  = Kokkos::View<char*>;

  ViewF d_dist_source   ("dist_source",   dst_matrix_elems);
  ViewI d_indr_mtrx     ("indr_mtrx",     (unsigned long int)point_count * max_degree);
  ViewI d_ungrpd_pnts   ("ungrpd_pnts",   point_count);
  ViewI d_degrees       ("degrees",        point_count);
  ViewC d_Ai_mask       ("Ai_mask",        (unsigned long int)thread_block_count * point_count);
  ViewF d_dist_to_clust ("dist_to_clust",  (unsigned long int)thread_block_count * max_degree);
  ViewC d_clustered     ("clustered",      point_count);
  ViewI d_cardnl        ("cardnl",         thread_block_count * 2);
  ViewI d_result        ("result",         point_count);

  // Host mirrors
  auto hm_dist_source = Kokkos::create_mirror_view(d_dist_source);
  auto hm_indr_mtrx   = Kokkos::create_mirror_view(d_indr_mtrx);
  auto hm_ungrpd_pnts = Kokkos::create_mirror_view(d_ungrpd_pnts);
  auto hm_degrees     = Kokkos::create_mirror_view(d_degrees);
  auto hm_clustered   = Kokkos::create_mirror_view(d_clustered);
  auto hm_cardnl      = Kokkos::create_mirror_view(d_cardnl);
  auto hm_result      = Kokkos::create_mirror_view(d_result);

  for (unsigned long int i = 0; i < dst_matrix_elems; i++) hm_dist_source[i] = dist_source[i];
  for (int i = 0; i < point_count * max_degree; i++)       hm_indr_mtrx[i]   = indr_mtrx_host[i];
  for (int i = 0; i < point_count; i++)                    hm_ungrpd_pnts[i] = ungrpd_pnts_indr_host[i];

  Kokkos::deep_copy(d_dist_source,   hm_dist_source);
  Kokkos::deep_copy(d_indr_mtrx,     hm_indr_mtrx);
  Kokkos::deep_copy(d_ungrpd_pnts,   hm_ungrpd_pnts);

  // Initialize clustered_pnts_mask to 0
  Kokkos::parallel_for("init_clustered", point_count,
      KOKKOS_LAMBDA(int i) { d_clustered[i] = 0; });

  // Initialize dist_to_clust to 0
  Kokkos::parallel_for("init_dist_to_clust",
      (int)(thread_block_count * max_degree),
      KOKKOS_LAMBDA(int i) { d_dist_to_clust[i] = 0.0f; });

  tpb = (point_count > THREADSPERBLOCK) ? THREADSPERBLOCK : point_count;

  // ---- Compute degrees ----
  Kokkos::parallel_for("degrees",
    TeamPol(thread_block_count, tpb),
    KOKKOS_LAMBDA(const MemberT &team) {
      int curThreadCount = team.team_size();
      int tid = team.team_rank();
      int tblock_id = team.league_rank();
      int TB_count  = team.league_size();
      int local_pc  = (point_count + TB_count - 1) / TB_count;
      int starting  = tblock_id * local_pc;
      int offset    = starting * max_degree;

      for (int i = 0; i + tid < local_pc; i += curThreadCount) {
        int cnt = 0;
        for (int j = 0; j < max_degree; j++) {
          if (d_indr_mtrx[offset + (i + tid) * max_degree + j] >= 0) cnt++;
        }
        d_degrees[starting + i + tid] = cnt;
      }
    });
  Kokkos::fence();

  cout << "\nInitial ThreadBlockCount: " << thread_block_count
       << " PointCount: " << point_count
       << " Max degree: " << max_degree << "\n" << endl;

  max_point_count = point_count;
  tpb = THREADSPERBLOCK;

  // Scratch memory: dist_array[THREADSPERBLOCK] + point_index_array[THREADSPERBLOCK]
  int scratch_per_team = ScratchF::shmem_size(THREADSPERBLOCK) +
                         ScratchI::shmem_size(THREADSPERBLOCK);

  double qtc_time = 0.0, trim_time = 0.0, update_time = 0.0;

  do {
    ++iter;
    int winner_node = -1, winner_index = -1;

    calculate_participants(point_count, node_count, cwrank,
                            &thread_block_count, &total_thread_block_count, &active_node_count);

    auto t_start = chrono::steady_clock::now();

    // ---- QTC kernel ----
    // Re-allocate d_cardnl if thread_block_count changed
    // (use oversized allocation initially — simpler for single-node run)
    Kokkos::parallel_for("qtc_kernel",
      TeamPol(thread_block_count, tpb).set_scratch_size(0, Kokkos::PerTeam(scratch_per_team)),
      KOKKOS_LAMBDA(const MemberT &team) {
        ScratchF dist_array(team.team_scratch(0), THREADSPERBLOCK);
        ScratchI point_index_array(team.team_scratch(0), THREADSPERBLOCK);

        int max_cardinality       = -1;
        int max_cardinality_index = -1;

        int tid       = team.team_rank();
        int tblock_id = team.league_rank();
        char  *Ai_mask_ptr       = d_Ai_mask.data() + (long)tblock_id * max_point_count;
        float *dist_to_clust_ptr = d_dist_to_clust.data() + (long)tblock_id * max_degree;
        int    base_offset       = tblock_id; // single node: cwrank=0

        for (int i = base_offset; i < point_count; i += total_thread_block_count) {
          int seed_index = d_ungrpd_pnts[i];
          int degree     = d_degrees[seed_index];
          if (degree <= max_cardinality) continue;

          int cnt = generate_candidate_cluster_compact_storage(
              team, dist_array, point_index_array,
              seed_index, degree, Ai_mask_ptr,
              d_dist_source.data(),
              d_clustered.data(),
              d_indr_mtrx.data(),
              dist_to_clust_ptr,
              point_count, (int)max_point_count, max_degree,
              nullptr, threshold);

          if (cnt > max_cardinality) {
            max_cardinality       = cnt;
            max_cardinality_index = seed_index;
          }
        }

        if (tid == 0) {
          d_cardnl[tblock_id * 2]     = max_cardinality;
          d_cardnl[tblock_id * 2 + 1] = max_cardinality_index;
        }
      });
    Kokkos::fence();

    auto t_end = chrono::steady_clock::now();
    qtc_time += chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count();

    // ---- Reduce cardinalities ----
    if (thread_block_count > 1) {
      Kokkos::parallel_for("reduce_card",
        TeamPol(1, 1),
        KOKKOS_LAMBDA(const MemberT &) {
          int mc = -1, wi = -1;
          for (int i = 0; i < thread_block_count * 2; i += 2) {
            if (d_cardnl[i] > mc) { mc = d_cardnl[i]; wi = d_cardnl[i+1]; }
          }
          d_cardnl[0] = mc;
          d_cardnl[1] = wi;
        });
      Kokkos::fence();
    }

    Kokkos::deep_copy(hm_cardnl, d_cardnl);
    max_card     = hm_cardnl[0];
    winner_index = hm_cardnl[1];
    winner_node  = 0;

    if (be_verbose)
      cout << "[0] Cluster Cardinality: " << max_card
           << " (index: " << winner_index << ")" << endl;

    // ---- Trim kernel ----
    t_start = chrono::steady_clock::now();

    // Scratch for trim: dist_array + point_index_array + tmp_pnts + cnt_sh + flag_sh
    int trim_scratch = ScratchF::shmem_size(THREADSPERBLOCK) +
                       ScratchI::shmem_size(THREADSPERBLOCK) +
                       ScratchI::shmem_size(THREADSPERBLOCK) + // tmp_pnts
                       ScratchI::shmem_size(2);                // cnt_sh, flag_sh

    Kokkos::parallel_for("trim_kernel",
      TeamPol(1, tpb).set_scratch_size(0, Kokkos::PerTeam(trim_scratch)),
      KOKKOS_LAMBDA(const MemberT &team) {
        ScratchF dist_array(team.team_scratch(0), THREADSPERBLOCK);
        ScratchI point_index_array(team.team_scratch(0), THREADSPERBLOCK);
        ScratchI tmp_pnts(team.team_scratch(0), THREADSPERBLOCK);
        ScratchI sh_state(team.team_scratch(0), 2); // [0]=cnt_sh, [1]=flag_sh

        int tid            = team.team_rank();
        int curThreadCount = team.team_size();
        int degree         = d_degrees[winner_index];

        generate_candidate_cluster_compact_storage(
            team, dist_array, point_index_array,
            winner_index, degree, d_Ai_mask.data(),
            d_dist_source.data(),
            d_clustered.data(),
            d_indr_mtrx.data(),
            d_dist_to_clust.data(),
            point_count, (int)max_point_count, max_degree,
            d_result.data(), threshold);

        if (tid == 0) { sh_state[0] = 0; sh_state[1] = 0; }
        team.team_barrier();

        for (int i = 0; i + tid < point_count; i += curThreadCount) {
          tmp_pnts[tid] = d_ungrpd_pnts[i + tid];
          int pnt = tmp_pnts[tid];

          if (1 == d_Ai_mask[pnt]) {
            sh_state[1] = 1; // flag_sh = true
            tmp_pnts[tid] = INVALID_POINT_MARKER;
          } else {
            d_ungrpd_pnts[sh_state[0] + tid] = pnt;
          }
          team.team_barrier();

          if (tid == 0) {
            if (sh_state[1]) {
              int cnt = sh_state[0];
              for (int j = 0; j < curThreadCount && i + j < point_count; j++) {
                if (INVALID_POINT_MARKER != tmp_pnts[j]) {
                  d_ungrpd_pnts[cnt] = tmp_pnts[j];
                  cnt++;
                }
              }
              sh_state[0] = cnt;
            } else {
              sh_state[0] += curThreadCount;
            }
            sh_state[1] = 0;
          }
          team.team_barrier();
        }
      });
    Kokkos::fence();

    t_end = chrono::steady_clock::now();
    trim_time += chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count();

    // Copy result back
    Kokkos::deep_copy(hm_result, d_result);

    t_start = chrono::steady_clock::now();

    // ---- Update clustered points mask ----
    Kokkos::parallel_for("update_clustered",
      TeamPol(1, tpb),
      KOKKOS_LAMBDA(const MemberT &team) {
        int tid            = team.team_rank();
        int curThreadCount = team.team_size();
        for (int i = 0; i + tid < (int)max_point_count; i += curThreadCount)
          d_clustered[i + tid] |= d_Ai_mask[i + tid];
      });
    Kokkos::fence();

    t_end = chrono::steady_clock::now();
    update_time += chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count();

    point_count -= max_card;

  } while (max_card > 1 && point_count);

  cout << "QTC complete. Iterations: " << iter << endl;
  cout << "qtc:    " << qtc_time    * 1e-9f << " s\n";
  cout << "trim:   " << trim_time   * 1e-9f << " s\n";
  cout << "update: " << update_time * 1e-9f << " s\n";
  cout << "total:  " << (qtc_time + trim_time + update_time) * 1e-9f << " s\n";

  free(pnts);
  free(dist_source);
  free(indr_mtrx_host);
  free(ungrpd_pnts_indr_host);
  free(cardnl);
  free(result);
  free(degrees);
  free(Ai_mask);
  free(dist_to_clust);
  free(clustered_pnts_mask);
}

// ============================================================
// main
// ============================================================
int main(int argc, char *argv[])
{
  int   point_count = 4096;
  float threshold   = 1.0f;
  bool  verbose     = false;

  // Accept --size N where:  1→4k, 2→8k, 3→16k, 4→16k, 5→26k
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--size" && i + 1 < argc) {
      int sz = atoi(argv[++i]);
      switch (sz) {
        case 1: point_count =  4 * 1024; break;
        case 2: point_count =  8 * 1024; break;
        case 3: point_count = 16 * 1024; break;
        case 4: point_count = 16 * 1024; break;
        case 5: point_count = 26 * 1024; break;
        default: fprintf(stderr, "unsupported size\n"); return 1;
      }
    } else if (std::string(argv[i]) == "--threshold" && i + 1 < argc) {
      threshold = atof(argv[++i]);
    } else if (std::string(argv[i]) == "--verbose") {
      verbose = true;
    }
  }

  Kokkos::initialize(argc, argv);
  {
    QTC(point_count, threshold, verbose);
  }
  Kokkos::finalize();
  return 0;
}
