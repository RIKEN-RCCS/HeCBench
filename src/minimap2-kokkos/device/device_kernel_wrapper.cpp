#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <Kokkos_Core.hpp>
#include "datatypes.h"
#include "kernel_common.h"
#include "memory_scheduler.h"

// ──────────────────────────────────────────────────────────────────────────────
// Device helper: integer log2
// ──────────────────────────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
score_dt device_ilog2(const score_dt v)
{
  if (v < 2)   return 0;
  if (v < 4)   return 1;
  if (v < 8)   return 2;
  if (v < 16)  return 3;
  if (v < 32)  return 4;
  if (v < 64)  return 5;
  if (v < 128) return 6;
  if (v < 256) return 7;
  return 8;
}

// ──────────────────────────────────────────────────────────────────────────────
// Device helper: chaining DP score
// ──────────────────────────────────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
score_dt chain_dp_score(const anchor_dt *active,
                        const anchor_dt  curr,
                        const float      avg_qspan,
                        const int        max_dist_x,
                        const int        max_dist_y,
                        const int        bw,
                        const int        id)
{
  const anchor_dt act = active[id];

  if (curr.tag != act.tag) return NEG_INF_SCORE_GPU;

  score_dt dist_x = (score_dt)act.x - (score_dt)curr.x;
  if (dist_x == 0 || dist_x > max_dist_x) return NEG_INF_SCORE_GPU;

  score_dt dist_y = (score_dt)act.y - (score_dt)curr.y;
  if (dist_y > max_dist_y || dist_y <= 0) return NEG_INF_SCORE_GPU;

  score_dt dd    = dist_x > dist_y ? dist_x - dist_y : dist_y - dist_x;
  if (dd > bw) return NEG_INF_SCORE_GPU;

  score_dt min_d  = dist_y < dist_x ? dist_y : dist_x;
  score_dt log_dd = device_ilog2(dd);

  score_dt sc = min_d > (score_dt)act.w ? (score_dt)act.w : min_d;
  sc -= (score_dt)(dd * (0.01f * avg_qspan)) + (log_dd >> 1);
  return sc;
}

// ──────────────────────────────────────────────────────────────────────────────
// Scratch memory view types for one team
// ──────────────────────────────────────────────────────────────────────────────
using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using AnchorScratch  = Kokkos::View<anchor_dt*,  ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ScoreScratch   = Kokkos::View<score_dt*,   ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ParentScratch  = Kokkos::View<parent_dt*,  ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// ──────────────────────────────────────────────────────────────────────────────
// Kokkos device kernel wrapper
// ──────────────────────────────────────────────────────────────────────────────
void device_chain_kernel_wrapper(
    std::vector<control_dt> &cont,
    std::vector<anchor_dt>  &arg,
    std::vector<return_dt>  &ret,
    int max_dist_x, int max_dist_y, int bw)
{
  const int batch_count = (int)(cont.size() / PE_NUM);

  // Allocate and fill host arrays
  const int control_n = (int)cont.size();
  const int arg_n     = (int)arg.size();
  const int ret_n     = batch_count * TILE_SIZE * PE_NUM;

  ret.resize(ret_n);

  // ── Upload to device ────────────────────────────────────────────────────────
  Kokkos::View<control_dt*> d_control("d_control", control_n);
  Kokkos::View<anchor_dt*>  d_arg    ("d_arg",     arg_n);
  Kokkos::View<return_dt*>  d_ret    ("d_ret",     ret_n);

  {
    auto h_control = Kokkos::create_mirror_view(d_control);
    auto h_arg     = Kokkos::create_mirror_view(d_arg);
    memcpy(h_control.data(), cont.data(), control_n * sizeof(control_dt));
    memcpy(h_arg.data(),     arg.data(),  arg_n     * sizeof(anchor_dt));
    Kokkos::deep_copy(d_control, h_control);
    Kokkos::deep_copy(d_arg,     h_arg);
  }

  // Global persistent state arrays across batches
  Kokkos::View<score_dt**>  d_max_tracker_g("max_g", PE_NUM, BACK_SEARCH_COUNT_GPU);
  Kokkos::View<parent_dt**> d_j_tracker_g  ("j_g",   PE_NUM, BACK_SEARCH_COUNT_GPU);
  Kokkos::deep_copy(d_max_tracker_g, 0);
  Kokkos::deep_copy(d_j_tracker_g, -1);

  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;

  const size_t shmem_size =
    AnchorScratch::shmem_size(BACK_SEARCH_COUNT_GPU)
    + ScoreScratch::shmem_size(BACK_SEARCH_COUNT_GPU)
    + ParentScratch::shmem_size(BACK_SEARCH_COUNT_GPU);

  auto k_start = std::chrono::steady_clock::now();

  for (int batch = 0; batch < batch_count; batch++) {
    const int batch_base_ctrl = batch * PE_NUM;
    const int batch_base_arg  = batch * PE_NUM * TILE_SIZE_ACTUAL;
    const int batch_base_ret  = batch * PE_NUM * TILE_SIZE;

    // BLOCK_NUM teams (one per PE), each with BACK_SEARCH_COUNT_GPU threads
    Kokkos::parallel_for("chain_kernel",
      team_policy(BLOCK_NUM, BACK_SEARCH_COUNT_GPU)
        .set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
      KOKKOS_LAMBDA(const member_type &team) {
        const int block = team.league_rank(); // which PE/team
        const int id    = team.team_rank();   // thread within team

        // Allocate scratch arrays (Kokkos advances bump pointer per allocation)
        AnchorScratch active_sm    (team.team_scratch(0), BACK_SEARCH_COUNT_GPU);
        ScoreScratch  max_tracker_sm(team.team_scratch(0), BACK_SEARCH_COUNT_GPU);
        ParentScratch j_tracker_sm(team.team_scratch(0), BACK_SEARCH_COUNT_GPU);

        const int ofs     = block;
        const auto ctrl   = d_control[batch_base_ctrl + ofs];
        const int arg_off = batch_base_arg + ofs * TILE_SIZE_ACTUAL;

        // Load initial active anchors
        active_sm[id] = d_arg[arg_off + id];

        if (ctrl.is_new_read) {
          max_tracker_sm[id] = 0;
          j_tracker_sm[id]   = -1;
        } else {
          max_tracker_sm[id] = d_max_tracker_g(ofs, id);
          j_tracker_sm[id]   = d_j_tracker_g  (ofs, id);
        }

        team.team_barrier();

        for (int i = BACK_SEARCH_COUNT_GPU, curr_idx = 0; curr_idx < TILE_SIZE; i++, curr_idx++) {

          // Read current anchor and its running score
          team.team_barrier();
          const anchor_dt curr    = active_sm[i % BACK_SEARCH_COUNT_GPU];
          score_dt  f_curr  = max_tracker_sm[i % BACK_SEARCH_COUNT_GPU];
          parent_dt p_curr  = j_tracker_sm  [i % BACK_SEARCH_COUNT_GPU];
          if ((score_dt)curr.w >= f_curr) {
            f_curr = (score_dt)curr.w;
            p_curr = (parent_dt)-1;
          }

          // Load new anchor into the slot that curr just vacated
          team.team_barrier();
          if (id == i % BACK_SEARCH_COUNT_GPU) {
            active_sm[id]     = d_arg[arg_off + i];
            max_tracker_sm[id] = 0;
            j_tracker_sm[id]   = -1;
          }

          // Compute DP score from curr against active[id]
          team.team_barrier();
          score_dt sc = chain_dp_score(active_sm.data(), curr,
                                       ctrl.avg_qspan,
                                       max_dist_x, max_dist_y, bw, id);

          // Update running max if this predecessor is better
          team.team_barrier();
          if (sc + f_curr >= max_tracker_sm[id]) {
            max_tracker_sm[id] = sc + f_curr;
            j_tracker_sm[id]   = (parent_dt)curr_idx
                                + (parent_dt)ctrl.tile_num * TILE_SIZE;
          }

          // The slot's original thread writes the result for curr
          team.team_barrier();
          if (id == curr_idx % BACK_SEARCH_COUNT_GPU) {
            return_dt tmp;
            tmp.score  = f_curr;
            tmp.parent = p_curr;
            d_ret[batch_base_ret + ofs*TILE_SIZE + curr_idx] = tmp;
          }
        } // curr_idx

        team.team_barrier();
        d_max_tracker_g(ofs, id) = max_tracker_sm[id];
        d_j_tracker_g  (ofs, id) = j_tracker_sm[id];
      });
    Kokkos::fence();
  } // batch

  auto k_end  = std::chrono::steady_clock::now();
  auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("Total kernel execution time: %f (s)\n", k_time * 1e-9);

  // Copy results back to host
  {
    auto h_ret = Kokkos::create_mirror_view(d_ret);
    Kokkos::deep_copy(h_ret, d_ret);
    memcpy(ret.data(), h_ret.data(), ret_n * sizeof(return_dt));
  }
}
