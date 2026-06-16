/*
 * Kokkos port of attentionMultiHead benchmark.
 * Uses TeamPolicy to replace CUDA shared memory + block reductions.
 * One Kokkos team per (beam, head) pair.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>

using TeamPol = Kokkos::TeamPolicy<>;
using TeamMem = TeamPol::member_type;

void mha(
    Kokkos::View<const float*> q,
    Kokkos::View<const float*> k,
    Kokkos::View<const float*> v,
    int beam_size, int n_steps, int qk_col, int v_col,
    int nhead, float scale, int THRESHOLD,
    Kokkos::View<float*> dst)
{
  int dim_per_head = qk_col / nhead;
  int nteams = beam_size * nhead;

  Kokkos::parallel_for("mha",
    TeamPol(nteams, Kokkos::AUTO, 1),
    KOKKOS_LAMBDA(const TeamMem& team) {
      int bid = team.league_rank();
      int candidate_id = bid / nhead;
      int head_id = bid % nhead;

      // QK^T / sqrt(d_k): dot each key row with the query slice
      // logits[step] = sum_i q[candidate*qk_col + head*dim + i] * k[candidate*qk_col*n_steps + head*dim + step*qk_col + i]
      Kokkos::View<float*, Kokkos::ScratchMemorySpace<>, Kokkos::MemoryUnmanaged>
        logits(team.team_scratch(0), n_steps);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_steps),
        [&](int step) {
          float s = 0.f;
          const float* q_ptr = q.data() + candidate_id * qk_col + head_id * dim_per_head;
          const float* k_ptr = k.data() + candidate_id * qk_col * n_steps
                             + head_id * dim_per_head + step * qk_col;
          for (int i = 0; i < dim_per_head; i++)
            s += q_ptr[i] * k_ptr[i];
          logits(step) = s * scale;
        });
      team.team_barrier();

      // Softmax over n_steps
      float max_val = -1e20f;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_steps),
        [&](int s, float& mx) { if (logits(s) > mx) mx = logits(s); },
        Kokkos::Max<float>(max_val));
      team.team_barrier();

      float sum_exp = 0.f;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_steps),
        [&](int s, float& se) {
          float e = Kokkos::exp(logits(s) - max_val);
          if (logits(s) - max_val < -(float)THRESHOLD) e = Kokkos::exp(-(float)THRESHOLD);
          logits(s) = e;
          se += e;
        }, sum_exp);
      team.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_steps),
        [&](int s) { logits(s) /= sum_exp; });
      team.team_barrier();

      // Weighted sum of values
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dim_per_head),
        [&](int feat) {
          float s = 0.f;
          int base = candidate_id * v_col * n_steps + head_id * dim_per_head + feat;
          for (int step = 0; step < n_steps; step++)
            s += logits(step) * v(base + step * v_col);
          dst(candidate_id * v_col + head_id * dim_per_head + feat) = s;
        });
    });
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int repeat = atoi(argv[1]);

    const int beamsize = 4;
    const int nhead = 16;
    const int dim_feature = nhead * 256;
    const int n_steps = 9;
    const float scaler = sqrtf(nhead * 1.f / dim_feature);
    const int qk_col = dim_feature;
    const int v_col  = dim_feature;
    const int THRESHOLD = 64;

    int q_size = beamsize * dim_feature;
    int k_size = beamsize * dim_feature * n_steps;
    int v_size = beamsize * dim_feature * n_steps;

    Kokkos::View<float*> d_q("q", q_size);
    Kokkos::View<float*> d_k("k", k_size);
    Kokkos::View<float*> d_v("v", v_size);
    Kokkos::View<float*> d_dst("dst", q_size);

    // Initialize
    Kokkos::parallel_for("init_q", q_size, KOKKOS_LAMBDA(int i) {
      // simple deterministic init
      d_q(i) = (float)(i % 100) / 100.f;
    });
    Kokkos::parallel_for("init_k", k_size, KOKKOS_LAMBDA(int i) {
      d_k(i) = (float)(i % 100) / 100.f;
    });
    Kokkos::parallel_for("init_v", v_size, KOKKOS_LAMBDA(int i) {
      d_v(i) = (float)(i % 100) / 100.f;
    });
    Kokkos::fence();

    // Compute scratch size: n_steps floats per team
    int scratch_size = Kokkos::View<float*, Kokkos::ScratchMemorySpace<>, Kokkos::MemoryUnmanaged>::shmem_size(n_steps);

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      using TeamPol2 = Kokkos::TeamPolicy<>;
      Kokkos::parallel_for("mha_timed",
        TeamPol2(beamsize * nhead, Kokkos::AUTO, 1).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const TeamMem& team) {
          int bid = team.league_rank();
          int candidate_id = bid / nhead;
          int head_id = bid % nhead;
          int dim_per_head = qk_col / nhead;

          Kokkos::View<float*, Kokkos::ScratchMemorySpace<>, Kokkos::MemoryUnmanaged>
            logits(team.team_scratch(0), n_steps);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_steps),
            [&](int step) {
              float s = 0.f;
              const float* q_ptr = d_q.data() + candidate_id * qk_col + head_id * dim_per_head;
              const float* k_ptr = d_k.data() + candidate_id * qk_col * n_steps
                                 + head_id * dim_per_head + step * qk_col;
              for (int ii = 0; ii < dim_per_head; ii++) s += q_ptr[ii] * k_ptr[ii];
              logits(step) = s * scaler;
            });
          team.team_barrier();

          float max_val = -1e20f;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_steps),
            [&](int s, float& mx) { if (logits(s) > mx) mx = logits(s); },
            Kokkos::Max<float>(max_val));
          team.team_barrier();

          float sum_exp = 0.f;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n_steps),
            [&](int s, float& se) {
              float diff = logits(s) - max_val;
              if (diff < -(float)THRESHOLD) diff = -(float)THRESHOLD;
              float e = Kokkos::exp(diff);
              logits(s) = e;
              se += e;
            }, sum_exp);
          team.team_barrier();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_steps),
            [&](int s) { logits(s) /= sum_exp; });
          team.team_barrier();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dim_per_head),
            [&](int feat) {
              float s = 0.f;
              int base = candidate_id * v_col * n_steps + head_id * dim_per_head + feat;
              for (int step = 0; step < n_steps; step++)
                s += logits(step) * d_v(base + step * v_col);
              d_dst(candidate_id * v_col + head_id * dim_per_head + feat) = s;
            });
        });
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    auto h_dst = Kokkos::create_mirror_view(d_dst);
    Kokkos::deep_copy(h_dst, d_dst);

    for (int i = 0; i < beamsize - 1; i++) {
      float sum = 0.f;
      for (int j = 0; j < dim_feature; j++) {
        float d = h_dst(i * dim_feature + j) - h_dst((i + 1) * dim_feature + j);
        sum += d * d;
      }
      printf("Distance between beams %d and %d: %f\n", i, i+1, sqrtf(sum));
    }
  }
  Kokkos::finalize();
  return 0;
}
