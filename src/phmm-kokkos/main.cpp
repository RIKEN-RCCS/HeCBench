#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>

// Compile-time constants inferred from the OMP source (constants_types.h)
// All internal array dims of size "2" in kernel.h match SM1=2 => states=3
static constexpr int x_dim  = 50;
static constexpr int y_dim  = 50;
static constexpr int BATCH  = 4;
static constexpr int STATES = 3;
static constexpr int SM1    = STATES - 1;  // = 2

// Flat-index helpers (host + device)
KOKKOS_INLINE_FUNCTION int fIdx(int i, int j, int b, int s) {
  return ((i * (y_dim+1) + j) * BATCH + b) * SM1 + s;
}
KOKKOS_INLINE_FUNCTION int tIdx(int i, int b, int s, int t) {
  return ((i * BATCH + b) * SM1 + s) * STATES + t;
}
KOKKOS_INLINE_FUNCTION int lIdx(int i, int j, int b, int s) {
  return ((i * 2 + j) * BATCH + b) * SM1 + s;
}
KOKKOS_INLINE_FUNCTION int sIdx(int b, int s) {
  return b * SM1 + s;
}

// Per-team scratch: e[SM1] + f01[SM1]
static constexpr int SCRATCH_DOUBLES = SM1 + SM1; // = 4

using ScratchPad = Kokkos::View<double *,
    Kokkos::DefaultExecutionSpace::scratch_memory_space,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void pair_HMM_forward(
    int cur_i, int cur_j,
    const Kokkos::View<double *> d_cur_fwd,
    const Kokkos::View<double *> d_trans,
    const Kokkos::View<double *> d_emis,
    const Kokkos::View<double *> d_like,
    const Kokkos::View<double *> d_start,
    Kokkos::View<double *>       d_next_fwd)
{
  using team_policy  = Kokkos::TeamPolicy<>;
  using member_type  = team_policy::member_type;

  const int scratch_bytes = ScratchPad::shmem_size(SCRATCH_DOUBLES);

  Kokkos::parallel_for(
    team_policy(BATCH, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
    KOKKOS_LAMBDA(const member_type &team) {
      const int batch_id = team.league_rank();

      // Scratch: e_sh[SM1] | f01_sh[SM1]
      ScratchPad scratch(team.team_scratch(0), SCRATCH_DOUBLES);
      double *e_sh   = scratch.data();
      double *f01_sh = e_sh + SM1;

      // Phase 1: fill e_sh in parallel over states
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SM1), [&](int states_id) {
        e_sh[states_id] = d_emis(fIdx(cur_i, cur_j, batch_id, states_id));
      });
      team.team_barrier();

      if (cur_j == 0) {
        if (cur_i == 1) {
          // Base case: f[1][0] = start * emis
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SM1), [&](int states_id) {
            d_next_fwd(fIdx(1, 0, batch_id, states_id)) =
              d_start(sIdx(batch_id, states_id)) * e_sh[states_id];
          });
        } else {
          // First column, rows > 1
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SM1), [&](int states_id) {
            f01_sh[states_id] = d_cur_fwd(fIdx(cur_i-1, 0, batch_id, states_id));
          });
          team.team_barrier();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SM1), [&](int states_id) {
            double s = 0.0;
            for (int k = 0; k < SM1; k++)
              s += f01_sh[k] * d_trans(tIdx(cur_i-1, batch_id, k, states_id));
            s *= e_sh[states_id] * d_like(lIdx(0, 1, batch_id, states_id));
            d_next_fwd(fIdx(cur_i, 0, batch_id, states_id)) = s;
          });
        }
      } else {
        // General case: cur_j > 0
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, SM1), [&](int states_id) {
          // Load f_loc for this states_id (all 4 neighbours)
          double f00[SM1], f01[SM1], f10[SM1], f11[SM1];
          for (int ii = 0; ii < SM1; ii++) {
            f00[ii] = d_cur_fwd(fIdx(cur_i-1, cur_j-1, batch_id, ii));
            f01[ii] = d_cur_fwd(fIdx(cur_i-1, cur_j,   batch_id, ii));
            f10[ii] = d_cur_fwd(fIdx(cur_i,   cur_j-1, batch_id, ii));
            f11[ii] = d_cur_fwd(fIdx(cur_i,   cur_j,   batch_id, ii));
          }
          double s0=0.0, s1=0.0, s2=0.0, s3=0.0;
          for (int k = 0; k < SM1; k++) {
            s0 += f00[k] * d_trans(tIdx(cur_i-1, batch_id, k, states_id));
            s1 += f01[k] * d_trans(tIdx(cur_i-1, batch_id, k, states_id));
            s2 += f10[k] * d_trans(tIdx(cur_i,   batch_id, k, states_id));
            s3 += f11[k] * d_trans(tIdx(cur_i,   batch_id, k, states_id));
          }
          s0 *= d_like(lIdx(0, 0, batch_id, states_id));
          s1 *= d_like(lIdx(0, 1, batch_id, states_id));
          s2 *= d_like(lIdx(1, 0, batch_id, states_id));
          s3 *= d_like(lIdx(1, 1, batch_id, states_id));
          d_next_fwd(fIdx(cur_i, cur_j, batch_id, states_id)) =
            (s0 + s1 + s2 + s3) * e_sh[states_id];
        });
      }
    });
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const int fwd_elem   = (x_dim+1)*(y_dim+1)*BATCH*SM1;
    const int emis_elem  = fwd_elem;
    const int trans_elem = (x_dim+1)*BATCH*SM1*STATES;
    const int like_elem  = 2*2*BATCH*SM1;
    const int start_elem = BATCH*SM1;

    // Host views (mirror of device data)
    Kokkos::View<double *> d_cur_fwd  ("cur_fwd",   fwd_elem);
    Kokkos::View<double *> d_next_fwd ("next_fwd",  fwd_elem);
    Kokkos::View<double *> d_emis     ("emis",      emis_elem);
    Kokkos::View<double *> d_trans    ("trans",     trans_elem);
    Kokkos::View<double *> d_like     ("like",      like_elem);
    Kokkos::View<double *> d_start    ("start",     start_elem);

    auto h_cur_fwd  = Kokkos::create_mirror_view(d_cur_fwd);
    auto h_next_fwd = Kokkos::create_mirror_view(d_next_fwd);
    auto h_emis     = Kokkos::create_mirror_view(d_emis);
    auto h_trans    = Kokkos::create_mirror_view(d_trans);
    auto h_like     = Kokkos::create_mirror_view(d_like);
    auto h_start    = Kokkos::create_mirror_view(d_start);

    std::default_random_engine rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < x_dim+1; i++)
      for (int j = 0; j < y_dim+1; j++)
        for (int b = 0; b < BATCH; b++)
          for (int s = 0; s < SM1; s++) {
            h_cur_fwd(fIdx(i,j,b,s)) = dist(rng);
            h_emis   (fIdx(i,j,b,s)) = dist(rng);
          }

    for (int i = 0; i < x_dim+1; i++)
      for (int b = 0; b < BATCH; b++)
        for (int s = 0; s < SM1; s++)
          for (int t = 0; t < STATES; t++)
            h_trans(tIdx(i,b,s,t)) = dist(rng);

    for (int b = 0; b < BATCH; b++)
      for (int s = 0; s < SM1; s++)
        h_start(sIdx(b,s)) = dist(rng);

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int b = 0; b < BATCH; b++)
          for (int s = 0; s < SM1; s++)
            h_like(lIdx(i,j,b,s)) = dist(rng);

    Kokkos::deep_copy(d_cur_fwd, h_cur_fwd);
    Kokkos::deep_copy(d_emis,    h_emis);
    Kokkos::deep_copy(d_trans,   h_trans);
    Kokkos::deep_copy(d_like,    h_like);
    Kokkos::deep_copy(d_start,   h_start);

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int count = 0; count < repeat; count++) {
      for (int i = 1; i < x_dim + 1; i++) {
        for (int j = 1; j < y_dim + 1; j++) {
          pair_HMM_forward(i, j,
                           d_cur_fwd, d_trans, d_emis, d_like, d_start,
                           d_next_fwd);
          Kokkos::fence();
          Kokkos::deep_copy(d_cur_fwd, d_next_fwd);
        }
      }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = (t2 - t1);
    std::cout << "Total execution time " << ms.count() << " milliseconds\n";

    // Copy back and compute checksum
    Kokkos::deep_copy(h_cur_fwd, d_cur_fwd);
    double checkSum = 0.0;
    for (int i = 0; i < x_dim+1; i++)
      for (int j = 0; j < y_dim+1; j++)
        for (int b = 0; b < BATCH; b++)
          for (int s = 0; s < SM1; s++)
            checkSum += h_cur_fwd(fIdx(i,j,b,s));

    std::cout << "Checksum " << checkSum << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
