#include <vector>

extern bool useLmem;

using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;
using ScratchView = Kokkos::View<float*, ScratchSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

KOKKOS_INLINE_FUNCTION int getLocalIdx(int i, int k, int num_systems) {
  return i + num_systems * k;
}

// ─────────────────────────── Transpose ───────────────────────────────────────

static double transpose_kokkos(
    Kokkos::View<float*, mem_space> odata,
    Kokkos::View<const float*, mem_space> idata,
    int width, int height)
{
  int nteamX = (width  + BLOCK_DIM - 1) / BLOCK_DIM;
  int nteamY = (height + BLOCK_DIM - 1) / BLOCK_DIM;
  int nteam  = nteamX * nteamY;
  int tpb    = TRANSPOSE_BLOCK_DIM * TRANSPOSE_BLOCK_DIM;
  int scratch_bytes = TRANSPOSE_BLOCK_DIM * (TRANSPOSE_BLOCK_DIM + 1) * (int)sizeof(float);

  Kokkos::TeamPolicy<> policy(nteam, tpb);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  shrDeltaT(0);
  for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
    Kokkos::parallel_for("transpose", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        ScratchView block_s(team.team_scratch(0),
                            TRANSPOSE_BLOCK_DIM * (TRANSPOSE_BLOCK_DIM + 1));
        float* block = block_s.data();

        int tid = team.team_rank();
        int bid = team.league_rank();

        int bix = bid % nteamX;
        int biy = bid / nteamX;
        int tix = tid % TRANSPOSE_BLOCK_DIM;
        int tiy = tid / TRANSPOSE_BLOCK_DIM;

        int i0 = bix * BLOCK_DIM + tix;
        int j0 = biy * BLOCK_DIM + tiy;

        int i1 = biy * BLOCK_DIM + tix;
        int j1 = bix * BLOCK_DIM + tiy;

        if (i0 < width && j0 < height && i1 < height && j1 < width) {
          block[tiy * (BLOCK_DIM + 1) + tix] = idata(i0 + j0 * width);
        }
        team.team_barrier();
        if (i0 < width && j0 < height && i1 < height && j1 < width) {
          odata(i1 + j1 * height) = block[tix * (BLOCK_DIM + 1) + tiy];
        }
      });
    Kokkos::fence();
  }
  return shrDeltaT(0);
}

// ─────────────────────────── Sweep solver ────────────────────────────────────

static double sweep_local_kokkos(
    Kokkos::View<const float*, mem_space> a_d,
    Kokkos::View<const float*, mem_space> b_d,
    Kokkos::View<const float*, mem_space> c_d,
    Kokkos::View<const float*, mem_space> d_d,
    Kokkos::View<float*, mem_space>       x_d,
    int system_size, int num_systems, bool reorder)
{
  shrDeltaT(0);
  for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
    Kokkos::parallel_for("sweep_local",
      Kokkos::RangePolicy<>(0, num_systems),
      KOKKOS_LAMBDA(int i) {
        int stride   = reorder ? num_systems : 1;
        int base_idx = reorder ? i : i * system_size;

        float a_arr[128];

        float c1, c2, c3, f_i, x_prev, x_next;

        c1 = c_d(base_idx);
        c2 = b_d(base_idx);
        f_i= d_d(base_idx);

        a_arr[1] = -c1 / c2;
        x_prev   = f_i / c2;

        int idx = base_idx;
        x_d(base_idx) = x_prev;

        for (int k = 1; k < system_size - 1; k++) {
          idx += stride;
          c1  = c_d(idx);
          c2  = b_d(idx);
          c3  = a_d(idx);
          f_i = d_d(idx);
          float q = c3 * a_arr[k] + c2;
          float t = 1.0f / q;
          x_next = (f_i - c3 * x_prev) * t;
          x_d(idx) = x_prev = x_next;
          a_arr[k + 1] = -c1 * t;
        }

        idx += stride;
        c2  = b_d(idx);
        c3  = a_d(idx);
        f_i = d_d(idx);
        float q = c3 * a_arr[system_size - 1] + c2;
        float t = 1.0f / q;
        x_next = (f_i - c3 * x_prev) * t;
        x_d(idx) = x_prev = x_next;

        for (int k = system_size - 2; k >= 0; k--) {
          idx -= stride;
          x_next  = x_d(idx);
          x_next += x_prev * a_arr[k + 1];
          x_d(idx) = x_prev = x_next;
        }
      });
    Kokkos::fence();
  }
  return shrDeltaT(0);
}

static double sweep_global_kokkos(
    Kokkos::View<const float*, mem_space> a_d,
    Kokkos::View<const float*, mem_space> b_d,
    Kokkos::View<const float*, mem_space> c_d,
    Kokkos::View<const float*, mem_space> d_d,
    Kokkos::View<float*, mem_space>       x_d,
    Kokkos::View<float*, mem_space>       w_d,
    int system_size, int num_systems, bool reorder)
{
  shrDeltaT(0);
  for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
    Kokkos::parallel_for("sweep_global",
      Kokkos::RangePolicy<>(0, num_systems),
      KOKKOS_LAMBDA(int i) {
        int stride   = reorder ? num_systems : 1;
        int base_idx = reorder ? i : i * system_size;

        float c1, c2, c3, f_i, x_prev, x_next;

        c1 = c_d(base_idx);
        c2 = b_d(base_idx);
        f_i= d_d(base_idx);

        w_d(getLocalIdx(i, 1, num_systems)) = -c1 / c2;
        x_prev = f_i / c2;

        int idx = base_idx;
        x_d(base_idx) = x_prev;

        for (int k = 1; k < system_size - 1; k++) {
          idx += stride;
          c1  = c_d(idx);
          c2  = b_d(idx);
          c3  = a_d(idx);
          f_i = d_d(idx);
          float q = c3 * w_d(getLocalIdx(i, k, num_systems)) + c2;
          float t = 1.0f / q;
          x_next = (f_i - c3 * x_prev) * t;
          x_d(idx) = x_prev = x_next;
          w_d(getLocalIdx(i, k + 1, num_systems)) = -c1 * t;
        }

        idx += stride;
        c2  = b_d(idx);
        c3  = a_d(idx);
        f_i = d_d(idx);
        float q = c3 * w_d(getLocalIdx(i, system_size - 1, num_systems)) + c2;
        float t = 1.0f / q;
        x_next = (f_i - c3 * x_prev) * t;
        x_d(idx) = x_prev = x_next;

        for (int k = system_size - 2; k >= 0; k--) {
          idx -= stride;
          x_next  = x_d(idx);
          x_next += x_prev * w_d(getLocalIdx(i, k + 1, num_systems));
          x_d(idx) = x_prev = x_next;
        }
      });
    Kokkos::fence();
  }
  return shrDeltaT(0);
}

double sweep_small_systems(float* a, float* b, float* c, float* d, float* x,
    int system_size, int num_systems, bool reorder = false)
{
  if (reorder)  shrLog("sweep_data_reorder_kernel\n");
  if (useLmem)  shrLog("sweep_small_systems_local_kernel\n");
  else          shrLog("sweep_small_systems_global_kernel\n");

  int mem_size = num_systems * system_size;

  Kokkos::View<float*, mem_space> a_d("a", mem_size);
  Kokkos::View<float*, mem_space> b_d("b", mem_size);
  Kokkos::View<float*, mem_space> c_d("c", mem_size);
  Kokkos::View<float*, mem_space> d_d("d", mem_size);
  Kokkos::View<float*, mem_space> x_d("x", mem_size);
  Kokkos::View<float*, mem_space> t_d("t", mem_size);
  Kokkos::View<float*, mem_space> w_d("w", mem_size);

  // Host aliases for the input arrays (so we can swap pointers for reorder)
  std::vector<float> a_buf(a, a + mem_size);
  std::vector<float> b_buf(b, b + mem_size);
  std::vector<float> c_buf(c, c + mem_size);
  std::vector<float> d_buf(d, d + mem_size);

  auto upload = [&]() {
    auto ah = Kokkos::create_mirror_view(a_d);
    auto bh = Kokkos::create_mirror_view(b_d);
    auto ch = Kokkos::create_mirror_view(c_d);
    auto dh = Kokkos::create_mirror_view(d_d);
    for (int i = 0; i < mem_size; i++) {
      ah(i)=a_buf[i]; bh(i)=b_buf[i]; ch(i)=c_buf[i]; dh(i)=d_buf[i];
    }
    Kokkos::deep_copy(a_d, ah);
    Kokkos::deep_copy(b_d, bh);
    Kokkos::deep_copy(c_d, ch);
    Kokkos::deep_copy(d_d, dh);
  };
  upload();

  double reorder_time = 0.0, solver_time = 0.0;

  if (reorder) {
    // transpose a
    reorder_time += transpose_kokkos(t_d, a_d, system_size, num_systems);
    Kokkos::deep_copy(a_d, t_d);
    reorder_time += transpose_kokkos(t_d, b_d, system_size, num_systems);
    Kokkos::deep_copy(b_d, t_d);
    reorder_time += transpose_kokkos(t_d, c_d, system_size, num_systems);
    Kokkos::deep_copy(c_d, t_d);
    reorder_time += transpose_kokkos(t_d, d_d, system_size, num_systems);
    Kokkos::deep_copy(d_d, t_d);
  }

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);

  if (useLmem)
    solver_time = sweep_local_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, reorder);
  else
    solver_time = sweep_global_kokkos(a_d, b_d, c_d, d_d, x_d, w_d, system_size, num_systems, reorder);

  if (reorder) {
    reorder_time += transpose_kokkos(t_d, x_d, num_systems, system_size);
    Kokkos::deep_copy(x_d, t_d);
  }

  auto xh = Kokkos::create_mirror_view(x_d);
  Kokkos::deep_copy(xh, x_d);
  for (int i = 0; i < mem_size; i++) x[i] = xh(i);

  return solver_time + reorder_time;
}
