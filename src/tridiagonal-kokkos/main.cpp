/*
 * Tridiagonal solvers - Kokkos port
 * Ported from tridiagonal-omp
 * Implements PCR, CR (Cyclic), and Sweep solvers using Kokkos
 */

#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../tridiagonal-omp/tridiagonal.h"
#include "../tridiagonal-omp/shrUtils.h"
#include "../tridiagonal-omp/shrUtils.cpp"
#include "../tridiagonal-omp/test_gen_result_check.h"
#include "../tridiagonal-omp/cpu_solvers.h"

bool             useLmem = false;
bool             useVec4 = false;
int              SWEEP_BLOCK_SIZE = 256;

using exec_space  = Kokkos::DefaultExecutionSpace;
using mem_space   = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;
using ScratchView  = Kokkos::View<float*, ScratchSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

KOKKOS_INLINE_FUNCTION int getLocalIdx(int i, int k, int num_systems) {
  return i + num_systems * k;
}

// ─────────────────────────── PCR solvers ────────────────────────────────────

static double pcr_small_systems_kernel_kokkos(
    Kokkos::View<const float*, mem_space> a_d,
    Kokkos::View<const float*, mem_space> b_d,
    Kokkos::View<const float*, mem_space> c_d,
    Kokkos::View<const float*, mem_space> d_d,
    Kokkos::View<float*, mem_space>       x_d,
    int system_size, int num_systems, int iterations)
{
  int scratch_bytes = (system_size + 1) * 5 * (int)sizeof(float);
  Kokkos::TeamPolicy<> policy(num_systems, system_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  shrDeltaT(0);
  for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
    Kokkos::parallel_for("pcr_base", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        ScratchView sv(team.team_scratch(0), (system_size + 1) * 5);
        float* sh = sv.data();

        int thid = team.team_rank();
        int blid = team.league_rank();

        float* a = sh;
        float* b = a + system_size + 1;
        float* c = b + system_size + 1;
        float* d = c + system_size + 1;
        float* x = d + system_size + 1;

        a[thid] = a_d(thid + blid * system_size);
        b[thid] = b_d(thid + blid * system_size);
        c[thid] = c_d(thid + blid * system_size);
        d[thid] = d_d(thid + blid * system_size);

        float aNew, bNew, cNew, dNew;
        team.team_barrier();

        int delta = 1;
        for (int j = 0; j < iterations; j++) {
          int i = thid;
          if (i < delta) {
            float tmp2 = c[i] / b[i + delta];
            bNew = b[i] - a[i + delta] * tmp2;
            dNew = d[i] - d[i + delta] * tmp2;
            aNew = 0;
            cNew = -c[i + delta] * tmp2;
          } else if ((system_size - i - 1) < delta) {
            float tmp = a[i] / b[i - delta];
            bNew = b[i] - c[i - delta] * tmp;
            dNew = d[i] - d[i - delta] * tmp;
            aNew = -a[i - delta] * tmp;
            cNew = 0;
          } else {
            float tmp1 = a[i] / b[i - delta];
            float tmp2 = c[i] / b[i + delta];
            bNew = b[i] - c[i - delta] * tmp1 - a[i + delta] * tmp2;
            dNew = d[i] - d[i - delta] * tmp1 - d[i + delta] * tmp2;
            aNew = -a[i - delta] * tmp1;
            cNew = -c[i + delta] * tmp2;
          }
          team.team_barrier();
          b[i] = bNew; d[i] = dNew; a[i] = aNew; c[i] = cNew;
          delta *= 2;
          team.team_barrier();
        }

        if (thid < delta) {
          int addr1 = thid, addr2 = thid + delta;
          float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
          x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
          x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
        }
        team.team_barrier();
        x_d(thid + blid * system_size) = x[thid];
      });
    Kokkos::fence();
  }
  return shrDeltaT(0);
}

static double pcr_branch_free_kernel_kokkos(
    Kokkos::View<const float*, mem_space> a_d,
    Kokkos::View<const float*, mem_space> b_d,
    Kokkos::View<const float*, mem_space> c_d,
    Kokkos::View<const float*, mem_space> d_d,
    Kokkos::View<float*, mem_space>       x_d,
    int system_size, int num_systems, int iterations)
{
  int scratch_bytes = (system_size + 1) * 5 * (int)sizeof(float);
  Kokkos::TeamPolicy<> policy(num_systems, system_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  shrDeltaT(0);
  for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
    Kokkos::parallel_for("pcr_bf", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        ScratchView sv(team.team_scratch(0), (system_size + 1) * 5);
        float* sh = sv.data();

        int thid = team.team_rank();
        int blid = team.league_rank();

        float* a = sh;
        float* b = a + system_size + 1;
        float* c = b + system_size + 1;
        float* d = c + system_size + 1;
        float* x = d + system_size + 1;

        a[thid] = a_d(thid + blid * system_size);
        b[thid] = b_d(thid + blid * system_size);
        c[thid] = c_d(thid + blid * system_size);
        d[thid] = d_d(thid + blid * system_size);

        float aNew, bNew, cNew, dNew;
        team.team_barrier();

        int delta = 1;
        for (int j = 0; j < iterations; j++) {
          int i = thid;
          int iRight = (i + delta) & (system_size - 1);
          int iLeft  = (i - delta) & (system_size - 1);

          float tmp1 = a[i] / b[iLeft];
          float tmp2 = c[i] / b[iRight];

          bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
          dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
          aNew = -a[iLeft] * tmp1;
          cNew = -c[iRight] * tmp2;

          team.team_barrier();
          b[i] = bNew; d[i] = dNew; a[i] = aNew; c[i] = cNew;
          delta *= 2;
          team.team_barrier();
        }

        if (thid < delta) {
          int addr1 = thid, addr2 = thid + delta;
          float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
          x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
          x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
        }
        team.team_barrier();
        x_d(thid + blid * system_size) = x[thid];
      });
    Kokkos::fence();
  }
  return shrDeltaT(0);
}

double pcr_small_systems(float* a, float* b, float* c, float* d, float* x,
    int system_size, int num_systems, int id = 0)
{
  const char* names[] = { "pcr_small_systems_kernel", "pcr_branch_free_kernel" };
  shrLog(" %s\n", names[id]);

  int mem_size = num_systems * system_size;
  int iterations = my_log2(system_size / 2);

  // allocate device views
  Kokkos::View<float*, mem_space> a_d("a", mem_size);
  Kokkos::View<float*, mem_space> b_d("b", mem_size);
  Kokkos::View<float*, mem_space> c_d("c", mem_size);
  Kokkos::View<float*, mem_space> d_d("d", mem_size);
  Kokkos::View<float*, mem_space> x_d("x", mem_size);

  // host mirrors
  auto a_h = Kokkos::create_mirror_view(a_d);
  auto b_h = Kokkos::create_mirror_view(b_d);
  auto c_h = Kokkos::create_mirror_view(c_d);
  auto d_h = Kokkos::create_mirror_view(d_d);

  for (int i = 0; i < mem_size; i++) {
    a_h(i) = a[i]; b_h(i) = b[i]; c_h(i) = c[i]; d_h(i) = d[i];
  }
  Kokkos::deep_copy(a_d, a_h);
  Kokkos::deep_copy(b_d, b_h);
  Kokkos::deep_copy(c_d, c_h);
  Kokkos::deep_copy(d_d, d_h);

  // warm up
  if (id == 0)
    pcr_small_systems_kernel_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);
  else
    pcr_branch_free_kernel_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);

  double sum_time;
  if (id == 0)
    sum_time = pcr_small_systems_kernel_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);
  else
    sum_time = pcr_branch_free_kernel_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations);

  auto x_h = Kokkos::create_mirror_view(x_d);
  Kokkos::deep_copy(x_h, x_d);
  for (int i = 0; i < mem_size; i++) x[i] = x_h(i);

  return sum_time / BENCH_ITERATIONS;
}

// ─────────────────────────── Cyclic solvers ──────────────────────────────────

static double cyclic_kernel_kokkos(
    Kokkos::View<const float*, mem_space> a_d,
    Kokkos::View<const float*, mem_space> b_d,
    Kokkos::View<const float*, mem_space> c_d,
    Kokkos::View<const float*, mem_space> d_d,
    Kokkos::View<float*, mem_space>       x_d,
    int system_size, int num_systems, int iterations, bool branch_free)
{
  int half_size = system_size / 2;
  // Each team: num_systems teams, half_size threads each
  int scratch_bytes = system_size * 5 * (int)sizeof(float);
  Kokkos::TeamPolicy<> policy(num_systems, half_size);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  shrDeltaT(0);
  for (int iter = 0; iter < BENCH_ITERATIONS; iter++) {
    Kokkos::parallel_for("cyclic", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        ScratchView sv(team.team_scratch(0), system_size * 5);
        float* sh = sv.data();

        int thid = team.team_rank();
        int blid = team.league_rank();
        int thid_num = half_size;

        float* a = sh;
        float* b = a + system_size;
        float* c = b + system_size;
        float* d = c + system_size;
        float* x = d + system_size;

        int base_off = blid * system_size;
        a[thid]           = a_d(thid + base_off);
        a[thid + thid_num]= a_d(thid + thid_num + base_off);
        b[thid]           = b_d(thid + base_off);
        b[thid + thid_num]= b_d(thid + thid_num + base_off);
        c[thid]           = c_d(thid + base_off);
        c[thid + thid_num]= c_d(thid + thid_num + base_off);
        d[thid]           = d_d(thid + base_off);
        d[thid + thid_num]= d_d(thid + thid_num + base_off);

        team.team_barrier();

        int stride = 1;
        thid_num = half_size;

        // forward elimination
        for (int j = 0; j < iterations; j++) {
          team.team_barrier();
          stride <<= 1;
          int delta = stride >> 1;
          if (thid < thid_num) {
            int i = stride * thid + stride - 1;
            if (i == system_size - 1) {
              float tmp = a[i] / b[i - delta];
              b[i] = b[i] - c[i - delta] * tmp;
              d[i] = d[i] - d[i - delta] * tmp;
              a[i] = -a[i - delta] * tmp;
              c[i] = 0;
            } else {
              float tmp1 = a[i] / b[i - delta];
              float tmp2 = c[i] / b[i + delta];
              b[i] = b[i] - c[i - delta] * tmp1 - a[i + delta] * tmp2;
              d[i] = d[i] - d[i - delta] * tmp1 - d[i + delta] * tmp2;
              a[i] = -a[i - delta] * tmp1;
              c[i] = -c[i + delta] * tmp2;
            }
          }
          thid_num >>= 1;
        }

        team.team_barrier();

        if (thid < 2) {
          int addr1 = stride - 1;
          int addr2 = (stride << 1) - 1;
          float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
          x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
          x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
        }
        team.team_barrier();

        // backward substitution
        stride >>= 1;
        thid_num = 2;
        for (int j = 0; j < iterations; j++) {
          int delta = stride >> 1;
          team.team_barrier();
          if (thid < thid_num - 1) {
            int i = stride * thid + delta - 1;
            x[i] = (d[i] - a[i] * x[i - delta] - c[i] * x[i + delta]) / b[i];
          }
          stride >>= 1;
          thid_num <<= 1;
        }

        team.team_barrier();
        x_d(thid + base_off)           = x[thid];
        x_d(thid + thid_num/2 + base_off) = x[thid + thid_num/2];
      });
    Kokkos::fence();
  }
  return shrDeltaT(0);
}

double cyclic_small_systems(float* a, float* b, float* c, float* d, float* x,
    int system_size, int num_systems, int id = 0)
{
  const char* names[] = { "cyclic_small_systems_kernel", "cyclic_branch_free_kernel" };
  shrLog(" %s\n", names[id]);

  int mem_size = num_systems * system_size;
  int iterations = my_log2(system_size / 2);

  Kokkos::View<float*, mem_space> a_d("a", mem_size);
  Kokkos::View<float*, mem_space> b_d("b", mem_size);
  Kokkos::View<float*, mem_space> c_d("c", mem_size);
  Kokkos::View<float*, mem_space> d_d("d", mem_size);
  Kokkos::View<float*, mem_space> x_d("x", mem_size);

  auto a_h = Kokkos::create_mirror_view(a_d);
  auto b_h = Kokkos::create_mirror_view(b_d);
  auto c_h = Kokkos::create_mirror_view(c_d);
  auto d_h = Kokkos::create_mirror_view(d_d);

  for (int i = 0; i < mem_size; i++) {
    a_h(i) = a[i]; b_h(i) = b[i]; c_h(i) = c[i]; d_h(i) = d[i];
  }
  Kokkos::deep_copy(a_d, a_h);
  Kokkos::deep_copy(b_d, b_h);
  Kokkos::deep_copy(c_d, c_h);
  Kokkos::deep_copy(d_d, d_h);

  // warm up
  cyclic_kernel_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations, id != 0);

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);

  double sum_time = cyclic_kernel_kokkos(a_d, b_d, c_d, d_d, x_d, system_size, num_systems, iterations, id != 0);

  auto x_h = Kokkos::create_mirror_view(x_d);
  Kokkos::deep_copy(x_h, x_d);
  for (int i = 0; i < mem_size; i++) x[i] = x_h(i);

  return sum_time / BENCH_ITERATIONS;
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

// ─────────────────────────── run() ───────────────────────────────────────────

void run(int system_size, int num_systems)
{
  double time_spent_gpu[3];
  double time_spent_cpu[1];

  const unsigned int mem_size = sizeof(float) * num_systems * system_size;

  float* a  = (float*)malloc(mem_size);
  float* b  = (float*)malloc(mem_size);
  float* c  = (float*)malloc(mem_size);
  float* d  = (float*)malloc(mem_size);
  float* x1 = (float*)malloc(mem_size);
  float* x2 = (float*)malloc(mem_size);

  for (int i = 0; i < num_systems; i++)
    test_gen_cyclic(&a[i * system_size], &b[i * system_size], &c[i * system_size],
                    &d[i * system_size], &x1[i * system_size], system_size, 0);

  shrLog("  Num_systems = %d, system_size = %d\n", num_systems, system_size);

  time_spent_cpu[0] = serial_small_systems(a, b, c, d, x2, system_size, num_systems);

  shrLog("\n----- CPU  solvers -----\n");
  shrLog("  CPU Time =    %.5f s\n", time_spent_cpu[0]);
  shrLog("  Throughput =  %.4f systems/sec\n",
         (float)num_systems / (time_spent_cpu[0] * 1000.0));

  shrLog("\n----- optimized GPU solvers -----\n\n");

  time_spent_gpu[0] = pcr_small_systems(a, b, c, d, x1, system_size, num_systems, 0);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-pcrsmall-base, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[0] = pcr_small_systems(a, b, c, d, x1, system_size, num_systems, 1);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-pcrsmall-optimized, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[1] = cyclic_small_systems(a, b, c, d, x1, system_size, num_systems, 0);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-cyclicsmall-base, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[1] = cyclic_small_systems(a, b, c, d, x1, system_size, num_systems, 1);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-cyclicsmall-optimized, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  if (!useVec4) {
    time_spent_gpu[2] = sweep_small_systems(a, b, c, d, x1, system_size, num_systems, false);
    shrLogEx(LOGBOTH | MASTER, 0,
      "Tridiagonal-sweepsmall-noreorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
      (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems);
    compare_small_systems(x1, x2, system_size, num_systems);
  }

  time_spent_gpu[2] = sweep_small_systems(a, b, c, d, x1, system_size, num_systems, true);
  shrLogEx(LOGBOTH | MASTER, 0,
    "Tridiagonal-sweepsmall-reorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
    (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  free(a); free(b); free(c); free(d); free(x1); free(x2);
}

// ─────────────────────────── main() ──────────────────────────────────────────

int main(int argc, const char** argv)
{
  shrSetLogFileName("oclTridiagonal.txt");
  shrLog("%s Starting...\n\n", argv[0]);

  int num_systems = 128 * 128;
  int system_size = SYSTEM_SIZE;

  if (shrCheckCmdLineFlag(argc, argv, "num_systems")) {
    char* ctaList; char* ctaStr;
    shrGetCmdLineArgumentstr(argc, argv, "num_systems", &ctaList);
    ctaStr = strtok(ctaList, " ,.-");
    num_systems = atoi(ctaStr);
  }

  if (shrCheckCmdLineFlag(argc, argv, "system_size")) {
    char* ctaList; char* ctaStr;
    shrGetCmdLineArgumentstr(argc, argv, "system_size", &ctaList);
    ctaStr = strtok(ctaList, " ,.-");
    system_size = atoi(ctaStr);
    if (system_size > 128) {
      shrLog("system size must be no more than 128\n");
      return -1;
    }
  }

  if (shrCheckCmdLineFlag(argc, argv, "lmem"))  useLmem = true;
  if (shrCheckCmdLineFlag(argc, argv, "vec4"))  useVec4 = true;

  if (shrCheckCmdLineFlag(argc, argv, "sweep-cta")) {
    char* ctaList; char* ctaStr;
    shrGetCmdLineArgumentstr(argc, argv, "sweep-cta", &ctaList);
    ctaStr = strtok(ctaList, " ,.-");
    SWEEP_BLOCK_SIZE = atoi(ctaStr);
  }

  if (useVec4) shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE / 4);
  else         shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE);

  Kokkos::initialize(argc, const_cast<char**>(argv));
  {
    run(system_size, num_systems);
  }
  Kokkos::finalize();

  return 0;
}
