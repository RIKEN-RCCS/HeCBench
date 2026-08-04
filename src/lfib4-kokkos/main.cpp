#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define P1 55
#define P2 119
#define P3 179
#define P4 256
#define LWDR 32
#define LKNB 8

typedef unsigned int uint32_t_l;

// Reference host implementation
static void LFIB4(uint32_t_l n, uint32_t_l *x) {
  for (uint32_t_l k = P4; k < n; k++) {
    x[k] = x[k-P1] + x[k-P2] + x[k-P3] + x[k-P4];
  }
}

using team_policy_t = Kokkos::TeamPolicy<>;
using team_member_t = team_policy_t::member_type;
using scratch_space_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using scratch_view_t = Kokkos::View<uint32_t_l*, scratch_space_t, Kokkos::MemoryUnmanaged>;

static void firstColGPU(Kokkos::View<uint32_t_l*> x, int s) {
  Kokkos::parallel_for(
    "firstColGPU",
    team_policy_t(1, P4).set_scratch_size(0, Kokkos::PerTeam(2 * P4 * sizeof(uint32_t_l))),
    KOKKOS_LAMBDA(const team_member_t& team) {
      scratch_view_t cx(team.team_scratch(0), 2 * P4);
      int myid = team.team_rank();
      int px = P4;

      cx(myid) = x(myid);
      team.team_barrier();

      for (int k = 1; k < s / P4; k++) {
        for (int i = 0; i < P4; i += LWDR) {
          if (myid < LWDR) {
            cx(px + i + myid) = cx(px + i + myid - P1)
                              + cx(px + i + myid - P2)
                              + cx(px + i + myid - P3)
                              + cx(px + i + myid - P4);
          }
          team.team_barrier();
        }

        x(k * P4 + myid) = cx(myid) = cx(px + myid);
        team.team_barrier();
      }
    });
}

static void colYGPU(Kokkos::View<uint32_t_l*> y, int s) {
  Kokkos::parallel_for(
    "colYGPU",
    team_policy_t(1, P4).set_scratch_size(0, Kokkos::PerTeam(3 * P4 * sizeof(uint32_t_l))),
    KOKKOS_LAMBDA(const team_member_t& team) {
      scratch_view_t cy(team.team_scratch(0), 3 * P4);
      int myid = team.team_rank();
      int ay = 2 * P4;

      cy(ay + myid) = y(2 * P4 + myid);
      team.team_barrier();

      for (int k = 0; k < s / P4; k++) {
        cy(myid) = cy(myid + P4);
        cy(myid + P4) = cy(ay + myid);
        team.team_barrier();

        for (int i = 0; i < P4; i += LWDR) {
          if (myid < LWDR) {
            cy(ay + i + myid) = cy(ay + i + myid - P1)
                              + cy(ay + i + myid - P2)
                              + cy(ay + i + myid - P3)
                              + cy(ay + i + myid - P4);
          }
          team.team_barrier();
        }
      }

      y(2 * P4 + myid) = cy(2 * P4 + myid);
      y(P4 + myid) = cy(P4 + myid);
      y(myid) = cy(myid);
    });
}

static void lastEntGPU(Kokkos::View<uint32_t_l*> x, Kokkos::View<uint32_t_l*> y,
                       int s, int r) {
  constexpr int a0_off = 0;
  constexpr int b0_off = a0_off + 3 * P4;
  constexpr int c0_off = b0_off + 2 * P4;
  constexpr int d0_off = c0_off + 2 * P4;
  constexpr int scratch_len = d0_off + 2 * P4;

  Kokkos::parallel_for(
    "lastEntGPU",
    team_policy_t(1, 2 * P4).set_scratch_size(0, Kokkos::PerTeam(scratch_len * sizeof(uint32_t_l))),
    KOKKOS_LAMBDA(const team_member_t& team) {
      scratch_view_t sh(team.team_scratch(0), scratch_len);
      int myid = team.team_rank();
      int a = a0_off + P4;
      int b = b0_off + P4;
      int c = c0_off + P4;
      int d = d0_off + P4;

      sh(a0_off + myid) = y(myid);
      team.team_barrier();

      if (myid < P4) sh(a0_off + myid + 2 * P4) = y(myid + 2 * P4);
      team.team_barrier();

      sh(d0_off + myid) = sh(c0_off + myid) = sh(b0_off + myid) = sh(a + myid);
      team.team_barrier();

      sh(b + myid - P4) += sh(a - (P4 - P3) + myid);
      team.team_barrier();

      sh(c + myid - P4) += sh(a - (P3 - P2) + myid) + sh(a - (P4 - P2) + myid);
      team.team_barrier();

      sh(d + myid - P4) += sh(a - (P2 - P1) + myid)
                         + sh(a - (P3 - P1) + myid)
                         + sh(a - (P4 - P1) + myid);
      team.team_barrier();

      a += P4;

      for (int i = 1; i < r; i++) {
        int xc = i * s;
        uint32_t_l tmp = 0;

        if (myid < P4) {
          for (int k = 0; k < P4 - P3; k++) tmp += x(xc - P4 + k) * sh(a + myid - k);
          for (int k = 0; k < P3 - P2; k++) tmp += x(xc - P3 + k) * sh(b + myid - k);
          for (int k = 0; k < P2 - P1; k++) tmp += x(xc - P2 + k) * sh(c + myid - k);
          for (int k = 0; k < P1; k++) tmp += x(xc - P1 + k) * sh(d + myid - k);

          x(xc + s - P4 + myid) = tmp;
        }
        team.team_barrier();
      }
    });
}

static void colsGPU(Kokkos::View<uint32_t_l*> x, int s, int r) {
  int league = r / LKNB + (r % LKNB ? 1 : 0);
  Kokkos::parallel_for(
    "colsGPU",
    team_policy_t(league, P4).set_scratch_size(0, Kokkos::PerTeam(LKNB * 2 * P4 * sizeof(uint32_t_l))),
    KOKKOS_LAMBDA(const team_member_t& team) {
      scratch_view_t cx(team.team_scratch(0), LKNB * 2 * P4);
      int block = team.league_rank();
      int tid = team.team_rank();
      int k0 = block * LKNB;
      int k1 = tid / LWDR;
      int k2 = tid % LWDR;

      int fcol = (block == 0) ? 1 : 0;
      int ecol = (block == league - 1 && r % LKNB) ? r % LKNB : LKNB;

      for (int i = fcol; i < ecol; i++)
        cx(i * 2 * P4 + tid) = x((k0 + i) * s - P4 + tid);

      team.team_barrier();

      int pcx = k1 * 2 * P4 + P4;

      for (int k = 0; k < s / P4 - 1; k++) {
        for (int i = 0; i < P4; i += LWDR) {
          if (!(block == 0 && tid == 0) && !(block == league - 1 && k1 >= ecol)) {
            cx(pcx + i + k2) = cx(pcx + i + k2 - P1)
                             + cx(pcx + i + k2 - P2)
                             + cx(pcx + i + k2 - P3)
                             + cx(pcx + i + k2 - P4);
          }

          team.team_barrier();
        }

        for (int i = fcol; i < ecol; i++) {
          x((k0 + i) * s + k * P4 + tid) = cx(i * 2 * P4 + P4 + tid);
          cx(i * 2 * P4 + tid) = cx(i * 2 * P4 + P4 + tid);
        }

        team.team_barrier();
      }
    });
}

static void gLFIB4(Kokkos::View<uint32_t_l*> x, int s, int r, const uint32_t_l *seed) {
  Kokkos::View<uint32_t_l*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      h_seed(const_cast<uint32_t_l*>(seed), P4);
  auto d_seed = Kokkos::subview(x, Kokkos::make_pair((size_t)0, (size_t)P4));
  Kokkos::deep_copy(d_seed, h_seed);

  Kokkos::View<uint32_t_l*> y("y", 3 * P4);
  auto h_y = Kokkos::create_mirror_view(y);
  for (int i = 0; i < 3 * P4; i++) h_y(i) = 0;
  h_y(2 * P4) = 1;
  Kokkos::deep_copy(y, h_y);

  firstColGPU(x, s);
  colYGPU(y, s);
  Kokkos::fence();

  lastEntGPU(x, y, s, r);
  colsGPU(x, s, r);
  Kokkos::fence();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: ./main <n>\n");
    return 1;
  }

  uint32_t_l n = (uint32_t_l)atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    srand(1234);

    uint32_t_l *x = (uint32_t_l*)malloc(n * sizeof(uint32_t_l));

    for (uint32_t_l r = 16; r <= 4096; r = r * 2) {

      // Compute s the same way as the CUDA reference.
      // Guard: when n/r < P4, rounding down to a P4-multiple yields s=0
      // and the while-loop would spin forever, so stop early.
      uint32_t_l s = n / r;
      s -= (s % P4 == 0 ? 0 : s % P4);
      if (s == 0) break;
      while (s * r < n) r++;

      printf("n=%u r=%u s=%u\n", n, r, s);

      // Seed for this run
      uint32_t_l *seed = (uint32_t_l*)malloc(P4 * sizeof(uint32_t_l));
      for (uint32_t_l k = 0; k < P4; k++) x[k] = seed[k] = (uint32_t_l)rand();

      // ---- Host computation ----
      auto h_start = std::chrono::steady_clock::now();
      LFIB4(n, x);
      auto h_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> host_time = h_end - h_start;

      // ---- Device computation via Kokkos ----
      Kokkos::View<uint32_t_l*> d_x("d_x", (size_t)r * s);
      auto d_start = std::chrono::steady_clock::now();
      gLFIB4(d_x, s, r, seed);
      auto d_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> device_time = d_end - d_start;

      // ---- Verify ----
      auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_x);
      bool ok = true;
      for (uint32_t_l i = 0; i < n; i++) {
        if (x[i] != h_x(i)) { ok = false; break; }
      }

      double speedup = (device_time.count() > 0.0)
                       ? host_time.count() / device_time.count()
                       : 1.0;

      printf("r = %u | host time = %lf | device time = %lf | speedup = %.1f "
             "check = %s\n",
             r, host_time.count(), device_time.count(), speedup,
             ok ? "PASS" : "FAIL");

      free(seed);
    }
    free(x);
  }
  Kokkos::finalize();
  return 0;
}
