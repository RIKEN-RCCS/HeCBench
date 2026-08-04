using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;
using ScratchView = Kokkos::View<float*, ScratchSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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
