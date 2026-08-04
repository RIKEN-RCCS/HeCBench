using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;
using ScratchView = Kokkos::View<float*, ScratchSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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
