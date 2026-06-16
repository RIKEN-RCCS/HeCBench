#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <vector>

// Threads per team (matches block size in the original)
#define BS 256

static double LCG_random_double(uint64_t* seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int n      = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int m      = (n + BS - 1) / BS;  // number of nodes/groups

  std::vector<int> h_nlist(n);
  std::vector<int> h_family(m);

  uint64_t seed = 123ULL;
  for (int i = 0; i < n; i++)
    h_nlist[i] = (LCG_random_double(&seed) > 0.5) ? 1 : -1;

  for (int i = 0; i < m; i++) {
    int s = 0;
    for (int j = 0; j < BS; j++) {
      int idx = i * BS + j;
      if (idx < n) s += (h_nlist[idx] != -1) ? 1 : 0;
    }
    h_family[i] = s + 1 + (int)(s * LCG_random_double(&seed));
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*>    nlist ("nlist",  n);
    Kokkos::View<int*>    family("family", m);
    Kokkos::View<int*>    n_neigh("n_neigh", m);
    Kokkos::View<double*> damage ("damage",  m);

    {
      auto hv_nlist  = Kokkos::create_mirror_view(nlist);
      auto hv_family = Kokkos::create_mirror_view(family);
      for (int i = 0; i < n; i++) hv_nlist(i)  = h_nlist[i];
      for (int i = 0; i < m; i++) hv_family(i) = h_family[i];
      Kokkos::deep_copy(nlist,  hv_nlist);
      Kokkos::deep_copy(family, hv_family);
    }

    using TeamPolicy = Kokkos::TeamPolicy<>;
    using TeamMember = TeamPolicy::member_type;

    Kokkos::fence();
    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("damage_of_node",
        TeamPolicy(m, Kokkos::AUTO),
        KOKKOS_LAMBDA(const TeamMember& team) {
          const int nid = team.league_rank();

          // Reduce: count bonds (nlist != -1) within this node's BS elements
          int neighbours = 0;
          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, BS),
            [=](int i, int& local_sum) {
              int gid = nid * BS + i;
              if (gid < (int)nlist.extent(0))
                local_sum += (nlist(gid) != -1) ? 1 : 0;
            },
            neighbours);

          // One thread per team writes the result (neighbours is broadcast)
          if (team.team_rank() == 0) {
            n_neigh(nid) = neighbours;
            damage(nid)  = 1.0 - (double)neighbours / (double)family(nid);
          }
        });
    }

    Kokkos::fence();
    auto t_end = std::chrono::steady_clock::now();
    auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

    auto h_damage = Kokkos::create_mirror_view(damage);
    Kokkos::deep_copy(h_damage, damage);

    double sum = 0.0;
    for (int i = 0; i < m; i++) sum += h_damage(i);
    printf("Checksum: total damage = %lf\n", sum);
  }
  Kokkos::finalize();
  return 0;
}
