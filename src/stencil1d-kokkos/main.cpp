/*
  1D stencil: adds 7 neighbors on each side (RADIUS=7) using Kokkos TeamPolicy
  with scratch memory to hold the halo-padded tile per team.
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define RADIUS     7
#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int pad_size = length + RADIUS;

  int* h_a = (int*)malloc(pad_size * sizeof(int));
  int* h_b = (int*)malloc(length  * sizeof(int));

  for (int i = 0; i < pad_size; i++) h_a[i] = i;

  Kokkos::initialize(argc, argv);
  {
    using ExecSpace   = Kokkos::DefaultExecutionSpace;
    using ScratchSpace = ExecSpace::scratch_memory_space;
    using ScratchView  = Kokkos::View<int*, ScratchSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    Kokkos::View<int*> a("a", pad_size);
    Kokkos::View<int*> b("b", length);

    // Copy host data into device views
    {
      auto h_a_v = Kokkos::View<int*, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_a, pad_size);
      Kokkos::deep_copy(a, h_a_v);
    }

    const int scratch_size = ScratchView::shmem_size(BLOCK_SIZE + 2 * RADIUS);
    const int num_teams    = length / BLOCK_SIZE;

    using policy_t = Kokkos::TeamPolicy<ExecSpace>;
    // Use AUTO team size so the backend can choose an optimal, supported value
    policy_t policy(num_teams, Kokkos::AUTO);

    Kokkos::fence();
    auto t_start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(
        "stencil1d",
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const policy_t::member_type& team) {
          ScratchView temp(team.team_scratch(0), BLOCK_SIZE + 2 * RADIUS);

          const int block_start = team.league_rank() * BLOCK_SIZE;

          // Load BLOCK_SIZE center elements distributed across threads
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, BLOCK_SIZE),
            [&](const int j) { temp(j + RADIUS) = a(block_start + j); });

          // Load RADIUS halo elements on each side
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, RADIUS),
            [&](const int j) {
              const int gindex          = block_start + j;
              temp(j)                   = (gindex < RADIUS) ? 0 : a(gindex - RADIUS);
              temp(j + RADIUS + BLOCK_SIZE) = a(gindex + BLOCK_SIZE);
            });

          team.team_barrier();

          // Compute stencil sum for each output element
          Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, BLOCK_SIZE),
            [&](const int j) {
              int result = 0;
              for (int offset = -RADIUS; offset <= RADIUS; offset++)
                result += temp(j + RADIUS + offset);
              b(block_start + j) = result;
            });
        });
    }

    Kokkos::fence();
    auto t_end = std::chrono::steady_clock::now();
    auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    // Copy result back to host
    {
      auto h_b_v = Kokkos::View<int*, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_b, length);
      Kokkos::deep_copy(h_b_v, b);
    }
  }
  Kokkos::finalize();

  // Verification against CPU reference
  bool ok = true;
  for (int i = 0; i < 2 * RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i + 2 * RADIUS; j++)
      s += j < RADIUS ? 0 : (h_a[j] - RADIUS);
    if (s != h_b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, h_b[i]);
      ok = false;
      break;
    }
  }
  if (ok) {
    for (int i = 2 * RADIUS; i < length; i++) {
      int s = 0;
      for (int j = i - RADIUS; j <= i + RADIUS; j++)
        s += h_a[j];
      if (s != h_b[i]) {
        printf("Error at %d: %d (host) != %d (device)\n", i, s, h_b[i]);
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_a);
  free(h_b);
  return 0;
}
