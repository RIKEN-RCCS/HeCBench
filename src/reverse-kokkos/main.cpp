#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n", argv[0]);
    return 1;
  }
  const int iteration = atoi(argv[1]);
  const int len       = 256;

  // Expected results for even/odd number of reverses
  int gold_odd[len], gold_even[len];
  for (int i = 0; i < len; i++) {
    gold_odd[i]  = len - i - 1;
    gold_even[i] = i;
  }

  std::default_random_engine generator(123);
  std::uniform_int_distribution<int> distribution(100, 9999);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> test("test", len);
    auto h_test = Kokkos::create_mirror_view(test);

    using TeamPolicy  = Kokkos::TeamPolicy<>;
    using TeamMember  = TeamPolicy::member_type;
    using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchView  = Kokkos::View<int*, ScratchSpace, Kokkos::MemoryUnmanaged>;

    const size_t scratch_size = ScratchView::shmem_size(len);

    long total_time = 0;
    int  error      = 0;

    for (int i = 0; i < iteration; i++) {
      const int count = distribution(generator);

      // Initialise device array to [0, 1, 2, ..., len-1]
      for (int j = 0; j < len; j++) h_test(j) = gold_even[j];
      Kokkos::deep_copy(test, h_test);

      Kokkos::fence();
      auto t_start = std::chrono::steady_clock::now();

      for (int j = 0; j < count; j++) {
        Kokkos::parallel_for("reverse",
          TeamPolicy(1, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
          KOKKOS_LAMBDA(const TeamMember& team) {
            ScratchView s(team.team_scratch(0), len);
            // Load phase: each thread loads its share of elements into scratch
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, len), [=](int t) {
              s(t) = test(t);
            });
            team.team_barrier();
            // Store phase: write reversed elements back
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, len), [=](int t) {
              test(t) = s(len - t - 1);
            });
          });
      }

      Kokkos::fence();
      auto t_end = std::chrono::steady_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

      Kokkos::deep_copy(h_test, test);

      const int* expected = (count % 2 == 0) ? gold_even : gold_odd;
      for (int j = 0; j < len; j++) {
        if (h_test(j) != expected[j]) { error = 1; break; }
      }
      if (error) break;
    }

    printf("Total kernel execution time: %f (s)\n", total_time * 1e-9f);
    printf("%s\n", error ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return 0;
}
