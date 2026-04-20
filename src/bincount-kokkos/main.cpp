/*
 * Kokkos port of bincount benchmark.
 * Histogram computation using Kokkos atomic_fetch_add.
 * Two modes: global atomics only, and local (team-shared) + global atomics.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <Kokkos_Core.hpp>

template <typename input_t, typename output_t, typename IndexType>
KOKKOS_INLINE_FUNCTION
IndexType getBin(input_t v, input_t minvalue, input_t maxvalue, IndexType nbins) {
  IndexType bin = (IndexType)((v - minvalue) * nbins / (maxvalue - minvalue));
  if (bin == nbins) bin--;
  return bin;
}

// Global atomics version
template <typename output_t, typename input_t, typename IndexType>
void bincount_global(
    Kokkos::View<output_t*> output,
    Kokkos::View<const input_t*> input,
    IndexType nbins, input_t minvalue, input_t maxvalue, IndexType input_size,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("bincount_global", input_size, KOKKOS_LAMBDA(IndexType i) {
      input_t v = input(i);
      if (v >= minvalue && v <= maxvalue) {
        IndexType bin = getBin(v, minvalue, maxvalue, nbins);
        Kokkos::atomic_fetch_add(&output(bin), (output_t)1);
      }
    });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of bincount kernel (global): %f (us)\n",
         (time * 1e-3f) / repeat);
}

// Shared (team-local) + global atomics version using TeamPolicy + scratch
template <typename output_t, typename input_t, typename IndexType>
void bincount_shared(
    Kokkos::View<output_t*> output,
    Kokkos::View<const input_t*> input,
    IndexType nbins, input_t minvalue, input_t maxvalue, IndexType input_size,
    int repeat)
{
  using TeamPol = Kokkos::TeamPolicy<>;
  using TeamMem = TeamPol::member_type;
  using ScratchView = Kokkos::View<output_t*, Kokkos::ScratchMemorySpace<>, Kokkos::MemoryUnmanaged>;

  const int block_size = 256;
  int nteams = (input_size + block_size - 1) / block_size;
  int scratch_size = ScratchView::shmem_size(nbins);

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("bincount_shared",
      TeamPol(nteams, block_size).set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const TeamMem& team) {
        ScratchView smem(team.team_scratch(0), nbins);
        // init shared histogram
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nbins),
          [&](IndexType i) { smem(i) = 0; });
        team.team_barrier();

        // accumulate in shared
        int start_i = team.league_rank() * block_size;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, block_size),
          [&](int t) {
            IndexType li = (IndexType)(start_i + t);
            if (li < input_size) {
              input_t v = input(li);
              if (v >= minvalue && v <= maxvalue) {
                IndexType bin = getBin(v, minvalue, maxvalue, nbins);
                Kokkos::atomic_fetch_add(&smem(bin), (output_t)1);
              }
            }
          });
        team.team_barrier();

        // flush to global
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nbins),
          [&](IndexType i) {
            Kokkos::atomic_fetch_add(&output(i), smem(i));
          });
      });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of bincount kernel (shared+global): %f (us)\n",
         (time * 1e-3f) / repeat);
}

template <typename output_t, typename input_t, typename IndexType>
void eval(IndexType input_size, int repeat) {
  std::vector<input_t> h_input(input_size);
  std::default_random_engine gen(123);
  std::normal_distribution<input_t> dist(5.0, 2.0);
  for (auto& v : h_input) v = dist(gen);

  input_t minval = *std::min_element(h_input.begin(), h_input.end());
  input_t maxval = *std::max_element(h_input.begin(), h_input.end());
  printf("Input min, max values: (%f %f)\n", (float)minval, (float)maxval);

  Kokkos::View<input_t*> d_input("input", input_size);
  auto h_in = Kokkos::create_mirror_view(d_input);
  for (IndexType i = 0; i < input_size; i++) h_in(i) = h_input[i];
  Kokkos::deep_copy(d_input, h_in);

  for (IndexType nbins = 768; nbins <= 768 * 32; nbins *= 2) {
    printf("\nNumber of bins: %d\n", (int)nbins);

    Kokkos::View<output_t*> d_output("output", nbins);

    printf("bincount using global atomics\n");
    Kokkos::deep_copy(d_output, (output_t)0);
    bincount_global<output_t, input_t, IndexType>(
        d_output, Kokkos::View<const input_t*>(d_input), nbins, minval, maxval, input_size, repeat);

    auto h_out = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_out, d_output);
    output_t omin = *std::min_element(h_out.data(), h_out.data()+nbins);
    output_t omax = *std::max_element(h_out.data(), h_out.data()+nbins);
    printf("Output min, median, max values: (%ld %ld %ld)\n",
           (int64_t)omin/repeat, (int64_t)h_out(nbins/2)/repeat, (int64_t)omax/repeat);

    // Shared version (only when nbins is small enough for scratch)
    printf("\nbincount using global and local atomics\n");
    Kokkos::deep_copy(d_output, (output_t)0);
    bincount_shared<output_t, input_t, IndexType>(
        d_output, Kokkos::View<const input_t*>(d_input), nbins, minval, maxval, input_size, repeat);
    Kokkos::deep_copy(h_out, d_output);
    omin = *std::min_element(h_out.data(), h_out.data()+nbins);
    omax = *std::max_element(h_out.data(), h_out.data()+nbins);
    printf("Output min, median, max values: (%ld %ld %ld)\n\n",
           (int64_t)omin/repeat, (int64_t)h_out(nbins/2)/repeat, (int64_t)omax/repeat);
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 3) {
      printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    const int n = atoi(argv[1]);
    const int repeat = atoi(argv[2]);
    eval<int, float, int>(n, repeat);
  }
  Kokkos::finalize();
  return 0;
}
