#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <Kokkos_Core.hpp>

#define VALUE 1

static void segreduce(const size_t num_elements, const int repeat)
{
  printf("num_elements = %zu\n", num_elements);

  // Host input: all ones
  Kokkos::View<int*> d_in("d_in", num_elements);
  Kokkos::deep_copy(d_in, VALUE);

  for (size_t segment_size = 16; segment_size <= 16384; segment_size *= 2) {
    const size_t num_segments = num_elements / segment_size;

    // Build keys on host (key[i] = i / segment_size) – not needed for the
    // simple per-segment approach below, but kept for conceptual parity.
    Kokkos::View<int*> d_out("d_out", num_segments);

    // Warmup
    {
      const size_t seg = segment_size;
      const size_t nsegs = num_segments;
      Kokkos::parallel_for("segreduce_warmup", (int)nsegs,
          KOKKOS_LAMBDA(const int s) {
            int sum = 0;
            for (size_t i = 0; i < seg; i++) sum += d_in[(size_t)s * seg + i];
            d_out[s] = sum;
          });
      Kokkos::fence();
    }

    auto start = std::chrono::steady_clock::now();

    const size_t seg   = segment_size;
    const size_t nsegs = num_segments;

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("segreduce", (int)nsegs,
          KOKKOS_LAMBDA(const int s) {
            int sum = 0;
            for (size_t i = 0; i < seg; i++) sum += d_in[(size_t)s * seg + i];
            d_out[s] = sum;
          });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    printf("num_segments = %zu segment_size = %zu Throughput = %f (G/s)\n",
           num_segments, segment_size,
           1.0 * num_elements * repeat / ns);

    // Verify
    auto h_out = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out, d_out);

    int expected = (int)segment_size * VALUE;
    int errors = 0;
    for (size_t i = 0; i < num_segments; i++) {
      if (h_out((int)i) != expected) {
        errors++;
        if (errors < 10)
          printf("segment %zu has sum %d (expected %d)\n", i, h_out((int)i), expected);
      }
    }
    if (errors > 0)
      printf("segmented reduction does not agree with the reference! %d errors!\n", errors);
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }

  const int multiplier = atoi(argv[1]);
  const int repeat     = atoi(argv[2]);
  size_t num_elements  = 16384ULL * multiplier;

  Kokkos::initialize(argc, argv);
  {
    segreduce(num_elements, repeat);
  }
  Kokkos::finalize();
  return 0;
}
