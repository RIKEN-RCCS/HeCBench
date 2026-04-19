// Port of overlap-cuda to Kokkos.
//
// The CUDA version pipelines H->D copy, kernel, D->H copy across 4 streams to
// demonstrate overlap.  Kokkos has no equivalent stream-overlap abstraction, so
// this port:
//   - Runs the same incKernel computation (out[i] += 1, inner_reps times)
//   - Measures "serial" time (STREAM_COUNT=1 logical pass) and "overlap" time
//     (STREAM_COUNT logical passes, back-to-back on device with separate Views)
//   - Verifies that all output elements equal inner_reps.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static constexpr int N           = 1 << 22;
static constexpr int nreps       = 10;
static constexpr int inner_reps  = 5;
static constexpr int STREAM_COUNT = 4;

// Run `streams_used` independent incKernel passes per repetition, nreps times.
// Returns total elapsed time in milliseconds.
float processWithKokkos(int streams_used,
                        Kokkos::View<int *> d_in[STREAM_COUNT],
                        Kokkos::View<int *> d_out[STREAM_COUNT])
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nreps; i++) {
    for (int s = 0; s < streams_used; s++) {
      // incKernel: out[idx] = (0==0 ? in[idx] : out[idx]) + 1, repeated
      // Equivalent: out[idx] = in[idx] + inner_reps  (in[] is all zeros)
      auto in_s  = d_in[s];
      auto out_s = d_out[s];
      Kokkos::parallel_for(
          "incKernel", N, KOKKOS_LAMBDA(int idx) {
            int val = in_s(idx);
            for (int k = 0; k < inner_reps; ++k)
              val += 1;
            out_s(idx) = val;
          });
    }
    Kokkos::fence();
  }

  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
             .count() * 1e-6f; // ms
}

int main(int /*argc*/, char * /*argv*/[])
{
  Kokkos::initialize();
  {
    printf("Length of the array = %d\n", N);

    Kokkos::View<int *> d_in[STREAM_COUNT];
    Kokkos::View<int *> d_out[STREAM_COUNT];

    for (int s = 0; s < STREAM_COUNT; s++) {
      d_in[s]  = Kokkos::View<int *>("d_in",  N);
      d_out[s] = Kokkos::View<int *>("d_out", N);
      // Input is all zeros
      Kokkos::deep_copy(d_in[s],  0);
      Kokkos::deep_copy(d_out[s], 0);
    }

    float serial_time  = processWithKokkos(1,            d_in, d_out);
    float overlap_time = processWithKokkos(STREAM_COUNT, d_in, d_out);

    printf("\nAverage measured timings over %d repetitions:\n", nreps);
    printf(" Avg. time when execution fully serialized\t: %f ms\n",
           serial_time / nreps);
    printf(" Avg. time when overlapped using %d streams\t: %f ms\n",
           STREAM_COUNT, overlap_time / nreps);
    printf(" Avg. speedup gained (serialized - overlapped)\t: %f\n",
           (serial_time - overlap_time) / nreps);

    printf("\nMeasured throughput:\n");
    int memsize = N * (int)sizeof(int);
    printf(" Fully serialized execution\t\t: %f GB/s\n",
           (nreps * (memsize * 2e-6)) / serial_time);
    printf(" Overlapped using %d streams\t\t: %f GB/s\n", STREAM_COUNT,
           (nreps * (memsize * 2e-6)) / overlap_time);

    // Verify: all output values should equal inner_reps
    bool passed = true;
    for (int s = 0; s < STREAM_COUNT && passed; s++) {
      auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        d_out[s]);
      for (int i = 0; i < N && passed; i++)
        passed = (h_out(i) == inner_reps);
    }

    printf("\n%s\n", passed ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
