#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define LAUNCH_ITERATIONS 3

static void init_input(float *a, size_t size)
{
  srand(123);
  for (size_t i = 0; i < size; i++)
    a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

// Mirrors "usingGraph" / "usingStream": copy input then run 100 reduce rounds.
// Without CUDA Graphs both implementations are identical in Kokkos.
static void run_reduce(const char *label,
                       Kokkos::View<float *> d_input,
                       typename Kokkos::View<float *>::HostMirror h_input,
                       size_t size)
{
  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    Kokkos::deep_copy(d_input, h_input);

    auto start = std::chrono::steady_clock::now();

    double result = 0.0;
    for (int i = 0; i < 100; i++) {
      double partial = 0.0;
      Kokkos::parallel_reduce(
        "reduce",
        Kokkos::RangePolicy<>(0, size),
        KOKKOS_LAMBDA(size_t j, double &sum) { sum += (double)d_input(j); },
        partial);
      result = partial;
    }
    Kokkos::fence();

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("[%s] final reduced sum = %lf\n", label, result);
    printf("Execution time: %f (us)\n\n", time * 1e-3f);
  }
}

int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  {
    for (size_t size = 512; size <= (size_t)(1 << 27); size *= 512) {
      printf("\n-----------------------------\n");
      printf("%zu elements\n", size);
      printf("Launch iterations = %d\n", LAUNCH_ITERATIONS);

      Kokkos::View<float *> d_input("input", size);
      auto h_input = Kokkos::create_mirror_view(d_input);
      init_input(h_input.data(), size);

      run_reduce("usingGraph",  d_input, h_input, size);
      run_reduce("UsingStream", d_input, h_input, size);
    }
  }
  Kokkos::finalize();
  return 0;
}
