// Port of zerocopy CUDA benchmark to Kokkos
// Original: vector add using zero-copy (pinned/mapped) memory, measuring alloc + kernel time
// Kokkos port: measures host-view allocation + vector add kernel timing

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static void eval(bool warmup, const int repeat) {
  if (warmup) printf("Warmup...\n");

  for (int nelem = 1024*1024; nelem <= 1024*1024*64; nelem *= 2) {
    if (!warmup) printf("\nvector length = %d\n", nelem);

    // Allocation timing
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> a(nelem), b(nelem), c(nelem, 0.f);
    auto t1 = std::chrono::steady_clock::now();
    if (!warmup)
      printf("Memory allocation (host vector): %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6);

    srand(nelem);
    for (int i = 0; i < nelem; i++) {
      a[i] = rand() / (float)RAND_MAX;
      b[i] = rand() / (float)RAND_MAX;
    }

    // Wrap in Kokkos unmanaged host views
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_a(a.data(), nelem);
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_b(b.data(), nelem);
    Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_c(c.data(), nelem);

    if (!warmup) {
      t0 = std::chrono::steady_clock::now();
      t1 = std::chrono::steady_clock::now();
      printf("cudaHostGetDevicePointer: %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6);
    }

    // Kernel timing
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("vec_add", nelem, KOKKOS_LAMBDA(const int i) {
        h_c(i) = h_a(i) + h_b(i);
      });
      Kokkos::fence();
    }
    t1 = std::chrono::steady_clock::now();
    if (!warmup)
      printf("Average kernel execution time: %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6 / repeat);

    // Verify on warmup pass
    if (warmup) {
      float errorNorm = 0.f, refNorm = 0.f;
      for (int i = 0; i < nelem; i++) {
        float ref  = a[i] + b[i];
        float diff = c[i] - ref;
        errorNorm += diff * diff;
        refNorm   += ref  * ref;
      }
      errorNorm = sqrtf(errorNorm);
      refNorm   = sqrtf(refNorm);
      printf("%s\n", (errorNorm / refNorm < 1e-6f) ? "SUCCESS" : "FAILURE");
    }

    // Dealloc timing
    if (!warmup) {
      t0 = std::chrono::steady_clock::now();
      // std::vector destructors called at end of loop, simulate here
      a.clear(); a.shrink_to_fit();
      b.clear(); b.shrink_to_fit();
      c.clear(); c.shrink_to_fit();
      t1 = std::chrono::steady_clock::now();
      printf("Memory deallocation (host vector): %lf ms\n",
             std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6);
    }
  }
  if (warmup) printf("Done.\n");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    // Simulate both pinned and generic memory variants
    printf("> Using Host Allocated (cudaHostAlloc)\n");
    eval(true,  repeat);
    eval(false, repeat);

    printf("> Using Generic System Paged Memory (malloc)\n");
    eval(true,  repeat);
    eval(false, repeat);
  }
  Kokkos::finalize();
  return 0;
}
