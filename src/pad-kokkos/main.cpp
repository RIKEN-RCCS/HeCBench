#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Padding kernel: write a 2D matrix of shape (m x n) into an output of shape
// (m x (n+pad)).  Columns [0, n) are copied from input; columns [n, n+pad)
// are set to zero.

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <m> <n> <pad> <repeat>\n", argv[0]);
    return 1;
  }
  const int m      = atoi(argv[1]);
  const int n      = atoi(argv[2]);
  const int pad    = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int n_out    = n + pad;
  const int in_size  = m * n;
  const int out_size = m * n_out;

  printf("m=%d n=%d pad=%d n_out=%d repeat=%d\n", m, n, pad, n_out, repeat);

  // Host allocation and initialisation
  float* h_in  = new float[in_size];
  float* h_out = new float[out_size]();

  srand(123);
  for (int i = 0; i < in_size; i++)
    h_in[i] = (float)rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_in ("d_in",  in_size);
    Kokkos::View<float*> d_out("d_out", out_size);

    // Copy input H→D
    {
      auto hv = Kokkos::View<const float*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_in, in_size);
      Kokkos::deep_copy(d_in, hv);
    }

    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for(
          "pad", out_size,
          KOKKOS_LAMBDA(int idx) {
            const int row = idx / n_out;
            const int col = idx % n_out;
            d_out[idx] = (col < n) ? d_in[row * n + col] : 0.f;
          });
    }

    Kokkos::fence();
    auto t1   = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Total padding execution time for %d iterations: %f (ms)\n",
           repeat, time * 1e-6f);
    printf("Average per iteration: %f (ms)\n", time * 1e-6f / repeat);

    // Copy result D→H
    {
      auto hv = Kokkos::View<float*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(h_out, out_size);
      Kokkos::deep_copy(hv, d_out);
    }

    // Verify
    int errors = 0;
    for (int row = 0; row < m && errors < 10; row++) {
      for (int col = 0; col < n_out; col++) {
        float expected = (col < n) ? h_in[row * n + col] : 0.f;
        if (h_out[row * n_out + col] != expected) {
          printf("Mismatch at (%d,%d): got %f expected %f\n",
                 row, col, h_out[row * n_out + col], expected);
          if (++errors >= 10) break;
        }
      }
    }
    if (errors == 0)
      printf("Verification PASSED\n");
    else
      printf("Verification FAILED (%d errors)\n", errors);
  }
  Kokkos::finalize();

  delete[] h_in;
  delete[] h_out;
  return 0;
}
