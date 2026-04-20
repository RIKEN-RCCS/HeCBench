// Port of tensorAccessor CUDA benchmark to Kokkos
// Both kernels (raw pointer and accessor-style) collapse to the same Kokkos implementation

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <nrows> <ncols> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t nrow   = atol(argv[1]);
  const int64_t ncol   = atol(argv[2]);
  const int      repeat = atoi(argv[3]);

  const int64_t numel   = nrow * ncol;
  const int64_t m_bytes = numel * sizeof(float);
  const int64_t v_bytes = ncol  * sizeof(float);
  const int64_t r_bytes = nrow  * sizeof(float);

  std::vector<float> h_m(numel), h_v(ncol), h_r(nrow), h_r_ref(nrow);
  srand(123);
  for (int64_t i = 0; i < numel; i++) h_m[i] = rand() / (float)RAND_MAX;
  for (int64_t i = 0; i < ncol;  i++) h_v[i] = rand() / (float)RAND_MAX;

  // Reference on CPU
  for (int64_t i = 0; i < nrow; i++) {
    float val = 0.f;
    for (int64_t j = 0; j < ncol; j++) val += h_m[i * ncol + j] * h_v[j];
    h_r_ref[i] = val;
  }

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_m("m", numel);
    Kokkos::View<float*> d_v("v", ncol);
    Kokkos::View<float*> d_r("r", nrow);

    {
      auto hm = Kokkos::create_mirror_view(d_m);
      auto hv = Kokkos::create_mirror_view(d_v);
      for (int64_t i = 0; i < numel; i++) hm(i) = h_m[i];
      for (int64_t i = 0; i < ncol;  i++) hv(i) = h_v[i];
      Kokkos::deep_copy(d_m, hm);
      Kokkos::deep_copy(d_v, hv);
    }

    // Warmup
    printf("Warmup..\n");
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("raw_matvec", nrow, KOKKOS_LAMBDA(const int64_t row) {
        float val = 0.f;
        for (int64_t j = 0; j < ncol; j++) val += d_m(row * ncol + j) * d_v(j);
        d_r(row) = val;
      });
    }
    Kokkos::fence();

    // Benchmark raw accessor kernel
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("raw_matvec", nrow, KOKKOS_LAMBDA(const int64_t row) {
        float val = 0.f;
        for (int64_t j = 0; j < ncol; j++) val += d_m(row * ncol + j) * d_v(j);
        d_r(row) = val;
      });
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average execution time of raw_accessor_kernel: %f (us)\n", ns * 1e-3f / repeat);

    // Benchmark tensor accessor kernel (same computation, different label)
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("tensor_matvec", nrow, KOKKOS_LAMBDA(const int64_t row) {
        float val = 0.f;
        for (int64_t j = 0; j < ncol; j++) val += d_m(row * ncol + j) * d_v(j);
        d_r(row) = val;
      });
    }
    Kokkos::fence();
    t1 = std::chrono::steady_clock::now();
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("Average execution time of tensor_packed_accessor_kernel: %f (us)\n", ns * 1e-3f / repeat);

    // Copy back and verify
    auto hr = Kokkos::create_mirror_view(d_r);
    Kokkos::deep_copy(hr, d_r);

    bool ok = true;
    for (int64_t i = 0; i < nrow; i++) {
      if (fabsf(hr(i) - h_r_ref[i]) > 1e-3f) { ok = false; break; }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
