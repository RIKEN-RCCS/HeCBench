#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <Kokkos_Core.hpp>

#define N 2048
#define IDX(i,j) ((i)+(j)*N)

int main() {
  std::clock_t start_time = std::clock();

  Kokkos::initialize();
  {
    Kokkos::View<float*> d_f("f", N*N), d_f_old("f_old", N*N);
    auto h_f     = Kokkos::create_mirror_view(d_f);
    auto h_f_old = Kokkos::create_mirror_view(d_f_old);

    // Initialize boundary conditions
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < N; ++i) {
        float val = 0.f;
        if      (i == 0 || i == N-1) val = sinf(j * 2 * (float)M_PI / (N-1));
        else if (j == 0 || j == N-1) val = sinf(i * 2 * (float)M_PI / (N-1));
        h_f[IDX(i,j)] = h_f_old[IDX(i,j)] = val;
      }
    Kokkos::deep_copy(d_f, h_f);
    Kokkos::deep_copy(d_f_old, h_f_old);

    float error      = std::numeric_limits<float>::max();
    const float tol  = 1.e-5f;
    const int max_it = 10000;
    int num_iters    = 0;

    auto start = std::chrono::steady_clock::now();

    while (error > tol && num_iters < max_it) {
      error = 0.f;

      Kokkos::parallel_reduce("jacobi_step",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{N-1,N-1}),
        KOKKOS_LAMBDA(int i, int j, float& err) {
          float t = 0.25f * (d_f_old[IDX(i-1,j)] + d_f_old[IDX(i+1,j)] +
                             d_f_old[IDX(i,j-1)] + d_f_old[IDX(i,j+1)]);
          float df = t - d_f_old[IDX(i,j)];
          d_f[IDX(i,j)] = t;
          err += df * df;
        }, error);

      Kokkos::parallel_for("swap",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{N-1,N-1}),
        KOKKOS_LAMBDA(int i, int j) {
          d_f_old[IDX(i,j)] = d_f[IDX(i,j)];
        });

      error = sqrtf(error / (N * N));

      if (num_iters % 1000 == 0)
        std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
      ++num_iters;
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average execution time per iteration: "
              << (time * 1e-9f) / num_iters << " (s)\n";

    if (error <= tol && num_iters < max_it)
      std::cout << "PASS" << std::endl;
    else
      std::cout << "FAIL" << std::endl;
  }
  Kokkos::finalize();

  double duration = (std::clock() - start_time) / (double)CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds\n";
  return 0;
}
