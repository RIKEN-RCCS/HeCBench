/*
 * Kokkos port of blas-gemmBatched.
 * Original used cublasSgemmBatched / cublasDgemmBatched / cublasHgemmBatched.
 * This port implements batched matrix-vector multiply using Kokkos parallel_for.
 * Each batch element: C[i] = alpha * A[i] * B[i] + beta * C[i]
 * where A[i] is size x size and B[i] is size x 1.
 */

#include <assert.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

void performance(int m, int n, int k, double avg_time_us) {
  double total_ops = 2.0 * m * n * k;
  double perf_gflops = total_ops / (avg_time_us * 1e-6) * 1e-9;
  if (perf_gflops >= 1000)
    printf("%.3lf TFLOP/s\n", perf_gflops / 1000.0);
  else
    printf("%.3lf GFLOP/s\n", perf_gflops);
}

void gemmBatched_ref(int num, int lda, int ldb, int ldc,
                     int m, int k, int n, double alpha, double beta,
                     const double* A, int ldA, const double* B, int ldB,
                     double* C, int ldC) {
  for (int b = 0; b < num; b++) {
    const double* Ab = A + b * lda * lda;
    const double* Bb = B + b * ldb;
    double* Cb = C + b * ldc;
    for (int i = 0; i < m; i++) {
      double s = 0.0;
      for (int j = 0; j < k; j++) s += Ab[i + j * ldA] * Bb[j];
      Cb[i] = alpha * s + beta * Cb[i];
    }
  }
}

template <typename T>
void gemmBatched(int lower, int upper, int num, int reps, int verbose) {
  using std::cout;
  if (verbose) cout << "initializing inputs" << std::endl;

  size_t mat_size = (size_t)upper * upper * num;
  size_t vec_size = (size_t)upper * num;

  std::vector<T> h_matrices(mat_size);
  std::vector<T> h_vectors(vec_size);
  std::vector<T> h_result(vec_size, T(0));
  std::vector<double> h_result_ref(vec_size, 0.0);

  srand48(48);
  for (size_t i = 0; i < mat_size; i++) h_matrices[i] = (T)drand48();
  for (size_t i = 0; i < vec_size; i++)  h_vectors[i] = (T)drand48();

  Kokkos::View<T*> d_matrices("matrices", mat_size);
  Kokkos::View<T*> d_vectors("vectors", vec_size);
  Kokkos::View<T*> d_result("result", vec_size);

  auto hm = Kokkos::create_mirror_view(d_matrices);
  auto hv = Kokkos::create_mirror_view(d_vectors);
  for (size_t i = 0; i < mat_size; i++) hm(i) = h_matrices[i];
  for (size_t i = 0; i < vec_size; i++) hv(i) = h_vectors[i];
  Kokkos::deep_copy(d_matrices, hm);
  Kokkos::deep_copy(d_vectors, hv);

  const T alpha = T(1), beta = T(0);

  for (int size = lower; size <= upper; size++) {
    if (verbose) cout << "running with <" << size << " x " << size << "> x <" << size << " x 1>" << std::endl;
    const int m = size, n = 1, k = size;
    const int lda = upper, ldb = upper, ldc = upper;
    double sum = 0.0;

    for (int rep = 0; rep <= reps; rep++) {
      Kokkos::deep_copy(d_result, T(0));
      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();

      Kokkos::parallel_for("gemmBatched",
        Kokkos::RangePolicy<>(0, num * m),
        KOKKOS_LAMBDA(int idx) {
          int b = idx / m;
          int row = idx % m;
          T s = T(0);
          const T* Ab = d_matrices.data() + b * (size_t)lda * lda;
          const T* Bb = d_vectors.data() + b * ldb;
          for (int j = 0; j < k; j++)
            s += Ab[row + j * lda] * Bb[j];
          d_result(b * ldc + row) = alpha * s + beta * d_result(b * ldc + row);
        });
      Kokkos::fence();

      auto end = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;
      if (rep != 0) sum += elapsed;
      if (verbose) cout << "size " << size << ": " << elapsed << " us\n";
    }

    cout << "size " << size << " average execution time: " << sum/reps << " us; "
         << sum/reps/num << " us per operation; floating-point operations per second: ";
    performance(m, n, k, sum/reps/num);

    // Verify double precision
    if constexpr (std::is_same_v<T, double>) {
      auto hr = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(hr, d_result);
      std::vector<double> ref(vec_size, 0.0);
      std::vector<double> dmat(mat_size), dvec(vec_size);
      for (size_t i = 0; i < mat_size; i++) dmat[i] = (double)h_matrices[i];
      for (size_t i = 0; i < vec_size; i++) dvec[i] = (double)h_vectors[i];
      gemmBatched_ref(num, lda, ldb, ldc, m, k, n, 1.0, 0.0,
                      dmat.data(), lda, dvec.data(), ldb, ref.data(), ldc);
      for (int b = 0; b < num; b++) {
        for (int j = 0; j < m; j++) {
          if (std::abs((double)hr(b*upper+j) - ref[b*upper+j]) > 1e-6) {
            cout << "Mismatch at batch " << b << ": " << hr(b*upper+j) << " != " << ref[b*upper+j] << "\n";
            break;
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    int lower = 2, upper = 100, num = 25000, reps = 10, verbose = 0;
    int status;
    while ((status = getopt(argc, argv, "l:u:n:r:v")) != -1) {
      switch (status) {
        case 'l': lower = strtoul(optarg, 0, 0); break;
        case 'u': upper = strtoul(optarg, 0, 0); break;
        case 'n': num   = strtoul(optarg, 0, 0); break;
        case 'r': reps  = strtoul(optarg, 0, 0); break;
        case 'v': verbose = 1; break;
        default: std::cerr << "invalid argument\n"; exit(1);
      }
    }
    std::cout << "running with lower: " << lower << " upper: " << upper
              << " num: " << num << " reps: " << reps << "\n";
    std::cout << ">>>>>>>>>>>>>>> Single precision gemmBatched >>>>>>>>>>>>>>> \n";
    gemmBatched<float>(lower, upper, num, reps, verbose);
    std::cout << ">>>>>>>>>>>>>>> Double precision gemmBatched >>>>>>>>>>>>>>> \n";
    gemmBatched<double>(lower, upper, num, reps, verbose);
  }
  Kokkos::finalize();
  return 0;
}
