// OpenACC implementation of GEMM: C = alpha * A * B + beta * C
// A is (M x K), B is (K x N), C is (M x N)  — all row-major
//
// Compile example (NVIDIA target via NVHPC/PGI):
//   nvc++ -acc=gpu -gpu=cc80 -O3 -o main main.cpp
//
// Compile example (GCC with OpenACC support):
//   g++ -fopenacc -O3 -o main main.cpp

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "utils.h"

// -----------------------------------------------------------------------
// Reference (host) GEMM — used for correctness check
// -----------------------------------------------------------------------
template <typename fp>
void ref_gemm(const fp *A, const fp *B, fp *C,
              int M, int K, int N, fp alpha, fp beta)
{
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      fp s = fp(0);
      for (int k = 0; k < K; k++)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = alpha * s + beta * C[i * N + j];
    }
  }
}

// -----------------------------------------------------------------------
// OpenACC GEMM kernel: C = alpha * A * B + beta * C
// -----------------------------------------------------------------------
template <typename fp>
void openacc_gemm(const fp *A, const fp *B, fp *C,
                  int M, int K, int N, fp alpha, fp beta)
{
  #pragma acc parallel loop collapse(2) \
      present(A[0:M*K], B[0:K*N], C[0:M*N])
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      fp s = fp(0);
      #pragma acc loop reduction(+:s)
      for (int k = 0; k < K; k++)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = alpha * s + beta * C[i * N + j];
    }
  }
}

// -----------------------------------------------------------------------
// Full benchmark: initialise, verify, measure
// -----------------------------------------------------------------------
template <typename fp>
void run_gemm_example(int m, int k, int n, int repeat)
{
  const fp alpha = fp(2.0);
  const fp beta  = fp(0.5);

  const size_t A_size = sizeof(fp) * m * k;
  const size_t B_size = sizeof(fp) * k * n;
  const size_t C_size = sizeof(fp) * m * n;

  fp *a = static_cast<fp *>(aligned_alloc(64, A_size));
  fp *b = static_cast<fp *>(aligned_alloc(64, B_size));
  fp *c = static_cast<fp *>(aligned_alloc(64, C_size));
  fp *r = static_cast<fp *>(aligned_alloc(64, C_size));   // reference result

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);
  memcpy(r, c, C_size);   // r starts from the same initial C

  // Compute reference result on the host
  ref_gemm(a, b, r, m, k, n, alpha, beta);

  // ---- OpenACC data region: keep A, B, C on device throughout ----------
  #pragma acc data copyin(a[0:m*k], b[0:k*n]) copy(c[0:m*n])
  {
    // Correctness check (single call)
    openacc_gemm(a, b, c, m, k, n, alpha, beta);
  }

  std::cout << "Checking OpenACC GEMM.. ";
  int error = memcmp(c, r, C_size);
  std::cout << (error ? "FAIL" : "PASS") << std::endl;

  // Re-initialise c to a fresh random state before benchmarking
  rand_matrix(c, m, n);

  // ---- Performance benchmark -------------------------------------------
  #pragma acc data copyin(a[0:m*k], b[0:k*n]) copy(c[0:m*n])
  {
    // Warm-up
    openacc_gemm(a, b, c, m, k, n, alpha, beta);
    #pragma acc wait

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      openacc_gemm(a, b, c, m, k, n, alpha, beta);

    #pragma acc wait
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    performance(m, n, k, false, static_cast<double>(time) / repeat);
  }

#ifdef DEBUG
  std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;
  print_2x2_matrix_values(a, k, "A");
  print_2x2_matrix_values(b, n, "B");
  print_2x2_matrix_values(c, n, "C");
#endif

  free(a);
  free(b);
  free(c);
  free(r);
}

// -----------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------
int main(int argc, char **argv)
{
  if (argc != 5) {
    printf("Usage: %s <m> <k> <n> <repeat>\n", argv[0]);
    return 1;
  }
  const int m      = atoi(argv[1]);
  const int k      = atoi(argv[2]);
  const int n      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  std::cout << "\tRunning with single precision data type:" << std::endl;
  run_gemm_example<float>(m, k, n, repeat);

  std::cout << "\tRunning with double precision data type:" << std::endl;
  run_gemm_example<double>(m, k, n, repeat);

  return 0;
}
