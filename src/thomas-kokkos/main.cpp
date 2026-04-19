#include <chrono>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "ThomasMatrix.hpp"
#include "utils.hpp"

void solve_seq(const double* l, const double* d, double* u, double* rhs, const int n, const int N)
{
  for (int j = 0; j < N; ++j) {
    int first = j*n;
    int last  = first + n - 1;

    u[first]   /= d[first];
    rhs[first] /= d[first];

    for (int i = first+1; i < last; i++) {
      u[i]   /= d[i] - l[i]*u[i-1];
      rhs[i]  = (rhs[i] - l[i]*rhs[i-1]) / (d[i] - l[i]*u[i-1]);
    }

    rhs[last] = (rhs[last] - l[last]*rhs[last-1]) / (d[last] - l[last]*u[last-1]);

    for (int i = last-1; i >= first; i--)
      rhs[i] -= u[i]*rhs[i+1];
  }
}

int main(int argc, char const *argv[])
{
  if (argc != 5) {
    std::cout << "Usage: " << argv[0]
              << " [system size] [#systems] [thread block size] [repeat]\n";
    return -1;
  }

  const int M         = std::stoi(argv[1]);
  const int N         = std::stoi(argv[2]);
  const int repeat    = std::stoi(argv[4]);

  const size_t matrix_size       = (size_t)M * N;
  const size_t matrix_size_bytes = matrix_size * sizeof(double);

  ThomasMatrix params = loadThomasMatrixSyn(M);

  double* u_seq   = (double*) malloc(matrix_size_bytes);
  double* d_seq   = (double*) malloc(matrix_size_bytes);
  double* l_seq   = (double*) malloc(matrix_size_bytes);
  double* rhs_seq = (double*) malloc(matrix_size_bytes);

  double* u_input   = (double*) malloc(matrix_size_bytes);
  double* d_input   = (double*) malloc(matrix_size_bytes);
  double* l_input   = (double*) malloc(matrix_size_bytes);
  double* rhs_input = (double*) malloc(matrix_size_bytes);

  double* u_Thomas_host   = (double*) malloc(matrix_size_bytes);
  double* d_Thomas_host   = (double*) malloc(matrix_size_bytes);
  double* l_Thomas_host   = (double*) malloc(matrix_size_bytes);
  double* rhs_Thomas_host = (double*) malloc(matrix_size_bytes);

  double* rhs_seq_output    = (double*) malloc(matrix_size_bytes);
  double* rhs_seq_interleave= (double*) malloc(matrix_size_bytes);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      u_seq[(i*M)+j]   = u_input[(i*M)+j]   = params.a[j];
      d_seq[(i*M)+j]   = d_input[(i*M)+j]   = params.d[j];
      l_seq[(i*M)+j]   = l_input[(i*M)+j]   = params.b[j];
      rhs_seq[(i*M)+j] = rhs_input[(i*M)+j] = params.rhs[j];
    }

  auto t0 = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++)
    solve_seq(l_seq, d_seq, u_seq, rhs_seq, M, N);
  auto t1 = std::chrono::steady_clock::now();
  printf("Average serial execution time: %f (ms)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1e-6 / repeat);

  for (size_t i = 0; i < matrix_size; ++i)
    rhs_seq_output[i] = rhs_seq[i];

  // Re-initialize
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      u_seq[(i*M)+j]   = u_input[(i*M)+j]   = params.a[j];
      d_seq[(i*M)+j]   = d_input[(i*M)+j]   = params.d[j];
      l_seq[(i*M)+j]   = l_input[(i*M)+j]   = params.b[j];
      rhs_seq[(i*M)+j] = rhs_input[(i*M)+j] = params.rhs[j];
    }

  // Transpose for sequential GPU access (interleaved layout)
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      u_Thomas_host[i*N+j]   = u_input[j*M+i];
      l_Thomas_host[i*N+j]   = l_input[j*M+i];
      d_Thomas_host[i*N+j]   = d_input[j*M+i];
      rhs_Thomas_host[i*N+j] = rhs_input[j*M+i];
      rhs_seq_interleave[i*N+j] = rhs_seq_output[j*M+i];
    }

  Kokkos::initialize(argc, const_cast<char**>(argv));
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = Kokkos::DefaultExecutionSpace::memory_space;

    Kokkos::View<double*, MemSpace> U("U", matrix_size);
    Kokkos::View<double*, MemSpace> D("D", matrix_size);
    Kokkos::View<double*, MemSpace> L("L", matrix_size);
    Kokkos::View<double*, MemSpace> RHS("RHS", matrix_size);

    auto U_h   = Kokkos::create_mirror_view(U);
    auto D_h   = Kokkos::create_mirror_view(D);
    auto L_h   = Kokkos::create_mirror_view(L);
    auto RHS_h = Kokkos::create_mirror_view(RHS);

    for (size_t i = 0; i < matrix_size; ++i) {
      U_h(i)   = u_Thomas_host[i];
      D_h(i)   = d_Thomas_host[i];
      L_h(i)   = l_Thomas_host[i];
      RHS_h(i) = rhs_Thomas_host[i];
    }

    Kokkos::deep_copy(U,   U_h);
    Kokkos::deep_copy(D,   D_h);
    Kokkos::deep_copy(L,   L_h);
    Kokkos::deep_copy(RHS, RHS_h);

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      Kokkos::parallel_for("thomas", Kokkos::RangePolicy<ExecSpace>(0, N),
        KOKKOS_LAMBDA(const int tid) {
          int first = tid;
          int last  = N*(M-1) + tid;

          U(first)   /= D(first);
          RHS(first) /= D(first);

          for (int i = first + N; i < last; i += N) {
            U(i)   /= D(i) - L(i) * U(i-N);
            RHS(i)  = (RHS(i) - L(i) * RHS(i-N)) / (D(i) - L(i) * U(i-N));
          }

          RHS(last) = (RHS(last) - L(last) * RHS(last-N)) /
                      (D(last)   - L(last) * U(last-N));

          for (int i = last-N; i >= first; i -= N)
            RHS(i) -= U(i) * RHS(i+N);
        });
      Kokkos::fence();
    }

    auto end = std::chrono::steady_clock::now();
    printf("Average kernel execution time: %f (ms)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-6 / repeat);

    Kokkos::deep_copy(RHS_h, RHS);
    for (size_t i = 0; i < matrix_size; ++i)
      rhs_Thomas_host[i] = RHS_h(i);
  }
  Kokkos::finalize();

  calcError(rhs_seq_interleave, rhs_Thomas_host, (int)matrix_size);

  free(u_seq); free(u_Thomas_host); free(u_input);
  free(d_seq); free(d_Thomas_host); free(d_input);
  free(l_seq); free(l_Thomas_host); free(l_input);
  free(rhs_seq); free(rhs_Thomas_host); free(rhs_input);
  free(rhs_seq_output); free(rhs_seq_interleave);

  return 0;
}
