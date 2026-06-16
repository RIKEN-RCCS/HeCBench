#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <Kokkos_Core.hpp>

// Thomas tridiagonal matrix structure
struct ThomasMatrix {
  double *a, *b, *d, *rhs;
  int M;
};

static double fRand(double fMin, double fMax) {
  double f = (double)rand() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

ThomasMatrix loadThomasMatrixSyn(int size) {
  ThomasMatrix tm;
  tm.M = size;
  tm.a   = (double*)malloc(size * sizeof(double));
  tm.b   = (double*)malloc(size * sizeof(double));
  tm.d   = (double*)malloc(size * sizeof(double));
  tm.rhs = (double*)malloc(size * sizeof(double));
  for (int i = 0; i < size; ++i) {
    tm.a[i]   = fRand(-2.0, 2.0);
    tm.b[i]   = fRand(-2.0, 2.0);
    tm.d[i]   = fRand(5.0, 10.0);
    tm.rhs[i] = fRand(-2.0, 2.0);
  }
  return tm;
}

// Sequential CPU solver for correctness check
void solve_seq(const double* l, const double* d, double* u, double* rhs,
               const int n, const int N) {
  for (int j = 0; j < N; ++j) {
    int first = j * n;
    int last  = first + n - 1;

    u[first]   /= d[first];
    rhs[first] /= d[first];

    for (int i = first + 1; i < last; i++) {
      u[i]   /= d[i] - l[i] * u[i-1];
      rhs[i]  = (rhs[i] - l[i] * rhs[i-1]) / (d[i] - l[i] * u[i-1]);
    }
    rhs[last] = (rhs[last] - l[last] * rhs[last-1]) /
                (d[last] - l[last] * u[last-1]);

    for (int i = last - 1; i >= first; i--)
      rhs[i] -= u[i] * rhs[i+1];
  }
}

template <typename T>
void calcError(T* src, T* dst, int size) {
  double error = 0;
  for (int i = 0; i < size; ++i) {
    double e = std::fabs(std::fabs(src[i]) - std::fabs(dst[i]));
    if (error < e) error = e;
  }
  printf("Maximum error: %e\n", error);
}

int main(int argc, char const* argv[]) {
  if (argc != 5) {
    std::cout << "Usage: " << argv[0]
              << " [system size] [#systems] [thread block size] [repeat]\n";
    return -1;
  }

  const int M         = std::stoi(argv[1]);
  const int N         = std::stoi(argv[2]);
  const int BlockSize = std::stoi(argv[3]);
  const int repeat    = std::stoi(argv[4]);

  const size_t matrix_size       = (size_t)M * N;
  const size_t matrix_size_bytes = matrix_size * sizeof(double);

  ThomasMatrix params = loadThomasMatrixSyn(M);

  // Host arrays for sequential reference
  double* u_seq   = (double*)malloc(matrix_size_bytes);
  double* d_seq   = (double*)malloc(matrix_size_bytes);
  double* l_seq   = (double*)malloc(matrix_size_bytes);
  double* rhs_seq = (double*)malloc(matrix_size_bytes);

  double* u_input   = (double*)malloc(matrix_size_bytes);
  double* d_input   = (double*)malloc(matrix_size_bytes);
  double* l_input   = (double*)malloc(matrix_size_bytes);
  double* rhs_input = (double*)malloc(matrix_size_bytes);

  double* rhs_seq_output    = (double*)malloc(matrix_size_bytes);
  double* rhs_seq_interleave = (double*)malloc(matrix_size_bytes);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      u_seq[(i*M)+j]   = u_input[(i*M)+j]   = params.a[j];
      d_seq[(i*M)+j]   = d_input[(i*M)+j]   = params.d[j];
      l_seq[(i*M)+j]   = l_input[(i*M)+j]   = params.b[j];
      rhs_seq[(i*M)+j] = rhs_input[(i*M)+j] = params.rhs[j];
    }
  }

  // CPU sequential reference
  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++)
    solve_seq(l_seq, d_seq, u_seq, rhs_seq, M, N);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average serial execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  for (size_t i = 0; i < matrix_size; ++i)
    rhs_seq_output[i] = rhs_seq[i];

  // Transpose for coalesced GPU access (each row = elements of one position
  // across all systems)
  double* U_host   = (double*)malloc(matrix_size_bytes);
  double* D_host   = (double*)malloc(matrix_size_bytes);
  double* L_host   = (double*)malloc(matrix_size_bytes);
  double* RHS_host = (double*)malloc(matrix_size_bytes);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      U_host[i*N+j]   = u_input[j*M+i];
      D_host[i*N+j]   = d_input[j*M+i];
      L_host[i*N+j]   = l_input[j*M+i];
      RHS_host[i*N+j] = rhs_input[j*M+i];
      rhs_seq_interleave[i*N+j] = rhs_seq_output[j*M+i];
    }
  }

  Kokkos::initialize(argc, (char**)argv);
  {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using mem_space  = typename exec_space::memory_space;

    Kokkos::View<double*, mem_space> dU("U", matrix_size);
    Kokkos::View<double*, mem_space> dD("D", matrix_size);
    Kokkos::View<double*, mem_space> dL("L", matrix_size);
    Kokkos::View<double*, mem_space> dRHS("RHS", matrix_size);

    auto hU   = Kokkos::create_mirror_view(dU);
    auto hD   = Kokkos::create_mirror_view(dD);
    auto hL   = Kokkos::create_mirror_view(dL);
    auto hRHS = Kokkos::create_mirror_view(dRHS);

    for (size_t i = 0; i < matrix_size; i++) {
      hU(i)   = U_host[i];
      hD(i)   = D_host[i];
      hL(i)   = L_host[i];
      hRHS(i) = RHS_host[i];
    }

    Kokkos::deep_copy(dU,   hU);
    Kokkos::deep_copy(dD,   hD);
    Kokkos::deep_copy(dL,   hL);
    Kokkos::deep_copy(dRHS, hRHS);

    Kokkos::fence();
    start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      // Each thread handles one of the N tridiagonal systems
      // Systems are stored column-major: element [pos][sys] = pos*N + sys
      Kokkos::parallel_for(
        "thomas",
        Kokkos::RangePolicy<exec_space>(0, N),
        KOKKOS_LAMBDA(const int tid) {
          int first = tid;
          int last  = N * (M - 1) + tid;

          dU(first)   /= dD(first);
          dRHS(first) /= dD(first);

          for (int i = first + N; i < last; i += N) {
            dU(i)   /= dD(i) - dL(i) * dU(i - N);
            dRHS(i)  = (dRHS(i) - dL(i) * dRHS(i - N)) /
                       (dD(i)   - dL(i) * dU(i - N));
          }

          dRHS(last) = (dRHS(last) - dL(last) * dRHS(last - N)) /
                       (dD(last)   - dL(last) * dU(last - N));

          for (int i = last - N; i >= first; i -= N)
            dRHS(i) -= dU(i) * dRHS(i + N);
        });
    }

    Kokkos::fence();
    end  = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

    Kokkos::deep_copy(hRHS, dRHS);
    for (size_t i = 0; i < matrix_size; i++)
      RHS_host[i] = hRHS(i);
  }
  Kokkos::finalize();

  calcError(rhs_seq_interleave, RHS_host, (int)matrix_size);

  free(u_seq); free(d_seq); free(l_seq); free(rhs_seq);
  free(u_input); free(d_input); free(l_input); free(rhs_input);
  free(U_host); free(D_host); free(L_host); free(RHS_host);
  free(rhs_seq_output); free(rhs_seq_interleave);
  free(params.a); free(params.b); free(params.d); free(params.rhs);
  return 0;
}
