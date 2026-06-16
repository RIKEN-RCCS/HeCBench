// SVD 3x3 benchmark ported to Kokkos
// The portable svd() implementation is shared from svd3x3-omp/kernels.cpp.
// That file has no OpenMP directives; only arithmetic, so it compiles for
// any Kokkos backend when included here.
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>

// Provide compatibility shims used inside kernels.cpp
#ifndef __fadd_rn
#define __fadd_rn(a,b) ((a)+(b))
#endif
#ifndef __fsub_rn
#define __fsub_rn(a,b) ((a)-(b))
#endif
#ifndef __frsqrt_rn
#define __frsqrt_rn(a) (1.f / sqrtf(a))
#endif

// Include the SVD implementation (no OMP directives inside)
#include "kernels.cpp"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to file> <repeat>\n";
    return 1;
  }
  const char* filename = argv[1];
  const int   repeat   = atoi(argv[2]);

  std::ifstream myfile(filename);
  if (!myfile.is_open()) {
    std::cout << "ERROR: failed to open " << filename << "\n";
    return -1;
  }
  int testsSize;
  myfile >> testsSize;
  std::cout << "dataset size: " << testsSize << "\n";
  if (testsSize <= 0) return -1;

  float* input    = (float*)malloc(sizeof(float) * 9  * testsSize);
  float* result   = (float*)malloc(sizeof(float) * 21 * testsSize);
  float* result_h = (float*)malloc(sizeof(float) * 21 * testsSize);

  int count = 0;
  for (int i = 0; i < testsSize; i++)
    for (int j = 0; j < 9; j++) myfile >> input[count++];
  myfile.close();

  Kokkos::initialize(argc, argv);
  {
    using exec_space = Kokkos::DefaultExecutionSpace;
    using mem_space  = typename exec_space::memory_space;
    const int ts = testsSize;

    Kokkos::View<float*, mem_space> d_input ("input",  9*ts);
    Kokkos::View<float*, mem_space> d_result("result", 21*ts);

    {
      auto h_in = Kokkos::create_mirror_view(d_input);
      for (int i = 0; i < 9*ts; i++) h_in(i) = input[i];
      Kokkos::deep_copy(d_input, h_in);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < repeat; rep++) {
      Kokkos::parallel_for(
        "svd3x3",
        Kokkos::RangePolicy<exec_space>(0, ts),
        KOKKOS_LAMBDA(int tid) {
          svd(d_input(tid+0*ts), d_input(tid+1*ts), d_input(tid+2*ts),
              d_input(tid+3*ts), d_input(tid+4*ts), d_input(tid+5*ts),
              d_input(tid+6*ts), d_input(tid+7*ts), d_input(tid+8*ts),
              d_result(tid+ 0*ts), d_result(tid+ 1*ts), d_result(tid+ 2*ts),
              d_result(tid+ 3*ts), d_result(tid+ 4*ts), d_result(tid+ 5*ts),
              d_result(tid+ 6*ts), d_result(tid+ 7*ts), d_result(tid+ 8*ts),
              d_result(tid+ 9*ts), d_result(tid+10*ts), d_result(tid+11*ts),
              d_result(tid+12*ts), d_result(tid+13*ts), d_result(tid+14*ts),
              d_result(tid+15*ts), d_result(tid+16*ts), d_result(tid+17*ts),
              d_result(tid+18*ts), d_result(tid+19*ts), d_result(tid+20*ts));
        });
    }

    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    auto h_result = Kokkos::create_mirror_view(d_result);
    Kokkos::deep_copy(h_result, d_result);
    for (int i = 0; i < 21*ts; i++) result[i] = h_result(i);
  }
  Kokkos::finalize();

  // Reference computation on CPU
  for (int tid = 0; tid < testsSize; tid++)
    svd(input[tid+0*testsSize], input[tid+1*testsSize], input[tid+2*testsSize],
        input[tid+3*testsSize], input[tid+4*testsSize], input[tid+5*testsSize],
        input[tid+6*testsSize], input[tid+7*testsSize], input[tid+8*testsSize],
        result_h[tid+ 0*testsSize], result_h[tid+ 1*testsSize], result_h[tid+ 2*testsSize],
        result_h[tid+ 3*testsSize], result_h[tid+ 4*testsSize], result_h[tid+ 5*testsSize],
        result_h[tid+ 6*testsSize], result_h[tid+ 7*testsSize], result_h[tid+ 8*testsSize],
        result_h[tid+ 9*testsSize], result_h[tid+10*testsSize], result_h[tid+11*testsSize],
        result_h[tid+12*testsSize], result_h[tid+13*testsSize], result_h[tid+14*testsSize],
        result_h[tid+15*testsSize], result_h[tid+16*testsSize], result_h[tid+17*testsSize],
        result_h[tid+18*testsSize], result_h[tid+19*testsSize], result_h[tid+20*testsSize]);

  bool ok = true;
  for (int i = 0; i < testsSize; i++) {
    if (fabsf(result[i] - result_h[i]) > 1e-3f) {
      std::cout << result[i] << " " << result_h[i] << "\n";
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << "\n";

  free(input); free(result); free(result_h);
  return 0;
}
