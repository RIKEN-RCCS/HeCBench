#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "reference.h"

#define A_VAL 0
#define B_VAL 15
#define ROW_SIZE 17
#define EPS 1e-7

// Host-only version for reference()
inline double f(double x) { return exp(x)*sin(x); }

KOKKOS_INLINE_FUNCTION double f_dev(double x) { return exp(x)*sin(x); }

KOKKOS_INLINE_FUNCTION unsigned int getFirstSetBitPos(int n) {
  return (unsigned int)(logf((float)(n & -n)) / logf(2.f)) + 1;
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of work-groups> <work-group size> <repeat>\n", argv[0]);
    return 1;
  }
  const int nwg = atoi(argv[1]);
  const int wgs = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const double a = A_VAL, b = B_VAL;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> d_result("result", nwg);
    auto h_result = Kokkos::create_mirror_view(d_result);

    using ScratchPad = Kokkos::View<double*,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryUnmanaged>;
    int scratch_size = ScratchPad::shmem_size(ROW_SIZE * wgs);
    auto policy = Kokkos::TeamPolicy<>(nwg, wgs)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    double d_sum = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < repeat; rep++) {
      Kokkos::parallel_for("romberg", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
          const int blockIdx_x  = team.league_rank();
          const int gridDim_x   = team.league_size();
          const int blockDim_x  = team.team_size();
          const int threadIdx_x = team.team_rank();

          ScratchPad smem(team.team_scratch(0), ROW_SIZE * blockDim_x);

          double diff   = (b - a) / gridDim_x;
          double local_a = a + blockIdx_x * diff;
          double local_b = a + (blockIdx_x+1) * diff;
          int max_eval  = (1 << (ROW_SIZE-1));
          double step   = (local_b - local_a) / max_eval;

          double local_col[ROW_SIZE];
          for (int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;

          int k;
          if (threadIdx_x == 0) {
            k = blockDim_x;
            local_col[0] = f_dev(local_a) + f_dev(local_b);
          } else {
            k = threadIdx_x;
          }

          for (; k < max_eval; k += blockDim_x)
            local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0 * f_dev(local_a + step * k);

          for (int i = 0; i < ROW_SIZE; i++)
            smem[ROW_SIZE * threadIdx_x + i] = local_col[i];

          team.team_barrier();

          if (threadIdx_x < ROW_SIZE) {
            double sum = 0.0;
            for (int i = threadIdx_x; i < blockDim_x * ROW_SIZE; i += ROW_SIZE)
              sum += smem[i];
            smem[threadIdx_x] = sum;
          }

          team.team_barrier();

          if (threadIdx_x == 0) {
            double table[ROW_SIZE];
            table[0] = smem[0];
            for (int k2 = 1; k2 < ROW_SIZE; k2++)
              table[k2] = table[k2-1] + smem[k2];
            for (int k2 = 0; k2 < ROW_SIZE; k2++)
              table[k2] *= (local_b - local_a) / (1 << (k2+1));
            for (int col = 0; col < ROW_SIZE-1; col++)
              for (int row = ROW_SIZE-1; row > col; row--)
                table[row] = table[row] + (table[row] - table[row-1]) / ((1<<(2*col+1)) - 1);
            d_result[blockIdx_x] = table[ROW_SIZE-1];
          }
        });
      Kokkos::fence();

      Kokkos::deep_copy(h_result, d_result);
      d_sum = 0.0;
      for (int k = 0; k < nwg; k++) d_sum += h_result[k];
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

    double ref_sum = reference(f, A_VAL, B_VAL, ROW_SIZE, EPS);
    printf("%s\n", (fabs(d_sum - ref_sum) > EPS) ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  return 0;
}
