#include "sobol_gpu.h"

#include <chrono>

#include "sobol.h"

#define k_2powneg32 2.3283064E-10F

KOKKOS_INLINE_FUNCTION
int _ffs(const int x) {
  for (int i = 0; i < 32; i++)
    if ((x >> i) & 1) return (i + 1);
  return 0;
}

double sobolGPU(int repeat, int n_vectors, int n_dimensions,
                Kokkos::View<unsigned int*> d_dir,
                Kokkos::View<float*> d_out)
{
  const int threadsperblock = 64;

  size_t dimGrid_y = n_dimensions;
  size_t dimGrid_x;

  if (n_dimensions < (4 * 24))
    dimGrid_x = 4 * 24;
  else
    dimGrid_x = 1;

  if (dimGrid_x > (unsigned int)(n_vectors / threadsperblock))
    dimGrid_x = (n_vectors + threadsperblock - 1) / threadsperblock;

  unsigned int targetDimGridX = (unsigned int)dimGrid_x;
  for (dimGrid_x = 1; dimGrid_x < targetDimGridX; dimGrid_x *= 2);

  size_t numTeam = dimGrid_x * dimGrid_y;

  using ScratchUInt = Kokkos::View<unsigned int*,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryUnmanaged>;
  size_t scratch_bytes = ScratchUInt::shmem_size(n_directions);

  using TeamPolicy = Kokkos::TeamPolicy<>;
  TeamPolicy policy((int)numTeam, threadsperblock);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  const int nv = n_vectors;
  const size_t dgx = dimGrid_x;

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
    Kokkos::parallel_for("sobolGPU", policy,
      KOKKOS_LAMBDA(const TeamPolicy::member_type& team) {
        const unsigned int teamX = (unsigned int)team.league_rank() % (unsigned int)dgx;
        const unsigned int teamY = (unsigned int)team.league_rank() / (unsigned int)dgx;

        ScratchUInt v(team.team_scratch(0), n_directions);

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, n_directions),
          [&](int i) {
            v(i) = d_dir(n_directions * teamY + i);
          });
        team.team_barrier();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, threadsperblock),
          [&](int tidX) {
            const unsigned int threadSizeX = (unsigned int)threadsperblock;

            int i0 = (int)(teamX * threadSizeX) + tidX;
            int stride = (int)(dgx * threadSizeX);

            unsigned int g = (unsigned int)i0 ^ ((unsigned int)i0 >> 1);
            unsigned int X = 0;
            unsigned int mask = 0;
            for (unsigned int k = 0; k < (unsigned int)(_ffs(stride) - 1); k++) {
              mask = -(g & 1u);
              X ^= mask & v(k);
              g >>= 1;
            }

            if (i0 < nv)
              d_out(nv * (int)teamY + i0) = (float)X * k_2powneg32;

            unsigned int v_log2stridem1 = v((unsigned int)(_ffs(stride) - 2));
            unsigned int v_stridemask = (unsigned int)stride - 1u;

            for (unsigned int i = (unsigned int)i0 + (unsigned int)stride;
                 i < (unsigned int)nv;
                 i += (unsigned int)stride)
            {
              X ^= v_log2stridem1 ^
                   v((unsigned int)(_ffs(~((int)(i - (unsigned int)stride) |
                                           (int)v_stridemask)) - 1));
              d_out(nv * (int)teamY + (int)i) = (float)X * k_2powneg32;
            }
          });
      });
  }
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
