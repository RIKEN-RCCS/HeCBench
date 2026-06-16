// ISO2DFD – 2D Acoustic Isotropic Wave Propagation
// Kokkos port from the OpenMP target version.

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <Kokkos_Core.hpp>

#define DT          0.002f
#define DXY         20.0f
#define HALF_LENGTH 1

static void usage(const char* prog) {
  std::cout << "Usage: " << prog << " n1 n2 nIterations\n";
}

static void initialize(float* prev, float* next, float* vel,
                       size_t nRows, size_t nCols)
{
  float wavelet[12] = {
    0.016387336f, -0.041464937f, -0.067372555f, 0.386110067f,
    0.812723635f,  0.416998396f,  0.076488599f,-0.059434419f,
    0.023680172f,  0.005611435f,  0.001823209f,-0.000720549f
  };

  for (size_t i = 0; i < nRows * nCols; i++) {
    prev[i] = 0.f;
    next[i] = 0.f;
    vel[i]  = 2250000.0f;  // v=1500 m/s, v^2
  }
  for (int s = 11; s >= 0; s--) {
    for (size_t i = nRows/2 - s; i < nRows/2 + s; i++) {
      for (size_t k = nCols/2 - s; k < nCols/2 + s; k++) {
        prev[i * nCols + k] = wavelet[s];
      }
    }
  }
}

static void iso_2dfd_cpu(float* next, float* prev, const float* vel,
                         float dtDIVdxy, int nRows, int nCols, int nIter)
{
  for (int k = 0; k < nIter; k++) {
    for (int i = HALF_LENGTH; i < nRows - HALF_LENGTH; i++) {
      for (int j = HALF_LENGTH; j < nCols - HALF_LENGTH; j++) {
        size_t gid = (size_t)i * nCols + j;
        float v = 0.f;
        v += prev[gid+1]     - 2.f*prev[gid] + prev[gid-1];
        v += prev[gid+nCols] - 2.f*prev[gid] + prev[gid-nCols];
        v *= dtDIVdxy * vel[gid];
        next[gid] = 2.f*prev[gid] - next[gid] + v;
      }
    }
    std::swap(next, prev);
  }
}

static bool within_epsilon(const float* output, const float* reference,
                            size_t dimx, size_t dimy, float delta = 0.1f)
{
  double norm2 = 0.0;
  bool   error = false;
  for (size_t iy = 0; iy < dimy; iy++) {
    for (size_t ix = 0; ix < dimx; ix++) {
      if (ix >= HALF_LENGTH && ix < dimx - HALF_LENGTH &&
          iy >= HALF_LENGTH && iy < dimy - HALF_LENGTH) {
        float diff = fabsf(reference[iy*dimx+ix] - output[iy*dimx+ix]);
        norm2 += (double)diff * diff;
        if (diff > delta) error = true;
      }
    }
  }
  if (error) printf("Euclidean error norm: %.9e\n", sqrt(norm2));
  return error;
}

int main(int argc, char* argv[])
{
  if (argc < 4) { usage(argv[0]); return 1; }

  int nRows = atoi(argv[1]);
  int nCols = atoi(argv[2]);
  int nIter = atoi(argv[3]);

  size_t nsize = (size_t)nRows * nCols;
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  float* prev_base = new float[nsize];
  float* next_base = new float[nsize];
  float* vel_base  = new float[nsize];
  float* next_cpu  = new float[nsize];

  initialize(prev_base, next_base, vel_base, nRows, nCols);

  std::cout << "Grid: " << nRows << " x " << nCols
            << "  Iterations: " << nIter << "\n";

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_prev("prev", nsize);
    Kokkos::View<float*> d_next("next", nsize);
    Kokkos::View<float*> d_vel ("vel",  nsize);

    {
      auto h_prev = Kokkos::create_mirror_view(d_prev);
      auto h_next = Kokkos::create_mirror_view(d_next);
      auto h_vel  = Kokkos::create_mirror_view(d_vel);
      for (size_t i = 0; i < nsize; i++) {
        h_prev(i) = prev_base[i];
        h_next(i) = next_base[i];
        h_vel (i) = vel_base [i];
      }
      Kokkos::deep_copy(d_prev, h_prev);
      Kokkos::deep_copy(d_next, h_next);
      Kokkos::deep_copy(d_vel,  h_vel);
    }

    using Range2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    auto t0 = std::chrono::steady_clock::now();

    for (int k = 0; k < nIter; k++) {
      // Alternate which buffer is "next" each iteration
      auto& cur_next = (k % 2 == 0) ? d_next : d_prev;
      auto& cur_prev = (k % 2 == 0) ? d_prev : d_next;

      Kokkos::View<float*> lcur_next = cur_next;
      Kokkos::View<float*> lcur_prev = cur_prev;
      int lnCols = nCols, lnRows = nRows;
      float ldtDIVdxy = dtDIVdxy;

      Kokkos::parallel_for("iso2dfd",
        Range2D({0, 0}, {lnRows, lnCols}),
        KOKKOS_LAMBDA(int gidRow, int gidCol) {
          int gid = gidRow * lnCols + gidCol;
          if (gidCol >= HALF_LENGTH && gidCol < lnCols - HALF_LENGTH &&
              gidRow >= HALF_LENGTH && gidRow < lnRows - HALF_LENGTH) {
            float v = 0.f;
            v += lcur_prev(gid+1)       - 2.f*lcur_prev(gid) + lcur_prev(gid-1);
            v += lcur_prev(gid+lnCols)  - 2.f*lcur_prev(gid) + lcur_prev(gid-lnCols);
            v *= ldtDIVdxy * d_vel(gid);
            lcur_next(gid) = 2.f*lcur_prev(gid) - lcur_next(gid) + v;
          }
        });
    }
    Kokkos::fence();

    auto t1 = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::cout << "Total kernel time:   " << ns * 1e-6f << " ms\n";
    std::cout << "Average kernel time: " << (ns * 1e-3f) / nIter << " us\n";

    // Copy result back (final "next" buffer)
    auto& final_next = (nIter % 2 == 0) ? d_next : d_prev;
    auto h_out = Kokkos::create_mirror_view(final_next);
    Kokkos::deep_copy(h_out, final_next);
    for (size_t i = 0; i < nsize; i++) next_base[i] = h_out(i);
  }
  Kokkos::finalize();

  // Write device result
  {
    std::ofstream f("wavefield_snapshot.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(next_base), nsize * sizeof(float));
  }

  // CPU reference
  std::cout << "Computing CPU reference...\n";
  initialize(prev_base, next_cpu, vel_base, nRows, nCols);
  auto cs = std::chrono::steady_clock::now();
  iso_2dfd_cpu(next_cpu, prev_base, vel_base, dtDIVdxy, nRows, nCols, nIter);
  auto ce = std::chrono::steady_clock::now();
  std::cout << "CPU time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ce-cs).count()
            << " ms\n";

  {
    std::ofstream f("wavefield_snapshot_cpu.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(next_cpu), nsize * sizeof(float));
  }

  bool err = within_epsilon(next_base, next_cpu, nCols, nRows);
  std::cout << (err ? "FAIL: wavefields differ\n"
                    : "PASS: wavefields match\n");

  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;
  delete[] next_cpu;
  return err ? 1 : 0;
}
