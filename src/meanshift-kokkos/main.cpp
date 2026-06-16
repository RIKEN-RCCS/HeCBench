/*
 * meanshift – Mean Shift Clustering
 * Kokkos port from the OMP offload version.
 *
 * Original work licensed under MIT (see /workspace/src/meanshift-omp/LICENSE)
 *
 * Usage: ./main <path_to_data.csv> <path_to_centroids.csv>
 *
 * Data files (CSV): dataset.tar.gz in the CUDA version contains sample data.
 * N=10000 points in D=3 dimensions; M=3 real centroids.
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
// Constants (from constants.h)
// ============================================================
namespace ms {
  constexpr float  RADIUS       = 60.f;
  constexpr float  SIGMA        = 4.f;
  constexpr float  DBL_SIGMA_SQ = 2.f * SIGMA * SIGMA;   // = 32
  constexpr float  MIN_DISTANCE = 60.f;
  constexpr size_t NUM_ITER     = 1000;
  constexpr float  DIST_TO_REAL = 10.f;
  constexpr int    N            = 10000;
  constexpr int    D            = 3;
  constexpr int    M            = 3;
  constexpr int    THREADS      = 64;
  constexpr int    BLOCKS       = (N + THREADS - 1) / THREADS;  // 157
  constexpr int    TILE_WIDTH   = THREADS;                       // 64
}

// ============================================================
// CSV utilities (from utils.h)
// ============================================================
template <const size_t Npt, const size_t Dim>
std::array<float, Npt * Dim> load_csv(const std::string& path, char delim)
{
  assert(std::filesystem::exists(path));
  std::ifstream file(path);
  std::string line;
  std::array<float, Npt * Dim> data_matrix;
  for (size_t i = 0; i < Npt; ++i) {
    std::getline(file, line);
    std::stringstream ss(line);
    std::string cell;
    for (size_t j = 0; j < Dim; ++j) {
      std::getline(ss, cell, delim);
      data_matrix[i * Dim + j] = std::stof(cell);
    }
  }
  return data_matrix;
}

template <const size_t Npt, const size_t Dim>
std::vector<std::array<float, Dim>> reduce_to_centroids(
    std::array<float, Npt * Dim>& data, float min_distance)
{
  std::vector<std::array<float, Dim>> centroids;
  std::array<float, Dim> first;
  for (size_t j = 0; j < Dim; j++) first[j] = data[j];
  centroids.push_back(first);
  for (size_t i = 0; i < Npt; i++) {
    bool close = false;
    for (const auto& c : centroids) {
      float dist = 0.f;
      for (size_t j = 0; j < Dim; j++) {
        float diff = data[i*Dim+j] - c[j];
        dist += diff * diff;
      }
      if (dist <= min_distance) { close = true; break; }
    }
    if (!close) {
      std::array<float, Dim> nc;
      for (size_t j = 0; j < Dim; j++) nc[j] = data[i*Dim+j];
      centroids.push_back(nc);
    }
  }
  return centroids;
}

template <const size_t Mreal, const size_t Dim>
bool are_close_to_real(const std::vector<std::array<float, Dim>>& centroids,
                       const std::array<float, Mreal * Dim>& real,
                       float eps)
{
  std::array<bool, Mreal> ok{};
  for (size_t i = 0; i < centroids.size() && i < Mreal; i++) {
    for (size_t j = 0; j < Mreal; j++) {
      float dist = 0.f;
      for (size_t k = 0; k < Dim; k++) {
        float diff = centroids[i][k] - real[j*Dim+k];
        dist += diff * diff;
      }
      if (dist <= eps) ok[i] = true;
    }
  }
  return std::all_of(ok.begin(), ok.end(), [](bool b){ return b; });
}

// ============================================================
// Kokkos mean_shift kernel (base)
// ============================================================
void mean_shift_base(Kokkos::View<float*> d_data,
                     Kokkos::View<float*> d_data_next)
{
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  constexpr int N = ms::N, D = ms::D;
  constexpr float RADIUS       = ms::RADIUS;
  constexpr float DBL_SIGMA_SQ = ms::DBL_SIGMA_SQ;

  Kokkos::parallel_for(
    "mean_shift_base",
    Kokkos::RangePolicy<ExecSpace>(0, N),
    KOKKOS_LAMBDA(int tid) {
      int row = tid * D;
      float new_position[D];
      for (int j = 0; j < D; j++) new_position[j] = 0.f;
      float tot_weight = 0.f;

      for (int i = 0; i < N; i++) {
        int row_n = i * D;
        float sq_dist = 0.f;
        for (int j = 0; j < D; j++) {
          float diff = d_data(row + j) - d_data(row_n + j);
          sq_dist += diff * diff;
        }
        if (sq_dist <= RADIUS) {
          float weight = Kokkos::exp(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; j++)
            new_position[j] += weight * d_data(row_n + j);
          tot_weight += weight;
        }
      }
      for (int j = 0; j < D; j++)
        d_data_next(row + j) = new_position[j] / tot_weight;
    });
}

// ============================================================
// Kokkos mean_shift_tiling kernel (optimized, tiled)
// ============================================================
void mean_shift_tiling(Kokkos::View<float*> d_data,
                       Kokkos::View<float*> d_data_next)
{
  using ExecSpace    = Kokkos::DefaultExecutionSpace;
  using ScratchSpace = ExecSpace::scratch_memory_space;
  using ScratchViewF = Kokkos::View<float*, ScratchSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  constexpr int N = ms::N, D = ms::D;
  constexpr int THREADS     = ms::THREADS;
  constexpr int BLOCKS      = ms::BLOCKS;
  constexpr int TILE_WIDTH  = ms::TILE_WIDTH;
  constexpr float RADIUS       = ms::RADIUS;
  constexpr float DBL_SIGMA_SQ = ms::DBL_SIGMA_SQ;

  // Scratch: local_data[TILE_WIDTH * D] + valid_data[TILE_WIDTH] as one flat view
  const int scratch_bytes = (TILE_WIDTH * D + TILE_WIDTH) * sizeof(float);

  Kokkos::TeamPolicy<> policy(BLOCKS, THREADS);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  Kokkos::parallel_for(
    "mean_shift_tiling",
    policy,
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      // Single flat scratch view: [0..TILE_WIDTH*D-1] = local_data,
      //                           [TILE_WIDTH*D .. TILE_WIDTH*D+TILE_WIDTH-1] = valid_data
      ScratchViewF scratch(team.team_scratch(0), TILE_WIDTH * D + TILE_WIDTH);
      const int valid_offset = TILE_WIDTH * D;

      const int bid = team.league_rank();
      const int lid = team.team_rank();
      const int tid = bid * THREADS + lid;
      const int row = tid * D;

      float new_position[D];
      for (int j = 0; j < D; j++) new_position[j] = 0.f;
      float tot_weight = 0.f;

      for (int t = 0; t < BLOCKS; t++) {
        // Load tile into scratch
        int tid_in_tile = t * TILE_WIDTH + lid;
        if (tid_in_tile < N) {
          for (int j = 0; j < D; j++)
            scratch(lid * D + j) = d_data(tid_in_tile * D + j);
          scratch(valid_offset + lid) = 1.f;
        } else {
          for (int j = 0; j < D; j++) scratch(lid * D + j) = 0.f;
          scratch(valid_offset + lid) = 0.f;
        }
        team.team_barrier();

        if (tid < N) {
          for (int i = 0; i < TILE_WIDTH; i++) {
            float valid_radius = RADIUS * scratch(valid_offset + i);
            float sq_dist = 0.f;
            for (int j = 0; j < D; j++) {
              float diff = d_data(row + j) - scratch(i * D + j);
              sq_dist += diff * diff;
            }
            if (sq_dist <= valid_radius) {
              float weight = Kokkos::exp(-sq_dist / DBL_SIGMA_SQ);
              for (int j = 0; j < D; j++)
                new_position[j] += weight * scratch(i * D + j);
              tot_weight += weight * scratch(valid_offset + i);
            }
          }
        }
        team.team_barrier();
      }

      if (tid < N) {
        for (int j = 0; j < D; j++)
          d_data_next(row + j) = new_position[j] / tot_weight;
      }
    });
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[])
{
  if (argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " <path_to_data.csv> <path_to_centroids.csv>\n";
    return 1;
  }
  const std::string path_to_data      = argv[1];
  const std::string path_to_centroids = argv[2];

  constexpr int N = ms::N, D = ms::D, M = ms::M;
  constexpr float DIST_TO_REAL = ms::DIST_TO_REAL;

  std::cout << "\nDATASET:    " << path_to_data  << "\n"
            << "NUM POINTS: " << N              << "\n"
            << "DIM:        " << D              << "\n"
            << "BLOCKS:     " << ms::BLOCKS     << "\n"
            << "THREADS:    " << ms::THREADS    << "\n"
            << "TILE WIDTH: " << ms::TILE_WIDTH << "\n";

  const auto real   = load_csv<M, D>(path_to_centroids, ',');
  auto data_host    = load_csv<N, D>(path_to_data, ',');
  auto result       = data_host;   // working copy

  Kokkos::initialize(argc, argv);
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = ExecSpace::memory_space;

    Kokkos::View<float*, MemSpace> d_data     ("d_data",      N * D);
    Kokkos::View<float*, MemSpace> d_data_next("d_data_next", N * D);

    auto h_data = Kokkos::create_mirror_view(d_data);

    // -------- base kernel --------
    for (int i = 0; i < N * D; i++) h_data(i) = result[i];
    Kokkos::deep_copy(d_data, h_data);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < ms::NUM_ITER; it++) {
      mean_shift_base(d_data, d_data_next);
      Kokkos::fence();
      // swap views
      auto tmp = d_data; d_data = d_data_next; d_data_next = tmp;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "\nAverage execution time of mean-shift (base) "
              << (ns * 1e-6 / ms::NUM_ITER) << " ms\n\n";

    Kokkos::deep_copy(h_data, d_data);
    for (int i = 0; i < N * D; i++) result[i] = h_data(i);

    auto centroids = reduce_to_centroids<N, D>(result, ms::MIN_DISTANCE);
    bool ok = are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    if ((int)centroids.size() == M && ok) std::cout << "PASS\n";
    else                                  std::cout << "FAIL\n";

    // -------- tiling kernel --------
    result = data_host;
    for (int i = 0; i < N * D; i++) h_data(i) = result[i];
    Kokkos::deep_copy(d_data, h_data);

    t0 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < ms::NUM_ITER; it++) {
      mean_shift_tiling(d_data, d_data_next);
      Kokkos::fence();
      auto tmp = d_data; d_data = d_data_next; d_data_next = tmp;
    }
    t1 = std::chrono::steady_clock::now();
    ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "\nAverage execution time of mean-shift (opt) "
              << (ns * 1e-6 / ms::NUM_ITER) << " ms\n\n";

    Kokkos::deep_copy(h_data, d_data);
    for (int i = 0; i < N * D; i++) result[i] = h_data(i);

    centroids = reduce_to_centroids<N, D>(result, ms::MIN_DISTANCE);
    ok = are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
    if ((int)centroids.size() == M && ok) std::cout << "PASS\n";
    else                                  std::cout << "FAIL\n";
  }
  Kokkos::finalize();
  return 0;
}
