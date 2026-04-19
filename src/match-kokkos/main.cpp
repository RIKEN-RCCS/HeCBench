//**********************************************************//
//   Matching test code - Kokkos port                      //
//   Ports GPU1, GPU2, GPU3, GPU4, GPU5                    //
//**********************************************************//

#include <Kokkos_Core.hpp>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

#define NPTS (2048 * 8)
#define NDIM 128

#define M1W 128
#define M2W  16
#define M2H  16
#define M5W  16
#define M5H  16
#define M5R   4

struct alignas(16) float4 {
  float x, y, z, w;
};

// ---- Host reference -------------------------------------------------------
void MatchC1(float *h_pts1, float *h_pts2, float *h_score, int *h_index) {
  memset(h_score, 0, sizeof(float) * NPTS);
  for (int p1 = 0; p1 < NPTS; p1++) {
    for (int p2 = 0; p2 < NPTS; p2++) {
      float score = 0.0f;
      for (int d = 0; d < NDIM; d++)
        score += h_pts1[p1 * NDIM + d] * h_pts2[p2 * NDIM + d];
      if (score > h_score[p1]) {
        h_score[p1] = score;
        h_index[p1] = p2;
      }
    }
  }
}

void CheckMatches(int *h_index, int *h_index2, float *h_score, float *h_score2) {
  int ndiff = 0;
  for (int i = 0; i < NPTS; i++) {
    ndiff += (h_index[i] != h_index2[i]);
    if (h_index[i] != h_index2[i])
      std::cout << "  " << i << " " << h_index[i] << " " << h_index2[i]
                << " " << h_score[i] << " " << h_score2[i] << std::endl;
  }
  std::cout << "Number of incorrect matches: " << ndiff << std::endl;
}

// ---- GPU1: one thread per p1 point ----------------------------------------
void MatchGPU1(Kokkos::View<float *> d_pts1, Kokkos::View<float *> d_pts2,
               Kokkos::View<float *> d_score, Kokkos::View<int *> d_index,
               int repeat) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "MatchGPU1",
        Kokkos::RangePolicy<>(0, NPTS),
        KOKKOS_LAMBDA(const int p1) {
          float max_score = 0.0f;
          int   best      = -1;
          for (int p2 = 0; p2 < NPTS; p2++) {
            float score = 0.0f;
            for (int d = 0; d < NDIM; d++)
              score += d_pts1(p1 * NDIM + d) * d_pts2(p2 * NDIM + d);
            if (score > max_score) { max_score = score; best = p2; }
          }
          d_score(p1) = max_score;
          d_index(p1) = best;
        });
    Kokkos::fence();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6 / repeat;
  std::cout << "MatchGPU1:   " << ms << " ms  "
            << 2.0 * NPTS * NPTS * NDIM / ms / 1024 / 1024 << " Gflops" << std::endl;
}

// ---- GPU2: tiled team-based ------------------------------------------------
void MatchGPU2(Kokkos::View<float *> d_pts1, Kokkos::View<float *> d_pts2,
               Kokkos::View<float *> d_score, Kokkos::View<int *> d_index,
               int repeat) {
  // scratch: buffer1[M2W*NDIM] + buffer2[M2H*NDIM] + scores[M2W*M2H]
  const int scratch_size =
      (M2W * NDIM + M2H * NDIM + M2W * M2H) * (int)sizeof(float);

  using TeamPolicy = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<float *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(NPTS / M2W, M2W * M2H, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "MatchGPU2", policy,
        KOKKOS_LAMBDA(const member_type &team) {
          int bp1 = M2W * team.league_rank();
          int idx = team.team_rank();          // 0..M2W*M2H-1
          int tx  = idx % M2W;
          int ty  = idx / M2W;

          ScratchView sdata(team.team_scratch(0),
                            M2W * NDIM + M2H * NDIM + M2W * M2H);
          float *buffer1 = sdata.data();
          float *buffer2 = buffer1 + M2W * NDIM;
          float *scores  = buffer2 + M2H * NDIM;

          // Load buffer1: pts1 block
          if (ty < M2W)
            for (int d = tx; d < NDIM; d += M2W)
              for (int j = ty; j < M2W; j += M2H)
                buffer1[j * NDIM + d] = d_pts1((bp1 + j) * NDIM + d);
          team.team_barrier();

          float max_score = 0.0f;
          int   best      = -1;

          for (int bp2 = 0; bp2 < NPTS; bp2 += M2H) {
            for (int d = tx; d < NDIM; d += M2W)
              buffer2[ty * NDIM + d] = d_pts2((bp2 + ty) * NDIM + d);
            team.team_barrier();

            float score = 0.0f;
            for (int d = 0; d < NDIM; d++)
              score += buffer1[tx * NDIM + d] * buffer2[ty * NDIM + d];
            scores[idx] = score;
            team.team_barrier();

            if (ty == 0) {
              for (int i = 0; i < M2H; i++) {
                if (scores[i * M2W + tx] > max_score) {
                  max_score = scores[i * M2W + tx];
                  best      = bp2 + i;
                }
              }
            }
            team.team_barrier();
          }

          if (ty == 0) {
            d_score(bp1 + tx) = max_score;
            d_index(bp1 + tx) = best;
          }
        });
    Kokkos::fence();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6 / repeat;
  std::cout << "MatchGPU2:   " << ms << " ms  "
            << 2.0 * NPTS * NPTS * NDIM / ms / 1024 / 1024 << " Gflops" << std::endl;
}

// ---- GPU3: padded buffer1 (stride M2W*(NDIM+1)) ---------------------------
void MatchGPU3(Kokkos::View<float *> d_pts1, Kokkos::View<float *> d_pts2,
               Kokkos::View<float *> d_score, Kokkos::View<int *> d_index,
               int repeat) {
  const int NDIM1 = NDIM + 1; // padded stride for buffer1
  const int scratch_size =
      (M2W * NDIM1 + M2H * NDIM + M2W * M2H) * (int)sizeof(float);

  using TeamPolicy = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView  = Kokkos::View<float *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(NPTS / M2W, M2W * M2H, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "MatchGPU3", policy,
        KOKKOS_LAMBDA(const member_type &team) {
          int bp1 = M2W * team.league_rank();
          int idx = team.team_rank();
          int tx  = idx % M2W;
          int ty  = idx / M2W;

          ScratchView sdata(team.team_scratch(0),
                            M2W * NDIM1 + M2H * NDIM + M2W * M2H);
          float *buffer1 = sdata.data();
          float *buffer2 = buffer1 + M2W * NDIM1;
          float *scores  = buffer2 + M2H * NDIM;

          if (ty < M2W)
            for (int d = tx; d < NDIM; d += M2W)
              for (int j = ty; j < M2W; j += M2H)
                buffer1[j * NDIM1 + d] = d_pts1((bp1 + j) * NDIM + d);
          team.team_barrier();

          float max_score = 0.0f;
          int   best      = -1;

          for (int bp2 = 0; bp2 < NPTS; bp2 += M2H) {
            for (int d = tx; d < NDIM; d += M2W)
              buffer2[ty * NDIM + d] = d_pts2((bp2 + ty) * NDIM + d);
            team.team_barrier();

            float score = 0.0f;
            for (int d = 0; d < NDIM; d++)
              score += buffer1[tx * NDIM1 + d] * buffer2[ty * NDIM + d];
            scores[idx] = score;
            team.team_barrier();

            if (ty == 0)
              for (int i = 0; i < M2H; i++)
                if (scores[i * M2W + tx] > max_score) {
                  max_score = scores[i * M2W + tx];
                  best      = bp2 + i;
                }
            team.team_barrier();
          }

          if (ty == 0) {
            d_score(bp1 + tx) = max_score;
            d_index(bp1 + tx) = best;
          }
        });
    Kokkos::fence();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6 / repeat;
  std::cout << "MatchGPU3:   " << ms << " ms  "
            << 2.0 * NPTS * NPTS * NDIM / ms / 1024 / 1024 << " Gflops" << std::endl;
}

// ---- GPU4: float4 vectorised version --------------------------------------
void MatchGPU4(Kokkos::View<float *> d_pts1, Kokkos::View<float *> d_pts2,
               Kokkos::View<float *> d_score, Kokkos::View<int *> d_index,
               int repeat) {
  const int NDIM4  = NDIM / 4;
  const int NDIM41 = NDIM4 + 1; // padded
  // buffer1: M2W*(NDIM/4+1) float4, buffer2: M2H*(NDIM/4) float4, scores: M2W*M2H float
  const int scratch_bytes =
      (M2W * NDIM41 + M2H * NDIM4) * (int)sizeof(float4) +
      M2W * M2H * (int)sizeof(float);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchViewB = Kokkos::View<char *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(NPTS / M2W, M2W * M2H, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "MatchGPU4", policy,
        KOKKOS_LAMBDA(const member_type &team) {
          int bp1 = M2W * team.league_rank();
          int idx = team.team_rank();
          int tx  = idx % M2W;
          int ty  = idx / M2W;

          ScratchViewB raw(team.team_scratch(0), scratch_bytes);
          float4 *buffer1 = reinterpret_cast<float4 *>(raw.data());
          float4 *buffer2 = buffer1 + M2W * NDIM41;
          float  *scores  = reinterpret_cast<float *>(buffer2 + M2H * NDIM4);

          // Cast pts to float4 pointers via raw data
          const float4 *f4_pts1 = reinterpret_cast<const float4 *>(&d_pts1(0));
          const float4 *f4_pts2 = reinterpret_cast<const float4 *>(&d_pts2(0));

          if (ty < M2W)
            for (int d = tx; d < NDIM4; d += M2W)
              for (int j = ty; j < M2W; j += M2H)
                buffer1[j * NDIM41 + d] = f4_pts1[(bp1 + j) * NDIM4 + d];
          team.team_barrier();

          float max_score = 0.0f;
          int   best      = -1;

          for (int bp2 = 0; bp2 < NPTS; bp2 += M2H) {
            for (int d = tx; d < NDIM4; d += M2W)
              buffer2[ty * NDIM4 + d] = f4_pts2[(bp2 + ty) * NDIM4 + d];
            team.team_barrier();

            float score = 0.0f;
            for (int d = 0; d < NDIM4; d++) {
              float4 v1 = buffer1[tx * NDIM41 + d];
              float4 v2 = buffer2[ty * NDIM4 + d];
              score += v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
            }
            scores[idx] = score;
            team.team_barrier();

            if (ty == 0)
              for (int i = 0; i < M2H; i++)
                if (scores[i * M2W + tx] > max_score) {
                  max_score = scores[i * M2W + tx];
                  best      = bp2 + i;
                }
            team.team_barrier();
          }

          if (ty == 0) {
            d_score(bp1 + tx) = max_score;
            d_index(bp1 + tx) = best;
          }
        });
    Kokkos::fence();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6 / repeat;
  std::cout << "MatchGPU4:   " << ms << " ms  "
            << 2.0 * NPTS * NPTS * NDIM / ms / 1024 / 1024 << " Gflops" << std::endl;
}

// ---- GPU5: float4 with M5R loop unrolling ---------------------------------
void MatchGPU5(Kokkos::View<float *> d_pts1, Kokkos::View<float *> d_pts2,
               Kokkos::View<float *> d_score, Kokkos::View<int *> d_index,
               int repeat) {
  const int NDIM4  = NDIM / 4;
  const int NDIM41 = NDIM4 + 1;
  // buffer1: M5W*(NDIM/4+1) float4, buffer2: M5H*(NDIM/4) float4
  // scores: M5W*M5H float, indices: M5W*M5H/M5R ints
  const int scratch_bytes =
      (M5W * NDIM41 + M5H * NDIM4) * (int)sizeof(float4) +
      (M5W * M5H) * (int)sizeof(float) +
      (M5W * M5H / M5R) * (int)sizeof(int);

  using TeamPolicy  = Kokkos::TeamPolicy<>;
  using member_type = TeamPolicy::member_type;
  using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchViewB = Kokkos::View<char *, ScratchSpace, Kokkos::MemoryUnmanaged>;

  TeamPolicy policy(NPTS / M5W, M5W * M5H, 1);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for(
        "MatchGPU5", policy,
        KOKKOS_LAMBDA(const member_type &team) {
          int bp1 = M5W * team.league_rank();
          int idx = team.team_rank();
          int tx  = idx % M5W;
          int ty  = idx / M5W;

          ScratchViewB raw(team.team_scratch(0), scratch_bytes);
          float4 *buffer1 = reinterpret_cast<float4 *>(raw.data());
          float4 *buffer2 = buffer1 + M5W * NDIM41;
          float  *scores  = reinterpret_cast<float *>(buffer2 + M5H * NDIM4);
          int    *indices = reinterpret_cast<int *>(scores + M5W * M5H);

          const float4 *f4_pts1 = reinterpret_cast<const float4 *>(&d_pts1(0));
          const float4 *f4_pts2 = reinterpret_cast<const float4 *>(&d_pts2(0));

          if (ty < M5W)
            for (int d = tx; d < NDIM4; d += M5W)
              for (int j = ty; j < M5W; j += M5H)
                buffer1[j * NDIM41 + d] = f4_pts1[(bp1 + j) * NDIM4 + d];
          team.team_barrier();

          float max_score = 0.0f;
          int   best      = -1;

          for (int bp2 = 0; bp2 < NPTS; bp2 += M5H) {
            for (int d = tx; d < NDIM4; d += M5W)
              buffer2[ty * NDIM4 + d] = f4_pts2[(bp2 + ty) * NDIM4 + d];
            team.team_barrier();

            if (ty < M5H / M5R) {
              float score[M5R];
              for (int dy = 0; dy < M5R; dy++) score[dy] = 0.0f;
              for (int d = 0; d < NDIM4; d++) {
                float4 v1 = buffer1[tx * NDIM41 + d];
                for (int dy = 0; dy < M5R; dy++) {
                  float4 v2 = buffer2[(M5R * ty + dy) * NDIM4 + d];
                  score[dy] += v1.x * v2.x + v1.y * v2.y +
                               v1.z * v2.z + v1.w * v2.w;
                }
              }
              for (int dy = 0; dy < M5R; dy++) {
                scores[tx + M5W * (M5R * ty + dy)] = score[dy];
              }
            }
            team.team_barrier();

            if (ty == 0)
              for (int i = 0; i < M5H; i++)
                if (scores[i * M5W + tx] > max_score) {
                  max_score = scores[i * M5W + tx];
                  best      = bp2 + i;
                }
            team.team_barrier();
          }

          if (ty < M5H / M5R) {
            scores[ty * M5W + tx]  = max_score;
            indices[ty * M5W + tx] = best;
          }
          team.team_barrier();

          if (ty == 0) {
            float  ms2 = scores[tx];
            int    bi2 = indices[tx];
            for (int y = 0; y < M5H / M5R; y++)
              if (scores[y * M5W + tx] > ms2) {
                ms2 = scores[y * M5W + tx];
                bi2 = indices[y * M5W + tx];
              }
            d_score(bp1 + tx) = ms2;
            d_index(bp1 + tx) = bi2;
          }
        });
    Kokkos::fence();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6 / repeat;
  std::cout << "MatchGPU5:   " << ms << " ms  "
            << 2.0 * NPTS * NPTS * NDIM / ms / 1024 / 1024 << " Gflops" << std::endl;
}

// ---- Main -----------------------------------------------------------------
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    std::vector<float> data(NPTS * NDIM * 2);
    float *h_pts1 = data.data();
    float *h_pts2 = data.data() + NPTS * NDIM;

    std::cout << std::endl;
    std::cout << "Data size:   " << 2.0 * NPTS * NDIM * sizeof(float) / 1024 / 1024 << " MB" << std::endl;

    for (int i = 0; i < NPTS; i++) {
      float sum1 = 0.0f, sum2 = 0.0f;
      for (int d = 0; d < NDIM; d++) {
        sum1 += h_pts1[i * NDIM + d] = (float)rand() / RAND_MAX;
        sum2 += h_pts2[i * NDIM + d] = (float)rand() / RAND_MAX;
      }
      sum1 = sqrtf((float)NDIM) / sum1;
      sum2 = sqrtf((float)NDIM) / sum2;
      for (int d = 0; d < NDIM; d++) {
        h_pts1[i * NDIM + d] *= sum1;
        h_pts2[i * NDIM + d] *= sum2;
      }
    }

    std::vector<int>   h_index(NPTS), h_index2(NPTS);
    std::vector<float> h_score(NPTS), h_score2(NPTS);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    MatchC1(h_pts1, h_pts2, h_score.data(), h_index.data());
    auto t1  = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-6;
    std::cout << "MatchCPU1:   " << ms << " ms  "
              << 2.0 * NPTS * NPTS * NDIM / ms / 1024 / 1024 << " Gflops" << std::endl;

    // Device views
    Kokkos::View<float *> d_pts1("d_pts1", NPTS * NDIM);
    Kokkos::View<float *> d_pts2("d_pts2", NPTS * NDIM);
    Kokkos::View<float *> d_score("d_score", NPTS);
    Kokkos::View<int *>   d_index("d_index", NPTS);

    {
      auto h_d_pts1 = Kokkos::create_mirror_view(d_pts1);
      auto h_d_pts2 = Kokkos::create_mirror_view(d_pts2);
      memcpy(h_d_pts1.data(), h_pts1, sizeof(float) * NPTS * NDIM);
      memcpy(h_d_pts2.data(), h_pts2, sizeof(float) * NPTS * NDIM);
      Kokkos::deep_copy(d_pts1, h_d_pts1);
      Kokkos::deep_copy(d_pts2, h_d_pts2);
    }

    auto copy_back = [&]() {
      auto h_s = Kokkos::create_mirror_view(d_score);
      auto h_i = Kokkos::create_mirror_view(d_index);
      Kokkos::deep_copy(h_s, d_score);
      Kokkos::deep_copy(h_i, d_index);
      memcpy(h_score2.data(), h_s.data(), sizeof(float) * NPTS);
      memcpy(h_index2.data(), h_i.data(), sizeof(int)   * NPTS);
    };

    MatchGPU1(d_pts1, d_pts2, d_score, d_index, repeat);
    copy_back();
    CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

    MatchGPU2(d_pts1, d_pts2, d_score, d_index, repeat);
    copy_back();
    CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

    MatchGPU3(d_pts1, d_pts2, d_score, d_index, repeat);
    copy_back();
    CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

    MatchGPU4(d_pts1, d_pts2, d_score, d_index, repeat);
    copy_back();
    CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

    MatchGPU5(d_pts1, d_pts2, d_score, d_index, repeat);
    copy_back();
    CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());
  }
  Kokkos::finalize();
  return 0;
}
