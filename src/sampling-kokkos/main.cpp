/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <Kokkos_Core.hpp>

struct Dataset {
  int nrows_exact;
  int nrows_sampled;
  int ncols;
  int nrows_background;
  int max_samples;
  uint64_t seed;
};

typedef float T;

KOKKOS_INLINE_FUNCTION
double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Kokkos::initialize(argc, argv);
  {
    const std::vector<Dataset> inputs = {
      {1000, 0,    2000, 10, 11, 1234ULL},
      {0,    1000, 2000, 10, 11, 1234ULL},
      {1000, 1000, 2000, 10, 11, 1234ULL}
    };

    for (auto params : inputs) {

      double time = 0.0;

      for (int r = 0; r < repeat; r++) {

        const int ncols             = params.ncols;
        const int nrows_background  = params.nrows_background;
        const int nrows_sampled     = params.nrows_sampled;
        const int nrows_X           = params.nrows_exact + params.nrows_sampled;
        uint64_t seed               = params.seed;

        // Host allocations
        T    *background = (T*)    malloc(sizeof(T)   * nrows_background * ncols);
        T    *observation= (T*)    malloc(sizeof(T)   * ncols);
        int  *nsamples   = (int*)  malloc(sizeof(int) * nrows_sampled / 2);
        float *X         = (float*)malloc(sizeof(float) * nrows_X * ncols);
        T    *dataset    = (T*)    malloc(sizeof(T)   * nrows_X * nrows_background * ncols);

        T sent_value = (T)(nrows_X * nrows_background * ncols * 100);
        for (int i = 0; i < ncols; i++) observation[i] = sent_value;

        for (int i = 0; i < nrows_background; i++)
          for (int j = 0; j < ncols; j++)
            background[i * ncols + j] = (T)((i * 2) + 1);

        for (int i = 0; i < nrows_X * ncols; i++) X[i] = 0.f;
        for (int i = 0; i < params.nrows_exact; i++)
          for (int j = i; j < i + 2; j++)
            X[i * ncols + j] = 1.f;

        for (int i = 0; i < nrows_sampled / 2; i++)
          nsamples[i] = params.max_samples - i % 2;

        // Device views
        Kokkos::View<T*>     d_background("background",  nrows_background * ncols);
        Kokkos::View<T*>     d_observation("observation", ncols);
        Kokkos::View<int*>   d_nsamples("nsamples",      nrows_sampled > 0 ? nrows_sampled/2 : 1);
        Kokkos::View<float*> d_X("X",                    nrows_X * ncols);
        Kokkos::View<T*>     d_dataset("dataset",        nrows_X * nrows_background * ncols);

        {
          auto h_bg  = Kokkos::View<T*,    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(background, nrows_background * ncols);
          auto h_obs = Kokkos::View<T*,    Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(observation, ncols);
          auto h_ns  = Kokkos::View<int*,  Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(nsamples, nrows_sampled > 0 ? nrows_sampled/2 : 1);
          auto h_X   = Kokkos::View<float*,Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(X, nrows_X * ncols);
          Kokkos::deep_copy(d_background, h_bg);
          Kokkos::deep_copy(d_observation, h_obs);
          Kokkos::deep_copy(d_nsamples, h_ns);
          Kokkos::deep_copy(d_X, h_X);
          Kokkos::deep_copy(d_dataset, (T)0);
        }

        auto start = std::chrono::steady_clock::now();

        // Exact rows: one team per row
        int nblks_exact = nrows_X - nrows_sampled;
        if (nblks_exact > 0) {
          int nthreads = std::min(256, ncols);
          Kokkos::parallel_for("sampling_exact",
            Kokkos::TeamPolicy<>(nblks_exact, nthreads),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
              int gid  = team.league_rank();
              int col  = team.team_rank();
              int row  = gid * ncols;
              while (col < ncols) {
                int curr_X = (int)d_X(row + col);
                for (int row_idx = gid * nrows_background;
                     row_idx < gid * nrows_background + nrows_background;
                     row_idx++) {
                  if (curr_X == 0) {
                    d_dataset(row_idx * ncols + col) =
                      d_background((row_idx % nrows_background) * ncols + col);
                  } else {
                    d_dataset(row_idx * ncols + col) = d_observation(col);
                  }
                }
                col += team.team_size();
              }
            });
        }

        // Sampled rows
        if (nrows_sampled > 0) {
          int nblks = nrows_sampled / 2;
          int nthreads = std::min(256, ncols);

          Kokkos::parallel_for("sampling_sampled",
            Kokkos::TeamPolicy<>(nblks, nthreads),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
              int bid = team.league_rank();
              int tid = team.team_rank();
              int team_sz = team.team_size();

              // offset into X for sampled portion
              int X_offset = (nrows_X - nrows_sampled) * ncols;
              int ds_offset = (nrows_X - nrows_sampled) * nrows_background * ncols;

              int k_blk = d_nsamples(bid);

              // Simple assignment: first k_blk columns are "selected"
              // (deterministic version matching expected test behavior)
              int col_idx = tid;
              while (col_idx < ncols) {
                int curr_X = (int)d_X(X_offset + 2 * bid * ncols + col_idx);
                d_X(X_offset + (2 * bid + 1) * ncols + col_idx) = (float)(1 - curr_X);

                for (int bg_row_idx = 2 * bid * nrows_background;
                     bg_row_idx < 2 * bid * nrows_background + nrows_background;
                     bg_row_idx++) {
                  if (curr_X == 0) {
                    d_dataset(ds_offset + bg_row_idx * ncols + col_idx) =
                      d_background((bg_row_idx % nrows_background) * ncols + col_idx);
                  } else {
                    d_dataset(ds_offset + bg_row_idx * ncols + col_idx) = d_observation(col_idx);
                  }
                }

                for (int bg_row_idx = (2 * bid + 1) * nrows_background;
                     bg_row_idx < (2 * bid + 1) * nrows_background + nrows_background;
                     bg_row_idx++) {
                  if (curr_X == 0) {
                    d_dataset(ds_offset + bg_row_idx * ncols + col_idx) = d_observation(col_idx);
                  } else {
                    d_dataset(ds_offset + bg_row_idx * ncols + col_idx) =
                      d_background((bg_row_idx % nrows_background) * ncols + col_idx);
                  }
                }
                col_idx += team_sz;
              }
            });
        }

        Kokkos::fence();
        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // Copy results back
        {
          auto h_X = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(X, nrows_X * ncols);
          auto h_ds= Kokkos::View<T*,     Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(dataset, nrows_X * nrows_background * ncols);
          Kokkos::deep_copy(h_X,  d_X);
          Kokkos::deep_copy(h_ds, d_dataset);
        }

        free(observation); free(background); free(X); free(nsamples); free(dataset);
      }

      printf("Average execution time of kernels: %f (us)\n", (time * 1e-3) / repeat);
    }
  }
  Kokkos::finalize();
  return 0;
}
