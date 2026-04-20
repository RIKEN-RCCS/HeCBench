/*
TSP_GPU: This code is a GPU-accelerated heuristic solver for the
symmetric Traveling Salesman Problem that is based on iterative hill
climbing with 2-opt local search.

Copyright (c) 2014-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/TSP_GPU/.

Publication: This work is described in detail in the following paper.
Molly A. O'Neil and Martin Burtscher. Rethinking the Parallelization of
Random-Restart Hill Climbing. Proceedings of the Eighth Workshop on General
Purpose Processing Using GPUs (10 pages). February 2015.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <chrono>
#include <Kokkos_Core.hpp>

#define tilesize 128

using exec_space   = Kokkos::DefaultExecutionSpace;
using mem_space    = exec_space::memory_space;
using ScratchSpace = exec_space::scratch_memory_space;
// Scratch layout per team: px_s[128] | py_s[128] | bf_s[128] | buf_s[128]  (all float)
using ScratchView  = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged>;

KOKKOS_INLINE_FUNCTION
float LCG_random(unsigned int *seed)
{
  const unsigned int m = 2147483648u;
  const unsigned int a = 26757677u;
  const unsigned int c = 1u;
  *seed = (a * (*seed) + c) % m;
  return (float)(*seed) / (float)m;
}

static int best_thread_count(int cities)
{
  int max = cities - 2;
  if (max > 256) max = 256;
  int best = 0, bthr = 4;
  for (int threads = 1; threads <= max; threads++) {
    int smem   = (int)(sizeof(int) * threads + 2 * sizeof(float) * tilesize + sizeof(int) * tilesize);
    int blocks = (16384 * 2) / smem;
    if (blocks > 16) blocks = 16;
    int thr    = (threads + 31) / 32 * 32;
    while (blocks * thr > 2048) blocks--;
    int perf   = threads * blocks;
    if (perf > best) {
      best = perf;
      bthr = threads;
    }
  }
  return bthr;
}

int main(int argc, char *argv[])
{
  printf("2-opt TSP Kokkos GPU code v2.3\n");
  printf("Copyright (c) 2014-2020, Texas State University. All rights reserved.\n");

  if (argc != 4) {
    fprintf(stderr, "\narguments: <input_file> <restart_count> <repeat>\n");
    return -1;
  }

  FILE *f = fopen(argv[1], "rt");
  if (f == NULL) { fprintf(stderr, "could not open file %s\n", argv[1]); return -1; }

  int restarts = atoi(argv[2]);
  if (restarts < 1) { fprintf(stderr, "restart_count is too small: %d\n", restarts); return -1; }

  int repeat = atoi(argv[3]);

  // Read TSP input file
  int ch;
  float in2, in3;
  char str[256];

  ch = getc(f); while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f); while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f); while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f); while ((ch != EOF) && (ch != ':'))  ch = getc(f);
  fscanf(f, "%s\n", str);

  int cities = atoi(str);
  if (cities < 100) {
    fprintf(stderr, "the problem size must be at least 100 for this version of the code\n");
    fclose(f); return -1;
  }

  ch = getc(f); while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  fscanf(f, "%s\n", str);
  if (strcmp(str, "NODE_COORD_SECTION") != 0) {
    fprintf(stderr, "wrong file format\n"); fclose(f); return -1;
  }

  float *posx = (float*)malloc(sizeof(float) * cities);
  float *posy = (float*)malloc(sizeof(float) * cities);
  if (!posx || !posy) { fprintf(stderr, "allocation error\n"); return -1; }

  int cnt = 0, in1;
  while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
    posx[cnt] = in2; posy[cnt] = in3; cnt++;
    if (cnt > cities) fprintf(stderr, "input too long\n");
    if (cnt != in1)   fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);
  }
  if (cnt != cities) fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);
  fscanf(f, "%s", str);
  if (strcmp(str, "EOF") != 0) fprintf(stderr, "didn't see 'EOF' at end of file\n");
  fclose(f);

  printf("configuration: %d cities, %d restarts, %s input\n", cities, restarts, argv[1]);

  int threads = best_thread_count(cities);
  printf("number of threads per team: %d\n", threads);

  // Device allocations
  // glob: per-team workspace: [buf(cities ints) | px(cities+1 floats) | py(cities+1 floats)]
  const int stride  = (3 * cities + 2 + 31) / 32 * 32;  // in ints, padded
  const int glob_sz = restarts * stride;

  // Scratch memory per team: px_s[128], py_s[128], bf_s[128], buf_s[128]
  const int scratch_size = ScratchView::shmem_size(4 * tilesize);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*, mem_space> d_posx("posx", cities);
    Kokkos::View<float*, mem_space> d_posy("posy", cities);
    Kokkos::View<int*,   mem_space> d_glob("glob", glob_sz);
    Kokkos::View<int[1], mem_space> d_climbs("climbs");
    Kokkos::View<int[1], mem_space> d_best("best");

    {
      auto hx = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(posx, cities);
      auto hy = Kokkos::View<float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(posy, cities);
      Kokkos::deep_copy(d_posx, hx);
      Kokkos::deep_copy(d_posy, hy);
    }

    double ktime = 0.0;

    for (int rep = 0; rep < repeat; rep++) {
      // Reset climbs and best
      auto h_climbs = Kokkos::create_mirror_view(d_climbs);
      auto h_best   = Kokkos::create_mirror_view(d_best);
      h_climbs(0) = 0;
      h_best(0)   = INT_MAX;
      Kokkos::deep_copy(d_climbs, h_climbs);
      Kokkos::deep_copy(d_best,   h_best);
      Kokkos::fence();

      auto kstart = std::chrono::steady_clock::now();

      Kokkos::parallel_for("tsp_kernel",
        Kokkos::TeamPolicy<exec_space>(restarts, threads)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<exec_space>::member_type& team) {

          ScratchView sm(team.team_scratch(0), 4 * tilesize);
          float* px_s  = sm.data();
          float* py_s  = sm.data() + tilesize;
          float* bf_s  = sm.data() + 2 * tilesize;
          float* buf_s = sm.data() + 3 * tilesize;

          const int lid  = team.team_rank();
          const int bid  = team.league_rank();
          const int dim  = team.team_size();

          // Per-team workspace in global memory
          int   *buf = &d_glob[bid * stride];
          float *px  = (float*)(&buf[cities]);
          float *py  = &px[cities + 1];

          // Copy city coordinates into per-team workspace
          for (int i = lid; i < cities; i += dim) px[i] = d_posx[i];
          for (int i = lid; i < cities; i += dim) py[i] = d_posy[i];
          team.team_barrier();

          // Serial random permutation (thread 0 only)
          if (lid == 0) {
            unsigned int seed = (unsigned int)bid;
            for (unsigned int i = 1; i < (unsigned int)cities; i++) {
              int j = (int)(LCG_random(&seed) * (cities - 1)) + 1;
              float tmp;
              tmp = px[i]; px[i] = px[j]; px[j] = tmp;
              tmp = py[i]; py[i] = py[j]; py[j] = tmp;
            }
            px[cities] = px[0];
            py[cities] = py[0];
          }
          team.team_barrier();

          // 2-opt hill climbing
          int minchange;
          do {
            // Compute initial edge costs into buf[]
            for (int i = lid; i < cities; i += dim) {
              float dx = px[i] - px[i+1], dy = py[i] - py[i+1];
              buf[i] = -int(sqrtf(dx*dx + dy*dy));
            }
            team.team_barrier();

            minchange = 0;
            int mini = 1, minj = 0;

            for (int ii = 0; ii < cities - 2; ii += dim) {
              int i = ii + lid;
              float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;
              if (i < cities - 2) {
                minchange -= buf[i];
                pxi0 = px[i];   pyi0 = py[i];
                pxi1 = px[i+1]; pyi1 = py[i+1];
                pxj1 = px[cities]; pyj1 = py[cities];
              }

              for (int jj = cities - 1; jj >= ii + 2; jj -= tilesize) {
                int bound = jj - tilesize + 1;
                // Load tile into shared memory
                for (int k = lid; k < tilesize; k += dim) {
                  if (k + bound >= ii + 2) {
                    px_s[k] = px[k + bound];
                    py_s[k] = py[k + bound];
                    bf_s[k] = (float)buf[k + bound];
                  }
                }
                team.team_barrier();

                int lower = bound;
                if (lower < i + 2) lower = i + 2;
                for (int j = jj; j >= lower; j--) {
                  int jm = j - bound;
                  float pxj0 = px_s[jm], pyj0 = py_s[jm];
                  float dx0 = pxi0 - pxj0, dy0 = pyi0 - pyj0;
                  float dx1 = pxi1 - pxj1, dy1 = pyi1 - pyj1;
                  int change = (int)bf_s[jm]
                             + int(sqrtf(dx0*dx0 + dy0*dy0))
                             + int(sqrtf(dx1*dx1 + dy1*dy1));
                  pxj1 = pxj0; pyj1 = pyj0;
                  if (minchange > change) {
                    minchange = change;
                    mini = i; minj = j;
                  }
                }
                team.team_barrier();
              }

              if (i < cities - 2) {
                minchange += buf[i];
              }
            }
            team.team_barrier();

            // Store thread's local minimum into buf_s for tree reduction
            int change = minchange;
            buf_s[lid] = (float)change;

            if (lid == 0) {
              Kokkos::atomic_increment(&d_climbs(0));
            }
            team.team_barrier();

            // Tree reduction to find global minimum
            int j = dim;
            do {
              int k = (j + 1) / 2;
              if ((lid + k) < j) {
                int tmp = (int)buf_s[lid + k];
                if (change > tmp) change = tmp;
                buf_s[lid] = (float)change;
              }
              j = k;
              team.team_barrier();
            } while (j > 1);

            // Non-deterministically pick a winner among those with global min
            if (minchange == (int)buf_s[0]) {
              buf_s[1] = (float)lid;
            }
            team.team_barrier();

            if (lid == (int)buf_s[1]) {
              buf_s[2] = (float)(mini + 1);
              buf_s[3] = (float)minj;
            }
            team.team_barrier();

            minchange = (int)buf_s[0];
            mini      = (int)buf_s[2];
            int sum   = (int)buf_s[3] + mini;

            // Reverse segment [mini..minj] in the tour
            for (int i = lid; (i + i) < sum; i += dim) {
              if (mini <= i) {
                int jrev = sum - i;
                float tmp;
                tmp = px[i]; px[i] = px[jrev]; px[jrev] = tmp;
                tmp = py[i]; py[i] = py[jrev]; py[jrev] = tmp;
              }
            }
            team.team_barrier();
          } while (minchange < 0);

          // Compute tour length
          int term = 0;
          for (int i = lid; i < cities; i += dim) {
            float dx = px[i] - px[i+1], dy = py[i] - py[i+1];
            term += int(sqrtf(dx*dx + dy*dy));
          }
          buf_s[lid] = (float)term;
          team.team_barrier();

          // Tree reduction for tour length sum
          int j = dim;
          do {
            int k = (j + 1) / 2;
            if ((lid + k) < j) {
              term += (int)buf_s[lid + k];
            }
            team.team_barrier();
            if ((lid + k) < j) {
              buf_s[lid] = (float)term;
            }
            j = k;
            team.team_barrier();
          } while (j > 1);

          if (lid == 0) {
            Kokkos::atomic_min(&d_best(0), (int)buf_s[0]);
          }
        });

      Kokkos::fence();
      auto kend = std::chrono::steady_clock::now();
      if (rep > 0)
        ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    }

    // Copy results back
    auto h_climbs = Kokkos::create_mirror_view(d_climbs);
    auto h_best   = Kokkos::create_mirror_view(d_best);
    Kokkos::deep_copy(h_climbs, d_climbs);
    Kokkos::deep_copy(h_best,   d_best);

    long long moves = 1LL * h_climbs(0) * (cities - 2) * (cities - 1) / 2;

    printf("Average kernel time: %.4f s\n", ktime * 1e-9f / repeat);
    printf("%.3f Gmoves/s\n", moves * repeat / ktime);
    printf("Best found tour length is %d with %d climbers\n", h_best(0), h_climbs(0));

    if (h_best(0) < 38000 && h_best(0) >= 35002)
      printf("PASS\n");
    else
      printf("FAIL\n");
  }
  Kokkos::finalize();

  free(posx);
  free(posy);
  return 0;
}
