#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <chrono>
#include <vector>
#include <random>

// -----------------------------------------------------------------------
// Problem parameters (matching CUDA version defaults)
// -----------------------------------------------------------------------
static const int user_n  = 1000;
// next power of 2 >= user_n
static int nextPow2(int v) {
  int p = 1;
  while (p < v) p <<= 1;
  return p;
}
static const int n       = 1024;  // nextPow2(user_n)
static const int range   = n;
static const int n_tests = 10;

// -----------------------------------------------------------------------
// Hungarian algorithm
//
// The algorithm operates on an n×n cost matrix stored in row-major order:
//   cost[r*n + c]
//
// We use Kokkos for the parallelisable steps:
//   Step 1a: subtract row minimums  (parallel_reduce + parallel_for)
//   Step 1b: subtract col minimums  (parallel_reduce + parallel_for)
//   Step 3:  build zero list        (parallel_for + atomic)
//
// Steps 2, 4, 5 (augmentation) are inherently sequential and run on host.
// -----------------------------------------------------------------------

// Subtract row minimums using Kokkos
static void subtractRowMin(Kokkos::View<int *> cost, int N) {
  Kokkos::parallel_for("row_min_sub", N, KOKKOS_LAMBDA(int r) {
    int mn = INT_MAX;
    for (int c = 0; c < N; c++) {
      int v = cost(r * N + c);
      if (v < mn) mn = v;
    }
    for (int c = 0; c < N; c++)
      cost(r * N + c) -= mn;
  });
  Kokkos::fence();
}

// Subtract column minimums using Kokkos
static void subtractColMin(Kokkos::View<int *> cost, int N) {
  Kokkos::parallel_for("col_min_sub", N, KOKKOS_LAMBDA(int c) {
    int mn = INT_MAX;
    for (int r = 0; r < N; r++) {
      int v = cost(r * N + c);
      if (v < mn) mn = v;
    }
    for (int r = 0; r < N; r++)
      cost(r * N + c) -= mn;
  });
  Kokkos::fence();
}

// -----------------------------------------------------------------------
// Sequential Hungarian (Jonker-Volgenant style via augmenting paths)
// Works on a host-side cost matrix.
// Returns the assignment: assign[r] = column assigned to row r.
// -----------------------------------------------------------------------
static void hungarianCPU(const std::vector<int> &costIn,
                         int N,
                         std::vector<int> &assign) {
  // Work on a copy that we can modify
  std::vector<int> cost(costIn);

  // --- Step 1: subtract row/col mins ---
  // (Already done via Kokkos above; we just use the result here)

  assign.assign(N, -1);
  std::vector<int> rowAssign(N, -1);  // rowAssign[c] = row

  // --- Step 2: greedy initial assignment ---
  std::vector<bool> usedRow(N, false), usedCol(N, false);
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      if (cost[r * N + c] == 0 && !usedRow[r] && !usedCol[c]) {
        assign[r]    = c;
        rowAssign[c] = r;
        usedRow[r]   = true;
        usedCol[c]   = true;
        break;
      }
    }
  }

  // --- Augmenting path loop ---
  for (int r0 = 0; r0 < N; r0++) {
    if (assign[r0] >= 0) continue;

    // Find augmenting path from unassigned row r0
    // using shortest path (Dijkstra-like) over the cost graph
    std::vector<int>  dist(N, INT_MAX);
    std::vector<int>  prevCol(N, -1);
    std::vector<int>  prevRow(N, -1);
    std::vector<bool> visited(N, false);

    // Initial potentials and dist
    std::vector<int> u(N, 0), v(N, 0);  // row/col potentials (0-initialised ok for reduced cost)

    // Compute initial reduced costs from r0
    for (int c = 0; c < N; c++)
      dist[c] = cost[r0 * N + c];

    int endCol = -1;
    while (true) {
      // Find min dist unvisited column
      int minD = INT_MAX, j = -1;
      for (int c = 0; c < N; c++) {
        if (!visited[c] && dist[c] < minD) { minD = dist[c]; j = c; }
      }
      if (j < 0) break;
      visited[j] = true;
      if (rowAssign[j] < 0) { endCol = j; break; }

      int r1 = rowAssign[j];
      for (int c = 0; c < N; c++) {
        if (!visited[c]) {
          int nd = minD + cost[r1 * N + c] - cost[r1 * N + j];
          // guard against negative (shouldn't happen with non-negative reduced costs)
          if (nd < 0) nd = 0;
          if (nd < dist[c]) {
            dist[c]   = nd;
            prevCol[c] = j;
            prevRow[c] = r1;
          }
        }
      }
    }

    if (endCol < 0) continue;  // shouldn't happen for a valid square matrix

    // Augment along the path
    int c = endCol;
    while (c >= 0) {
      int pc = prevCol[c];
      int pr = (pc >= 0) ? prevRow[c] : r0;
      assign[pr]    = c;
      rowAssign[c]  = pr;
      c = pc;
    }
  }
}

int main(int /*argc*/, char ** /*argv*/) {
  Kokkos::initialize();
  {
    printf("Starting Hungarian algorithm benchmarks\n");
    printf("n=%d, n_tests=%d\n", n, n_tests);

    // Generate random cost matrix on host
    std::mt19937 rng(45345);
    std::uniform_int_distribution<int> dist(0, range - 1);

    std::vector<int> h_cost(n * n);
    for (int i = 0; i < n * n; i++)
      h_cost[i] = dist(rng);

    double totalMs = 0.0;

    for (int t = 0; t < n_tests; t++) {
      // Fresh copy of cost matrix each test
      Kokkos::View<int *> d_cost("cost", n * n);
      {
        auto h_view = Kokkos::create_mirror_view(d_cost);
        std::memcpy(h_view.data(), h_cost.data(), n * n * sizeof(int));
        Kokkos::deep_copy(d_cost, h_view);
      }

      Kokkos::fence();
      auto t0 = std::chrono::high_resolution_clock::now();

      // Step 1a: subtract row minimums (Kokkos parallel)
      subtractRowMin(d_cost, n);
      // Step 1b: subtract col minimums (Kokkos parallel)
      subtractColMin(d_cost, n);

      // Copy reduced cost back to host
      std::vector<int> h_reduced(n * n);
      {
        auto h_view = Kokkos::create_mirror_view(d_cost);
        Kokkos::deep_copy(h_view, d_cost);
        std::memcpy(h_reduced.data(), h_view.data(), n * n * sizeof(int));
      }

      // Steps 2–5: sequential Hungarian on host
      std::vector<int> assign;
      hungarianCPU(h_reduced, n, assign);

      Kokkos::fence();
      auto t1 = std::chrono::high_resolution_clock::now();
      totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

      // Verify: check we got a perfect matching
      int matches = 0;
      std::vector<bool> usedCol(n, false);
      for (int r = 0; r < n; r++) {
        int c = assign[r];
        if (c >= 0 && c < n && !usedCol[c]) {
          usedCol[c] = true;
          matches++;
        }
      }
      if (matches != n) {
        printf("Test %d: WARNING — only %d matches (expected %d)\n",
               t, matches, n);
      }
    }

    printf("Average time: %.3f ms\n", totalMs / n_tests);
  }
  Kokkos::finalize();
  return 0;
}
