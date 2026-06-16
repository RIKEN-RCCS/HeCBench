#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#define VALUE_TYPE double

// Generate an n×n lower-triangular random sparse matrix in CSR format.
// Each row has exactly one diagonal entry (value in [2,10]) and up to
// nnz_per_row-1 random off-diagonal lower entries (values in (0,1]).
static void generate_lower_triangular(
    int n, int nnz_per_row,
    std::vector<int>        &csrRowPtr,
    std::vector<int>        &csrColIdx,
    std::vector<VALUE_TYPE> &csrVal)
{
  csrRowPtr.resize(n + 1, 0);
  csrColIdx.clear();
  csrVal.clear();

  for (int row = 0; row < n; row++) {
    int off_diag = std::min(nnz_per_row - 1, row);

    // Select off_diag distinct columns uniformly from [0, row-1]
    std::vector<int> pool(row);
    std::iota(pool.begin(), pool.end(), 0);
    for (int k = 0; k < off_diag; k++) {
      int idx = k + rand() % (row - k);
      std::swap(pool[k], pool[idx]);
    }
    std::vector<int> cols(pool.begin(), pool.begin() + off_diag);
    std::sort(cols.begin(), cols.end());

    for (int c : cols) {
      csrColIdx.push_back(c);
      csrVal.push_back((VALUE_TYPE)(rand() % 9 + 1) / 10.0);
    }
    // Diagonal: value in [2, 10] to keep the system well-conditioned
    csrColIdx.push_back(row);
    csrVal.push_back((VALUE_TYPE)(rand() % 9 + 2));

    csrRowPtr[row + 1] = (int)csrColIdx.size();
  }
}

// Sequential forward substitution (reference).
static void forward_sub_ref(
    int n,
    const std::vector<int>        &csrRowPtr,
    const std::vector<int>        &csrColIdx,
    const std::vector<VALUE_TYPE> &csrVal,
    const std::vector<VALUE_TYPE> &b,
    std::vector<VALUE_TYPE>       &x)
{
  for (int row = 0; row < n; row++) {
    VALUE_TYPE xi = b[row];
    int start = csrRowPtr[row];
    int end   = csrRowPtr[row + 1];
    for (int j = start; j < end - 1; j++)        // skip diagonal (last entry)
      xi -= csrVal[j] * x[csrColIdx[j]];
    x[row] = xi / csrVal[end - 1];               // divide by diagonal
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: ./main <n> <nnz_per_row> <repeat>\n");
    return 1;
  }

  int n           = atoi(argv[1]);
  int nnz_per_row = atoi(argv[2]);
  int repeat      = atoi(argv[3]);

  srand(42);

  // ---- Build matrix ----
  std::vector<int>        csrRowPtr, csrColIdx;
  std::vector<VALUE_TYPE> csrVal;
  generate_lower_triangular(n, nnz_per_row, csrRowPtr, csrColIdx, csrVal);
  int nnz = csrRowPtr[n];

  // ---- Build rhs and reference solution ----
  std::vector<VALUE_TYPE> x_ref(n), b(n, 0.0);
  for (int i = 0; i < n; i++) x_ref[i] = (VALUE_TYPE)(rand() % 10 + 1);
  // b = L * x_ref
  for (int row = 0; row < n; row++)
    for (int j = csrRowPtr[row]; j < csrRowPtr[row + 1]; j++)
      b[row] += csrVal[j] * x_ref[csrColIdx[j]];

  // ---- Level scheduling (host preprocessing) ----
  // level[row] = 0 if no off-diagonal deps, else 1 + max(level[dep])
  std::vector<int> level(n, 0);
  for (int row = 0; row < n; row++) {
    int start = csrRowPtr[row];
    int end   = csrRowPtr[row + 1] - 1;           // exclude diagonal
    for (int j = start; j < end; j++) {
      int col = csrColIdx[j];                      // col < row (lower triangular)
      level[row] = std::max(level[row], level[col] + 1);
    }
  }
  int maxLevel = *std::max_element(level.begin(), level.end());

  // Group rows by level
  std::vector<std::vector<int>> levelRows(maxLevel + 1);
  for (int row = 0; row < n; row++)
    levelRows[level[row]].push_back(row);

  Kokkos::initialize(argc, argv);
  {
    // ---- Create device views ----
    Kokkos::View<int*>        d_csrRowPtr("csrRowPtr", n + 1);
    Kokkos::View<int*>        d_csrColIdx("csrColIdx", nnz);
    Kokkos::View<VALUE_TYPE*> d_csrVal   ("csrVal",    nnz);
    Kokkos::View<VALUE_TYPE*> d_b        ("b",         n);
    Kokkos::View<VALUE_TYPE*> d_x        ("x",         n);

    {
      auto h_rp  = Kokkos::create_mirror_view(d_csrRowPtr);
      auto h_ci  = Kokkos::create_mirror_view(d_csrColIdx);
      auto h_cv  = Kokkos::create_mirror_view(d_csrVal);
      auto h_b   = Kokkos::create_mirror_view(d_b);
      for (int i = 0; i <= n;   i++) h_rp(i) = csrRowPtr[i];
      for (int i = 0; i < nnz;  i++) h_ci(i) = csrColIdx[i];
      for (int i = 0; i < nnz;  i++) h_cv(i) = csrVal[i];
      for (int i = 0; i < n;    i++) h_b(i)  = b[i];
      Kokkos::deep_copy(d_csrRowPtr, h_rp);
      Kokkos::deep_copy(d_csrColIdx, h_ci);
      Kokkos::deep_copy(d_csrVal,    h_cv);
      Kokkos::deep_copy(d_b,         h_b);
    }

    // Pre-build device views for each level's row list
    std::vector<Kokkos::View<int*>> d_levelRows(maxLevel + 1);
    for (int l = 0; l <= maxLevel; l++) {
      int lsize = (int)levelRows[l].size();
      d_levelRows[l] = Kokkos::View<int*>("lrows", lsize);
      auto h_lr = Kokkos::create_mirror_view(d_levelRows[l]);
      for (int k = 0; k < lsize; k++) h_lr(k) = levelRows[l][k];
      Kokkos::deep_copy(d_levelRows[l], h_lr);
    }

    // ---- Timed solve (repeated) ----
    double total_ns = 0.0;
    for (int rep = 0; rep < repeat; rep++) {
      Kokkos::deep_copy(d_x, (VALUE_TYPE)0.0);

      auto t_start = std::chrono::steady_clock::now();

      for (int l = 0; l <= maxLevel; l++) {
        int lsize = (int)levelRows[l].size();
        auto lrows = d_levelRows[l];            // capture view handle by value

        Kokkos::parallel_for("sptrsv_level", lsize, KOKKOS_LAMBDA(const int k) {
          int row = lrows(k);
          VALUE_TYPE xi = d_b(row);
          int start = d_csrRowPtr(row);
          int end   = d_csrRowPtr(row + 1);
          for (int j = start; j < end - 1; j++)
            xi -= d_csrVal(j) * d_x(d_csrColIdx(j));
          d_x(row) = xi / d_csrVal(end - 1);
        });
        Kokkos::fence();
      }

      auto t_end = std::chrono::steady_clock::now();
      total_ns += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_end - t_start).count();
    }

    printf("Average kernel execution time: %f (us)\n",
           (total_ns * 1e-3) / repeat);

    // ---- Verify ----
    auto h_x = Kokkos::create_mirror_view(d_x);
    Kokkos::deep_copy(h_x, d_x);

    double ref = 0.0, res = 0.0;
    for (int i = 0; i < n; i++) {
      ref += std::abs(x_ref[i]);
      res += std::abs(h_x(i) - x_ref[i]);
    }
    res = (ref > 0.0) ? res / ref : res;

    printf("|x-xref|/|xref| = %8.2e\n", res);
    printf("%s\n", (res < 1e-4) ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
