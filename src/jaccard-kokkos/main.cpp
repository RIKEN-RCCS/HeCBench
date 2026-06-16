#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

typedef float vtype;

// ---- fill kernel ----
// weight_j[j] = (j+1)/e (weighted) or 1.0 (unweighted)
// weight_i[j] = 0
template <bool weighted, typename T>
void fill_weights(int e, Kokkos::View<T*> weight_j, Kokkos::View<T*> weight_i) {
  Kokkos::parallel_for("fill_wj", e, KOKKOS_LAMBDA(const int j) {
    weight_j(j) = weighted ? (T)(j + 1) / e : (T)1.0;
  });
  Kokkos::parallel_for("fill_wi", e, KOKKOS_LAMBDA(const int j) {
    weight_i(j) = (T)0.0;
  });
}

// ---- jaccard_row_sum ----
// Unweighted: work[row] = degree of row
// Weighted  : work[row] = sum of weight_j[col] for each col in row's neighbors
template <bool weighted, typename T>
void jaccard_row_sum(int n,
                     Kokkos::View<const int*> csrPtr,
                     Kokkos::View<const int*> csrInd,
                     Kokkos::View<const T*>   weight_j,
                     Kokkos::View<T*>         work) {
  if (weighted) {
    Kokkos::parallel_for("row_sum_w", n, KOKKOS_LAMBDA(const int row) {
      int start = csrPtr(row);
      int end   = csrPtr(row + 1);
      T sum = (T)0.0;
      for (int k = start; k < end; k++)
        sum += weight_j(csrInd(k));
      work(row) = sum;
    });
  } else {
    Kokkos::parallel_for("row_sum_uw", n, KOKKOS_LAMBDA(const int row) {
      work(row) = (T)(csrPtr(row + 1) - csrPtr(row));
    });
  }
}

// ---- jaccard_is ----
// For each row, for each edge j in the row:
//   weight_s[j] = work[row] + work[col]
//   weight_i[j] = number (or sum) of shared neighbors
// Parallelized over rows; inner loops over edges and neighbor search are serial.
template <bool weighted, typename T>
void jaccard_is(int n,
                Kokkos::View<const int*> csrPtr,
                Kokkos::View<const int*> csrInd,
                Kokkos::View<const T*>   weight_j,
                Kokkos::View<const T*>   work,
                Kokkos::View<T*>         weight_i,
                Kokkos::View<T*>         weight_s) {
  Kokkos::parallel_for("jaccard_is", n, KOKKOS_LAMBDA(const int row) {
    for (int j = csrPtr(row); j < csrPtr(row + 1); j++) {
      int col = csrInd(j);
      int Ni  = csrPtr(row + 1) - csrPtr(row);
      int Nj  = csrPtr(col + 1) - csrPtr(col);
      int ref = (Ni < Nj) ? row : col;
      int cur = (Ni < Nj) ? col : row;

      weight_s(j) = work(row) + work(col);

      // Iterate over ref's neighbors; binary-search for each in cur's neighbors.
      // No atomics needed: each edge j is owned exclusively by this row thread.
      for (int i = csrPtr(ref); i < csrPtr(ref + 1); i++) {
        int ref_col = csrInd(i);
        T   ref_val = weighted ? weight_j(ref_col) : (T)1.0;

        int left  = csrPtr(cur);
        int right = csrPtr(cur + 1) - 1;
        while (left <= right) {
          int middle  = (left + right) >> 1;
          int cur_col = csrInd(middle);
          if      (cur_col > ref_col) right = middle - 1;
          else if (cur_col < ref_col) left  = middle + 1;
          else {
            weight_i(j) += ref_val;
            break;
          }
        }
      }
    }
  });
}

// ---- jaccard_jw ----
// weight_j[j] = gamma * csrVal[j] * weight_i[j] / (weight_s[j] - weight_i[j])
template <typename T>
void jaccard_jw(int e, T gamma,
                Kokkos::View<const T*> csrVal,
                Kokkos::View<const T*> weight_i,
                Kokkos::View<const T*> weight_s,
                Kokkos::View<T*>       weight_j) {
  Kokkos::parallel_for("jaccard_jw", e, KOKKOS_LAMBDA(const int j) {
    T Wi = weight_i(j);
    T Ws = weight_s(j);
    weight_j(j) = gamma * csrVal(j) * Wi / (Ws - Wi);
  });
}

// ---- Main driver for one pass (weighted or unweighted) ----
template <bool weighted, typename T>
void jaccard_weight(int iteration, int n, int e,
                    int* csr_ptr, int* csr_ind, T* csr_val) {
  const T gamma = (T)0.46;

  // Device views
  Kokkos::View<int*> d_csrPtr("csrPtr", n + 1);
  Kokkos::View<int*> d_csrInd("csrInd", e);
  Kokkos::View<T*>   d_csrVal("csrVal", e);
  Kokkos::View<T*>   d_weight_j("weight_j", e);
  Kokkos::View<T*>   d_weight_i("weight_i", e);
  Kokkos::View<T*>   d_weight_s("weight_s", e);
  Kokkos::View<T*>   d_work("work", n);

  // Copy CSR data to device
  {
    auto h_ptr = Kokkos::create_mirror_view(d_csrPtr);
    auto h_ind = Kokkos::create_mirror_view(d_csrInd);
    auto h_val = Kokkos::create_mirror_view(d_csrVal);
    for (int i = 0; i <= n; i++) h_ptr(i) = csr_ptr[i];
    for (int i = 0; i < e;  i++) h_ind(i) = csr_ind[i];
    for (int i = 0; i < e;  i++) h_val(i) = csr_val[i];
    Kokkos::deep_copy(d_csrPtr, h_ptr);
    Kokkos::deep_copy(d_csrInd, h_ind);
    Kokkos::deep_copy(d_csrVal, h_val);
  }
  Kokkos::fence();

  auto t_start = std::chrono::steady_clock::now();

  for (int it = 0; it < iteration; it++) {
    fill_weights<weighted, T>(e, d_weight_j, d_weight_i);

    jaccard_row_sum<weighted, T>(n,
      Kokkos::View<const int*>(d_csrPtr),
      Kokkos::View<const int*>(d_csrInd),
      Kokkos::View<const T*>(d_weight_j),
      d_work);

    jaccard_is<weighted, T>(n,
      Kokkos::View<const int*>(d_csrPtr),
      Kokkos::View<const int*>(d_csrInd),
      Kokkos::View<const T*>(d_weight_j),
      Kokkos::View<const T*>(d_work),
      d_weight_i,
      d_weight_s);

    jaccard_jw<T>(e, gamma,
      Kokkos::View<const T*>(d_csrVal),
      Kokkos::View<const T*>(d_weight_i),
      Kokkos::View<const T*>(d_weight_s),
      d_weight_j);

    Kokkos::fence();
  }

  auto t_end = std::chrono::steady_clock::now();
  double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_end - t_start).count();
  printf("Average execution time of kernels: %f (s)\n", (ns * 1e-9) / iteration);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: ./main <numRow> <numCol> <iteration>\n");
    return 1;
  }

  int numRow   = atoi(argv[1]);
  int numCol   = atoi(argv[2]);
  int iteration = atoi(argv[3]);

  srand(2);

  // Build dense random matrix and convert to CSR
  printf("Number of matrix rows and cols: %d %d\n", numRow, numCol);

  std::vector<vtype> csr_val;
  std::vector<int>   csr_ptr = {0};
  std::vector<int>   csr_ind;
  int nnz = 0;

  for (int i = 0; i < numRow; i++) {
    for (int j = 0; j < numCol; j++) {
      vtype v = (vtype)(rand() % 10);
      if (v != (vtype)0) {
        csr_val.push_back(v);
        csr_ind.push_back(j);
        nnz++;
      }
    }
    csr_ptr.push_back(nnz);
  }

  Kokkos::initialize(argc, argv);
  {
    jaccard_weight<true,  vtype>(iteration, numRow, nnz,
                                 csr_ptr.data(), csr_ind.data(), csr_val.data());
    jaccard_weight<false, vtype>(iteration, numRow, nnz,
                                 csr_ptr.data(), csr_ind.data(), csr_val.data());
  }
  Kokkos::finalize();
  return 0;
}
