/*
 * Kokkos port of reverse2D-cuda benchmark.
 *
 * Reverses a 2D matrix along rows or columns for both row-major and
 * column-major layouts.  Four combinations are benchmarked for each
 * element type (unsigned char, unsigned short, unsigned int, unsigned long).
 *
 * Each thread handles one (srcIdx, dstIdx) pair and writes:
 *   out[dstIdx] = in[srcIdx]
 *   out[srcIdx] = in[dstIdx]
 * effectively swapping the two elements, covering exactly half the matrix.
 *
 * Index calculations (Ratio = 1, no vectorisation):
 *
 *   rowMajor=T, alongRows=T  (reverse rows, row-major)
 *     len = ceil(nrows/2) * ncols
 *     srcRow = idx / ncols,  srcCol = idx % ncols
 *     dstRow = nrows - srcRow - 1,  dstCol = srcCol
 *     srcIdx = srcRow*ncols + srcCol,  dstIdx = dstRow*ncols + dstCol
 *
 *   rowMajor=T, alongRows=F  (reverse columns within each row, row-major)
 *     len = nrows * ceil(ncols/2)
 *     srcRow = idx / ceil(ncols/2),  srcCol = idx % ceil(ncols/2)
 *     dstRow = srcRow,  dstCol = ncols - srcCol - 1
 *     srcIdx = srcRow*ncols + srcCol,  dstIdx = dstRow*ncols + dstCol
 *
 *   rowMajor=F, alongRows=T  (reverse rows within each column, col-major)
 *     len = ceil(nrows/2) * ncols
 *     mod = ceil(nrows/2)
 *     srcRow = idx % mod,  srcCol = idx / mod
 *     dstRow = nrows - srcRow - 1,  dstCol = srcCol
 *     srcIdx = srcCol*nrows + srcRow,  dstIdx = dstCol*nrows + dstRow
 *
 *   rowMajor=F, alongRows=F  (reverse columns, col-major)
 *     len = nrows * ceil(ncols/2)
 *     srcRow = idx % nrows,  srcCol = idx / nrows
 *     dstRow = srcRow,  dstCol = ncols - srcCol - 1
 *     srcIdx = srcCol*nrows + srcRow,  dstIdx = dstCol*nrows + dstRow
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

// ---------------------------------------------------------------------------
// Reverse kernel (single element per thread, no vectorisation)
// ---------------------------------------------------------------------------
template <typename T>
static long reverseKokkos(Kokkos::View<T*> d_out,
                           Kokkos::View<const T*> d_in,
                           int nrows, int ncols,
                           bool rowMajor, bool alongRows)
{
  const int half_rows = (nrows + 1) / 2; // ceil(nrows/2)
  const int half_cols = (ncols + 1) / 2; // ceil(ncols/2)
  const int len       = alongRows ? half_rows * ncols : nrows * half_cols;

  Kokkos::fence();
  auto t0 = std::chrono::steady_clock::now();

  if (rowMajor && alongRows) {
    // Reverse rows in row-major layout
    Kokkos::parallel_for(
        "reverse_rm_ar",
        Kokkos::RangePolicy<>(0, len),
        KOKKOS_LAMBDA(int idx) {
          const int srcRow = idx / ncols;
          const int srcCol = idx % ncols;
          const int dstRow = nrows - srcRow - 1;
          const int srcIdx = srcRow * ncols + srcCol;
          const int dstIdx = dstRow * ncols + srcCol;
          const T a = d_in(srcIdx);
          const T b = d_in(dstIdx);
          d_out(dstIdx) = a;
          d_out(srcIdx) = b;
        });

  } else if (rowMajor && !alongRows) {
    // Reverse columns within each row, row-major
    Kokkos::parallel_for(
        "reverse_rm_ac",
        Kokkos::RangePolicy<>(0, len),
        KOKKOS_LAMBDA(int idx) {
          const int srcRow = idx / half_cols;
          const int srcCol = idx % half_cols;
          const int dstCol = ncols - srcCol - 1;
          const int srcIdx = srcRow * ncols + srcCol;
          const int dstIdx = srcRow * ncols + dstCol;
          const T a = d_in(srcIdx);
          const T b = d_in(dstIdx);
          d_out(dstIdx) = a;
          d_out(srcIdx) = b;
        });

  } else if (!rowMajor && alongRows) {
    // Reverse rows within each column, col-major
    Kokkos::parallel_for(
        "reverse_cm_ar",
        Kokkos::RangePolicy<>(0, len),
        KOKKOS_LAMBDA(int idx) {
          const int srcRow = idx % half_rows;
          const int srcCol = idx / half_rows;
          const int dstRow = nrows - srcRow - 1;
          const int srcIdx = srcCol * nrows + srcRow;
          const int dstIdx = srcCol * nrows + dstRow;
          const T a = d_in(srcIdx);
          const T b = d_in(dstIdx);
          d_out(dstIdx) = a;
          d_out(srcIdx) = b;
        });

  } else {
    // !rowMajor && !alongRows: reverse columns, col-major
    Kokkos::parallel_for(
        "reverse_cm_ac",
        Kokkos::RangePolicy<>(0, len),
        KOKKOS_LAMBDA(int idx) {
          const int srcRow = idx % nrows;
          const int srcCol = idx / nrows;
          const int dstCol = ncols - srcCol - 1;
          const int srcIdx = srcCol * nrows + srcRow;
          const int dstIdx = dstCol * nrows + srcRow;
          const T a = d_in(srcIdx);
          const T b = d_in(dstIdx);
          d_out(dstIdx) = a;
          d_out(srcIdx) = b;
        });
  }

  Kokkos::fence();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Evaluate one case
// ---------------------------------------------------------------------------
template <typename T>
static void eval_case(Kokkos::View<T*> d_in,
                      Kokkos::View<T*> d_out,
                      Kokkos::View<T*, Kokkos::HostSpace> h_in,
                      int nrows, int ncols,
                      bool rowMajor, bool alongRows,
                      int repeat)
{
  printf("\nInput matrix is %s-major, reverse along %s\n",
         rowMajor ? "row" : "column",
         alongRows ? "rows" : "columns");

  Kokkos::deep_copy(d_in, h_in);

  long total_ns = 0;
  for (int i = 0; i < repeat; ++i)
    total_ns += reverseKokkos<T>(d_out,
                                  Kokkos::View<const T*>(d_in),
                                  nrows, ncols, rowMajor, alongRows);

  printf("Average kernel execution time: %f (ms)\n",
         (double)total_ns * 1e-6 / repeat);
}

// ---------------------------------------------------------------------------
// Evaluate all four cases for a given element type
// ---------------------------------------------------------------------------
template <typename T>
static void eval(int nrows, int ncols, int repeat)
{
  const size_t matrix_size = (size_t)nrows * ncols;

  Kokkos::View<T*>                  d_in ("d_in",  matrix_size);
  Kokkos::View<T*>                  d_out("d_out", matrix_size);
  Kokkos::View<T*, Kokkos::HostSpace> h_in("h_in", matrix_size);

  std::default_random_engine rng(123);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < matrix_size; ++i)
    h_in(i) = static_cast<T>(dist(rng));

  eval_case(d_in, d_out, h_in, nrows, ncols, /*rowMajor=*/true,  /*alongRows=*/true,  repeat);
  eval_case(d_in, d_out, h_in, nrows, ncols, /*rowMajor=*/true,  /*alongRows=*/false, repeat);
  eval_case(d_in, d_out, h_in, nrows, ncols, /*rowMajor=*/false, /*alongRows=*/true,  repeat);
  eval_case(d_in, d_out, h_in, nrows, ncols, /*rowMajor=*/false, /*alongRows=*/false, repeat);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <nrows> <ncols> <iterations>\n", argv[0]);
    return 1;
  }
  const int nrows  = atoi(argv[1]);
  const int ncols  = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  Kokkos::initialize(argc, argv);
  {
    printf("\nElement size: %zu byte\n", sizeof(unsigned char));
    eval<unsigned char>(nrows, ncols, repeat);

    printf("\nElement size: %zu bytes\n", sizeof(unsigned short));
    eval<unsigned short>(nrows, ncols, repeat);

    printf("\nElement size: %zu bytes\n", sizeof(unsigned int));
    eval<unsigned int>(nrows, ncols, repeat);

    printf("\nElement size: %zu bytes\n", sizeof(unsigned long));
    eval<unsigned long>(nrows, ncols, repeat);
  }
  Kokkos::finalize();
  return 0;
}
