// Kokkos port of spmv-cuda
// Sparse matrix-vector multiplication (CSR and COO formats)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <chrono>

#define REAL float

static void init_vector(REAL *v, size_t n) {
  srand48(1 << 12);
  for (size_t i = 0; i < n; i++) v[i] = (REAL)drand48();
}

static void init_matrix(REAL *matrix, size_t num_rows, size_t nnz) {
  size_t num_elems = num_rows * num_rows;
  size_t *perm = (size_t*)malloc(num_elems * sizeof(size_t));
  for (size_t i = 0; i < num_elems; i++) perm[i] = i;
  for (size_t i = num_elems; i > 0; i--) {
    size_t a = i-1, b = (size_t)(drand48() * i);
    if (a != b) { auto t = perm[a]; perm[a] = perm[b]; perm[b] = t; }
  }
  for (size_t i = 0; i < num_elems; i++) matrix[i] = 0;
  for (size_t i = 0; i < nnz; i++) matrix[perm[i]] = (REAL)drand48();
  free(perm);
}

static void init_csr(size_t *row_indices, REAL *values, size_t *col_indices,
                     REAL *matrix, size_t num_rows, size_t nnz) {
  row_indices[0] = 0;
  size_t *cnts = (size_t*)calloc(num_rows, sizeof(size_t));
  size_t tmp = 0;
  for (size_t i = 0; i < num_rows; i++)
    for (size_t j = 0; j < num_rows; j++)
      if (matrix[i*num_rows+j] != 0) {
        values[tmp] = matrix[i*num_rows+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (size_t i = 1; i <= num_rows; i++) row_indices[i] = row_indices[i-1] + cnts[i-1];
  free(cnts);
}

static void init_coo(size_t *row_indices, REAL *values, size_t *col_indices,
                     REAL *matrix, size_t num_rows, size_t nnz) {
  size_t tmp = 0;
  for (size_t i = 0; i < num_rows; i++)
    for (size_t j = 0; j < num_rows; j++)
      if (matrix[i*num_rows+j] != 0) {
        row_indices[tmp] = i;
        values[tmp] = matrix[i*num_rows+j];
        col_indices[tmp] = j;
        tmp++;
      }
}

static float check(REAL *A, REAL *B, size_t n) {
  float err = 0;
  for (size_t i = 0; i < n; i++) err += fabsf(A[i] - B[i]);
  return err / n;
}

// Reference: dense matrix-vector multiply
static long reference(int repeat, size_t num_rows, const REAL *x, REAL *matrix, REAL *y) {
  Kokkos::View<REAL*> d_x("d_x", num_rows);
  Kokkos::View<REAL*> d_mat("d_mat", num_rows * num_rows);
  Kokkos::View<REAL*> d_y("d_y", num_rows);
  {
    auto hx = Kokkos::create_mirror_view(d_x);
    auto hm = Kokkos::create_mirror_view(d_mat);
    for (size_t i = 0; i < num_rows; i++) hx(i) = x[i];
    for (size_t i = 0; i < num_rows*num_rows; i++) hm(i) = matrix[i];
    Kokkos::deep_copy(d_x, hx); Kokkos::deep_copy(d_mat, hm);
  }

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("dense_mv", num_rows, KOKKOS_LAMBDA(const size_t i) {
      REAL temp = 0;
      for (size_t j = 0; j < num_rows; j++)
        if (d_mat(i*num_rows+j) != 0) temp += d_mat(i*num_rows+j) * d_x(j);
      d_y(i) = temp;
    });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();

  auto hy = Kokkos::create_mirror_view(d_y);
  Kokkos::deep_copy(hy, d_y);
  for (size_t i = 0; i < num_rows; i++) y[i] = hy(i);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

// CSR SpMV
static long spmv_csr(int repeat, size_t num_rows, const REAL *x, size_t nnz,
                     REAL *matrix, REAL *y) {
  size_t *row_idx=(size_t*)malloc((num_rows+1)*sizeof(size_t));
  size_t *col_idx=(size_t*)malloc(nnz*sizeof(size_t));
  REAL   *vals   =(REAL*)malloc(nnz*sizeof(REAL));
  init_csr(row_idx, vals, col_idx, matrix, num_rows, nnz);

  Kokkos::View<size_t*> d_row("d_row", num_rows+1);
  Kokkos::View<size_t*> d_col("d_col", nnz);
  Kokkos::View<REAL*>   d_val("d_val", nnz);
  Kokkos::View<REAL*>   d_x("d_x", num_rows);
  Kokkos::View<REAL*>   d_y("d_y", num_rows);
  {
    auto hr=Kokkos::create_mirror_view(d_row); for(size_t i=0;i<=num_rows;i++) hr(i)=row_idx[i]; Kokkos::deep_copy(d_row,hr);
    auto hc=Kokkos::create_mirror_view(d_col); for(size_t i=0;i<nnz;i++) hc(i)=col_idx[i]; Kokkos::deep_copy(d_col,hc);
    auto hv=Kokkos::create_mirror_view(d_val); for(size_t i=0;i<nnz;i++) hv(i)=vals[i]; Kokkos::deep_copy(d_val,hv);
    auto hx=Kokkos::create_mirror_view(d_x); for(size_t i=0;i<num_rows;i++) hx(i)=x[i]; Kokkos::deep_copy(d_x,hx);
  }

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();
  for (int r=0;r<repeat;r++) {
    Kokkos::parallel_for("spmv_csr", num_rows, KOKKOS_LAMBDA(const size_t i) {
      REAL temp = 0;
      for (size_t idx=d_row(i); idx<d_row(i+1); idx++) temp += d_val(idx)*d_x(d_col(idx));
      d_y(i) = temp;
    });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();

  auto hy=Kokkos::create_mirror_view(d_y); Kokkos::deep_copy(hy,d_y);
  for(size_t i=0;i<num_rows;i++) y[i]=hy(i);
  free(row_idx); free(col_idx); free(vals);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}

// COO SpMV
static long spmv_coo(int repeat, size_t num_rows, const REAL *x, size_t nnz,
                     REAL *matrix, REAL *y) {
  size_t *row_idx=(size_t*)malloc(nnz*sizeof(size_t));
  size_t *col_idx=(size_t*)malloc(nnz*sizeof(size_t));
  REAL   *vals   =(REAL*)malloc(nnz*sizeof(REAL));
  init_coo(row_idx, vals, col_idx, matrix, num_rows, nnz);

  Kokkos::View<size_t*> d_row("d_row", nnz);
  Kokkos::View<size_t*> d_col("d_col", nnz);
  Kokkos::View<REAL*>   d_val("d_val", nnz);
  Kokkos::View<REAL*>   d_x("d_x", num_rows);
  Kokkos::View<REAL*>   d_y("d_y", num_rows);
  {
    auto hr=Kokkos::create_mirror_view(d_row); for(size_t i=0;i<nnz;i++) hr(i)=row_idx[i]; Kokkos::deep_copy(d_row,hr);
    auto hc=Kokkos::create_mirror_view(d_col); for(size_t i=0;i<nnz;i++) hc(i)=col_idx[i]; Kokkos::deep_copy(d_col,hc);
    auto hv=Kokkos::create_mirror_view(d_val); for(size_t i=0;i<nnz;i++) hv(i)=vals[i]; Kokkos::deep_copy(d_val,hv);
    auto hx=Kokkos::create_mirror_view(d_x); for(size_t i=0;i<num_rows;i++) hx(i)=x[i]; Kokkos::deep_copy(d_x,hx);
  }
  Kokkos::deep_copy(d_y, (REAL)0);

  Kokkos::fence();
  auto start = std::chrono::steady_clock::now();
  for (int r=0;r<repeat;r++) {
    Kokkos::deep_copy(d_y, (REAL)0);
    Kokkos::parallel_for("spmv_coo", nnz, KOKKOS_LAMBDA(const size_t i) {
      Kokkos::atomic_add(&d_y(d_row(i)), d_val(i)*d_x(d_col(i)));
    });
  }
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();

  auto hy=Kokkos::create_mirror_view(d_y); Kokkos::deep_copy(hy,d_y);
  for(size_t i=0;i<num_rows;i++) y[i]=hy(i);
  free(row_idx); free(col_idx); free(vals);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage %s <number of non-zero elements> <number of rows in a square matrix> <repeat>\n", argv[0]);
    return 1;
  }
  size_t nnz = atol(argv[1]);
  size_t num_rows = atol(argv[2]);
  int repeat = atoi(argv[3]);
  assert(nnz > 0 && num_rows > 0);

  size_t num_elems = num_rows * num_rows;
  assert(nnz <= num_elems);

  REAL *values = (REAL*)malloc(nnz * sizeof(REAL));
  REAL *x = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *y_ref = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *y = (REAL*)malloc(num_rows * sizeof(REAL));
  REAL *matrix = (REAL*)malloc(num_elems * sizeof(REAL));

  srand48(1 << 12);
  init_matrix(matrix, num_rows, nnz);
  init_vector(x, num_rows);

  Kokkos::initialize(argc, argv);
  {
    long elapsed = reference(repeat, num_rows, x, matrix, y_ref);

    printf("Number of non-zero elements: %zu\n", nnz);
    printf("Number of rows in a square matrix: %zu\n", num_rows);
    printf("Sparsity: %lf%%\n", (num_elems - nnz) * 1.0 / num_elems * 100.0);

    elapsed = spmv_csr(repeat, num_rows, x, nnz, matrix, y);
    printf("Average kernel (CSR) execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
    printf("Error rate: %f\n", check(y, y_ref, num_rows));

    elapsed = spmv_coo(repeat, num_rows, x, nnz, matrix, y);
    printf("Average kernel (COO) execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
    printf("Error rate: %f\n", check(y, y_ref, num_rows));
  }
  Kokkos::finalize();

  free(values); free(x); free(y); free(y_ref); free(matrix);
  return 0;
}
