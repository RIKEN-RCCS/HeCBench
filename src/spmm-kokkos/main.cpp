// Kokkos port of spmm-cuda
// Sparse-sparse matrix multiply: C(sparse) = A(sparse, CSR) * B(sparse, CSR)

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static void init_matrix(float *matrix, int num_rows, int num_cols, int nnz) {
  int n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  for (int i = n; i > 0; i--) {
    int a = i-1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

static void init_csr(int *row_offsets, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int /*nnz*/) {
  row_offsets[0] = 0;
  int *cnts = (int*)calloc(num_rows, sizeof(int));
  int tmp = 0;
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (int i = 1; i <= num_rows; i++) row_offsets[i] = row_offsets[i-1] + cnts[i-1];
  free(cnts);
}

// Reference sparse*sparse: compute C = A * B, return nnz of C
static int spmm_ref(float *A, float *B, float *&values, int *&offsets, int *&columns,
                    int A_num_cols, int C_num_rows, int C_num_cols) {
  int C_nnz = 0;
  float *C = (float*)malloc(C_num_rows * C_num_cols * sizeof(float));
  for (int i = 0; i < C_num_rows; i++)
    for (int j = 0; j < C_num_cols; j++) {
      double s = 0;
      for (int k = 0; k < A_num_cols; k++) s += A[i*A_num_cols+k] * B[k*C_num_cols+j];
      if (s != 0) C_nnz++;
      C[i*C_num_cols+j] = (float)s;
    }
  values  = (float*)malloc(C_nnz * sizeof(float));
  columns = (int*)malloc(C_nnz * sizeof(int));
  offsets = (int*)malloc((C_num_rows+1) * sizeof(int));
  init_csr(offsets, values, columns, C, C_num_rows, C_num_cols, C_nnz);
  free(C);
  return C_nnz;
}

// SpSpMM kernel: C(m,n)[CSR] = A(m,k)[CSR] * B(k,n)[CSR]
// We compute row by row; result stored in dense intermediate
void spmm_sparse(Kokkos::View<int*> dA_off, Kokkos::View<int*> dA_col, Kokkos::View<float*> dA_val,
                 Kokkos::View<int*> dB_off, Kokkos::View<int*> dB_col, Kokkos::View<float*> dB_val,
                 Kokkos::View<int*> dC_off, Kokkos::View<int*> dC_col, Kokkos::View<float*> dC_val,
                 int m, int k, int n, int c_nnz)
{
  // We'll compute via an intermediate dense row buffer per row
  Kokkos::View<float*> dense_row("dense_row", m * n);
  Kokkos::deep_copy(dense_row, 0.0f);

  Kokkos::parallel_for("spmm_sparse", m, KOKKOS_LAMBDA(const int row) {
    float* row_buf = &dense_row(row * n);
    for (int idxA = dA_off(row); idxA < dA_off(row+1); idxA++) {
      int kk = dA_col(idxA);
      float va = dA_val(idxA);
      for (int idxB = dB_off(kk); idxB < dB_off(kk+1); idxB++) {
        int j = dB_col(idxB);
        row_buf[j] += va * dB_val(idxB);
      }
    }
  });

  // Convert dense_row to CSR C
  Kokkos::parallel_for("count_C_nnz", m, KOKKOS_LAMBDA(const int row) {
    int cnt = 0;
    for (int j = 0; j < n; j++) if (dense_row(row*n+j) != 0.0f) cnt++;
    dC_off(row + 1) = cnt;
  });
  Kokkos::fence();

  // Prefix sum on host
  auto h_off = Kokkos::create_mirror_view(dC_off);
  Kokkos::deep_copy(h_off, dC_off);
  h_off(0) = 0;
  for (int i = 0; i < m; i++) h_off(i+1) += h_off(i);
  Kokkos::deep_copy(dC_off, h_off);

  Kokkos::parallel_for("fill_C", m, KOKKOS_LAMBDA(const int row) {
    int pos = dC_off(row);
    for (int j = 0; j < n; j++) {
      float v = dense_row(row*n+j);
      if (v != 0.0f) {
        dC_col(pos) = j;
        dC_val(pos) = v;
        pos++;
      }
    }
  });
}

int main(int argc, char *argv[]) {
  if (argc != 8) {
    printf("Sparse matrix * sparse matrix = sparse matrix (SpSpMM)\n");
    printf("Usage %s <M> <K> <N> <A_nnz> <B_nnz> <repeat> <verify>\n", argv[0]);
    return 1;
  }
  int m=atoi(argv[1]), k=atoi(argv[2]), n=atoi(argv[3]);
  int a_nnz=atoi(argv[4]), b_nnz=atoi(argv[5]);
  int repeat=atoi(argv[6]), verify=atoi(argv[7]);

  float *hA = (float*)malloc(m*k*sizeof(float));
  float *hB = (float*)malloc(k*n*sizeof(float));
  float *hA_v=(float*)malloc(a_nnz*sizeof(float)); int *hA_c=(int*)malloc(a_nnz*sizeof(int)); int *hA_o=(int*)malloc((m+1)*sizeof(int));
  float *hB_v=(float*)malloc(b_nnz*sizeof(float)); int *hB_c=(int*)malloc(b_nnz*sizeof(int)); int *hB_o=(int*)malloc((k+1)*sizeof(int));

  init_matrix(hA, m, k, a_nnz); init_csr(hA_o, hA_v, hA_c, hA, m, k, a_nnz);
  init_matrix(hB, k, n, b_nnz); init_csr(hB_o, hB_v, hB_c, hB, k, n, b_nnz);

  // Pre-compute C nnz for allocation
  float *ref_v=nullptr; int *ref_c=nullptr, *ref_o=nullptr;
  int c_nnz = spmm_ref(hA, hB, ref_v, ref_o, ref_c, k, m, n);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<int*> dA_o("dA_o",m+1), dA_c("dA_c",a_nnz); Kokkos::View<float*> dA_v("dA_v",a_nnz);
    Kokkos::View<int*> dB_o("dB_o",k+1), dB_c("dB_c",b_nnz); Kokkos::View<float*> dB_v("dB_v",b_nnz);
    Kokkos::View<int*> dC_o("dC_o",m+1), dC_c("dC_c",c_nnz); Kokkos::View<float*> dC_v("dC_v",c_nnz);

    { auto h=Kokkos::create_mirror_view(dA_o); for(int i=0;i<=m;i++) h(i)=hA_o[i]; Kokkos::deep_copy(dA_o,h); }
    { auto h=Kokkos::create_mirror_view(dA_c); for(int i=0;i<a_nnz;i++) h(i)=hA_c[i]; Kokkos::deep_copy(dA_c,h); }
    { auto h=Kokkos::create_mirror_view(dA_v); for(int i=0;i<a_nnz;i++) h(i)=hA_v[i]; Kokkos::deep_copy(dA_v,h); }
    { auto h=Kokkos::create_mirror_view(dB_o); for(int i=0;i<=k;i++) h(i)=hB_o[i]; Kokkos::deep_copy(dB_o,h); }
    { auto h=Kokkos::create_mirror_view(dB_c); for(int i=0;i<b_nnz;i++) h(i)=hB_c[i]; Kokkos::deep_copy(dB_c,h); }
    { auto h=Kokkos::create_mirror_view(dB_v); for(int i=0;i<b_nnz;i++) h(i)=hB_v[i]; Kokkos::deep_copy(dB_v,h); }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();
    for (int i=0;i<repeat;i++)
      spmm_sparse(dA_o,dA_c,dA_v, dB_o,dB_c,dB_v, dC_o,dC_c,dC_v, m,k,n,c_nnz);
    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    printf("Average execution time of SPMM compute: %f (us)\n",
           (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-3f)/repeat);

    if (verify) {
      auto h_o=Kokkos::create_mirror_view(dC_o); Kokkos::deep_copy(h_o,dC_o);
      auto h_c=Kokkos::create_mirror_view(dC_c); Kokkos::deep_copy(h_c,dC_c);
      auto h_v=Kokkos::create_mirror_view(dC_v); Kokkos::deep_copy(h_v,dC_v);
      int correct=1;
      if (h_o(m) != c_nnz) { correct=0; printf("C nnz mismatch %d vs %d\n", h_o(m), c_nnz); }
      if (correct) for (int i=0;i<c_nnz;i++) if (fabsf(h_v(i)-ref_v[i])>1e-2f||h_c(i)!=ref_c[i]) { correct=0; break; }
      printf("%s\n", correct ? "spmm_example test PASSED" : "spmm_example test FAILED: wrong result");
    }
  }
  Kokkos::finalize();

  free(hA); free(hB); free(hA_v); free(hA_c); free(hA_o);
  free(hB_v); free(hB_c); free(hB_o); free(ref_v); free(ref_c); free(ref_o);
  return 0;
}
