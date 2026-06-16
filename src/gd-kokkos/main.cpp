#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include "utils.h"

int main(int argc, const char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <path to file> <lambda> <alpha> <repeat>\n", argv[0]);
    return 1;
  }
  const std::string file_path = argv[1];
  const float lambda = atof(argv[2]);
  const float alpha  = atof(argv[3]);
  const int iters    = (int)atof(argv[4]);

  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m;
  const int n = A.n;
  printf("m=%d n=%d nnz=%ld\n", m, n, A.nzmax);

  Kokkos::initialize(argc, (char**)argv);
  {
    Kokkos::View<float*> d_x("x", n);
    Kokkos::View<float*> d_grad("grad", n);
    Kokkos::View<int*>   d_row_ptr("row_ptr",   A.row_ptr.size());
    Kokkos::View<int*>   d_col_index("col_index", A.col_index.size());
    Kokkos::View<float*> d_value("value",      A.values.size());
    Kokkos::View<int*>   d_y_label("y_label",  A.y_label.size());

    // Initialize x to 0, copy sparse matrix data to device
    Kokkos::deep_copy(d_x, 0.f);
    {
      auto hrp = Kokkos::create_mirror_view(d_row_ptr);
      auto hci = Kokkos::create_mirror_view(d_col_index);
      auto hv  = Kokkos::create_mirror_view(d_value);
      auto hy  = Kokkos::create_mirror_view(d_y_label);
      for (size_t i = 0; i < A.row_ptr.size();   i++) hrp[i] = A.row_ptr[i];
      for (size_t i = 0; i < A.col_index.size(); i++) hci[i] = A.col_index[i];
      for (size_t i = 0; i < A.values.size();    i++) hv[i]  = A.values[i];
      for (int    i = 0; i < m;                  i++) hy[i]  = A.y_label[i];
      Kokkos::deep_copy(d_row_ptr,   hrp);
      Kokkos::deep_copy(d_col_index, hci);
      Kokkos::deep_copy(d_value,     hv);
      Kokkos::deep_copy(d_y_label,   hy);
    }

    long long train_start = get_time();

    float obj_val    = 0.f;
    float train_error = 0.f;

    for (int k = 0; k < iters; k++) {

      // Reset gradient
      Kokkos::deep_copy(d_grad, 0.f);

      // Compute gradients via atomic adds
      Kokkos::parallel_for("grad", m,
        KOKKOS_LAMBDA(int i) {
          float xp = 0.f;
          for (int j = d_row_ptr[i]; j < d_row_ptr[i+1]; j++)
            xp += d_value[j] * d_x[d_col_index[j]];
          float accum = expf(-(float)d_y_label[i] * xp);
          accum = accum / (1.f + accum);
          for (int j = d_row_ptr[i]; j < d_row_ptr[i+1]; j++) {
            float temp = -accum * d_value[j] * (float)d_y_label[i];
            Kokkos::atomic_add(&d_grad[d_col_index[j]], temp);
          }
        });
      Kokkos::fence();

      // Compute objective value
      float total_obj_val = 0.f;
      Kokkos::parallel_reduce("obj", m,
        KOKKOS_LAMBDA(int i, float& obj) {
          float xp = 0.f;
          for (int j = d_row_ptr[i]; j < d_row_ptr[i+1]; j++)
            xp += d_value[j] * d_x[d_col_index[j]];
          obj += logf(1.f + expf(-(float)d_y_label[i] * xp));
        }, total_obj_val);

      // Correct predictions count
      int correct = 0;
      Kokkos::parallel_reduce("correct", m,
        KOKKOS_LAMBDA(int i, int& cor) {
          float xp = 0.f;
          for (int j = d_row_ptr[i]; j < d_row_ptr[i+1]; j++)
            xp += d_value[j] * d_x[d_col_index[j]];
          float pred = 1.f / (1.f + expf(-xp));
          int pred_label = (pred >= 0.5f) ? 1 : -1;
          if (d_y_label[i] == pred_label) cor++;
        }, correct);

      // L2 norm of x
      float l2_norm = 0.f;
      Kokkos::parallel_reduce("l2", n,
        KOKKOS_LAMBDA(int i, float& norm) { norm += d_x[i] * d_x[i]; },
        l2_norm);

      obj_val     = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
      train_error = 1.f - (float)correct / (float)m;

      // Update x
      Kokkos::parallel_for("update", n,
        KOKKOS_LAMBDA(int i) {
          float g = d_grad[i] / (float)m + lambda * d_x[i];
          d_x[i] -= alpha * g;
        });
      Kokkos::fence();
    }

    long long train_end = get_time();
    printf("Training time takes %lld(us) for %d iterations\n\n",
           train_end - train_start, iters);
    printf("object value = %f  train_error = %f\n", obj_val, train_error);
  }
  Kokkos::finalize();
  return 0;
}
