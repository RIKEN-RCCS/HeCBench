// Port of unfold-cuda benchmark to Kokkos
// Implements PyTorch unfold backward pass

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelem  = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int64_t size                    = 2;
  const int64_t step                    = 1;
  const int64_t grad_in_dim_stride      = 1;
  const int64_t grad_in_last_dim_stride = 1;
  const int64_t grad_in_dim_size        = nelem;
  const int64_t h_idx_dim               = 0;

  using scalar_t = int;

  std::vector<scalar_t> h_grad_in(grad_in_dim_size);
  srand(123);
  for (int i = 0; i < grad_in_dim_size; i++) h_grad_in[i] = rand() % 256;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<scalar_t*> d_grad_in ("grad_in",  grad_in_dim_size);
    Kokkos::View<scalar_t*> d_grad_out("grad_out", grad_in_dim_size);
    Kokkos::View<int64_t*>  d_idx_dim ("idx_dim",  1);

    {
      auto h = Kokkos::create_mirror_view(d_grad_in);
      for (int i = 0; i < grad_in_dim_size; i++) h(i) = h_grad_in[i];
      Kokkos::deep_copy(d_grad_in, h);
      auto hi = Kokkos::create_mirror_view(d_idx_dim);
      hi(0) = h_idx_dim;
      Kokkos::deep_copy(d_idx_dim, hi);
    }
    // Zero out output
    Kokkos::deep_copy(d_grad_out, 0);

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::parallel_for("unfold_backward", grad_in_dim_size,
        KOKKOS_LAMBDA(const int i) {
          int64_t idx_dim = d_idx_dim(0);

          int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
          if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size))
            ++left_fold_idx;

          int64_t right_fold_idx = idx_dim / step;
          if (right_fold_idx >= grad_in_dim_size) right_fold_idx = grad_in_dim_size - 1;

          scalar_t acc = 0;
          for (int64_t fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
            int64_t idx_last_dim = idx_dim - fold_idx * step;
            acc += d_grad_in(fold_idx * grad_in_dim_stride +
                             idx_last_dim * grad_in_last_dim_stride);
          }
          d_grad_out(i) += acc;
        });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of unfold backward kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    auto h_out = Kokkos::create_mirror_view(d_grad_out);
    Kokkos::deep_copy(h_out, d_grad_out);

    bool ok = true;
    for (int i = 0; i < grad_in_dim_size; i++) {
      if (repeat * h_grad_in[i] != h_out(i)) { ok = false; break; }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
