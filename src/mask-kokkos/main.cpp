#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <Kokkos_Core.hpp>

// ─── CPU reference implementations ──────────────────────────────────────────

template <typename T>
void sequenceMaskKernel_cpu(int N, int M, int B, const T* in,
                             const int* seq_lengths, T fill_val, T* out) {
  for (int i = 0; i < B; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < M; k++) {
        int ind = N * M * i + M * j + k;
        out[ind] = (k >= seq_lengths[j] ? fill_val : in[ind]);
      }
}

template <typename T>
void windowMaskKernel_cpu(int N, int M, int B, const T* in,
                           const int* window_centers, int radius,
                           T fill_val, T* out) {
  for (int i = 0; i < B; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < M; k++) {
        int ind = N * M * i + M * j + k;
        out[ind] = (k < window_centers[j] - radius ||
                    k > window_centers[j] + radius ? fill_val : in[ind]);
      }
}

template <typename T>
void upperMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  for (int i = 0; i < B; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < M; k++) {
        int ind = N * M * i + M * j + k;
        out[ind] = (k > j ? fill_val : in[ind]);
      }
}

template <typename T>
void lowerMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  for (int i = 0; i < B; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < M; k++) {
        int ind = N * M * i + M * j + k;
        out[ind] = (k < j ? fill_val : in[ind]);
      }
}

template <typename T>
void upperDiagMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  for (int i = 0; i < B; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < M; k++) {
        int ind = N * M * i + M * j + k;
        out[ind] = (k >= j ? fill_val : in[ind]);
      }
}

template <typename T>
void lowerDiagMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  for (int i = 0; i < B; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < M; k++) {
        int ind = N * M * i + M * j + k;
        out[ind] = (k <= j ? fill_val : in[ind]);
      }
}

// ─── Kokkos device kernels ───────────────────────────────────────────────────

template <typename T>
void sequenceMaskKernel(int N, int M, int B,
                         Kokkos::View<const T*> d_in,
                         Kokkos::View<const int*> d_seq,
                         T fill_val,
                         Kokkos::View<T*> d_out) {
  if (B >= 0) {
    int total = B * N * M;
    Kokkos::parallel_for("sequenceMask_B", total, KOKKOS_LAMBDA(int index) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      d_out(ind) = (k >= d_seq(j) ? fill_val : d_in(ind));
    });
  } else {
    int total = N * M;
    Kokkos::parallel_for("sequenceMask", total, KOKKOS_LAMBDA(int index) {
      int i = index / M;
      int j = index % M;
      d_out(index) = (j >= d_seq(i) ? fill_val : d_in(index));
    });
  }
}

template <typename T>
void windowMaskKernel(int N, int M, int B,
                       Kokkos::View<const T*> d_in,
                       Kokkos::View<const int*> d_win,
                       int radius, T fill_val,
                       Kokkos::View<T*> d_out) {
  if (B >= 0) {
    int total = B * N * M;
    Kokkos::parallel_for("windowMask_B", total, KOKKOS_LAMBDA(int index) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      d_out(ind) = (k < d_win(j) - radius || k > d_win(j) + radius
                    ? fill_val : d_in(ind));
    });
  } else {
    int total = N * M;
    Kokkos::parallel_for("windowMask", total, KOKKOS_LAMBDA(int index) {
      int i = index / M;
      int j = index % M;
      d_out(index) = (j < d_win(i) - radius || j > d_win(i) + radius
                      ? fill_val : d_in(index));
    });
  }
}

template <typename T>
void upperMaskKernel(int N, int M, int B,
                      Kokkos::View<const T*> d_in, T fill_val,
                      Kokkos::View<T*> d_out) {
  if (B >= 0) {
    int total = B * N * M;
    Kokkos::parallel_for("upperMask_B", total, KOKKOS_LAMBDA(int index) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      d_out(ind) = (k > j ? fill_val : d_in(ind));
    });
  } else {
    int total = N * M;
    Kokkos::parallel_for("upperMask", total, KOKKOS_LAMBDA(int index) {
      int i = index / M;
      int j = index % M;
      d_out(index) = (j > i ? fill_val : d_in(index));
    });
  }
}

template <typename T>
void lowerMaskKernel(int N, int M, int B,
                      Kokkos::View<const T*> d_in, T fill_val,
                      Kokkos::View<T*> d_out) {
  if (B >= 0) {
    int total = B * N * M;
    Kokkos::parallel_for("lowerMask_B", total, KOKKOS_LAMBDA(int index) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      d_out(ind) = (k < j ? fill_val : d_in(ind));
    });
  } else {
    int total = N * M;
    Kokkos::parallel_for("lowerMask", total, KOKKOS_LAMBDA(int index) {
      int i = index / M;
      int j = index % M;
      d_out(index) = (j < i ? fill_val : d_in(index));
    });
  }
}

template <typename T>
void upperDiagMaskKernel(int N, int M, int B,
                          Kokkos::View<const T*> d_in, T fill_val,
                          Kokkos::View<T*> d_out) {
  if (B >= 0) {
    int total = B * N * M;
    Kokkos::parallel_for("upperDiagMask_B", total, KOKKOS_LAMBDA(int index) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      d_out(ind) = (k >= j ? fill_val : d_in(ind));
    });
  } else {
    int total = N * M;
    Kokkos::parallel_for("upperDiagMask", total, KOKKOS_LAMBDA(int index) {
      int i = index / M;
      int j = index % M;
      d_out(index) = (j >= i ? fill_val : d_in(index));
    });
  }
}

template <typename T>
void lowerDiagMaskKernel(int N, int M, int B,
                          Kokkos::View<const T*> d_in, T fill_val,
                          Kokkos::View<T*> d_out) {
  if (B >= 0) {
    int total = B * N * M;
    Kokkos::parallel_for("lowerDiagMask_B", total, KOKKOS_LAMBDA(int index) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      d_out(ind) = (k <= j ? fill_val : d_in(ind));
    });
  } else {
    int total = N * M;
    Kokkos::parallel_for("lowerDiagMask", total, KOKKOS_LAMBDA(int index) {
      int i = index / M;
      int j = index % M;
      d_out(index) = (j <= i ? fill_val : d_in(index));
    });
  }
}

// ─── Verification helper ─────────────────────────────────────────────────────

template <typename T>
void print_mask_ratio(Kokkos::View<T*> d_out, const T* out_ref,
                      T fill_val, int data_size) {
  auto h_out = Kokkos::create_mirror_view(d_out);
  Kokkos::deep_copy(h_out, d_out);

  int error = memcmp(h_out.data(), out_ref, data_size * sizeof(T));
  int cnt_fill = 0;
  for (int i = 0; i < data_size; i++)
    if (h_out(i) == fill_val) cnt_fill++;
  printf("%s, Mask ratio: %f\n", (error ? "FAIL" : "PASS"),
         (float)cnt_fill / data_size);
}

// ─── Evaluation driver ───────────────────────────────────────────────────────

template <typename T>
void eval_mask(const int M, const int N, const int B, const int repeat) {
  const T fill_val = -1;
  const int radius = M / 4;
  int batch_dim = (B <= 0) ? 1 : B;

  printf("\nM = %d, N = %d, B = %d\n", M, N, batch_dim);

  int data_size  = N * M * batch_dim;
  int window_size = N;
  int seq_len    = N;

  T   *h_in      = (T*)   malloc(data_size   * sizeof(T));
  T   *out_ref   = (T*)   malloc(data_size   * sizeof(T));
  int *h_seq_len = (int*) malloc(seq_len     * sizeof(int));
  int *h_window  = (int*) malloc(window_size * sizeof(int));

  srand(123);
  for (int i = 0; i < seq_len;    i++) h_seq_len[i] = rand() % (M / 2);
  for (int i = 0; i < window_size; i++) h_window[i]  = rand() % M;
  for (int i = 0; i < data_size;  i++) h_in[i]       = rand() % (M * N);

  // Allocate device views and copy input data
  Kokkos::View<T*>   d_in("d_in",   data_size);
  Kokkos::View<T*>   d_out("d_out", data_size);
  Kokkos::View<int*> d_seq("d_seq", seq_len);
  Kokkos::View<int*> d_win("d_win", window_size);

  {
    auto h = Kokkos::create_mirror_view(d_in);
    memcpy(h.data(), h_in, data_size * sizeof(T));
    Kokkos::deep_copy(d_in, h);
  }
  {
    auto h = Kokkos::create_mirror_view(d_seq);
    memcpy(h.data(), h_seq_len, seq_len * sizeof(int));
    Kokkos::deep_copy(d_seq, h);
  }
  {
    auto h = Kokkos::create_mirror_view(d_win);
    memcpy(h.data(), h_window, window_size * sizeof(int));
    Kokkos::deep_copy(d_win, h);
  }

  auto d_in_c  = Kokkos::View<const T*>(d_in);
  auto d_seq_c = Kokkos::View<const int*>(d_seq);
  auto d_win_c = Kokkos::View<const int*>(d_win);

  // ── sequenceMask ──────────────────────────────────────────────────────────
  sequenceMaskKernel_cpu(N, M, batch_dim, h_in, h_seq_len, fill_val, out_ref);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    sequenceMaskKernel(N, M, batch_dim, d_in_c, d_seq_c, fill_val, d_out);
  Kokkos::fence();
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of sequenceMask kernel: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         * 1e-3f / repeat);
  print_mask_ratio(d_out, out_ref, fill_val, data_size);

  // ── windowMask ────────────────────────────────────────────────────────────
  windowMaskKernel_cpu(N, M, batch_dim, h_in, h_window, radius, fill_val, out_ref);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    windowMaskKernel(N, M, batch_dim, d_in_c, d_win_c, radius, fill_val, d_out);
  Kokkos::fence();
  end = std::chrono::steady_clock::now();
  printf("Average execution time of windowMask kernel: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         * 1e-3f / repeat);
  print_mask_ratio(d_out, out_ref, fill_val, data_size);

  // ── upperMask ─────────────────────────────────────────────────────────────
  upperMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    upperMaskKernel(N, M, batch_dim, d_in_c, fill_val, d_out);
  Kokkos::fence();
  end = std::chrono::steady_clock::now();
  printf("Average execution time of upperMask kernel: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         * 1e-3f / repeat);
  print_mask_ratio(d_out, out_ref, fill_val, data_size);

  // ── lowerMask ─────────────────────────────────────────────────────────────
  lowerMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    lowerMaskKernel(N, M, batch_dim, d_in_c, fill_val, d_out);
  Kokkos::fence();
  end = std::chrono::steady_clock::now();
  printf("Average execution time of lowerMask kernel: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         * 1e-3f / repeat);
  print_mask_ratio(d_out, out_ref, fill_val, data_size);

  // ── upperDiagMask ─────────────────────────────────────────────────────────
  upperDiagMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    upperDiagMaskKernel(N, M, batch_dim, d_in_c, fill_val, d_out);
  Kokkos::fence();
  end = std::chrono::steady_clock::now();
  printf("Average execution time of upperDiagMask kernel: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         * 1e-3f / repeat);
  print_mask_ratio(d_out, out_ref, fill_val, data_size);

  // ── lowerDiagMask ─────────────────────────────────────────────────────────
  lowerDiagMaskKernel_cpu(N, M, batch_dim, h_in, fill_val, out_ref);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    lowerDiagMaskKernel(N, M, batch_dim, d_in_c, fill_val, d_out);
  Kokkos::fence();
  end = std::chrono::steady_clock::now();
  printf("Average execution time of lowerDiagMask kernel: %f (us)\n",
         std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
         * 1e-3f / repeat);
  print_mask_ratio(d_out, out_ref, fill_val, data_size);

  free(h_in);
  free(out_ref);
  free(h_window);
  free(h_seq_len);
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <B> <repeat>\n", argv[0]);
    return 1;
  }
  const int M      = atoi(argv[1]);
  const int N      = atoi(argv[2]);
  const int B      = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  Kokkos::initialize(argc, argv);
  eval_mask<int>(M, N, B, repeat);
  Kokkos::finalize();
  return 0;
}
