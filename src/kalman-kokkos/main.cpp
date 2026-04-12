/*
 * Kokkos port of kalman-sycl benchmark.
 * Original copyright (c) 2019-2021, NVIDIA CORPORATION (Apache 2.0).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <Kokkos_Core.hpp>

template <int n>
KOKKOS_INLINE_FUNCTION
void Mv_l(const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = sum;
  }
}

template <int n>
KOKKOS_INLINE_FUNCTION
void Mv_l(double alpha, const double* A, const double* v, double* out)
{
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i + j * n] * v[j];
    }
    out[i] = alpha * sum;
  }
}

template <int n, bool aT = false, bool bT = false>
KOKKOS_INLINE_FUNCTION
void MM_l(const double* A, const double* B, double* out)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double Aik = aT ? A[k + i * n] : A[i + k * n];
        double Bkj = bT ? B[j + k * n] : B[k + j * n];
        sum += Aik * Bkj;
      }
      out[i + j * n] = sum;
    }
  }
}

template <int rd>
KOKKOS_INLINE_FUNCTION
void kalman_kernel(
  int bid,
  const double* __restrict__ ys,
  int nobs,
  const double* __restrict__ T,
  const double* __restrict__ Z,
  const double* __restrict__ RQR,
  const double* __restrict__ P,
  const double* __restrict__ alpha,
  bool intercept,
  const double* __restrict__ d_mu,
  int batch_size,
  double* __restrict__ vs,
  double* __restrict__ Fs,
  double* __restrict__ sum_logFs,
  int n_diff,
  int fc_steps,
  double* __restrict__ d_fc,
  bool conf_int,
  double* d_F_fc)
{
  constexpr int rd2 = rd * rd;
  double l_RQR[rd2];
  double l_T[rd2];
  double l_Z[rd];
  double l_P[rd2];
  double l_alpha[rd];
  double l_K[rd];
  double l_tmp[rd2];
  double l_TP[rd2];

  if (bid < batch_size) {
    int b_rd_offset  = bid * rd;
    int b_rd2_offset = bid * rd2;
    for (int i = 0; i < rd2; i++) {
      l_RQR[i] = RQR[b_rd2_offset + i];
      l_T[i]   = T[b_rd2_offset + i];
      l_P[i]   = P[b_rd2_offset + i];
    }
    for (int i = 0; i < rd; i++) {
      if (n_diff > 0) l_Z[i] = Z[b_rd_offset + i];
      l_alpha[i] = alpha[b_rd_offset + i];
    }

    double b_sum_logFs = 0.0;
    const double* b_ys = ys + bid * nobs;
    double* b_vs       = vs + bid * nobs;
    double* b_Fs       = Fs + bid * nobs;

    double mu = intercept ? d_mu[bid] : 0.0;

    for (int it = 0; it < nobs; it++) {
      double vs_it = b_ys[it];
      if (n_diff == 0)
        vs_it -= l_alpha[0];
      else {
        for (int i = 0; i < rd; i++) {
          vs_it -= l_alpha[i] * l_Z[i];
        }
      }
      b_vs[it] = vs_it;

      double _Fs;
      if (n_diff == 0)
        _Fs = l_P[0];
      else {
        _Fs = 0.0;
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
          }
        }
      }
      b_Fs[it] = _Fs;
      if (it >= n_diff) b_sum_logFs += Kokkos::log(_Fs);

      MM_l<rd>(l_T, l_P, l_TP);
      double _1_Fs = 1.0 / _Fs;
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_K[i] = _1_Fs * l_TP[i];
        }
      } else
        Mv_l<rd>(_1_Fs, l_TP, l_Z, l_K);

      Mv_l<rd>(l_T, l_alpha, l_tmp);
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i] + l_K[i] * vs_it;
      }
      l_alpha[n_diff] += mu;

      for (int i = 0; i < rd2; i++) {
        l_tmp[i] = l_T[i];
      }
      if (n_diff == 0) {
        for (int i = 0; i < rd; i++) {
          l_tmp[i] -= l_K[i];
        }
      } else {
        for (int i = 0; i < rd; i++) {
          for (int j = 0; j < rd; j++) {
            l_tmp[j * rd + i] -= l_K[i] * l_Z[j];
          }
        }
      }

      MM_l<rd, false, true>(l_TP, l_tmp, l_P);
      for (int i = 0; i < rd2; i++) {
        l_P[i] += l_RQR[i];
      }
    }
    sum_logFs[bid] = b_sum_logFs;

    double* b_fc   = fc_steps ? d_fc + bid * fc_steps : nullptr;
    double* b_F_fc = conf_int ? d_F_fc + bid * fc_steps : nullptr;
    for (int it = 0; it < fc_steps; it++) {
      if (n_diff == 0)
        b_fc[it] = l_alpha[0];
      else {
        double pred = 0.0;
        for (int i = 0; i < rd; i++) {
          pred += l_alpha[i] * l_Z[i];
        }
        b_fc[it] = pred;
      }

      Mv_l<rd>(l_T, l_alpha, l_tmp);
      for (int i = 0; i < rd; i++) {
        l_alpha[i] = l_tmp[i];
      }
      l_alpha[n_diff] += mu;

      if (conf_int) {
        if (n_diff == 0)
          b_F_fc[it] = l_P[0];
        else {
          double _Fs = 0.0;
          for (int i = 0; i < rd; i++) {
            for (int j = 0; j < rd; j++) {
              _Fs += l_P[j * rd + i] * l_Z[i] * l_Z[j];
            }
          }
          b_F_fc[it] = _Fs;
        }

        MM_l<rd>(l_T, l_P, l_TP);
        MM_l<rd, false, true>(l_TP, l_T, l_P);
        for (int i = 0; i < rd2; i++) {
          l_P[i] += l_RQR[i];
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 5) {
      printf("Usage: %s <#series> <#observations> <forecast steps> <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }

    const int nseries    = atoi(argv[1]);
    const int nobs       = atoi(argv[2]);
    const int fc_steps   = atoi(argv[3]);
    const int repeat     = atoi(argv[4]);

    const int rd  = 8;
    const int rd2 = rd * rd;
    const int batch_size = nseries;

    const int rd2_word  = nseries * rd2;
    const int rd_word   = nseries * rd;
    const int nobs_word = nseries * nobs;
    const int ns_word   = nseries;
    const int fc_word   = fc_steps * nseries;

    srand(123);

    // Device Views
    Kokkos::View<double*> d_RQR("d_RQR", rd2_word);
    Kokkos::View<double*> d_T("d_T", rd2_word);
    Kokkos::View<double*> d_P("d_P", rd2_word);
    Kokkos::View<double*> d_Z("d_Z", rd_word);
    Kokkos::View<double*> d_alpha("d_alpha", rd_word);
    Kokkos::View<double*> d_ys("d_ys", nobs_word);
    Kokkos::View<double*> d_mu("d_mu", ns_word);
    Kokkos::View<double*> d_vs("d_vs", nobs_word);
    Kokkos::View<double*> d_Fs("d_Fs", nobs_word);
    Kokkos::View<double*> d_sum_logFs("d_sum_logFs", ns_word);
    Kokkos::View<double*> d_fc("d_fc", fc_word);
    Kokkos::View<double*> d_F_fc("d_F_fc", fc_word);

    // Host mirrors
    auto h_RQR   = Kokkos::create_mirror_view(d_RQR);
    auto h_T     = Kokkos::create_mirror_view(d_T);
    auto h_P     = Kokkos::create_mirror_view(d_P);
    auto h_Z     = Kokkos::create_mirror_view(d_Z);
    auto h_alpha = Kokkos::create_mirror_view(d_alpha);
    auto h_ys    = Kokkos::create_mirror_view(d_ys);
    auto h_mu    = Kokkos::create_mirror_view(d_mu);
    auto h_F_fc  = Kokkos::create_mirror_view(d_F_fc);

    for (int i = 0; i < rd2_word; i++) h_RQR(i)   = (double)rand() / RAND_MAX;
    for (int i = 0; i < rd2_word; i++) h_T(i)     = (double)rand() / RAND_MAX;
    for (int i = 0; i < rd2_word; i++) h_P(i)     = (double)rand() / RAND_MAX;
    for (int i = 0; i < rd_word;  i++) h_Z(i)     = (double)rand() / RAND_MAX;
    for (int i = 0; i < rd_word;  i++) h_alpha(i) = (double)rand() / RAND_MAX;
    for (int i = 0; i < nobs_word; i++) h_ys(i)   = (double)rand() / RAND_MAX;
    for (int i = 0; i < ns_word;  i++) h_mu(i)    = (double)rand() / RAND_MAX;

    Kokkos::deep_copy(d_RQR,   h_RQR);
    Kokkos::deep_copy(d_T,     h_T);
    Kokkos::deep_copy(d_P,     h_P);
    Kokkos::deep_copy(d_Z,     h_Z);
    Kokkos::deep_copy(d_alpha, h_alpha);
    Kokkos::deep_copy(d_ys,    h_ys);
    Kokkos::deep_copy(d_mu,    h_mu);

    for (int n_diff = 0; n_diff < rd; n_diff++) {
      Kokkos::fence();
      auto start = std::chrono::steady_clock::now();

      for (int iter = 0; iter < repeat; iter++) {
        Kokkos::parallel_for("kalman", nseries, KOKKOS_LAMBDA(int bid) {
          kalman_kernel<8>(
            bid,
            d_ys.data(),
            nobs,
            d_T.data(),
            d_Z.data(),
            d_RQR.data(),
            d_P.data(),
            d_alpha.data(),
            true,
            d_mu.data(),
            batch_size,
            d_vs.data(),
            d_Fs.data(),
            d_sum_logFs.data(),
            n_diff,
            fc_steps,
            d_fc.data(),
            true,
            d_F_fc.data());
        });
      }

      Kokkos::fence();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time (n_diff = %d): %f (s)\n",
             n_diff, (time * 1e-9f) / repeat);
    }

    Kokkos::deep_copy(h_F_fc, d_F_fc);

    double sum = 0.0;
    for (int i = 0; i < fc_steps * nseries - 1; i++)
      sum += (fabs(h_F_fc(i+1)) - fabs(h_F_fc(i))) /
             (fabs(h_F_fc(i+1)) + fabs(h_F_fc(i)));
    printf("Checksum: %lf\n", sum);
  }
  Kokkos::finalize();
  return 0;
}
