// Copyright (c) 2021 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
//
// Kokkos port of the LDA (Latent Dirichlet Allocation) E-step kernel.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <cmath>

#include <Kokkos_Core.hpp>

#define EPS 1e-6f

KOKKOS_INLINE_FUNCTION
float ReduceSum(const float* vec, const int length) {
  float s = 0.f;
  for (int i = 0; i < length; i++)
    s += vec[i];
  return s;
}

// reference: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
KOKKOS_INLINE_FUNCTION
float Digamma(float x) {
  float result = 0.0f, xx, xx2, xx4;
  for (; x < 7.0f; ++x)
    result -= 1.0f / x;
  x -= 0.5f;
  xx = 1.0f / x;
  xx2 = xx * xx;
  xx4 = xx2 * xx2;
  result += logf(x) + 1.0f / 24.0f * xx2
    - 7.0f / 960.0f * xx4 + 31.0f / 8064.0f * xx4 * xx2
    - 127.0f / 30720.0f * xx4 * xx4;
  return result;
}

int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);

  srand(123);

  const int num_topics = 1000;
  const int num_words  = 10266;
  const int block_cnt  = 500;
  const int num_indptr = block_cnt; // max: num_words
  const int block_dim  = 256;
  const int num_iters  = 64;

  std::vector<float> h_alpha(num_topics);
  for (int i = 0; i < num_topics; i++)
    h_alpha[i] = (float)rand() / (float)RAND_MAX;

  std::vector<float> h_beta((size_t)num_topics * num_words);
  for (int i = 0; i < num_topics * num_words; i++)
    h_beta[i] = (float)rand() / (float)RAND_MAX;

  std::vector<float> h_grad_alpha((size_t)num_topics * block_cnt, 0.0f);
  std::vector<float> h_new_beta((size_t)num_topics * num_words, 0.0f);
  std::vector<int>   h_locks(num_words, 0);

  std::vector<int> h_indptr(num_indptr + 1, 0);
  h_indptr[num_indptr] = num_words - 1;
  for (int i = num_indptr; i >= 1; i--) {
    int t = h_indptr[i] - 1 - (rand() % (num_words / num_indptr));
    if (t < 0) break;
    h_indptr[i - 1] = t;
  }
  const int num_cols = num_words;

  std::vector<int>   h_cols(num_cols);
  std::vector<float> h_counts(num_cols);
  for (int i = 0; i < num_cols; i++) {
    h_cols[i]   = i;
    h_counts[i] = 0.5f; // arbitrary
  }

  std::vector<float> h_train_losses(block_cnt, 0.f);
  std::vector<float> h_vali_losses(block_cnt, 0.f);

  Kokkos::initialize(argc, argv);
  {
    using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using ScratchView  = Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged>;

    // Allocate device views
    Kokkos::View<float*> d_alpha("alpha", num_topics);
    Kokkos::View<float*> d_beta("beta", (size_t)num_topics * num_words);
    Kokkos::View<float*> d_grad_alpha("grad_alpha", (size_t)num_topics * block_cnt);
    Kokkos::View<float*> d_new_beta("new_beta", (size_t)num_topics * num_words);
    Kokkos::View<int*>   d_locks("locks", num_words);
    Kokkos::View<int*>   d_cols("cols", num_cols);
    Kokkos::View<int*>   d_indptr("indptr", num_indptr + 1);
    Kokkos::View<float*> d_counts("counts", num_cols);
    Kokkos::View<bool*>  d_vali("vali", num_cols);
    Kokkos::View<float*> d_gamma("gamma", (size_t)num_indptr * num_topics);
    Kokkos::View<float*> d_train_losses("train_losses", block_cnt);
    Kokkos::View<float*> d_vali_losses("vali_losses", block_cnt);

    // Copy host data to device using unmanaged host views
    Kokkos::deep_copy(d_alpha,
      Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_alpha.data(), num_topics));
    Kokkos::deep_copy(d_beta,
      Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_beta.data(), (size_t)num_topics * num_words));
    Kokkos::deep_copy(d_grad_alpha,
      Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_grad_alpha.data(), (size_t)num_topics * block_cnt));
    Kokkos::deep_copy(d_new_beta,
      Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_new_beta.data(), (size_t)num_topics * num_words));
    Kokkos::deep_copy(d_locks,
      Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_locks.data(), num_words));
    Kokkos::deep_copy(d_cols,
      Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_cols.data(), num_cols));
    Kokkos::deep_copy(d_indptr,
      Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_indptr.data(), num_indptr + 1));
    Kokkos::deep_copy(d_counts,
      Kokkos::View<const float*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        h_counts.data(), num_cols));
    Kokkos::deep_copy(d_vali, false);
    Kokkos::deep_copy(d_train_losses, 0.f);
    Kokkos::deep_copy(d_vali_losses, 0.f);

    const size_t scratch_size = ScratchView::shmem_size(4 * num_topics);
    auto policy = Kokkos::TeamPolicy<>(block_cnt, block_dim)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    // Training
    bool init_gamma = false;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      init_gamma = (i == 0);

      Kokkos::parallel_for("EstepKernel", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
          ScratchView smem(team.team_scratch(0), 4 * num_topics);
          float* _new_gamma    = smem.data();
          float* _phi          = smem.data() + num_topics;
          float* _loss_vec     = smem.data() + num_topics * 2;
          float* _vali_phi_sum = smem.data() + num_topics * 3;

          const int blockIdx_x  = team.league_rank();
          const int threadIdx_x = team.team_rank();
          const int blockDim_x  = block_dim;
          const int gridDim_x   = block_cnt;

          float* _grad_alpha = d_grad_alpha.data() + num_topics * blockIdx_x;

          for (int idx = blockIdx_x; idx < num_indptr; idx += gridDim_x) {
            int beg = d_indptr(idx), end = d_indptr(idx + 1);
            float* _gamma = d_gamma.data() + num_topics * idx;

            if (init_gamma) {
              for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
                _gamma[j] = d_alpha(j) + (end - beg) / num_topics;
            }
            team.team_barrier();

            // Initialise phi sum for validation data
            for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
              _vali_phi_sum[j] = 0.0f;

            // E-step iterations
            for (int j = 0; j < num_iters; ++j) {
              for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
                _new_gamma[k] = 0.0f;
              team.team_barrier();

              for (int k = beg; k < end; ++k) {
                const int   w     = d_cols(k);
                const bool  _vali = d_vali(k);
                const float c     = d_counts(k);

                if (!_vali || j + 1 == num_iters) {
                  for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                    _phi[l] = d_beta(w * num_topics + l) * expf(Digamma(_gamma[l]));
                  team.team_barrier();

                  float phi_sum = ReduceSum(_phi, num_topics);

                  for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                    _phi[l] /= phi_sum;
                    if (_vali)
                      _vali_phi_sum[l] += _phi[l] * c;
                    else
                      _new_gamma[l] += _phi[l] * c;
                  }
                  team.team_barrier();
                }

                if (j + 1 == num_iters) {
                  if (!_vali) {
                    // Acquire lock: spin until CAS(0 -> 1) succeeds
                    if (threadIdx_x == 0) {
                      int exp_val = 0;
                      while (Kokkos::atomic_compare_exchange(&d_locks(w), exp_val, 1) != 0)
                        exp_val = 0;
                    }
                    team.team_barrier();

                    for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                      d_new_beta(w * num_topics + l) += _phi[l] * c;
                    team.team_barrier();

                    // Release lock
                    if (threadIdx_x == 0) d_locks(w) = 0;
                    team.team_barrier();
                  }

                  // Compute KL loss
                  // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
                  for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                    _loss_vec[l]  = logf(fmaxf(d_beta(w * num_topics + l), EPS));
                    _loss_vec[l] -= logf(fmaxf(_phi[l], EPS));
                    _loss_vec[l] *= _phi[l];
                  }
                  team.team_barrier();

                  float _loss = ReduceSum(_loss_vec, num_topics) * c;
                  if (threadIdx_x == 0) {
                    if (_vali)
                      d_vali_losses(blockIdx_x) += _loss;
                    else
                      d_train_losses(blockIdx_x) += _loss;
                  }
                  team.team_barrier();
                }
                team.team_barrier();
              } // end k loop

              // Update gamma
              for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
                _gamma[k] = _new_gamma[k] + d_alpha(k);
              team.team_barrier();
            } // end j (num_iters) loop

            // Update gradient of alpha and loss from E[log(theta)]
            float gamma_sum = ReduceSum(_gamma, num_topics);
            for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
              float Elogthetad = Digamma(_gamma[j]) - Digamma(gamma_sum);
              _grad_alpha[j]    += Elogthetad;
              _new_gamma[j]     *= Elogthetad;
              _vali_phi_sum[j]  *= Elogthetad;
            }

            // see Eq (15) in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
            float train_loss = ReduceSum(_new_gamma, num_topics);
            float vali_loss  = ReduceSum(_vali_phi_sum, num_topics);
            if (threadIdx_x == 0) {
              d_train_losses(blockIdx_x) += train_loss;
              d_vali_losses(blockIdx_x)  += vali_loss;
            }
            team.team_barrier();
          } // end idx loop
        }
      );
    }

    Kokkos::fence();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

    // Validation: mark every entry as validation data
    Kokkos::deep_copy(d_vali, true);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      Kokkos::parallel_for("EstepKernel", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
          ScratchView smem(team.team_scratch(0), 4 * num_topics);
          float* _new_gamma    = smem.data();
          float* _phi          = smem.data() + num_topics;
          float* _loss_vec     = smem.data() + num_topics * 2;
          float* _vali_phi_sum = smem.data() + num_topics * 3;

          const int blockIdx_x  = team.league_rank();
          const int threadIdx_x = team.team_rank();
          const int blockDim_x  = block_dim;
          const int gridDim_x   = block_cnt;

          float* _grad_alpha = d_grad_alpha.data() + num_topics * blockIdx_x;

          for (int idx = blockIdx_x; idx < num_indptr; idx += gridDim_x) {
            int beg = d_indptr(idx), end = d_indptr(idx + 1);
            float* _gamma = d_gamma.data() + num_topics * idx;

            if (init_gamma) {
              for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
                _gamma[j] = d_alpha(j) + (end - beg) / num_topics;
            }
            team.team_barrier();

            for (int j = threadIdx_x; j < num_topics; j += blockDim_x)
              _vali_phi_sum[j] = 0.0f;

            for (int j = 0; j < num_iters; ++j) {
              for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
                _new_gamma[k] = 0.0f;
              team.team_barrier();

              for (int k = beg; k < end; ++k) {
                const int   w     = d_cols(k);
                const bool  _vali = d_vali(k);
                const float c     = d_counts(k);

                if (!_vali || j + 1 == num_iters) {
                  for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                    _phi[l] = d_beta(w * num_topics + l) * expf(Digamma(_gamma[l]));
                  team.team_barrier();

                  float phi_sum = ReduceSum(_phi, num_topics);

                  for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                    _phi[l] /= phi_sum;
                    if (_vali)
                      _vali_phi_sum[l] += _phi[l] * c;
                    else
                      _new_gamma[l] += _phi[l] * c;
                  }
                  team.team_barrier();
                }

                if (j + 1 == num_iters) {
                  if (!_vali) {
                    if (threadIdx_x == 0) {
                      int exp_val = 0;
                      while (Kokkos::atomic_compare_exchange(&d_locks(w), exp_val, 1) != 0)
                        exp_val = 0;
                    }
                    team.team_barrier();

                    for (int l = threadIdx_x; l < num_topics; l += blockDim_x)
                      d_new_beta(w * num_topics + l) += _phi[l] * c;
                    team.team_barrier();

                    if (threadIdx_x == 0) d_locks(w) = 0;
                    team.team_barrier();
                  }

                  for (int l = threadIdx_x; l < num_topics; l += blockDim_x) {
                    _loss_vec[l]  = logf(fmaxf(d_beta(w * num_topics + l), EPS));
                    _loss_vec[l] -= logf(fmaxf(_phi[l], EPS));
                    _loss_vec[l] *= _phi[l];
                  }
                  team.team_barrier();

                  float _loss = ReduceSum(_loss_vec, num_topics) * c;
                  if (threadIdx_x == 0) {
                    if (_vali)
                      d_vali_losses(blockIdx_x) += _loss;
                    else
                      d_train_losses(blockIdx_x) += _loss;
                  }
                  team.team_barrier();
                }
                team.team_barrier();
              } // end k loop

              for (int k = threadIdx_x; k < num_topics; k += blockDim_x)
                _gamma[k] = _new_gamma[k] + d_alpha(k);
              team.team_barrier();
            } // end j loop

            float gamma_sum = ReduceSum(_gamma, num_topics);
            for (int j = threadIdx_x; j < num_topics; j += blockDim_x) {
              float Elogthetad = Digamma(_gamma[j]) - Digamma(gamma_sum);
              _grad_alpha[j]   += Elogthetad;
              _new_gamma[j]    *= Elogthetad;
              _vali_phi_sum[j] *= Elogthetad;
            }

            float train_loss = ReduceSum(_new_gamma, num_topics);
            float vali_loss  = ReduceSum(_vali_phi_sum, num_topics);
            if (threadIdx_x == 0) {
              d_train_losses(blockIdx_x) += train_loss;
              d_vali_losses(blockIdx_x)  += vali_loss;
            }
            team.team_barrier();
          } // end idx loop
        }
      );
    }

    Kokkos::fence();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);

    // Copy results back to host
    {
      auto mv = Kokkos::create_mirror_view(d_train_losses);
      Kokkos::deep_copy(mv, d_train_losses);
      for (int i = 0; i < block_cnt; i++) h_train_losses[i] = mv(i);
    }
    {
      auto mv = Kokkos::create_mirror_view(d_vali_losses);
      Kokkos::deep_copy(mv, d_vali_losses);
      for (int i = 0; i < block_cnt; i++) h_vali_losses[i] = mv(i);
    }
  }
  Kokkos::finalize();

  float total_train_loss = std::accumulate(h_train_losses.begin(), h_train_losses.end(), 0.0f);
  float total_vali_loss  = std::accumulate(h_vali_losses.begin(),  h_vali_losses.end(),  0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);

  return 0;
}
