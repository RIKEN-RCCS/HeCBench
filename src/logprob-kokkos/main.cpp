// logprob-kokkos/main.cpp
// Port of logprob-cuda: computes per-token log-probabilities from logits via
// log-softmax, then accumulates them per batch entry.
//
// log_probs_kernel    → Kokkos::TeamPolicy over (step, batch) pairs;
//                       TeamThreadRange reductions for max and sum-exp.
// accumulate_log_probs → Kokkos::TeamPolicy over batch entries;
//                        TeamThreadRange reduction for sum.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
static void log_probs_cpu(
    float*       log_probs,
    const float* logits,
    const int*   ids,
    const int*   lengths,
    int          max_input_length,
    int          batch_size,
    int          vocab_size,
    int          vocab_size_padded,
    float*       cum_log_probs)
{
  for (int bidx = 0; bidx < batch_size; ++bidx) {
    float accum = 0.f;
    for (int step = 0; step < lengths[bidx] - 1; ++step) {
      int         batch_offset = bidx * max_input_length * vocab_size_padded;
      int         step_offset  = step * vocab_size_padded;
      const float* lptr        = logits + batch_offset + step_offset;

      float max_val = -std::numeric_limits<float>::max();
      for (int i = 0; i < vocab_size; ++i)
        max_val = std::fmaxf(max_val, lptr[i]);

      float sum_exp = 0.f;
      for (int i = 0; i < vocab_size; ++i)
        sum_exp += expf(lptr[i] - max_val);

      int idx        = step + bidx * (max_input_length - 1);
      int token_idx  = step + 1 + bidx * max_input_length;
      log_probs[idx] = lptr[ids[token_idx]] - max_val - logf(sum_exp + 1e-9f);
      accum += log_probs[idx];
    }
    cum_log_probs[bidx] = accum;
  }
}

// ---------------------------------------------------------------------------
// Kokkos kernels
// ---------------------------------------------------------------------------
using TeamPolicy = Kokkos::TeamPolicy<>;
using MemberType = TeamPolicy::member_type;

// Kernel 1: compute per-(step,batch) log probability
static void log_probs_kernel(
    Kokkos::View<float*>       d_log_probs,
    Kokkos::View<const float*> d_logits,
    Kokkos::View<const int*>   d_ids,
    Kokkos::View<const int*>   d_lengths,
    int max_length,
    int batch_size,
    int vocab_size,
    int vocab_size_padded)
{
  const int num_teams = (max_length - 1) * batch_size;

  Kokkos::parallel_for(
      "log_probs_kernel",
      TeamPolicy(num_teams, Kokkos::AUTO),
      KOKKOS_LAMBDA(const MemberType& team) {
        const int league_rank = team.league_rank();
        const int step  = league_rank % (max_length - 1);
        const int bidx  = league_rank / (max_length - 1);

        if (bidx >= batch_size || step >= d_lengths(bidx) - 1) return;

        const int base = bidx * max_length * vocab_size_padded
                       + step * vocab_size_padded;

        // Reduce max over vocab
        float max_val = -FLT_MAX;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, vocab_size),
            [=](int i, float& lmax) {
              lmax = fmaxf(lmax, d_logits(base + i));
            },
            Kokkos::Max<float>(max_val));
        // max_val broadcast to all team threads after reduce

        // Reduce sum of exp(logit - max)
        float sum_exp = 0.f;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, vocab_size),
            [=](int i, float& lsum) {
              lsum += expf(d_logits(base + i) - max_val);
            },
            sum_exp);

        // Single thread writes log_prob
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
          const int idx       = step + bidx * (max_length - 1);
          const int token_idx = step + 1 + bidx * max_length;
          d_log_probs(idx) = d_logits(base + d_ids(token_idx))
                             - max_val
                             - logf(sum_exp + 1e-9f);
        });
      });
}

// Kernel 2: accumulate log_probs over steps for each batch entry
static void accumulate_log_probs(
    Kokkos::View<float*>       d_cum,
    Kokkos::View<const float*> d_log_probs,
    Kokkos::View<const int*>   d_lengths,
    int max_length,
    int batch_size)
{
  Kokkos::parallel_for(
      "accumulate_log_probs",
      TeamPolicy(batch_size, Kokkos::AUTO),
      KOKKOS_LAMBDA(const MemberType& team) {
        const int bidx   = team.league_rank();
        const int length = d_lengths(bidx);
        const int base   = bidx * (max_length - 1);

        float accum = 0.f;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, length - 1),
            [=](int step, float& lsum) { lsum += d_log_probs(base + step); },
            accum);

        Kokkos::single(Kokkos::PerTeam(team),
                       [=]() { d_cum(bidx) = accum; });
      });
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <max_length> <batch_size> <vocab_size> <repeat>\n",
           argv[0]);
    return 1;
  }
  const int max_length  = std::atoi(argv[1]);
  const int batch_size  = std::atoi(argv[2]);
  const int vocab_size  = std::atoi(argv[3]);
  const int repeat      = std::atoi(argv[4]);

  const int vocab_size_padded = (vocab_size + 31) / 32 * 32;

  const std::size_t logits_n   = static_cast<std::size_t>(batch_size) * max_length * vocab_size_padded;
  const std::size_t log_probs_n = static_cast<std::size_t>(batch_size) * (max_length - 1);
  const std::size_t ids_n       = static_cast<std::size_t>(batch_size) * max_length;

  // Generate host data
  std::vector<float> h_logits(logits_n);
  {
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> ud(-6.f, 6.f);
    for (auto& v : h_logits) v = ud(rng);
  }

  std::vector<int> h_lengths(batch_size, max_length);

  std::vector<int> h_ids(ids_n);
  srand(123);
  for (auto& v : h_ids) v = rand() % vocab_size;

  std::vector<float> h_log_probs_ref(log_probs_n);
  std::vector<float> h_cum_ref(batch_size);
  log_probs_cpu(h_log_probs_ref.data(), h_logits.data(), h_ids.data(),
                h_lengths.data(), max_length, batch_size,
                vocab_size, vocab_size_padded,
                h_cum_ref.data());

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<float*> d_logits("d_logits", logits_n);
    Kokkos::View<float*> d_log_probs("d_log_probs", log_probs_n);
    Kokkos::View<float*> d_cum("d_cum", batch_size);
    Kokkos::View<int*>   d_ids("d_ids", ids_n);
    Kokkos::View<int*>   d_lengths("d_lengths", batch_size);

    {
      auto ml  = Kokkos::create_mirror_view(d_logits);
      auto mi  = Kokkos::create_mirror_view(d_ids);
      auto mln = Kokkos::create_mirror_view(d_lengths);
      for (std::size_t k = 0; k < logits_n; ++k) ml(k) = h_logits[k];
      for (std::size_t k = 0; k < ids_n;    ++k) mi(k) = h_ids[k];
      for (int k = 0; k < batch_size;        ++k) mln(k) = h_lengths[k];
      Kokkos::deep_copy(d_logits,  ml);
      Kokkos::deep_copy(d_ids,     mi);
      Kokkos::deep_copy(d_lengths, mln);
    }

    // Read-only aliases for kernels
    auto rd_logits  = Kokkos::View<const float*>(d_logits);
    auto rd_ids     = Kokkos::View<const int*>(d_ids);
    auto rd_lengths = Kokkos::View<const int*>(d_lengths);
    auto rd_lp      = Kokkos::View<const float*>(d_log_probs);

    auto run_kernels = [&]() {
      log_probs_kernel(d_log_probs, rd_logits, rd_ids, rd_lengths,
                       max_length, batch_size, vocab_size, vocab_size_padded);
      accumulate_log_probs(d_cum, rd_lp, rd_lengths,
                           max_length, batch_size);
    };

    // Warmup + verify
    run_kernels();
    Kokkos::fence();

    std::vector<float> h_log_probs(log_probs_n);
    std::vector<float> h_cum(batch_size);
    {
      auto mlp = Kokkos::create_mirror_view(d_log_probs);
      auto mc  = Kokkos::create_mirror_view(d_cum);
      Kokkos::deep_copy(mlp, d_log_probs);
      Kokkos::deep_copy(mc,  d_cum);
      for (std::size_t k = 0; k < log_probs_n; ++k) h_log_probs[k] = mlp(k);
      for (int k = 0; k < batch_size;          ++k) h_cum[k] = mc(k);
    }

    bool error = false;
    for (std::size_t i = 0; i < log_probs_n && !error; ++i) {
      if (fabsf(h_log_probs[i] - h_log_probs_ref[i]) > 1e-3f) {
        printf("log_probs mismatch @%zu: %f vs %f\n",
               i, h_log_probs[i], h_log_probs_ref[i]);
        error = true;
      }
    }
    for (int i = 0; i < batch_size && !error; ++i) {
      if (fabsf(h_cum[i] - h_cum_ref[i]) > 1e-1f) {
        printf("cum_log_probs mismatch @%d: %f vs %f\n",
               i, h_cum[i], h_cum_ref[i]);
        error = true;
      }
    }
    printf("%s\n", error ? "FAIL" : "PASS");

    // Timed run
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_kernels();
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time of kernels: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);
  }
  Kokkos::finalize();
  return 0;
}
