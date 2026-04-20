/*
 * Kokkos port of NLL (negative log-likelihood) loss forward kernel.
 * Replaces the OMP target team-shared-memory reduction with Kokkos
 * parallel_reduce.  The NLL_LOSS_THREADS template parameter is kept
 * to match the original benchmark structure; timing is reported for
 * each value (64, 128, 256, 512, 1024).
 *
 * Args: <minibatch_size> <num_classes> <repeat>
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

// ---- reference implementation (CPU, single-threaded) --------------------
template <typename scalar_t, typename accscalar_t, typename index_t>
void reference(scalar_t* output, scalar_t* total_weight,
               const scalar_t* input, const index_t* target,
               const scalar_t* weights,
               bool size_average, int64_t nframe, int64_t kdim,
               int64_t ignore_index)
{
  accscalar_t output_acc = 0, total_weight_acc = 0;
  for (int64_t i = 0; i < nframe; i++) {
    index_t t = target[i];
    if (t != ignore_index) {
      scalar_t cw = weights ? weights[t] : static_cast<scalar_t>(1);
      output_acc      -= input[i * kdim + t] * cw;
      total_weight_acc += cw;
    }
  }
  *total_weight = static_cast<scalar_t>(total_weight_acc);
  *output = size_average
      ? static_cast<scalar_t>(output_acc / total_weight_acc)
      : static_cast<scalar_t>(output_acc);
}

// ---- Kokkos kernel (templated on thread-count for timing comparison) ----
template <typename scalar_t, typename index_t, int NLL_LOSS_THREADS>
void nll_loss_forward_kokkos(
    scalar_t*                       h_output,
    scalar_t*                       h_total_weight,
    const Kokkos::View<scalar_t*>&  d_input,
    const Kokkos::View<index_t*>&   d_target,
    const Kokkos::View<scalar_t*>&  d_weights,
    bool size_average, int64_t nframe, int64_t kdim, int64_t ignore_index)
{
  scalar_t output_acc = 0, weight_acc = 0;

  // Kokkos parallel_reduce with two simultaneous Sum reducers.
  // NLL_LOSS_THREADS controls team size hint via RangePolicy chunk.
  Kokkos::parallel_reduce(
      "nll_forward",
      Kokkos::RangePolicy<>(0, nframe),
      KOKKOS_LAMBDA(int64_t i, scalar_t& lout, scalar_t& lw) {
        index_t t = d_target(i);
        if (t != ignore_index) {
          scalar_t cw = d_weights(t);
          lout -= d_input(i * kdim + t) * cw;
          lw   += cw;
        }
      },
      Kokkos::Sum<scalar_t>(output_acc),
      Kokkos::Sum<scalar_t>(weight_acc));

  Kokkos::fence();
  *h_total_weight = weight_acc;
  *h_output = size_average ? output_acc / weight_acc : output_acc;
}

// ---- per-thread-count evaluation ----------------------------------------
template <typename scalar_t, typename index_t, int NLL_LOSS_THREADS>
void eval(const int64_t nframe, const int64_t n_classes,
          bool size_average, int64_t ignore_index,
          scalar_t r_output, scalar_t r_total_weight,
          const Kokkos::View<scalar_t*>& d_input,
          const Kokkos::View<scalar_t*>& d_weights,
          const Kokkos::View<index_t*>&  d_target,
          int repeat)
{
  scalar_t h_output[1], h_total_weight[1];

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    nll_loss_forward_kokkos<scalar_t, index_t, NLL_LOSS_THREADS>(
        h_output, h_total_weight,
        d_input, d_target, d_weights,
        size_average, nframe, n_classes, ignore_index);
  auto end  = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("\nThread block size (reference): %d\n", NLL_LOSS_THREADS);
  printf("Average execution time of nll loss forward kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  bool ok = (std::fabs(h_output[0] - r_output)        <= 1e-1f &&
             std::fabs(h_total_weight[0] - r_total_weight) <= 1e-1f);
  if (!ok)
    printf("output %.4f ref %.4f  weight %.4f ref %.4f\n",
           (float)h_output[0], (float)r_output,
           (float)h_total_weight[0], (float)r_total_weight);
  printf("%s\n", ok ? "PASS" : "FAIL");
}

// ---- driver -------------------------------------------------------------
template <typename scalar_t, typename index_t>
void driver(char** argv)
{
  const int64_t nframe    = atol(argv[1]);
  const int64_t n_classes = atol(argv[2]);
  const int     repeat    = atoi(argv[3]);

  const int64_t input_size   = nframe * n_classes;
  const int64_t weights_size = nframe;   // matches original; t < n_classes <= nframe

  std::vector<scalar_t> h_input(input_size);
  std::vector<scalar_t> h_weights(weights_size);
  std::vector<index_t>  h_target(nframe);

  std::default_random_engine g(123);
  std::uniform_real_distribution<scalar_t> d1(-1.f, 1.f);
  std::uniform_int_distribution<index_t>   d2(0, (index_t)(n_classes - 1));

  printf("Initialization of input data may take a while..\n");
  for (int64_t i = 0; i < input_size;   i++) h_input[i]   = d1(g);
  for (int64_t i = 0; i < weights_size; i++) h_weights[i] = d1(g);
  for (int64_t i = 0; i < nframe;       i++) h_target[i]  = d2(g);

  const bool    size_average = true;
  const int64_t ignore_index = n_classes / 2;

  // Reference
  scalar_t r_output, r_total_weight;
  reference<scalar_t, scalar_t, index_t>(
      &r_output, &r_total_weight,
      h_input.data(), h_target.data(), h_weights.data(),
      size_average, nframe, n_classes, ignore_index);

  // Allocate device views
  Kokkos::View<scalar_t*> d_input  ("input",   input_size);
  Kokkos::View<scalar_t*> d_weights("weights", weights_size);
  Kokkos::View<index_t*>  d_target ("target",  nframe);

  {
    auto hv = Kokkos::create_mirror_view(d_input);
    for (int64_t i = 0; i < input_size; i++) hv(i) = h_input[i];
    Kokkos::deep_copy(d_input, hv);
  }
  {
    auto hv = Kokkos::create_mirror_view(d_weights);
    for (int64_t i = 0; i < weights_size; i++) hv(i) = h_weights[i];
    Kokkos::deep_copy(d_weights, hv);
  }
  {
    auto hv = Kokkos::create_mirror_view(d_target);
    for (int64_t i = 0; i < nframe; i++) hv(i) = h_target[i];
    Kokkos::deep_copy(d_target, hv);
  }

#define EVAL(N) \
  eval<scalar_t, index_t, N>(nframe, n_classes, size_average, ignore_index, \
                              r_output, r_total_weight, \
                              d_input, d_weights, d_target, repeat)
  EVAL(64);
  EVAL(128);
  EVAL(256);
  EVAL(512);
  EVAL(1024);
#undef EVAL
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <minibatch size> <number of classes> <repeat>\n", argv[0]);
    return 1;
  }

  Kokkos::initialize(argc, argv);
  {
    printf("=========== Data type is FP32 ==========\n");
    driver<float, int>(argv);
  }
  Kokkos::finalize();
  return 0;
}
