#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <Kokkos_Core.hpp>

// sigmoid for use inside KOKKOS_LAMBDA
KOKKOS_INLINE_FUNCTION float gru_sigmoid(float x)
{
  return 1.f / (1.f + expf(-x));
}

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of sequences> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int vsz    = atoi(argv[1]);
  const int hsz    = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_size  = 3 * vsz * hsz;
  const int hidden_size = 3 * vsz * hsz;
  const int bias_size   = 3 * hsz;
  const int store_size  = 5 * vsz * hsz;
  const int state_size  = vsz;

  // Host arrays (float instead of half)
  float *h_input        = new float[input_size];
  float *h_hidden       = new float[hidden_size];
  float *h_input_bias   = new float[bias_size];
  float *h_hidden_bias  = new float[bias_size];
  float *h_hx           = new float[state_size];
  float *h_hy           = new float[state_size];

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-2.f, 2.f);

  for (int i = 0; i < input_size;  i++) h_input[i]       = distr(g);
  for (int i = 0; i < hidden_size; i++) h_hidden[i]      = distr(g);
  for (int i = 0; i < bias_size;   i++) { h_input_bias[i] = distr(g);
                                           h_hidden_bias[i] = distr(g); }
  for (int i = 0; i < state_size;  i++) h_hx[i] = distr(g);

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float *> d_input      ("input",       input_size);
    Kokkos::View<float *> d_hidden     ("hidden",      hidden_size);
    Kokkos::View<float *> d_input_bias ("input_bias",  bias_size);
    Kokkos::View<float *> d_hidden_bias("hidden_bias", bias_size);
    Kokkos::View<float *> d_hx        ("hx",           state_size);
    Kokkos::View<float *> d_hy        ("hy",           state_size);
    Kokkos::View<float *> d_store     ("store",        store_size);

    // Copy host → device
    {
      auto m_input       = Kokkos::create_mirror_view(d_input);
      auto m_hidden      = Kokkos::create_mirror_view(d_hidden);
      auto m_ibias       = Kokkos::create_mirror_view(d_input_bias);
      auto m_hbias       = Kokkos::create_mirror_view(d_hidden_bias);
      auto m_hx          = Kokkos::create_mirror_view(d_hx);

      for (int i = 0; i < input_size;  i++) m_input(i)  = h_input[i];
      for (int i = 0; i < hidden_size; i++) m_hidden(i) = h_hidden[i];
      for (int i = 0; i < bias_size;   i++) {
        m_ibias(i) = h_input_bias[i];
        m_hbias(i) = h_hidden_bias[i];
      }
      for (int i = 0; i < state_size; i++) m_hx(i) = h_hx[i];

      Kokkos::deep_copy(d_input,       m_input);
      Kokkos::deep_copy(d_hidden,      m_hidden);
      Kokkos::deep_copy(d_input_bias,  m_ibias);
      Kokkos::deep_copy(d_hidden_bias, m_hbias);
      Kokkos::deep_copy(d_hx,         m_hx);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      const int hsz_ = hsz;
      Kokkos::parallel_for("gru_cell_forward", vsz, KOKKOS_LAMBDA(int linearIndex) {
        // Offset into the 3-gated weight matrices
        const int offset = (linearIndex / hsz_) * 3 * hsz_ + linearIndex % hsz_;

        const float ir = d_input(offset + 0 * hsz_);
        const float ii = d_input(offset + 1 * hsz_);
        const float in = d_input(offset + 2 * hsz_);
        const float hr = d_hidden(offset + 0 * hsz_);
        const float hi = d_hidden(offset + 1 * hsz_);
        const float hn = d_hidden(offset + 2 * hsz_);
        const float hx = d_hx(linearIndex);

        const int bi = linearIndex % hsz_;
        const float b1r = d_input_bias(bi + 0 * hsz_);
        const float b1i = d_input_bias(bi + 1 * hsz_);
        const float b1n = d_input_bias(bi + 2 * hsz_);
        const float b2r = d_hidden_bias(bi + 0 * hsz_);
        const float b2i = d_hidden_bias(bi + 1 * hsz_);
        const float b2n = d_hidden_bias(bi + 2 * hsz_);

        // GRU gate computations
        const float rg = gru_sigmoid(ir + hr + b1r + b2r);  // reset gate
        const float ig = gru_sigmoid(ii + hi + b1i + b2i);  // update gate
        const float ng = tanhf(in + b1n + rg * (hn + b2n)); // new gate

        d_hy(linearIndex) = ng + ig * (hx - ng);

        // Save intermediate values for backward pass
        const int soffset = (linearIndex / hsz_) * 5 * hsz_ + linearIndex % hsz_;
        d_store(soffset + 0 * hsz_) = rg;
        d_store(soffset + 1 * hsz_) = ig;
        d_store(soffset + 2 * hsz_) = ng;
        d_store(soffset + 3 * hsz_) = hx;
        d_store(soffset + 4 * hsz_) = hn + b2n;
      });
    }

    Kokkos::fence();
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end - start).count();
    printf("Average execution time of gru_cell_forward: %f (us)\n",
           (time * 1e-3f) / repeat);

    // Copy result back and compute checksum
    auto m_hy = Kokkos::create_mirror_view(d_hy);
    Kokkos::deep_copy(m_hy, d_hy);

    float checksum = 0.f;
    for (int i = 0; i < state_size; i++) checksum += m_hy(i);
    printf("Checksum is %f\n", checksum / state_size);
  }
  Kokkos::finalize();

  delete[] h_input;
  delete[] h_hidden;
  delete[] h_input_bias;
  delete[] h_hidden_bias;
  delete[] h_hx;
  delete[] h_hy;
  return 0;
}
