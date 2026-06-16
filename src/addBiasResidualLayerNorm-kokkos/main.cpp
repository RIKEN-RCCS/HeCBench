/*
 * Kokkos port of addBiasResidualLayerNorm.
 * Uses float instead of half/bfloat16. Block-level reductions via TeamPolicy.
 */

#include <algorithm>
#include <chrono>
#include <random>
#include <cstdio>
#include <cmath>
#include <Kokkos_Core.hpp>

// Layer norm: out = (out + input + bias - mean) / sqrt(variance + eps) * gamma + beta
// Uses TeamPolicy: one team per row (m rows, n columns)
void addBiasResidualLayerNorm(
    Kokkos::View<float*> out,
    Kokkos::View<float*> input,
    Kokkos::View<float*> bias,
    Kokkos::View<float*> gamma,
    Kokkos::View<float*> beta,
    float eps, int m, int n, int repeat)
{
  using TeamPol = Kokkos::TeamPolicy<>;
  using TeamMem = TeamPol::member_type;

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    Kokkos::parallel_for("LayerNorm",
      TeamPol(m, Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamMem& team) {
        int row = team.league_rank();
        float mean = 0.f, var = 0.f;

        // Step 1: compute sum and add bias+residual
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
          [=](int col, float& sum) {
            float v = out(row * n + col) + input(row * n + col) + bias(col);
            out(row * n + col) = v;
            sum += v;
          }, mean);
        team.team_barrier();
        mean /= n;

        // Step 2: compute variance
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, n),
          [=](int col, float& s) {
            float d = out(row * n + col) - mean;
            s += d * d;
          }, var);
        team.team_barrier();
        float inv_std = 1.f / Kokkos::sqrt(var / n + eps);

        // Step 3: normalize
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n),
          [=](int col) {
            float v = (out(row * n + col) - mean) * inv_std;
            out(row * n + col) = v * gamma(col) + beta(col);
          });
      });
  }
  Kokkos::fence();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AddBiasResidualLayerNorm (m=%d, n=%d): %f (us)\n",
         m, n, (time * 1e-3f) / repeat);
}

void layer(int repeat) {
  const int m = 4096;
  int dims[] = {256, 512, 768, 1024, 2048, 4096, 8192};

  for (int dim : dims) {
    const int n = dim;
    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dis(0.f, 1.f);

    Kokkos::View<float*> h_input("h_input", m * n);
    Kokkos::View<float*> d_output("d_output", m * n);
    Kokkos::View<float*> d_input("d_input", m * n);
    Kokkos::View<float*> d_bias("d_bias", n);
    Kokkos::View<float*> d_gamma("d_gamma", n);
    Kokkos::View<float*> d_beta("d_beta", n);

    auto h_in = Kokkos::create_mirror_view(d_input);
    auto h_bias = Kokkos::create_mirror_view(d_bias);
    auto h_gamma = Kokkos::create_mirror_view(d_gamma);
    auto h_beta = Kokkos::create_mirror_view(d_beta);

    for (int i = 0; i < m * n; i++) h_in(i) = dis(gen);
    for (int i = 0; i < n; i++) {
      h_bias(i) = dis(gen);
      h_gamma(i) = dis(gen);
      h_beta(i) = dis(gen);
    }
    Kokkos::deep_copy(d_input, h_in);
    Kokkos::deep_copy(d_bias, h_bias);
    Kokkos::deep_copy(d_gamma, h_gamma);
    Kokkos::deep_copy(d_beta, h_beta);

    // Initialize output to zero (residual add uses out[] directly)
    Kokkos::deep_copy(d_output, 0.0f);

    float eps = 1e-6f;
    addBiasResidualLayerNorm(d_output, d_input, d_bias, d_gamma, d_beta, eps, m, n, repeat);

    auto h_out = Kokkos::create_mirror_view(d_output);
    Kokkos::deep_copy(h_out, d_output);
    float s = 0;
    for (int i = 0; i < m * n; i++) s += h_out(i);
    printf("Checksum = %f\n", s / ((float)n * n));
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      Kokkos::finalize();
      return 1;
    }
    int repeat = atoi(argv[1]);
    printf("---------------- float32 (Kokkos port) -------------\n");
    layer(repeat);
  }
  Kokkos::finalize();
  return 0;
}
