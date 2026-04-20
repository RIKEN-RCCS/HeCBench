// Port of wedford (Welford) CUDA benchmark to Kokkos
// Computes per-feature mean and biased variance via Welford's online algorithm

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <batch_size> <spatial_size> <feature_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size   = atoi(argv[1]);
  const int spatial_size = atoi(argv[2]);
  const int feature_size = atoi(argv[3]);
  const int repeat       = atoi(argv[4]);

  const size_t input_size = (size_t)batch_size * spatial_size * feature_size;
  std::vector<float> h_input(input_size);
  srand(123);
  for (size_t i = 0; i < input_size; i++) h_input[i] = rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_input("input", input_size);
    Kokkos::View<float*> d_mean ("mean",  feature_size);
    Kokkos::View<float*> d_var  ("var",   feature_size);

    {
      auto h = Kokkos::create_mirror_view(d_input);
      for (size_t i = 0; i < input_size; i++) h(i) = h_input[i];
      Kokkos::deep_copy(d_input, h);
    }

    Kokkos::fence();
    auto start = std::chrono::steady_clock::now();

    for (int rep = 0; rep < repeat; rep++) {
      // One thread per feature (block_idx.x equivalent)
      Kokkos::parallel_for("welford", feature_size, KOKKOS_LAMBDA(const int f) {
        int   count  = 0;
        float x_mean = 0.f;
        float m_2_n  = 0.f;

        for (int b = 0; b < batch_size; b++) {
          int base = f * spatial_size + b * spatial_size * feature_size;
          for (int s = 0; s < spatial_size; s++) {
            count++;
            float x_n = d_input(base + s);
            float d   = x_n - x_mean;
            x_mean   += d / count;
            m_2_n    += d * (x_n - x_mean);
          }
        }
        d_mean(f) = x_mean;
        d_var (f) = (count > 0) ? m_2_n / count : 0.f;
      });
      Kokkos::fence();
    }

    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

    auto h_var  = Kokkos::create_mirror_view(d_var);
    auto h_mean = Kokkos::create_mirror_view(d_mean);
    Kokkos::deep_copy(h_var,  d_var);
    Kokkos::deep_copy(h_mean, d_mean);

    double avg_var = 0.0, avg_mean = 0.0;
    for (int i = 0; i < feature_size; i++) {
      avg_var  += h_var(i);
      avg_mean += h_mean(i);
    }
    avg_var  /= feature_size;
    avg_mean /= feature_size;
    printf("Checksum: mean = %f and variance = %f\n", avg_var, avg_mean);
  }
  Kokkos::finalize();
  return 0;
}
