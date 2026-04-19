// Rowwise Moments benchmark (Kokkos port)
// Computes per-row mean and reciprocal std-dev using Welford's algorithm
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>


int main(int argc, char* argv[])
{
  if (argc != 7) {
    printf("Usage: %s <batch> <channel> <width> <height> <group> <repeat>\n", argv[0]);
    return 1;
  }
  const int N      = atoi(argv[1]);
  const int C      = atoi(argv[2]);
  const int W      = atoi(argv[3]);
  const int H      = atoi(argv[4]);
  const int G      = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  const int64_t D = C / G;                // channels per group
  const float eps  = 1e-6f;

  const size_t input_size       = (size_t)N * C * W * H;
  const size_t output_size      = (size_t)N * G;
  const size_t input_size_bytes = input_size * sizeof(float);
  const size_t output_size_bytes = output_size * sizeof(float);

  float* h_X    = (float*)malloc(input_size_bytes);
  float* h_mean = (float*)malloc(output_size_bytes);
  float* h_rstd = (float*)malloc(output_size_bytes);

  srand(123);
  for (size_t i = 0; i < input_size; i++)
    h_X[i] = (float)rand() / RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_X   ("X",    input_size);
    Kokkos::View<float*> d_mean("mean", output_size);
    Kokkos::View<float*> d_rstd("rstd", output_size);

    {
      auto hv = Kokkos::create_mirror_view(d_X);
      for (size_t i = 0; i < input_size; i++) hv(i) = h_X[i];
      Kokkos::deep_copy(d_X, hv);
    }

    const int64_t row_len = D * H * W;
    const int     num_rows = N * G;

    Kokkos::fence();
    auto t_start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < repeat; iter++) {
      Kokkos::parallel_for("RowwiseMoments", num_rows,
        KOKKOS_LAMBDA(const int row) {
          float mean = 0.f, m2 = 0.f, nf = 0.f;
          for (int64_t j = 0; j < row_len; j++) {
            float x     = d_X[row * row_len + j];
            nf          += 1.f;
            float delta  = x - mean;
            mean         += delta / nf;
            float delta2 = x - mean;
            m2           += delta * delta2;
          }
          float divisor = nf > 0.f ? nf : 1.f;
          d_rstd[row]   = 1.f / sqrtf(m2 / divisor + eps);
          d_mean[row]   = mean;
        });
      Kokkos::fence();
    }

    auto t_end = std::chrono::steady_clock::now();
    auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Average execution time of RowwiseMoments kernel: %f (us)\n",
           (float)time * 1e-3f / repeat);

    {
      auto hv_mean = Kokkos::create_mirror_view(d_mean);
      auto hv_rstd = Kokkos::create_mirror_view(d_rstd);
      Kokkos::deep_copy(hv_mean, d_mean);
      Kokkos::deep_copy(hv_rstd, d_rstd);
      for (size_t i = 0; i < output_size; i++) {
        h_mean[i] = hv_mean(i);
        h_rstd[i] = hv_rstd(i);
      }
    }

    double avg_mean = 0.0, avg_rstd = 0.0;
    for (size_t i = 0; i < output_size; i++) {
      avg_mean += h_mean[i];
      avg_rstd += h_rstd[i];
    }
    avg_mean /= output_size;
    avg_rstd /= output_size;
    printf("Checksum: mean = %lf and rstd = %lf\n", avg_mean, avg_rstd);
  }
  Kokkos::finalize();

  free(h_X); free(h_mean); free(h_rstd);
  return 0;
}
