#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const float threshold = 0.4f;
  const int classwise_topK = 10;
  const int num_classes = 1000;
  const int num_priors = 1024;
  const int batch_size = 512;
  const int BINS = 2048;

  const size_t scores_size = (size_t)batch_size * num_classes * num_priors;
  const size_t indices_size = (size_t)batch_size * num_classes * classwise_topK;
  const size_t count_size = (size_t)batch_size * num_classes;

  float* scores = (float*) malloc(scores_size * sizeof(float));
  int*   count  = (int*)   malloc(count_size  * sizeof(int));
  int*   indices = (int*)  malloc(indices_size * sizeof(int));

  srand(123);
  for (size_t i = 0; i < scores_size; i++)
    scores[i] = rand() / (float)RAND_MAX;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> d_scores("d_scores", scores_size);
    Kokkos::View<int*>   d_count("d_count",   count_size);
    Kokkos::View<int*>   d_indices("d_indices", indices_size);

    {
      auto hs = Kokkos::create_mirror_view(d_scores);
      for (size_t i = 0; i < scores_size; i++) hs(i) = scores[i];
      Kokkos::deep_copy(d_scores, hs);
    }

    Kokkos::deep_copy(d_count,   0);
    Kokkos::deep_copy(d_indices, 0);

    const int nc = num_classes;
    const int np = num_priors;
    const int topK = classwise_topK;
    const float thr = threshold;
    const int bins_count = BINS;

    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      Kokkos::deep_copy(d_count,   0);
      Kokkos::deep_copy(d_indices, 0);

      Kokkos::parallel_for("FindTopK", batch_size * nc, KOKKOS_LAMBDA(const int bc) {
        const int b = bc / nc;
        const int c = bc % nc;

        const float* sc = d_scores.data() + (b * nc + c) * np;
        int* cnt = d_count.data() + b * nc + c;
        int* idx = d_indices.data() + (b * nc + c) * topK;

        // Local bins array
        int bins[2048] = {};

        // Count scores per bin (shift left by 1)
        for (int i = 0; i < np; i++) {
          const float confidence = sc[i];
          if (confidence > thr) {
            float conf_scaled = (confidence - thr) / (1.0f - thr);
            int bin_index = (int)(conf_scaled * bins_count);
            if (bin_index >= bins_count) bin_index = bins_count - 1;
            bin_index -= 1; // shift left
            if (bin_index >= 0)
              bins[bin_index]++;
          }
        }

        // Compute suffix sum of bins
        for (int i = bins_count - 2; i >= 0; i--)
          bins[i] += bins[i + 1];

        // Count total candidates above threshold
        int total = 0;
        for (int i = 0; i < np; i++) {
          if (sc[i] > thr) total++;
        }
        *cnt = (total < topK) ? total : topK;

        // Scatter indices using bins as position counters
        // bins[bin_index] now holds the suffix sum (number of elements in bins >= bin_index)
        // We use bins as output positions
        int written = 0;
        for (int i = 0; i < np && written < *cnt; i++) {
          const float confidence = sc[i];
          if (confidence > thr) {
            float conf_scaled = (confidence - thr) / (1.0f - thr);
            int bin_index = (int)(conf_scaled * bins_count);
            if (bin_index >= bins_count) bin_index = bins_count - 1;
            // Place index at current bin position
            int pos = bins_count - 1 - bin_index; // map from high confidence to front
            if (pos < topK) {
              idx[written++] = i;
            }
          }
        }
      });
      Kokkos::fence();
    }

    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    printf("Average kernel execution time: %f (ms)\n", ns * 1e-6 / repeat);

    // Copy back count and compute checksum
    auto h_count = Kokkos::create_mirror_view(d_count);
    Kokkos::deep_copy(h_count, d_count);

    long checksum = 0;
    for (int b = 0; b < batch_size; b++)
      for (int c = 0; c < num_classes; c++)
        checksum += h_count(b * num_classes + c);
    printf("Checksum (count) = %ld\n", checksum);
  }
  Kokkos::finalize();

  free(scores);
  free(count);
  free(indices);
  return 0;
}
