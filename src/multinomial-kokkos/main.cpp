// multinomial-kokkos/main.cpp
// Port of multinomial-cuda: samples from multinomial distributions via CDF.
//
// Kokkos::parallel_for over distributions; each work item does the sequential
// prefix-sum scan over categories to find the sampled bucket.

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference (from reference.h)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename accscalar_t>
void sampleMultinomialOnce_cpu(
    int*           dest,
    int            distributions,
    int            categories,
    const scalar_t* sampled,
    const scalar_t* dist,
    int            stride_dist,
    int            stride_categories)
{
  for (int curDist = 0; curDist < distributions; ++curDist) {
    accscalar_t sum = accscalar_t{0};
    for (int cat = 0; cat < categories; ++cat)
      sum += static_cast<accscalar_t>(
          dist[curDist * stride_dist + cat * stride_categories]);

    if (sum == accscalar_t{0}) { dest[curDist] = 0; continue; }

    scalar_t  sample     = sampled[curDist];
    accscalar_t prevBucket = accscalar_t{0};
    accscalar_t curBucket  = accscalar_t{0};
    bool found = false;
    int  foundPos = 0;

    for (int cat = 0; cat < categories; ++cat) {
      accscalar_t dv = static_cast<accscalar_t>(
                           dist[curDist * stride_dist + cat * stride_categories])
                       / sum;
      prevBucket = curBucket;
      curBucket  = prevBucket + dv;
      if (sample < curBucket && sample >= prevBucket && dv > scalar_t{0}) {
        foundPos = cat; found = true; break;
      }
    }
    if (found) {
      dest[curDist] = foundPos;
    } else {
      for (int cat = categories - 1; cat >= 0; --cat) {
        if (dist[curDist * stride_dist + cat * stride_categories] > scalar_t{0}) {
          dest[curDist] = cat; break;
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <numDist> <numCategories> <repeat>\n", argv[0]);
    return 1;
  }
  const int numDist       = std::atoi(argv[1]);
  const int numCategories = std::atoi(argv[2]);
  const int repeat        = std::atoi(argv[3]);

  // Host data
  std::vector<float> h_sample(numDist);
  {
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (int i = 0; i < numDist; ++i) h_sample[i] = ud(rng);
  }

  std::vector<float> h_dist(static_cast<std::size_t>(numDist) * numCategories);
  {
    srand(123);
    for (int i = 0; i < numDist * numCategories; ++i)
      h_dist[i] = static_cast<float>(rand() % 100 + 1);
  }

  std::vector<int> h_result(numDist, 0);
  std::vector<int> h_result_ref(numDist, 0);

  Kokkos::initialize(argc, argv);
  {
    // Device views
    Kokkos::View<float*> d_sample("d_sample", numDist);
    Kokkos::View<float*> d_dist("d_dist", numDist * numCategories);
    Kokkos::View<int*>   d_result("d_result", numDist);

    {
      auto ms = Kokkos::create_mirror_view(d_sample);
      auto md = Kokkos::create_mirror_view(d_dist);
      for (int i = 0; i < numDist; ++i) ms(i) = h_sample[i];
      for (int i = 0; i < numDist * numCategories; ++i) md(i) = h_dist[i];
      Kokkos::deep_copy(d_sample, ms);
      Kokkos::deep_copy(d_dist,   md);
    }

    const int cats = numCategories;

    auto run_kernel = [&]() {
      Kokkos::parallel_for(
          "sampleMultinomial", numDist,
          KOKKOS_LAMBDA(int curDist) {
            // Sum of distribution weights
            float sum = 0.f;
            for (int cat = 0; cat < cats; ++cat)
              sum += d_dist[curDist * cats + cat];

            if (sum == 0.f) { d_result(curDist) = 0; return; }

            float sample     = d_sample(curDist);
            float prevBucket = 0.f;
            float curBucket  = 0.f;
            bool  found      = false;

            for (int cat = 0; cat < cats; ++cat) {
              float dv   = d_dist[curDist * cats + cat] / sum;
              prevBucket = curBucket;
              curBucket  = prevBucket + dv;
              if (sample < curBucket && sample >= prevBucket && dv > 0.f) {
                d_result(curDist) = cat;
                found = true;
                break;
              }
            }
            if (!found) {
              for (int cat = cats - 1; cat >= 0; --cat) {
                if (d_dist[curDist * cats + cat] > 0.f) {
                  d_result(curDist) = cat;
                  break;
                }
              }
            }
          });
    };

    // Warmup + verify
    run_kernel();
    Kokkos::fence();
    sampleMultinomialOnce_cpu<float, float>(
        h_result_ref.data(), numDist, numCategories,
        h_sample.data(), h_dist.data(), numCategories, 1);
    {
      auto mr = Kokkos::create_mirror_view(d_result);
      Kokkos::deep_copy(mr, d_result);
      for (int i = 0; i < numDist; ++i) h_result[i] = mr(i);
    }

    bool error = false;
    for (int i = 0; i < numDist; ++i) {
      if (std::abs(h_result[i] - h_result_ref[i]) > 1) {
        printf("results mismatch: dist=%d kokkos=%d ref=%d\n",
               i, h_result[i], h_result_ref[i]);
        error = true;
        break;
      }
    }
    printf("%s\n", error ? "FAIL" : "PASS");

    // Timed run
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) run_kernel();
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    printf("Average execution time of sampleMultinomialOnce kernel: %f (us)\n",
           std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
               * 1e-3 / repeat);
  }
  Kokkos::finalize();
  return 0;
}
