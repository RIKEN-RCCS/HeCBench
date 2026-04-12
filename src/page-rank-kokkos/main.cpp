#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <chrono>

#define D_FACTOR (0.85f)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

const int max_iter = 1000;
const float threshold = 1e-16f;

int *random_pages(int n, unsigned int *noutlinks, int divisor) {
  int *pages = (int *)malloc(sizeof(int) * n * n);
  if (divisor <= 0) {
    fprintf(stderr, "ERROR: Invalid divisor '%d'\n", divisor);
    exit(1);
  }
  for (int i = 0; i < n; ++i) {
    noutlinks[i] = 0;
    for (int j = 0; j < n; ++j) {
      if (i != j && (abs(rand()) % divisor == 0)) {
        pages[i * n + j] = 1;
        noutlinks[i] += 1;
      }
    }
    if (noutlinks[i] == 0) {
      int k;
      do { k = abs(rand()) % n; } while (k == i);
      pages[i * n + k] = 1;
      noutlinks[i] = 1;
    }
  }
  return pages;
}

void init_array(float *a, int n, float val) {
  for (int i = 0; i < n; ++i) a[i] = val;
}

float maximum_dif(float *difs, int n) {
  float max = 0.0f;
  for (int i = 0; i < n; ++i)
    max = difs[i] > max ? difs[i] : max;
  return max;
}

void usage(char *argv[]) {
  fprintf(stderr,
    "Usage: %s [-n number of pages] [-i max iterations]"
    " [-t threshold] [-q divisor for zero density]\n", argv[0]);
}

static struct option size_opts[] = {
  {"number of pages", 1, NULL, 'n'},
  {"max number of iterations", 1, NULL, 'i'},
  {"minimum threshold", 1, NULL, 't'},
  {"divisor for zero density", 1, NULL, 'q'},
  {0, 0, 0, 0}
};

int main(int argc, char *argv[]) {
  int n = 1000;
  int iter = max_iter;
  float thresh = threshold;
  int divisor = 2;

  int opt, opt_index = 0;
  while ((opt = getopt_long(argc, argv, "::n:i:t:q:", size_opts, &opt_index)) != -1) {
    switch (opt) {
      case 'n': n = atoi(optarg); break;
      case 'i': iter = atoi(optarg); break;
      case 't': thresh = atof(optarg); break;
      case 'q': divisor = atoi(optarg); break;
      default: usage(argv); exit(EXIT_FAILURE);
    }
  }

  Kokkos::initialize(argc, argv);
  {
    float *page_ranks = (float *)malloc(sizeof(float) * n);
    unsigned int *noutlinks = (unsigned int *)malloc(sizeof(unsigned int) * n);
    for (int i = 0; i < n; ++i) noutlinks[i] = 0;

    int *pages = random_pages(n, noutlinks, divisor);
    init_array(page_ranks, n, 1.0f / (float)n);

    float *diffs = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) diffs[i] = 0.0f;

    // Device views
    Kokkos::View<int *> d_pages("pages", n * n);
    Kokkos::View<float *> d_page_ranks("page_ranks", n);
    Kokkos::View<unsigned int *> d_noutlinks("noutlinks", n);
    Kokkos::View<float *> d_maps("maps", n * n);
    Kokkos::View<float *> d_diffs("diffs", n);

    auto h_pages = Kokkos::create_mirror_view(d_pages);
    auto h_page_ranks = Kokkos::create_mirror_view(d_page_ranks);
    auto h_noutlinks = Kokkos::create_mirror_view(d_noutlinks);
    auto h_diffs = Kokkos::create_mirror_view(d_diffs);

    for (int i = 0; i < n * n; i++) h_pages(i) = pages[i];
    for (int i = 0; i < n; i++) h_page_ranks(i) = page_ranks[i];
    for (int i = 0; i < n; i++) h_noutlinks(i) = noutlinks[i];
    for (int i = 0; i < n; i++) h_diffs(i) = 0.0f;

    Kokkos::deep_copy(d_pages, h_pages);
    Kokkos::deep_copy(d_page_ranks, h_page_ranks);
    Kokkos::deep_copy(d_noutlinks, h_noutlinks);
    Kokkos::deep_copy(d_diffs, h_diffs);

    float max_diff = 99.0f;
    int t = 1;
    double ktime = 0.0;

    for (; t <= iter && max_diff >= thresh; ++t) {
      auto start = std::chrono::high_resolution_clock::now();

      int local_n = n;
      Kokkos::parallel_for("outbound", local_n, KOKKOS_LAMBDA(const int i) {
        float outbound_rank = d_page_ranks(i) / (float)d_noutlinks(i);
        for (int j = 0; j < local_n; ++j)
          d_maps(i * local_n + j) = d_pages(i * local_n + j) * outbound_rank;
      });
      Kokkos::fence();

      Kokkos::parallel_for("rank_update", local_n, KOKKOS_LAMBDA(const int j) {
        float old_rank = d_page_ranks(j);
        float new_rank = 0.0f;
        for (int i = 0; i < local_n; ++i)
          new_rank += d_maps(i * local_n + j);
        new_rank = ((1.f - D_FACTOR) / local_n) + (D_FACTOR * new_rank);
        d_diffs(j) = fmaxf(fabsf(new_rank - old_rank), d_diffs(j));
        d_page_ranks(j) = new_rank;
      });
      Kokkos::fence();

      auto end = std::chrono::high_resolution_clock::now();
      ktime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

      Kokkos::deep_copy(h_diffs, d_diffs);
      for (int i = 0; i < n; i++) diffs[i] = h_diffs(i);
      max_diff = maximum_dif(diffs, n);

      // Reset diffs
      Kokkos::parallel_for("reset_diffs", local_n, KOKKOS_LAMBDA(const int i) {
        d_diffs(i) = 0.f;
      });
      Kokkos::fence();
    }

    fprintf(stderr, "Max difference %f is reached at iteration %d\n", max_diff, t);
    printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n",
           n, iter, thresh, ktime);

    free(pages);
    free(page_ranks);
    free(noutlinks);
    free(diffs);
  }
  Kokkos::finalize();
  return 0;
}
