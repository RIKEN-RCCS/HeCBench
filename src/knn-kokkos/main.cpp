// K-Nearest Neighbors - Kokkos port
// Original: knn-sycl/main.cpp

#include <sys/time.h>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <Kokkos_Core.hpp>

#define BLOCK_DIM 16

// ---- Serial CPU reference ----
static float compute_distance(const float *ref, int ref_nb, const float *query,
                               int query_nb, int dim, int ref_idx, int query_idx) {
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    float diff = ref[d * ref_nb + ref_idx] - query[d * query_nb + query_idx];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

static void modified_insertion_sort(float *dist, int *index, int length, int k) {
  index[0] = 0;
  for (int i = 1; i < length; ++i) {
    float curr_dist  = dist[i];
    int   curr_index = i;
    if (i >= k && curr_dist >= dist[k - 1]) continue;
    int j = std::min(i, k - 1);
    while (j > 0 && dist[j - 1] > curr_dist) {
      dist[j]  = dist[j - 1];
      index[j] = index[j - 1];
      --j;
    }
    dist[j]  = curr_dist;
    index[j] = curr_index;
  }
}

static bool knn_serial(const float *ref, int ref_nb, const float *query,
                       int query_nb, int dim, int k,
                       float *knn_dist, int *knn_index) {
  float *dist  = (float*)malloc(ref_nb * sizeof(float));
  int   *index = (int*)  malloc(ref_nb * sizeof(int));
  if (!dist || !index) {
    printf("Memory allocation error\n");
    free(dist); free(index); return false;
  }
  for (int i = 0; i < query_nb; ++i) {
    for (int j = 0; j < ref_nb; ++j) {
      dist[j]  = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
      index[j] = j;
    }
    modified_insertion_sort(dist, index, ref_nb, k);
    for (int j = 0; j < k; ++j) {
      knn_dist [j * query_nb + i] = dist[j];
      knn_index[j * query_nb + i] = index[j];
    }
  }
  free(dist); free(index);
  return true;
}

// ---- Kokkos parallel KNN ----
static void knn_parallel(float *ref_host, int ref_nb,
                          float *query_host, int query_nb,
                          int dim, int k,
                          float *dist_host, int *ind_host) {
  // Device views
  Kokkos::View<float*> d_ref  ("d_ref",   (size_t)ref_nb   * dim);
  Kokkos::View<float*> d_query("d_query", (size_t)query_nb * dim);
  Kokkos::View<float*> d_dist ("d_dist",  (size_t)ref_nb   * query_nb);
  Kokkos::View<int*>   d_ind  ("d_ind",   (size_t)query_nb * k);

  {
    auto h = Kokkos::create_mirror_view(d_ref);
    for (int i = 0; i < ref_nb * dim; i++) h(i) = ref_host[i];
    Kokkos::deep_copy(d_ref, h);
  }
  {
    auto h = Kokkos::create_mirror_view(d_query);
    for (int i = 0; i < query_nb * dim; i++) h(i) = query_host[i];
    Kokkos::deep_copy(d_query, h);
  }

  // Kernel 1: Compute all pairwise squared distances
  Kokkos::parallel_for("ComputeDistanceGlobal",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ref_nb, query_nb}),
    KOKKOS_LAMBDA(int ref_idx, int query_idx) {
      float ssd = 0.f;
      for (int d = 0; d < dim; d++) {
        float diff = d_ref(d * ref_nb + ref_idx) - d_query(d * query_nb + query_idx);
        ssd += diff * diff;
      }
      d_dist(ref_idx * query_nb + query_idx) = ssd;
    });
  Kokkos::fence();

  // Kernel 2: Insertion sort each column (per query point)
  Kokkos::parallel_for("insertionSort",
    Kokkos::RangePolicy<>(0, query_nb),
    KOKKOS_LAMBDA(int xIndex) {
      float *p_dist = d_dist.data() + xIndex;
      int   *p_ind  = d_ind.data()  + xIndex;
      float max_dist;
      int   max_row;

      // Initialise first index
      p_ind[0]  = 0;
      max_dist  = p_dist[0];

      // Sort first k elements
      for (int l = 1; l < k; l++) {
        int   curr_row  = l * query_nb;
        float curr_dist = p_dist[curr_row];
        if (curr_dist < max_dist) {
          int i = l - 1;
          for (int a = 0; a < l - 1; a++) {
            if (p_dist[a * query_nb] > curr_dist) { i = a; break; }
          }
          for (int j = l; j > i; j--) {
            p_dist[j * query_nb] = p_dist[(j - 1) * query_nb];
            p_ind [j * query_nb] = p_ind [(j - 1) * query_nb];
          }
          p_dist[i * query_nb] = curr_dist;
          p_ind [i * query_nb] = l;
        } else {
          p_ind[l * query_nb] = l;
        }
        max_dist = p_dist[curr_row];
      }

      // Insert remaining elements
      max_row = (k - 1) * query_nb;
      for (int l = k; l < ref_nb; l++) {
        float curr_dist = p_dist[l * query_nb];
        if (curr_dist < max_dist) {
          int i = k - 1;
          for (int a = 0; a < k - 1; a++) {
            if (p_dist[a * query_nb] > curr_dist) { i = a; break; }
          }
          for (int j = k - 1; j > i; j--) {
            p_dist[j * query_nb] = p_dist[(j - 1) * query_nb];
            p_ind [j * query_nb] = p_ind [(j - 1) * query_nb];
          }
          p_dist[i * query_nb] = curr_dist;
          p_ind [i * query_nb] = l;
          max_dist = p_dist[max_row];
        }
      }
    });
  Kokkos::fence();

  // Kernel 3: Square root of first k rows
  Kokkos::parallel_for("parallelSqrt",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {k, query_nb}),
    KOKKOS_LAMBDA(int yIndex, int xIndex) {
      d_dist(yIndex * query_nb + xIndex) =
        Kokkos::sqrt(d_dist(yIndex * query_nb + xIndex));
    });
  Kokkos::fence();

  // Copy results back (only k * query_nb elements)
  {
    auto h_dist = Kokkos::create_mirror_view(d_dist);
    auto h_ind  = Kokkos::create_mirror_view(d_ind);
    Kokkos::deep_copy(h_dist, d_dist);
    Kokkos::deep_copy(h_ind,  d_ind);
    for (int i = 0; i < query_nb * k; i++) dist_host[i] = h_dist(i);
    for (int i = 0; i < query_nb * k; i++) ind_host[i]  = h_ind(i);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int iterations = atoi(argv[1]);

  int ref_nb   = 4096;
  int query_nb = 4096;
  int dim      = 68;
  int k        = 20;
  int c_iterations = 1;
  const float precision = 0.001f;
  int nb_correct_precisions = 0;
  int nb_correct_indexes    = 0;

  float *ref   = (float*)malloc(ref_nb   * dim * sizeof(float));
  float *query = (float*)malloc(query_nb * dim * sizeof(float));
  float *dist  = (float*)malloc(query_nb * k   * sizeof(float));
  int   *ind   = (int*)  malloc(query_nb * k   * sizeof(int));

  srand(2);
  for (int i = 0; i < ref_nb * dim;   i++) ref[i]   = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < query_nb * dim; i++) query[i] = (float)rand() / (float)RAND_MAX;

  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float *knn_dist  = (float*)malloc(query_nb * k * sizeof(float));
  int   *knn_index = (int*)  malloc(query_nb * k * sizeof(int));
  printf("Ground truth computation in progress...\n\n");
  if (!knn_serial(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
    free(ref); free(query); free(knn_dist); free(knn_index);
    return EXIT_FAILURE;
  }

  struct timeval tic, toc;
  float elapsed_time;

  printf("On CPU:\n");
  gettimeofday(&tic, NULL);
  for (int i = 0; i < c_iterations; i++)
    knn_serial(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  gettimeofday(&toc, NULL);
  elapsed_time = (toc.tv_sec - tic.tv_sec) + (toc.tv_usec - tic.tv_usec) / 1000000.f;
  printf(" done in %f s for %d iterations (%f s by iteration)\n",
         elapsed_time, c_iterations, elapsed_time / c_iterations);

  Kokkos::initialize(argc, argv);
  {
    printf("on Kokkos:\n");
    gettimeofday(&tic, NULL);
    for (int i = 0; i < iterations; i++)
      knn_parallel(ref, ref_nb, query, query_nb, dim, k, dist, ind);
    gettimeofday(&toc, NULL);
    elapsed_time = (toc.tv_sec - tic.tv_sec) + (toc.tv_usec - tic.tv_usec) / 1000000.f;
    printf(" done in %f s for %d iterations (%f s by iteration)\n",
           elapsed_time, iterations, elapsed_time / iterations);
  }
  Kokkos::finalize();

  for (int i = 0; i < query_nb * k; ++i) {
    if (fabsf(dist[i] - knn_dist[i]) <= precision)
      nb_correct_precisions++;
    if (ind[i] == knn_index[i])
      nb_correct_indexes++;
    else
      printf("Mismatch @index %d: %d %d\n", i, ind[i], knn_index[i]);
  }
  float precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  float index_accuracy     = nb_correct_indexes    / ((float)query_nb * k);
  printf("Precision accuracy %f\nIndex accuracy %f\n",
         precision_accuracy, index_accuracy);

  free(ind); free(dist); free(query); free(ref);
  free(knn_dist); free(knn_index);
  return 0;
}
