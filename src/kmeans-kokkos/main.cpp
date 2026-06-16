#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int npoints   = 1000;
    const int nfeatures = 34;
    const int nclusters = 5;
    const int nloops    = 3;
    const float threshold = 0.001f;

    // Synthetic feature data
    std::vector<float> h_feature_vec(npoints * nfeatures);
    for (int i = 0; i < npoints; i++)
      for (int j = 0; j < nfeatures; j++)
        h_feature_vec[i * nfeatures + j] = (float)sin(i * 0.1 + j * 0.3);

    // Device Views
    Kokkos::View<float*> d_feature("feature", npoints * nfeatures);
    Kokkos::View<float*> d_feature_swap("feature_swap", nfeatures * npoints);
    Kokkos::View<float*> d_cluster("cluster", nclusters * nfeatures);
    Kokkos::View<int*>   d_membership("membership", npoints);

    auto h_feature      = Kokkos::create_mirror_view(d_feature);
    auto h_feature_swap = Kokkos::create_mirror_view(d_feature_swap);
    auto h_cluster      = Kokkos::create_mirror_view(d_cluster);
    auto h_membership   = Kokkos::create_mirror_view(d_membership);

    // Copy feature data to device
    for (int i = 0; i < npoints * nfeatures; i++) h_feature(i) = h_feature_vec[i];
    Kokkos::deep_copy(d_feature, h_feature);

    // Transpose kernel
    Kokkos::parallel_for("transpose", npoints, KOKKOS_LAMBDA(int tid) {
      for (int i = 0; i < nfeatures; i++)
        d_feature_swap(i * npoints + tid) = d_feature(tid * nfeatures + i);
    });
    Kokkos::fence();

    std::vector<int> h_membership_vec(npoints, -1);
    std::vector<float> h_cluster_vec(nclusters * nfeatures);

    auto t_start = std::chrono::steady_clock::now();

    for (int lp = 0; lp < nloops; lp++) {
      // Initialize cluster centers from first nclusters points
      for (int i = 0; i < nclusters; i++)
        for (int j = 0; j < nfeatures; j++)
          h_cluster_vec[i * nfeatures + j] = h_feature_vec[i * nfeatures + j];

      // Reset membership
      for (int i = 0; i < npoints; i++) h_membership_vec[i] = -1;

      // Copy cluster to device
      for (int i = 0; i < nclusters * nfeatures; i++) h_cluster(i) = h_cluster_vec[i];
      Kokkos::deep_copy(d_cluster, h_cluster);

      int loop = 0;
      float delta = 0.0f;
      do {
        delta = 0.0f;

        // Assignment kernel: find nearest cluster for each point
        Kokkos::parallel_for("assign", npoints, KOKKOS_LAMBDA(int point_id) {
          float min_dist = FLT_MAX;
          int index = 0;
          for (int i = 0; i < nclusters; i++) {
            float dist = 0.0f;
            for (int l = 0; l < nfeatures; l++) {
              float diff = d_feature_swap(l * npoints + point_id) - d_cluster(i * nfeatures + l);
              dist += diff * diff;
            }
            if (dist < min_dist) { min_dist = dist; index = i; }
          }
          d_membership(point_id) = index;
        });
        Kokkos::fence();

        // Copy membership back to host
        Kokkos::deep_copy(h_membership, d_membership);

        // Update cluster centers on host
        std::vector<int> new_centers_len(nclusters, 0);
        std::vector<float> new_centers(nclusters * nfeatures, 0.0f);

        for (int i = 0; i < npoints; i++) {
          int cluster_id = h_membership(i);
          new_centers_len[cluster_id]++;
          if (h_membership(i) != h_membership_vec[i]) {
            delta++;
            h_membership_vec[i] = h_membership(i);
          }
          for (int j = 0; j < nfeatures; j++)
            new_centers[cluster_id * nfeatures + j] += h_feature_vec[i * nfeatures + j];
        }

        for (int i = 0; i < nclusters; i++) {
          for (int j = 0; j < nfeatures; j++) {
            if (new_centers_len[i] > 0)
              h_cluster_vec[i * nfeatures + j] = new_centers[i * nfeatures + j] / new_centers_len[i];
          }
        }

        // Copy updated clusters back to device
        for (int i = 0; i < nclusters * nfeatures; i++) h_cluster(i) = h_cluster_vec[i];
        Kokkos::deep_copy(d_cluster, h_cluster);

      } while (delta > threshold && loop++ < 500);

      printf("Loop %d converged in %d iterations, delta=%f\n", lp, loop, delta);
    }

    auto t_end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Total time: %f s\n", time * 1e-9);

    // Print cluster sizes
    std::vector<int> cluster_sizes(nclusters, 0);
    for (int i = 0; i < npoints; i++) cluster_sizes[h_membership_vec[i]]++;
    for (int i = 0; i < nclusters; i++)
      printf("Cluster %d: %d points\n", i, cluster_sizes[i]);
  }
  Kokkos::finalize();
  return 0;
}
